"""Reconstruct the public ADB brief and compare sampling with exact contractions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.linalg import expm
from scipy.stats import qmc, t

ROOT = Path(__file__).resolve().parents[1]
STATES = np.array(list(np.ndindex(5, 5, 2, 2)), dtype=int)
CAP = np.array([4, 4, 1, 1])
COMBINE = np.ravel_multi_index(
    np.minimum(STATES[:, None, :] + STATES[None, :, :], CAP).transpose(2, 0, 1),
    (5, 5, 2, 2),
)
TARGETS = [
    "term_expected_principal_loss_usd",
    "term_joint_attachment_probability",
    "term_portfolio_loss_exceedance_probability",
    "term_portfolio_stop_loss_usd",
    "term_both_bonds_exhausted_probability",
]
PUBLISHED = {
    "adb-base": [10.06e6, .04414, .01205, 282355, .000032],
    "adb-independent": [10.06e6, .02716, .00719, 171266, .000004],
    "adb-common-shock": [10.06e6, .05053, .01607, 385043, .000044],
    "adb-rain-stress": [11.17e6, .07046, .01322, 306286, .000069],
    "adb-rain-cover": [9.97e6, .04414, .00700, 170927, .000032],
}


def compound_rates(tails):
    t = np.array(tails, dtype=float)
    p0 = 1 - t[0]
    p1, p2, p3 = t[:-1] - t[1:]
    total = -np.log(p0)
    l1 = p1 / p0
    l2 = p2 / p0 - l1**2 / 2
    l3 = p3 / p0 - l1 * l2 - l1**3 / 6
    rates = np.array([l1, l2, l3, total - l1 - l2 - l3])
    assert np.all(rates >= 0)
    return rates, total


def annual_model(scenario):
    rates1, l1 = compound_rates([.03883, .02331, .01790, .01184])
    rates2, l2 = compound_rates([.03925, .01743, .01332, .01043])
    multiplier = {"adb-independent": 0, "adb-common-shock": 2}.get(scenario, 1)
    common = multiplier * (.075 * l1 + .074 * l2) / 2
    assert common <= min(l1, l2)
    w1, w2 = rates1 / l1, rates2 / l2
    generator = np.zeros((25, 25))
    for a in range(5):
        for b in range(5):
            idx = a * 5 + b
            for k in range(1, 5):
                generator[idx, min(a+k, 4)*5+b] += (l1-common)*w1[k-1]
                generator[idx, a*5+min(b+k, 4)] += (l2-common)*w2[k-1]
                for j in range(1, 5):
                    generator[idx, min(a+k, 4)*5+min(b+j, 4)] += common*w1[k-1]*w2[j-1]
            generator[idx, idx] -= generator[idx].sum()
    eq = expm(generator)[0].reshape(5, 5)
    rain_rate = -np.log(.98) * (2 if scenario == "adb-rain-stress" else 1)
    shared_rain = 0 if scenario == "adb-independent" else .25 * rain_rate
    p00 = np.exp(-2*rain_rate + shared_rain)
    p10 = np.exp(-rain_rate) - p00
    rain = np.array([[p00, p10], [p10, 1-p00-2*p10]])
    annual = (eq[:, :, None, None] * rain[None, None, :, :]).ravel()
    assert abs(annual.sum()-1) < 1e-12 and annual.min() >= 0
    return annual, {"earthquake_rates": [rates1.tolist(), rates2.tolist()],
                    "shared_earthquake_rate": common, "rain_rate": rain_rate,
                    "shared_rain_rate": shared_rain}


def payout_features(states, scenario):
    eq_cap, rain_cap = (60e6, 20e6) if scenario == "adb-rain-cover" else (70e6, 10e6)
    one = states[..., 0] * eq_cap / 4 + states[..., 2] * rain_cap
    two = states[..., 1] * eq_cap / 4 + states[..., 3] * rain_cap
    loss = one + two
    return np.stack([loss, (one > 0) & (two > 0), loss > 80e6,
                     np.maximum(loss-80e6, 0), (one == 80e6) & (two == 80e6)], axis=-1)


def exact_term(annual):
    term = np.zeros(100)
    term[0] = 1
    for _ in range(3):
        term = np.bincount(COMBINE.ravel(), weights=(term[:, None]*annual).ravel(), minlength=100)
    return term


def factorized_term(annual):
    """Use independence between peril blocks, preserving country dependence."""
    cube = annual.reshape(25, 4)
    eq, rain = cube.sum(axis=1), cube.sum(axis=0)
    assert np.max(abs(cube - eq[:, None]*rain)) < 1e-12
    eq_states = np.array(list(np.ndindex(5, 5)))
    rain_states = np.array(list(np.ndindex(2, 2)))
    eq_map = np.ravel_multi_index(np.minimum(eq_states[:, None]+eq_states, 4).transpose(2, 0, 1), (5, 5))
    rain_map = np.ravel_multi_index(np.minimum(rain_states[:, None]+rain_states, 1).transpose(2, 0, 1), (2, 2))
    products = []
    for probabilities, mapping in [(eq, eq_map), (rain, rain_map)]:
        term = np.zeros(len(probabilities))
        term[0] = 1
        for _ in range(3):
            term = np.bincount(mapping.ravel(), weights=(term[:, None]*probabilities).ravel(), minlength=len(term))
        products.append(term)
    return np.outer(*products).ravel()


def paths_from_uniform(u, distribution):
    return np.minimum(np.searchsorted(np.cumsum(distribution), u), 99)


def sample_estimate(method, budget, seed, annual, scenario, return_scrambles=False):
    rng = np.random.default_rng(seed)
    if method == "ScrambledSobol":
        estimates = []
        for scramble in range(8):
            draws = qmc.Sobol(3, scramble=True, seed=seed*8+scramble).random_base2(int(np.log2(budget//8)))
            idx = paths_from_uniform(draws, annual)
            end = np.minimum(STATES[idx].sum(axis=1), CAP)
            estimates.append(payout_features(end, scenario).mean(axis=0))
        values = np.array(estimates)
        return values if return_scrambles else values.mean(axis=0)
    if method == "MonteCarlo":
        draws = rng.random((budget, 3))
        idx = paths_from_uniform(draws, annual)
        end = np.minimum(STATES[idx].sum(axis=1), CAP)
        return payout_features(end, scenario).mean(axis=0)
    if method == "ImportanceSampling":
        state_loss = payout_features(STATES, scenario)[:, 0]
        proposal = .5*annual + .5*annual*state_loss/(annual @ state_loss)
        idx = paths_from_uniform(rng.random((budget, 3)), proposal)
        weight = np.prod(annual[idx]/proposal[idx], axis=1)
        end = np.minimum(STATES[idx].sum(axis=1), CAP)
        return np.mean(payout_features(end, scenario)*weight[:, None], axis=0)
    if method == "StratifiedSampling":
        p0 = annual[0]
        conditional = annual.copy()
        conditional[0] = 0
        conditional /= conditional.sum()
        result = np.zeros(5)
        for first in range(3):
            n = budget//3 + (first < budget % 3)
            idx = np.zeros((n, 3), dtype=int)
            idx[:, first] = paths_from_uniform(rng.random(n), conditional)
            if first < 2:
                idx[:, first+1:] = paths_from_uniform(rng.random((n, 2-first)), annual)
            end = np.minimum(STATES[idx].sum(axis=1), CAP)
            result += p0**first * (1-p0) * payout_features(end, scenario).mean(axis=0)
        return result
    raise ValueError(method)


def run(budget=4096, repeats=32):
    results = {}
    for scenario in PUBLISHED:
        begin = time.perf_counter_ns()
        annual, calibration = annual_model(scenario)
        setup_ms = (time.perf_counter_ns()-begin)/1e6
        start = time.perf_counter_ns()
        term = exact_term(annual)
        exact_ms = (time.perf_counter_ns()-start)/1e6
        start = time.perf_counter_ns()
        factorized = factorized_term(annual)
        factor_ms = (time.perf_counter_ns()-start)/1e6
        features = payout_features(STATES, scenario)
        truth = term @ features
        assert np.max(abs(term-factorized)) < 1e-12
        assert np.allclose(term.sum(), 1)
        methods = {}
        for method in ["MonteCarlo", "ScrambledSobol", "StratifiedSampling", "ImportanceSampling"]:
            start = time.perf_counter_ns()
            if method == 'ScrambledSobol':
                scramble_estimates = np.array([sample_estimate(method, budget, 1729+r, annual, scenario, True) for r in range(repeats)])
                estimates = scramble_estimates.mean(axis=1)
            else:
                estimates = np.array([sample_estimate(method, budget, 1729+r, annual, scenario) for r in range(repeats)])
            runtime = (time.perf_counter_ns()-start)/1e6/repeats
            rmse = np.sqrt(np.mean((estimates-truth)**2, axis=0))
            methods[method] = {
                "mean_estimate": dict(zip(TARGETS, estimates.mean(axis=0).tolist())),
                "rmse": dict(zip(TARGETS, rmse.tolist())),
                "relative_rmse": dict(zip(TARGETS, (rmse/truth).tolist())),
                "zero_exhaustion_estimate_replicates": int(np.sum(estimates[:, 4] == 0)),
                "mean_runtime_ms": runtime,
                "replicate_estimates": estimates.tolist(),
                "paths_per_replicate": budget,
                "annual_state_draw_upper_bound": 3*budget,
            }
            if method == 'ScrambledSobol':
                halfwidth = t.ppf(.975,7)*scramble_estimates.std(axis=1,ddof=1)/np.sqrt(8)
                methods[method]['scrambles_per_replicate'] = 8
                methods[method]['paths_per_scramble'] = budget//8
                methods[method]['t_interval_degrees_of_freedom'] = 7
                methods[method]['replicate_95_ci_halfwidth'] = halfwidth.tolist()
                methods[method]['empirical_95_ci_coverage'] = np.mean(abs(estimates-truth) <= halfwidth,axis=0).tolist()
                methods[method]['raw_scramble_estimates'] = scramble_estimates.tolist()
        pub = np.array(PUBLISHED[scenario])
        # The PDF reports different decimal precision for each column.
        rounding_tolerance = np.array([5000, .000005, .000005, .51, .0000005])
        matches = abs(truth-pub) <= rounding_tolerance
        results[scenario] = {
            "calibration": calibration,
            "exact_targets": dict(zip(TARGETS, truth.tolist())),
            "pdf_rounded_targets": dict(zip(TARGETS, pub.tolist())),
            "pdf_rounding_match": dict(zip(TARGETS, matches.tolist())),
            "pdf_target_differences": dict(zip(TARGETS, (truth-pub).tolist())),
            "annual_zero_payout_probability": float(annual[0]),
            "annual_setup_ms": setup_ms,
            "full_100_state_convolution_ms": exact_ms,
            "peril_factorized_tensor_contraction_ms": factor_ms,
            "factorization_max_probability_error": float(np.max(abs(term-factorized))),
            "factorized_target_absolute_error": dict(zip(TARGETS, abs((factorized-term)@features).tolist())),
            "term_state_probabilities": term.tolist(),
            "sampling_baselines": methods,
        }
    return {
        "schema": "adb.official.s1.v1",
        "source": "benchmark/pyproject.toml",
        "source_sha256": hashlib.sha256((ROOT/'benchmark/pyproject.toml').read_bytes()).hexdigest(),
        "basis": "Official illustrative catastrophe model, not observed historical economic losses",
        "construction": "Equations 2 through 7, published annual tails with full floating point precision",
        "quantum_hardware_used": False,
        "determinism": "Fixed seeds and deterministic contractions; wall time is measured, not deterministic",
        "cost_scope": "Annual distribution setup is timed separately; exact methods access full model probabilities",
        "factorized_method": "Exact independent peril tensor contraction, a classical structural control, not learned MPS",
        "sampling_ci": "32 independent randomizations by default; raw replicates retained for uncertainty calculation",
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__},
        "budget_paths": budget, "repeats": repeats, "targets": TARGETS, "scenarios": results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=4096)
    parser.add_argument('--repeats', type=int, default=32)
    args = parser.parse_args()
    if args.budget < 8 or args.budget & (args.budget-1):
        parser.error('budget must be a power of two of at least eight')
    result = run(args.budget, args.repeats)
    output = ROOT/'data/probe_official_s1_results.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({"output": str(output), "base": result['scenarios']['adb-base']['exact_targets']}))
