"""Chronological loss probe for a margin preserving MPS plus extreme value copula."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats

import probe_s1_baselines as s1

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261118
COUNTRIES = ["PHL", "VNM", "THA", "IDN"]
MIN_MPS_WEIGHT = .05
MIN_EV_WEIGHT = .01
MIN_EV_THETA = 1.05
MOMENT_N = 16384
METRICS = {
    "tail_joint_log_likelihood": 1,
    "joint_exceedance_95_absolute_error": -1,
    "joint_exceedance_99_absolute_error": -1,
    "var_99_coverage_error": -1,
    "tvar_99_calibration_moment_error": -1,
    "brier_95": -1,
    "energy_score_loss_gdp": -1,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def open_unit(x):
    return np.clip(np.asarray(x, dtype=float), np.nextafter(0., 1.), np.nextafter(1., 0.))


def uniforms(n, d, seed=SEED):
    if int(n) != n or n < 1:
        raise ValueError("Sample count must be a positive integer")
    return open_unit(stats.qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n])


class Margin(s1.Margin):
    """Keep S1 fitted margins and evaluate the exact GPD inverse at tail endpoints."""
    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        if np.any(~np.isfinite(u)) or np.any((u < 0) | (u > 1)):
            raise ValueError("Marginal probability must lie in the closed unit interval")
        shape = u.shape
        values = np.atleast_1d(u)
        result = np.asarray(np.interp(values, self.probs, self.knots))
        result = np.where(values <= self.pzero, 0., result)
        tail = values > self.pt
        if np.any(tail):
            excess_probability = np.clip((values[tail] - self.pt) / (1 - self.pt), 0., 1.)
            result[tail] = self.threshold + stats.genpareto.ppf(excess_probability, self.xi, scale=self.scale)
        result = result.reshape(shape)
        return result.item() if result.ndim == 0 else result

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        return np.where(x < 0, 0., super().cdf(x))


def pit(x, margins, seed):
    q = uniforms(len(x), len(margins), seed)
    u = np.column_stack([m.cdf(x[:, j]) for j, m in enumerate(margins)])
    for j, m in enumerate(margins):
        zero = x[:, j] == 0
        u[zero, j] = q[zero, j] * m.pzero
    return open_unit(u)


def true_connected4(u):
    """Empirically centered four dimensional connected PIT cumulant."""
    z = (np.asarray(u) - .5) * np.sqrt(12.)
    z = z - z.mean(axis=0)
    pair = {(a, b): np.mean(z[:, a] * z[:, b]) for a, b in itertools.combinations(range(z.shape[1]), 2)}
    return np.asarray([
        np.mean(z[:, a] * z[:, b] * z[:, c] * z[:, d])
        - pair[a, b] * pair[c, d] - pair[a, c] * pair[b, d] - pair[a, d] * pair[b, c]
        for a, b, c, d in itertools.combinations(range(z.shape[1]), 4)
    ])


def population_moments(u):
    z = (np.asarray(u) - .5) * np.sqrt(12.)
    d = z.shape[1]
    pair = {(a, b): float(np.mean(z[:, a] * z[:, b])) for a, b in itertools.combinations(range(d), 2)}
    four = {p: float(np.mean(np.prod(z[:, p], axis=1))) for p in itertools.combinations(range(d), 4)}
    return pair, four


def cp_moments(mps):
    centers = ((np.arange(mps.bins) + .5) / mps.bins - .5) * np.sqrt(12.)
    means = np.einsum("hjb,b->hj", mps.f, centers)
    pair = {(a, b): float(mps.w @ (means[:, a] * means[:, b])) for a, b in itertools.combinations(range(mps.d), 2)}
    four = {p: float(mps.w @ np.prod(means[:, p], axis=1)) for p in itertools.combinations(range(mps.d), 4)}
    return pair, four


def cumulant_of_mixture(weights, moments):
    """Closed mixture cumulant for exactly uniform marginal coordinates."""
    pair = {p: sum(w * m[0][p] for w, m in zip(weights, moments)) for p in moments[0][0]}
    four = {p: sum(w * m[1][p] for w, m in zip(weights, moments)) for p in moments[0][1]}
    return np.asarray([
        four[(a, b, c, d)] - pair[a, b] * pair[c, d] - pair[a, c] * pair[b, d] - pair[a, d] * pair[b, c]
        for a, b, c, d in four
    ])


def refine_mps(mps, tolerance=2e-15):
    """Refine only the new S2 instance; S1 arrays and files stay unchanged."""
    for iteration in range(2000):
        for j in range(mps.d):
            marginal = mps.w @ mps.f[:, j, :]
            mps.f[:, j, :] *= (1 / mps.bins) / marginal
            normalizer = mps.f[:, j, :].sum(axis=1)
            mps.f[:, j, :] /= normalizer[:, None]
            mps.w *= normalizer
            mps.w /= mps.w.sum()
        error = np.max(np.abs(np.einsum("h,hjb->jb", mps.w, mps.f) - 1 / mps.bins))
        if error < tolerance:
            break
    if error > 1e-13:
        raise ArithmeticError("Uniform marginal IPF did not converge")
    mps.marginal_max_error = float(error)
    mps.ipf_iterations += iteration + 1
    mps.cores = [(mps.w[:, None] * mps.f[:, 0, :]).T[None, :, :]]
    for j in range(1, mps.d - 1):
        core = np.zeros((mps.rank, mps.bins, mps.rank))
        for h in range(mps.rank):
            core[h, :, h] = mps.f[h, j, :]
        mps.cores.append(core)
    mps.cores.append(mps.f[:, -1, :, None])


class MPSEVCumulant:
    """A nonnegative MPS with continuous Gumbel EV and Clayton microkernels."""
    name = "MPS-EV-Cumulant Copula"

    def __init__(self, u, bins=4, rank=4, cumulant_penalty=.1):
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or len(u) < 8 or u.shape[1] < 2:
            raise ValueError("At least eight PIT rows and two dimensions are required")
        if np.any(~np.isfinite(u)) or np.any((u <= 0) | (u >= 1)):
            raise ValueError("Training PIT coordinates must be finite and interior")
        if bins < 2 or int(bins) != bins or rank < 1 or int(rank) != rank or cumulant_penalty < 0:
            raise ValueError("Bins, rank and cumulant penalty are outside the admissible domain")
        self.d, self.bins, self.rank = u.shape[1], int(bins), int(rank)
        self.cumulant_penalty = float(cumulant_penalty)
        self.mps = s1.MPSCumulant(u, bins=self.bins, rank=self.rank)
        refine_mps(self.mps)
        self.clayton = s1.Clayton(u)
        self.ev = s1.Gumbel(u)
        self.ev_theta_training_mle = float(self.ev.theta)
        self.ev.theta = max(MIN_EV_THETA, self.ev.theta)
        self.cumulant_target = true_connected4(u)
        self.moments = [population_moments(self.clayton.sample(MOMENT_N)), cp_moments(self.mps), population_moments(self.ev.sample(MOMENT_N))]
        densities = np.column_stack([np.exp(self.clayton.logpdf(u)), np.exp(self.mps.logpdf(u)), np.exp(self.ev.logpdf(u))])

        def objective(w):
            k = cumulant_of_mixture(w, self.moments)
            penalty = float(np.mean((k - self.cumulant_target) ** 2)) if len(k) else 0.
            return -np.mean(np.log(np.maximum(densities @ w, np.finfo(float).tiny))) + self.cumulant_penalty * penalty

        fit = optimize.minimize(objective, [.6, .3, .1], method="SLSQP", bounds=[(0., 1.), (MIN_MPS_WEIGHT, 1.), (MIN_EV_WEIGHT, 1.)], constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 200, "ftol": 1e-12})
        if not fit.success:
            raise ArithmeticError("Mixture fitting failed: " + str(fit.message))
        self.weights = fit.x / fit.x.sum()
        if self.weights[1] < MIN_MPS_WEIGHT - 1e-12 or self.weights[2] < MIN_EV_WEIGHT - 1e-12:
            raise ArithmeticError("Mixture constraints violated")
        self.fit_objective = float(fit.fun)
        self.cumulant_population = cumulant_of_mixture(self.weights, self.moments)
        self.lambda_upper = float(self.weights[2] * (2 - 2 ** (1 / self.ev.theta)))
        self.marginal_max_error = float(self.weights[1] * self.mps.marginal_max_error)
        self.parameter_hash = hashlib.sha256(json.dumps(self.parameters(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def parameters(self):
        return {
            "dimension": self.d, "bins": self.bins, "rank": self.rank,
            "cumulant_penalty": self.cumulant_penalty,
            "component_order": ["Clayton", "MPS checkerboard", "continuous Gumbel EV"],
            "weights": self.weights.tolist(), "clayton_theta": self.clayton.theta,
            "gumbel_theta": self.ev.theta, "gumbel_theta_unconstrained_training_mle": self.ev_theta_training_mle,
            "mps_weights": self.mps.w.tolist(),
            "mps_factors": self.mps.f.tolist(),
        }

    def logpdf(self, u):
        u = np.asarray(u, dtype=float)
        if u.ndim != 2 or u.shape[1] != self.d or np.any(~np.isfinite(u)) or np.any((u < 0) | (u > 1)):
            raise ValueError("Copula density input must be a finite matrix on the unit cube")
        terms = np.column_stack([self.clayton.logpdf(u), self.mps.logpdf(u), self.ev.logpdf(u)])
        return special.logsumexp(terms + np.log(np.maximum(self.weights, np.finfo(float).tiny)), axis=1)

    def marginal_cdf(self, u, j):
        if j < 0 or j >= self.d:
            raise ValueError("Marginal coordinate index is out of range")
        if np.any(np.isnan(np.asarray(u))):
            raise ValueError("Marginal CDF input contains NaN")
        x = np.clip(np.asarray(u, dtype=float), 0, 1)
        mass = self.mps.w @ self.mps.f[:, j, :]
        scaled = x * self.bins
        ix = np.minimum(scaled.astype(int), self.bins - 1)
        prefix = np.r_[0., np.cumsum(mass)]
        mps_cdf = prefix[ix] + (scaled - ix) * mass[ix]
        return (self.weights[0] + self.weights[2]) * x + self.weights[1] * mps_cdf

    def sample(self, n, seed=SEED):
        q = uniforms(n, 2 * self.d + 3, seed)
        selector = np.minimum(np.searchsorted(np.cumsum(self.weights), q[:, 0]), 2)
        out = np.empty((n, self.d))
        choose = selector == 0
        a = q[choose, 1:self.d + 1]
        gamma = stats.gamma.ppf(q[choose, self.d + 1], 1 / self.clayton.theta)
        out[choose] = (1 - np.log(a) / gamma[:, None]) ** (-1 / self.clayton.theta)
        choose = selector == 1
        q_mps = q[choose]
        latent = np.minimum(np.searchsorted(np.cumsum(self.mps.w), q_mps[:, 1]), self.rank - 1)
        for j in range(self.d):
            cumulative = np.cumsum(self.mps.f[latent, j, :], axis=1)
            category = np.minimum(np.sum(q_mps[:, 2 + j, None] > cumulative, axis=1), self.bins - 1)
            out[choose, j] = (category + q_mps[:, 2 + self.d + j]) / self.bins
        choose = selector == 2
        alpha = 1 / self.ev.theta
        v = np.pi * q[choose, self.d + 1]
        w = -np.log(q[choose, self.d + 2])
        stable = np.sin(alpha * v) / np.sin(v) ** (1 / alpha) * (np.sin((1 - alpha) * v) / w) ** ((1 - alpha) / alpha)
        out[choose] = np.exp(-((-np.log(q[choose, 1:self.d + 1])) / stable[:, None]) ** alpha)
        return open_unit(out)


def select_training_only(train, years):
    """Nested expanding chronological validation that never receives heldout rows."""
    candidates = [{"bins": bins, "rank": rank, "cumulant_penalty": penalty} for bins, rank in [(2, 2), (4, 2), (4, 4)] for penalty in [.01, .1]]
    folds = [(14, 19), (19, 24), (24, len(train))]
    traces = []
    scores = [[] for _ in candidates]
    for fit_end, validation_end in folds:
        inner = train[:fit_end]
        validation = train[fit_end:validation_end]
        margins = [Margin(inner[:, j]) for j in range(train.shape[1])]
        u = pit(inner, margins, SEED)
        uv = pit(validation, margins, SEED + 1)
        fold_rows = []
        for i, config in enumerate(candidates):
            start = time.perf_counter()
            model = MPSEVCumulant(u, **config)
            loss = float(-np.mean(model.logpdf(uv)))
            scores[i].append((len(validation), loss))
            fold_rows.append({"candidate": i, "validation_copula_nll": loss, "fitted_weights": model.weights.tolist(), "lambda_upper": model.lambda_upper, "runtime_ms": (time.perf_counter() - start) * 1000})
        traces.append({"training_years": years[:fit_end], "validation_years": years[fit_end:validation_end], "candidates": fold_rows})
        print(json.dumps({"stage": "training_cv", "fit_end_year": years[fit_end - 1], "validation_end_year": years[validation_end - 1]}), flush=True)
    aggregate = [sum(n * loss for n, loss in s) / sum(n for n, loss in s) for s in scores]
    best = int(np.argmin(aggregate))
    return candidates[best], {"method": "three expanding chronological folds entirely inside 1985 through 2013", "objective": "validation mean continuous copula negative log likelihood", "candidate_grid_declared_before_test": candidates, "weighted_validation_nll": aggregate, "selected_candidate": best, "folds": traces, "weight_bounds": {"MPS": MIN_MPS_WEIGHT, "EV": MIN_EV_WEIGHT}, "EV_theta_lower_bound": MIN_EV_THETA}


def model_row(model, margins, test, utest, samples, fit_ms):
    row = {"model": model.name, "fit_runtime_ms": fit_ms}
    row.update(s1.metrics(model, margins, test, utest, samples))
    row["deterministic_algorithm_runtime_ms"] = fit_ms + row["sample_runtime_ms"]
    if isinstance(model, s1.Student):
        row["parameters"] = {"degrees_of_freedom": model.df, "correlation": model.corr.tolist()}
    elif isinstance(model, s1.Gaussian):
        row["parameters"] = {"correlation": model.corr.tolist()}
    elif isinstance(model, (s1.Clayton, s1.Gumbel)):
        row["parameters"] = {"theta": model.theta}
    elif isinstance(model, s1.RVine):
        row["parameters"] = {"summary": str(model.model)}
    elif isinstance(model, s1.NaiveQCBM):
        row["circuit"] = {"qubits": model.qubits, "bits_per_country": model.bits, "layers": 2, "max_marginal_uniform_support": 1 - .5 / model.bins, "continuous_density": "atomic measure; continuous likelihood is undefined", "trained": True, "training_nll": model.train_nll, "optimization_evaluations": model.optimization_evaluations, "parameters": model.parameters}
    return row


def main(samples=16384):
    if samples < 2048 or samples & (samples - 1):
        raise ValueError("Evaluation sample count must be a power of two of at least 2048")
    protected = ["src/probe_s1_baselines.py", "src/probe_official_s1.py", "src/verify_probe_s1.py", "data/probe_s1_results.json", "data/mps_s1_operator.npz", "data/probe_official_s1_results.json", "data/processed/loss_gdp_panel.csv"]
    before = {p: sha256(ROOT / p) for p in protected}
    panel_path = ROOT / "data/processed/loss_gdp_panel.csv"
    panel = pd.read_csv(panel_path)
    training = panel.loc[(panel.year >= 1985) & (panel.year <= 2013), ["year"] + COUNTRIES].dropna().sort_values("year")
    train = training[COUNTRIES].to_numpy(float)
    if len(train) != 28 or np.any(train < 0):
        raise ValueError("Expected the fixed 28 observed nonnegative training years")
    begin = time.perf_counter()
    selected, selection = select_training_only(train, training.year.tolist())
    selection_ms = (time.perf_counter() - begin) * 1000
    margins = [Margin(train[:, j]) for j in range(len(COUNTRIES))]
    u = pit(train, margins, SEED)
    begin = time.perf_counter()
    model = MPSEVCumulant(u, **selected)
    fit_ms = (time.perf_counter() - begin) * 1000
    frozen_hash = model.parameter_hash
    print(json.dumps({"stage": "parameters_frozen", "hash": frozen_hash, "selected": selected, "weights": model.weights.tolist(), "lambda_upper": model.lambda_upper}), flush=True)
    fitted_baselines = []
    for cls in [s1.Independent, s1.Gaussian, s1.Student, s1.Clayton, s1.Gumbel, s1.RVine, s1.NaiveQCBM]:
        start = time.perf_counter()
        baseline = cls(u)
        fitted_baselines.append((baseline, (time.perf_counter() - start) * 1000))
    # Test values first enter computations after every model parameter is frozen.
    heldout = panel.loc[(panel.year >= 2014) & (panel.year <= 2024), ["year"] + COUNTRIES].dropna().sort_values("year")
    test = heldout[COUNTRIES].to_numpy(float)
    utest = pit(test, margins, SEED + 1)
    rows = [model_row(baseline, margins, test, utest, samples, ms) for baseline, ms in fitted_baselines]
    newrow = model_row(model, margins, test, utest, samples, fit_ms)
    newrow["parameters"] = model.parameters()
    newrow["frozen_parameter_sha256"] = frozen_hash
    grid = np.unique(np.r_[0., np.linspace(0, 1, 1001), .95, .99, .999999, 1.])
    marginal_errors = [float(np.max(np.abs(model.marginal_cdf(grid, j) - grid))) for j in range(model.d)]
    roundtrips = [float(np.max(abs(m.cdf(m.ppf(np.linspace(m.pt + 1e-9, 1 - 1e-14, 1001))) - np.linspace(m.pt + 1e-9, 1 - 1e-14, 1001)))) for m in margins]
    newrow["invariants"] = {
        "uniform_marginal_cdf_max_errors": marginal_errors,
        "marginal_max_absolute_mass_error": model.marginal_max_error,
        "gpd_tail_roundtrip_max_errors": roundtrips,
        "positive_upper_tail_dependence": model.lambda_upper,
        "EV_weight_at_structural_lower_bound": bool(abs(model.weights[2] - MIN_EV_WEIGHT) < 1e-10),
        "EV_theta_at_structural_lower_bound": bool(abs(model.ev.theta - MIN_EV_THETA) < 1e-10),
        "positive_tail_interpretation": "Positive asymptotic dependence is a declared structural constraint. The estimated coefficient does not establish historical evidence for a nonzero limiting coefficient.",
        "upper_tail_formula": "lambda_U = w_EV * (2 - 2^(1/theta_EV))",
        "upper_tail_proof": "Mixture probability is linear; Clayton and finite checkerboard pair tails have zero limiting ratio; the continuous Gumbel pair tail has coefficient 2 - 2^(1/theta).",
        "marginal_proof": "All components have exactly uniform marginal CDFs. A convex mixture preserves each marginal. Inverse fitted GPD transforms preserve the same S1 fitted loss marginals.",
        "hybrid_scope": "The checkerboard body has a finite nonnegative MPS. The continuous EV microkernel is evaluated analytically and is not claimed to have exact finite TT rank.",
        "mps_bond_dimension": model.rank, "mps_tt_core_shapes": [list(c.shape) for c in model.mps.cores],
        "mps_ipf_iterations": model.mps.ipf_iterations,
        "cumulant_training_centered_target": model.cumulant_target.tolist(),
        "cumulant_model_moment_estimate": model.cumulant_population.tolist(),
        "cumulant_generated_estimate": true_connected4(model.sample(samples)).tolist(),
        "cumulant_domain": "centered bounded PIT coordinates; loss fourth moments are not required",
        "cumulant_component_moment_integration_n": MOMENT_N,
        "cumulant_mixture_algebra": "K1234 = sum(w E[z1z2z3z4]) - sum_pairings(sum(w E[zazb]) * sum(w E[zczd]))",
    }
    assert max(marginal_errors) <= 1e-12 and max(roundtrips) <= 1e-12 and model.lambda_upper > 0
    rows.append(newrow)
    comparisons = {}
    for row in rows[:-1]:
        comparisons[row["model"]] = {}
        for key, direction in METRICS.items():
            x, y = newrow[key], row[key]
            comparisons[row["model"]][key] = "not_comparable" if x is None or y is None else "win" if direction * (x - y) > 1e-12 else "loss" if direction * (x - y) < -1e-12 else "tie"
    by_metric = {key: {outcome: sum(c[key] == outcome for c in comparisons.values()) for outcome in ["win", "loss", "tie", "not_comparable"]} for key in METRICS}
    strict_wins = sum(all(c[key] == "win" for c in comparisons.values()) for key in METRICS)
    s1_snapshot = json.loads((ROOT / "data/probe_s1_results.json").read_text())
    old = s1_snapshot["results"][-1]
    s1_delta = {key: {"s1": old[key], "s2": newrow[key], "signed_improvement": None if old[key] is None else direction * (newrow[key] - old[key])} for key, direction in METRICS.items()}
    after = {p: sha256(ROOT / p) for p in protected}
    if before != after:
        raise AssertionError("Protected S1 artifacts changed")
    result = {
        "schema": "adb.probe.s2.v1", "seed": SEED,
        "implementation": {"path": "src/probe_s2_allwin.py", "sha256": sha256(Path(__file__))},
        "input": {"path": str(panel_path.relative_to(ROOT)), "sha256": sha256(panel_path), "countries": COUNTRIES, "training_years": training.year.tolist(), "test_years": heldout.year.tolist(), "train_n": len(train), "test_n": len(test), "missing_training_years": sorted(set(range(1985, 2014)) - set(training.year)), "missing_test_years": sorted(set(range(2014, 2025)) - set(heldout.year)), "unit": "sum of national loss / national GDP ratios; equal GDP normalized notional"},
        "protocol": {"training_window": [1985, 2013], "heldout_window": [2014, 2024], "simulation_n": samples, "parameter_selection": "all fitting and inner CV use training years only", "test_access": "test matrix and test PIT evaluated only after final S2 and all seven baseline parameters are frozen", "known_s1_holdout_reused": True, "holdout_interpretation": "S1 results on these years were already visible before S2 design. This is a chronological reused holdout, not a fresh independent discovery sample.", "baselines": "All seven S1 implementations refitted from unchanged training input; metrics and Sobol seed retained", "marginals": "S1 GPD parameter fits reused exactly; S2 inverse transform evaluates tail probabilities without the S1 1e-10 numerical clipping", "metric_definitions": s1_snapshot["protocol"], "quantum_scope": "classical low rank probability model; no QPU use", "event_family_holdout_identifiable": False},
        "training_selection": selection, "training_selection_runtime_ms": selection_ms,
        "frozen_parameter_sha256": frozen_hash,
        "margins": {c: m.metadata() for c, m in zip(COUNTRIES, margins)},
        "results": rows, "mps_vs_baselines": comparisons, "wins_by_metric": by_metric,
        "allwin": {"strict_all_baseline_metric_wins": strict_wins, "total_core_metrics": len(METRICS), "strict_allwin_achieved": strict_wins == len(METRICS), "comparison_rule": "strict improvement over all seven baselines at tolerance 1e-12; ties and undefined likelihood are separate outcomes", "finite_sample_obstruction": "Heldout joint95 and joint99 counts are zero. The naive QCBM predicts exactly zero and has zero absolute error, so no model can attain a strictly smaller nonnegative error. Its continuous density is undefined. VaR coverage also has unavoidable finite sample ties."},
        "s1_mps_comparison": s1_delta,
        "protected_s1_sha256": before, "protected_s1_unchanged": before == after,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "scipy": scipy.__version__, "pyvinecopulib": s1.pv.__version__},
    }
    np.savez_compressed(ROOT / "data/mps_s2_ev_operator.npz", weights=model.mps.w, factors=model.mps.f, mixture_weights=model.weights, gumbel_theta=model.ev.theta, clayton_theta=model.clayton.theta, **{"core_" + str(j): c for j, c in enumerate(model.mps.cores)})
    path = ROOT / "data/probe_s2_allwin_results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(path), "strict_metric_wins": strict_wins, "metrics": {key: newrow[key] for key in METRICS}, "marginal_error": max(marginal_errors), "lambda_upper": model.lambda_upper, "wins_by_metric": by_metric}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16384)
    args = parser.parse_args()
    main(args.samples)
