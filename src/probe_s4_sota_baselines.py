"""Head to head rare event baselines on the five scenario contract.

Methods compared under one contract, one reference and one budget:
MPS-QMC-IS (terminal diagonal MPO tilt, exact Markov bridge, scrambled Sobol),
multilevel Cross-Entropy IS (Rubinstein and Kroese 2004) with an oracle variant,
Subset Simulation (Au and Beck 2001) with adaptive conditional sampling,
and Iterative Quantum Amplitude Estimation (Grinko et al. 2021) driven by the
12 qubit purified loader and an explicit Grover operator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy
from scipy import stats
from scipy.linalg import expm
from scipy.special import ndtr

import probe_official_s1 as s1
import probe_official_s2_scaling as s2scale
import quantum_resources_s2 as qr
import s2_contract as contract

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = list(s1.PUBLISHED)
TARGETS = s1.TARGETS
SHORT = ["expected_loss", "joint_attachment", "loss_exceedance", "stop_loss", "dual_exhaustion"]
BUDGET = 4096
HORIZON = 3.
SEEDS = [1729 + r for r in range(32)]
MISS_REL = .10

CE_PILOT = 512
CE_MAX_ITER = 5
CE_RHO = .1
CE_RHO_MIN = .01
CE_SMOOTH = .8
CE_ORACLE_PILOT = 2 ** 17
CE_ORACLE_ITER = 10

SUS_N_GRID = [512, 1024]
SUS_P0 = .1
SUS_TARGET_ACC = .44

IQAE_ALPHA = .05
IQAE_SHOT_GRID = [32, 64, 128, 256]
IQAE_ACCURACY_GRID = [.05, .01, .005, .002, .001, .0005, .0002]
IQAE_ACCURACY_SHOTS = 100
IQAE_SHOT_CEILING = 2_000_000
GATE_CHECK_POWERS = 128


# ---------------------------------------------------------------- contract
class Scenario:
    """Latent compound Poisson description of one official scenario."""

    def __init__(self, name):
        self.name = name
        self.params = contract.parameters(name)
        exact = contract.exact_contract(**self.params)
        self.truth = np.asarray(exact["targets"], dtype=float)
        self.term = exact["term"]
        lam = contract.TOTAL_RATES
        c = self.params["multiplier"] * contract.BASE_SHARED_EQ
        r, s = self.params["rain_rate"], self.params["shared_rain"]
        # Poisson means over the three year horizon: own EQ 1, own EQ 2, shared EQ,
        # own rain 1, own rain 2, shared rain.
        self.mu = HORIZON * np.array([lam[0] - c, lam[1] - c, c, r - s, r - s, s])
        w = contract.RATES / lam[:, None]
        # Mark streams: own 1, own 2, shared mark seen by 1, shared mark seen by 2.
        self.w = np.array([w[0], w[1], w[0], w[1]])
        self.eq_cap = self.params["eq_cap_m"]
        self.rain_cap = 80. - self.eq_cap
        self.threshold = self.params["threshold_m"]
        a_levels = np.unique((np.arange(5)[:, None] * self.eq_cap / 4 + np.arange(2)[None] * self.rain_cap).ravel())
        l_levels = np.unique((a_levels[:, None] + a_levels[None]).ravel())
        gaps = np.r_[np.diff(a_levels), np.diff(l_levels)]
        self.delta = .25 * float(gaps.min())
        self.levels = {"L": self.threshold + self.delta, "J": self.delta, "X": 80.}

    def evaluate(self, n_eq, sev, n_rain):
        """Exact capped contract payout from latent counts and the first four marks."""
        used = np.minimum(np.c_[n_eq[:, 0], n_eq[:, 1], n_eq[:, 2], n_eq[:, 2]], 4)
        mask = np.arange(4)[None, None, :] < used[:, :, None]
        sums = (sev * mask).sum(axis=2)
        eq1 = np.minimum(sums[:, 0] + sums[:, 2], 4)
        eq2 = np.minimum(sums[:, 1] + sums[:, 3], 4)
        r1 = (n_rain[:, 0] + n_rain[:, 2]) > 0
        r2 = (n_rain[:, 1] + n_rain[:, 2]) > 0
        a = eq1 * self.eq_cap / 4 + r1 * self.rain_cap
        b = eq2 * self.eq_cap / 4 + r2 * self.rain_cap
        loss = a + b
        t = self.threshold
        f = np.stack([loss * 1e6, (a > 0) & (b > 0), loss > t, np.maximum(loss - t, 0) * 1e6,
                      (a >= 80 - 1e-12) & (b >= 80 - 1e-12)], axis=1).astype(float)
        state = ((eq1 * 5 + eq2) * 2 + r1) * 2 + r2
        return a, b, loss, f, state, used

    def scores(self, a, b, loss, t_a, t_b, t_all):
        """Plateau free scores whose exact thresholds equal the contract events."""
        d = self.delta
        joint = np.minimum(a + d * t_a, b + d * t_b)
        return {"L": loss + d * t_all, "J": joint, "X": joint}


TARGET_SPEC = [  # (score key, final weight function index, rare level?)
    ("L", 0, False),
    ("J", 1, True),
    ("L", 2, True),
    ("L", 3, True),
    ("X", 4, True),
]


# ---------------------------------------------------------------- CE-IS
def ce_draw(sc, mu, v, n, rng):
    """Draw n paths from the tilted product proposal with exact likelihood ratios."""
    d = LatentMap(sc, mu, v).from_uniform(rng.random((n, LatentMap.DIM)))
    counts, sev, mask = d["counts"], d["sev"], d["mask"]
    # Exact Radon Nikodym derivative dP/dQ over counts and the tilted first marks.
    logw = np.zeros(n)
    for j in range(6):
        if sc.mu[j] > 0:
            logw += counts[:, j] * np.log(sc.mu[j] / mu[j]) - (sc.mu[j] - mu[j])
    for s in range(4):
        ratio = np.log(sc.w[s] / v[s])
        logw += (ratio[sev[:, s] - 1] * mask[:, s]).sum(axis=1)
    d["w"] = np.exp(logw)
    return d


def ce_update(sc, mu, v, draw, h):
    omega = h * draw["w"]
    total = omega.sum()
    if not np.isfinite(total) or total <= 0:
        return mu, v
    new_mu = mu.copy()
    for j in range(6):
        if sc.mu[j] > 0:
            new_mu[j] = min(max((omega @ draw["counts"][:, j]) / total, 1e-12), 40.)
    new_v = v.copy()
    for s in range(4):
        cnt = np.array([(omega[:, None] * ((draw["sev"][:, s] == c + 1) & draw["mask"][:, s])).sum() for c in range(4)])
        if cnt.sum() > 0:
            new_v[s] = np.maximum(cnt / cnt.sum(), 1e-3)
            new_v[s] /= new_v[s].sum()
    return CE_SMOOTH * new_mu + (1 - CE_SMOOTH) * mu, CE_SMOOTH * new_v + (1 - CE_SMOOTH) * v


def ce_adapt(sc, target, pilot, max_iter, rng):
    key, fidx, rare = TARGET_SPEC[target]
    level = sc.levels[key]
    mu, v = sc.mu.copy(), sc.w.copy()
    used, trace, reached = 0, [], False
    for it in range(max_iter):
        draw = ce_draw(sc, mu, v, pilot, rng)
        used += pilot
        if rare:
            score = draw["score"][key]
            gamma = min(float(np.quantile(score, 1 - CE_RHO)), level)
            # Adaptive rho: once at least one percent of the pilot already hits the
            # target event, the event itself becomes the elite set.
            if np.sum(score >= level) >= max(5, math.ceil(CE_RHO_MIN * pilot)):
                gamma = level
            reached = gamma >= level
            h = draw["f"][:, fidx] if reached and fidx == 3 else (score >= gamma).astype(float)
        else:
            gamma, reached = None, it >= 1
            h = draw["f"][:, fidx]
        mu, v = ce_update(sc, mu, v, draw, h)
        trace.append({"iteration": it, "gamma_m": gamma})
        if reached:
            break
    return mu, v, used, trace, reached


def ce_is_run(sc, target, seed):
    rng = np.random.default_rng(seed)
    mu, v, used, trace, reached = ce_adapt(sc, target, CE_PILOT, CE_MAX_ITER, rng)
    final = BUDGET - used
    draw = ce_draw(sc, mu, v, final, rng)
    vals = draw["f"][:, target] * draw["w"]
    return vals.mean(), vals.std(ddof=1) / math.sqrt(final), {"pilot_draws": used, "final_draws": final,
                                                               "contract_evaluations": used + final,
                                                               "iterations": len(trace), "level_reached": reached}


# ---------------------------------------------------------------- Subset Simulation
class LatentMap:
    """Standard normal latent space of dimension 22 mapped exactly to the contract."""
    DIM = 22

    def __init__(self, sc, mu=None, v=None):
        self.sc = sc
        mu = sc.mu if mu is None else mu
        v = sc.w if v is None else v
        support = np.arange(120)
        self.cdf = [np.r_[stats.poisson.cdf(support, m)[:-1], 1.] if m > 0 else np.ones(120) for m in mu]
        self.wcdf = np.cumsum(v, axis=1)
        self.wcdf[:, -1] = 1.
    @staticmethod
    def _position(u, cdf, cell):
        """Fractional position of u inside its inverse CDF cell, continuous in u."""
        lo = np.where(cell > 0, cdf[np.maximum(cell - 1, 0)], 0.)
        width = cdf[cell] - lo
        return np.clip(np.divide(u - lo, width, out=np.zeros_like(u), where=width > 0), 0., 1 - 1e-12)

    def __call__(self, z):
        d = self.from_uniform(ndtr(z))
        return d["score"], d["f"], d["state"]

    def from_uniform(self, u):
        u = np.clip(u, 1e-300, 1 - 1e-16)
        n = len(u)
        counts = np.stack([np.searchsorted(self.cdf[j], u[:, j]) for j in range(6)], axis=1)
        pos_c = np.stack([self._position(u[:, j], self.cdf[j], counts[:, j]) for j in range(6)], axis=1)
        sev = np.empty((n, 4, 4), dtype=np.int64)
        pos_s = np.empty((n, 4, 4))
        for s in range(4):
            sev[:, s] = np.searchsorted(self.wcdf[s], u[:, 6 + 4 * s:10 + 4 * s]) + 1
            pos_s[:, s] = self._position(u[:, 6 + 4 * s:10 + 4 * s], self.wcdf[s], sev[:, s] - 1)
        a, b, loss, f, state, used = self.sc.evaluate(counts[:, :3], sev, counts[:, 3:])
        mask = np.arange(4)[None, None, :] < used[:, :, None]
        eq_open = [a < 80 - 1e-12, b < 80 - 1e-12]
        rain_off = [(counts[:, 3] + counts[:, 5]) == 0, (counts[:, 4] + counts[:, 5]) == 0]
        # Progress toward the next contract cell: positions of every coordinate that
        # can still raise the country payout (counts, active marks, absent rain).
        parts = []
        for i, (own, stream, rain) in enumerate([(0, 0, 3), (1, 1, 4)]):
            eq_capped = (np.minimum((sev[:, stream] * mask[:, stream]).sum(1)
                                    + (sev[:, 2 + i] * mask[:, 2 + i]).sum(1), 4) >= 4)
            live = eq_open[i] & ~eq_capped
            num = live * (pos_c[:, own] + pos_c[:, 2] + (pos_s[:, stream] * mask[:, stream]).sum(1)
                          + (pos_s[:, 2 + i] * mask[:, 2 + i]).sum(1))
            den = live * (2 + mask[:, stream].sum(1) + mask[:, 2 + i].sum(1))
            num = num + rain_off[i] * (pos_c[:, rain] + pos_c[:, 5])
            den = den + rain_off[i] * 2
            parts.append((num, den))
        t_a = np.divide(parts[0][0], parts[0][1], out=np.zeros(n), where=parts[0][1] > 0)
        t_b = np.divide(parts[1][0], parts[1][1], out=np.zeros(n), where=parts[1][1] > 0)
        den = parts[0][1] + parts[1][1]
        t_all = np.divide(parts[0][0] + parts[1][0], den, out=np.zeros(n), where=den > 0)
        return {"score": self.sc.scores(a, b, loss, t_a, t_b, t_all), "f": f, "state": state,
                "counts": counts, "sev": sev, "mask": mask}


def subset_simulation(lmap, key, seed, level_n=512, budget=BUDGET):
    """Au and Beck (2001) with conditional sampling in standard normal space.

    Levels use level_n samples (declared grid 512 or 1024) and p0=0.1. The preconditioned Crank Nicolson move
    z' = rho z + sqrt(1-rho^2) xi preserves the latent Gaussian, so a candidate
    is accepted exactly when it stays inside the current failure domain
    (Papaioannou et al. 2015). The proposal spread adapts toward 0.44 acceptance.
    After the final threshold is reached, the remaining budget extends the
    final level chains, so every run spends exactly the declared budget.
    """
    rng = np.random.default_rng(seed)
    level = lmap.sc.levels[key]
    ns = int(round(SUS_P0 * level_n))
    z = rng.standard_normal((level_n, lmap.DIM))
    score, f, _ = lmap(z)
    score = score[key]
    evals = level_n
    levels = [{"gamma": -np.inf, "z": z, "score": score, "f": f}]
    log_p = 0.
    thresholds, acceptance = [], []
    sigma = .6
    while True:
        cur = levels[-1]
        order = np.argsort(-cur["score"], kind="stable")
        if np.sum(cur["score"] >= level) >= ns or evals >= budget:
            break
        gamma = .5 * (cur["score"][order[ns - 1]] + cur["score"][order[ns]])
        frac = float(np.mean(cur["score"] > gamma))
        need = min(level_n - ns, budget - evals)
        if need <= 0:
            break
        seeds = order[:ns]
        chains = cur["z"][seeds].copy()
        cs, cf = cur["score"][seeds].copy(), cur["f"][seeds].copy()
        zs, ss, fs = [chains.copy()], [cs.copy()], [cf.copy()]
        made, acc_n, prop_n, step = 0, 0, 0, 0
        while made < need:
            m = min(ns, need - made)
            rho = math.sqrt(1 - sigma ** 2)
            cand = rho * chains[:m] + sigma * rng.standard_normal((m, lmap.DIM))
            csc, cfe, _ = lmap(cand)
            csc = csc[key]
            ok = csc > gamma
            chains[:m][ok], cs[:m][ok], cf[:m][ok] = cand[ok], csc[ok], cfe[ok]
            zs.append(chains[:m].copy()); ss.append(cs[:m].copy()); fs.append(cf[:m].copy())
            made += m; acc_n += int(ok.sum()); prop_n += m; step += 1
            sigma = float(np.clip(sigma * math.exp((ok.mean() - SUS_TARGET_ACC) / math.sqrt(step)), .05, .999))
        evals += need
        log_p += math.log(frac)
        thresholds.append(float(gamma))
        acceptance.append(acc_n / prop_n)
        levels.append({"gamma": gamma, "z": np.concatenate(zs), "score": np.concatenate(ss), "f": np.concatenate(fs),
                       "chains": chains, "chain_scores": cs, "chain_f": cf})
    # Spend any remaining budget on the final conditional level.
    cur = levels[-1]
    extra = budget - evals
    if extra > 0:
        if len(levels) == 1:
            zx = rng.standard_normal((extra, lmap.DIM))
            sx, fx, _ = lmap(zx)
            cur["z"] = np.concatenate([cur["z"], zx]); cur["score"] = np.concatenate([cur["score"], sx[key]])
            cur["f"] = np.concatenate([cur["f"], fx])
        else:
            chains, cs, cf = cur["chains"], cur["chain_scores"], cur["chain_f"]
            zs, ss, fs = [cur["z"]], [cur["score"]], [cur["f"]]
            made, step = 0, 0
            gamma = cur["gamma"]
            while made < extra:
                m = min(len(chains), extra - made)
                rho = math.sqrt(1 - sigma ** 2)
                cand = rho * chains[:m] + sigma * rng.standard_normal((m, lmap.DIM))
                csc, cfe, _ = lmap(cand)
                csc = csc[key]
                ok = csc > gamma
                chains[:m][ok], cs[:m][ok], cf[:m][ok] = cand[ok], csc[ok], cfe[ok]
                zs.append(chains[:m].copy()); ss.append(cs[:m].copy()); fs.append(cf[:m].copy())
                made += m; step += 1
                sigma = float(np.clip(sigma * math.exp((ok.mean() - SUS_TARGET_ACC) / math.sqrt(step)), .05, .999))
            cur["z"], cur["score"], cur["f"] = np.concatenate(zs), np.concatenate(ss), np.concatenate(fs)
        evals += extra
    if evals != budget:
        raise AssertionError("Subset Simulation budget is not conserved")
    # Band estimator: level i samples are conditional on score > gamma_i.
    probs = [1.]
    for t in thresholds:
        probs.append(probs[-1] * SUS_P0 if False else None)
    pf = [1.]
    for i in range(1, len(levels)):
        prev = levels[i - 1]
        pf.append(pf[-1] * float(np.mean(prev["score"][:level_n] > levels[i]["gamma"])))
    expect = np.zeros(5)
    for i, lv in enumerate(levels):
        if i + 1 < len(levels):
            band = lv["score"] <= levels[i + 1]["gamma"]
            expect += pf[i] * (lv["f"] * band[:, None]).mean(axis=0)
        else:
            expect += pf[i] * lv["f"].mean(axis=0)
    prob_target = pf[-1] * float(np.mean(levels[-1]["score"] >= level))
    return {"probability": prob_target, "band_expectations": expect, "evaluations": evals,
            "levels": len(levels), "thresholds_m": thresholds, "acceptance": acceptance,
            "final_level_samples": int(len(levels[-1]["score"]))}


# ---------------------------------------------------------------- IQAE
def scenario_factors(name):
    """Purified latent factors of qr.global_factors with scenario intensities."""
    p = contract.parameters(name)
    rates = contract.RATES
    totals = contract.TOTAL_RATES
    common = p["multiplier"] * contract.BASE_SHARED_EQ
    rain, shared_rain = p["rain_rate"], p["shared_rain"]
    pk = np.r_[stats.poisson.pmf(np.arange(4), 3 * common), stats.poisson.sf(3, 3 * common)]
    pr = np.array([math.exp(-3 * shared_rain), -math.expm1(-3 * shared_rain)])
    weights = np.outer(pk, pr).ravel()
    f = np.empty((10, 2, 10))
    for j in range(2):
        severity = rates[j] / totals[j]
        gen = np.zeros((5, 5))
        for x in range(5):
            for a in range(1, 5):
                gen[x, min(x + a, 4)] += 3 * (totals[j] - common) * severity[a - 1]
            gen[x, x] -= gen[x].sum()
        kr = expm(gen)[0]
        common_severity = np.r_[0., severity]
        for k in range(5):
            eq = kr if k < 4 else np.array([0., 0., 0., 0., 1.])
            for r in range(2):
                pp = 1. if r else -math.expm1(-3 * (rain - shared_rain))
                f[2 * k + r, j] = np.outer(eq, [1 - pp, pp]).ravel()
            if k < 3:
                kr = np.bincount(np.minimum(np.arange(5)[:, None] + np.arange(5), 4).ravel(),
                                 weights=np.outer(kr, common_severity).ravel(), minlength=5)
    return weights, f


class GateSim:
    """Exact real statevector over a fixed RY plus CNOT list, forward and inverse."""

    def __init__(self, gates, qubits):
        self.q = qubits
        idx = np.arange(2 ** qubits, dtype=np.int64)
        self.ops = []
        for g in gates:
            if g[0] == "ry":
                _, t, ang = g
                ix = idx[(idx & (1 << t)) == 0]
                self.ops.append(("ry", ix, ix | (1 << t), math.cos(ang / 2), math.sin(ang / 2)))
            else:
                _, c, t = g
                ix = idx[((idx & (1 << c)) != 0) & ((idx & (1 << t)) == 0)]
                self.ops.append(("cx", ix, ix | (1 << t), 0., 0.))

    def apply(self, state, inverse=False):
        state = state.copy()
        for kind, ix, other, c, s in (reversed(self.ops) if inverse else self.ops):
            a, z = state[ix], state[other]
            if kind == "ry":
                if inverse:
                    state[ix], state[other] = c * a + s * z, -s * a + c * z
                else:
                    state[ix], state[other] = c * a - s * z, s * a + c * z
            else:
                state[ix], state[other] = z, a
        return state


def build_quantum(name):
    weights, local = scenario_factors(name)
    gates, q = qr.compile_purified_loader(weights, local)
    sim = GateSim(gates, q)
    zero = np.zeros(2 ** q); zero[0] = 1.
    psi = sim.apply(zero)
    ref = qr.simulate_real_gates(gates, q)
    marginal = np.zeros((10, 10))
    good = []
    for h in range(16):
        for x in range(10):
            for y in range(10):
                seq = f"{h:04b}{x:04b}{y:04b}"
                ix = sum(int(bit) << i for i, bit in enumerate(seq))
                marginal[x, y] += psi[ix] ** 2
                if x == 9 and y == 9:
                    good.append(ix)
    good = np.array(good)
    # Local index 2*eq+rain against the exact contract order (eq1, eq2, rain1, rain2).
    exact = contract.exact_contract(**contract.parameters(name))
    official = exact["term"].reshape(5, 5, 2, 2).transpose(0, 2, 1, 3).reshape(10, 10)
    p_good = float(np.sum(psi[good] ** 2))
    return {"gates": gates, "qubits": q, "sim": sim, "psi": psi, "good": good, "p_good": p_good,
            "terminal_error": float(np.max(abs(marginal - official))),
            "simulator_crosscheck": float(np.max(abs(psi - ref))),
            "norm_error": float(abs(psi @ psi - 1)),
            "cnot": sum(g[0] == "cx" for g in gates), "ry": sum(g[0] == "ry" for g in gates),
            "truth": float(exact["targets"][4])}


def grover_check(qs, powers=GATE_CHECK_POWERS):
    """Apply Q = -A S0 A^dagger S_chi gate by gate and compare with sin^2((2k+1) theta)."""
    theta = math.asin(math.sqrt(qs["p_good"]))
    state = qs["psi"].copy()
    max_angle, max_norm, angle_64, norm_64 = 0., abs(state @ state - 1), 0., 0.
    for k in range(powers + 1):
        pk = float(np.sum(state[qs["good"]] ** 2))
        max_angle = max(max_angle, abs(math.sin((2 * k + 1) * theta) ** 2 - pk))
        max_norm = max(max_norm, abs(state @ state - 1))
        if k <= 64:
            angle_64, norm_64 = max_angle, max_norm
        if k == powers:
            break
        state[qs["good"]] *= -1                      # S_chi
        state = qs["sim"].apply(state, inverse=True)  # A^dagger
        state[0] *= -1                                # S_0 = I - 2|0><0|
        state = -qs["sim"].apply(state)               # -A
    return {"theta": theta, "powers_checked": powers, "max_angle_error_k_le_64": angle_64,
            "max_norm_error_k_le_64": norm_64, "max_angle_error_k_le_128": max_angle,
            "max_norm_error_k_le_128": max_norm, "gate_operations_per_Q": 2 * len(qs["sim"].ops)}


def clopper_pearson(k, n, alpha):
    lo = 0. if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1. if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def find_next_k(k, upper, interval, min_ratio=2.):
    lo, hi = interval
    old = 4 * k + 2
    max_scaling = int(1 / (2 * (hi - lo)))
    scaling = max_scaling - (max_scaling - 2) % 4
    while scaling >= min_ratio * old:
        tmin = scaling * lo - int(scaling * lo)
        tmax = scaling * hi - int(scaling * hi)
        if tmin <= tmax <= .5 and tmin <= .5:
            return int((scaling - 2) / 4), True
        if tmax >= .5 and tmax >= tmin >= .5:
            return int((scaling - 2) / 4), False
        scaling -= 4
    return int(k), upper


def iqae(theta, epsilon, alpha, shots, rng, method="clopper_pearson", oracle_cap=None, shot_cap=None, stop="angle"):
    """Grinko et al. (2021) Algorithm 1 with binomial shot noise at each power."""
    max_rounds = int(math.log(2 * math.pi / 8 / epsilon) / math.log(2)) + 1
    intervals = [[0., .25]]
    a_int = [0., 1.]
    powers, ones = [0], []
    upper, n_oracle, n_shots, k_max, it, stalled = True, 0, 0, 0, 0, False
    # stop="angle" is the Grinko et al. rule; stop="amplitude" halts once the amplitude CI half width is at most epsilon.
    while intervals[-1][1] - intervals[-1][0] > epsilon / math.pi and (stop == "angle" or (a_int[1] - a_int[0]) / 2 > epsilon):
        k, upper = find_next_k(powers[-1], upper, intervals[-1])
        if oracle_cap is not None and (n_oracle + k * shots > oracle_cap or n_shots + shots > shot_cap):
            break
        if n_shots + shots > IQAE_SHOT_CEILING:
            stalled = True
            break
        it += 1
        powers.append(k)
        p_k = math.sin((2 * k + 1) * theta) ** 2
        one = int(rng.binomial(shots, p_k))
        ones.append(one)
        n_oracle += k * shots
        n_shots += shots
        k_max = max(k_max, k)
        # Pool consecutive rounds at the same power (running sums, same result as rescanning).
        if it >= 2 and powers[-2] == k:
            rs, ro = rs + shots, ro + one
        else:
            rs, ro = shots, one
        if method == "chernoff":
            prob = ro / rs
            eps = math.sqrt(math.log(2 * max_rounds / alpha) / (2 * rs))
            lo, hi = max(0., prob - eps), min(1., prob + eps)
        else:
            lo, hi = clopper_pearson(ro, rs, alpha / max_rounds)
        if upper:
            tmin, tmax = math.acos(1 - 2 * lo) / 2 / math.pi, math.acos(1 - 2 * hi) / 2 / math.pi
        else:
            tmin, tmax = 1 - math.acos(1 - 2 * hi) / 2 / math.pi, 1 - math.acos(1 - 2 * lo) / 2 / math.pi
        scale = 4 * k + 2
        tu = (int(scale * intervals[-1][1]) + tmax) / scale
        tl = (int(scale * intervals[-1][0]) + tmin) / scale
        intervals.append([tl, tu])
        a_int = [math.sin(2 * math.pi * tl) ** 2, math.sin(2 * math.pi * tu) ** 2]
    est = float(np.mean(a_int))
    return {"estimate": est, "ci": [float(min(a_int)), float(max(a_int))], "oracle_calls": n_oracle,
            "shots": n_shots, "k_max": k_max, "rounds": it,
            "a_calls": 2 * n_oracle + n_shots, "stalled_at_shot_ceiling": stalled}


# ---------------------------------------------------------------- statistics
def summarize(values, truth, runtimes, ses=None, cis=None, extra=None):
    v = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(v)):
        raise AssertionError("Non finite estimate")
    err = v - truth
    row = {"truth": float(truth), "mean_estimate": float(v.mean()),
           "relative_bias": float(err.mean() / truth),
           "relative_bias_standard_error": float(v.std(ddof=1) / math.sqrt(len(v)) / truth),
           "relative_rmse": float(np.sqrt(np.mean(err ** 2)) / truth),
           "miss_rate": f"{int(np.sum(np.abs(err) > MISS_REL * truth))}/{len(v)}",
           "misses": int(np.sum(np.abs(err) > MISS_REL * truth)),
           "zero_estimates": int(np.sum(v == 0)),
           "runtime_ms": float(np.mean(runtimes))}
    if ses is not None:
        s = np.asarray(ses)
        row["ci95_noncoverage"] = int(np.sum(np.abs(err) > 1.96 * s))
        row["mean_reported_relative_se"] = float(np.mean(s) / truth)
    if cis is not None:
        c = np.asarray(cis)
        row["ci_noncoverage"] = int(np.sum((truth < c[:, 0]) | (truth > c[:, 1])))
        row["mean_relative_ci_halfwidth"] = float(np.mean(c[:, 1] - c[:, 0]) / 2 / truth)
    if extra:
        row.update(extra)
    return row


# ---------------------------------------------------------------- runners
def run_mps(sc):
    annual, _ = s1.annual_model(sc.name)
    start = time.perf_counter_ns()
    tilt = s2scale.TerminalTilt(annual, sc.name)
    setup_ms = (time.perf_counter_ns() - start) / 1e6
    ref = tilt.term @ tilt.features
    if np.max(np.abs(ref - sc.truth) / sc.truth) > 1e-9:
        raise AssertionError("MPS reference differs from the contract reference")
    est, rt = [], []
    for seed in SEEDS:
        t0 = time.perf_counter_ns()
        est.append(tilt.sample(BUDGET, seed, qmc=True))
        rt.append((time.perf_counter_ns() - t0) / 1e6)
    est = np.array(est)
    return {SHORT[j]: summarize(est[:, j], sc.truth[j], rt, extra={"contract_evaluations": BUDGET, "setup_ms": setup_ms,
                                                                    "budget_shared_across_targets": True})
            for j in range(5)}


def run_ce(sc):
    out = {}
    for j in range(5):
        est, se, rt, meta = [], [], [], []
        for seed in SEEDS:
            t0 = time.perf_counter_ns()
            e, s, m = ce_is_run(sc, j, seed * 10 + j)
            rt.append((time.perf_counter_ns() - t0) / 1e6)
            if m["contract_evaluations"] != BUDGET:
                raise AssertionError("CE budget is not conserved")
            est.append(e); se.append(s); meta.append(m)
        out[SHORT[j]] = summarize(est, sc.truth[j], rt, ses=se, extra={
            "contract_evaluations": BUDGET, "mean_pilot_draws": float(np.mean([m["pilot_draws"] for m in meta])),
            "mean_ce_iterations": float(np.mean([m["iterations"] for m in meta])),
            "level_reached_runs": int(sum(m["level_reached"] for m in meta))})
    return out


def run_oracle(sc):
    out, checks = {}, {}
    for j in range(5):
        rng = np.random.default_rng(900001 + 97 * j + SCENARIOS.index(sc.name))
        t0 = time.perf_counter_ns()
        mu, v, _, trace, reached = ce_adapt(sc, j, CE_ORACLE_PILOT, CE_ORACLE_ITER, rng)
        # Two refinement updates at the final level with the full offline sample.
        for _ in range(2):
            draw = ce_draw(sc, mu, v, CE_ORACLE_PILOT, rng)
            key, fidx, rare = TARGET_SPEC[j]
            h = draw["f"][:, fidx] if fidx in (0, 3) else draw["f"][:, fidx]
            mu, v = ce_update(sc, mu, v, draw, h)
        setup_ms = (time.perf_counter_ns() - t0) / 1e6
        parts = [ce_draw(sc, mu, v, 2 ** 16, rng) for _ in range(4)]
        w_all = np.concatenate([p["w"] for p in parts])
        fv = np.concatenate([p["f"][:, j] * p["w"] for p in parts])
        del parts
        w_mean, w_se = float(w_all.mean()), float(w_all.std(ddof=1) / 2 ** 9)
        checks[SHORT[j]] = {"likelihood_ratio_mean": w_mean, "likelihood_ratio_se": w_se,
                            "likelihood_ratio_z": (w_mean - 1) / w_se,
                            "large_sample_relative_error": float(fv.mean() / sc.truth[j] - 1),
                            "large_sample_z": float((fv.mean() - sc.truth[j]) / (fv.std(ddof=1) / 2 ** 9)),
                            "proposal_poisson_means": mu.tolist(), "nominal_poisson_means": sc.mu.tolist(),
                            "proposal_mark_probabilities": v.tolist(), "offline_level_reached": bool(reached)}
        est, se, rt = [], [], []
        for seed in SEEDS:
            r = np.random.default_rng(seed * 10 + j + 500000)
            t1 = time.perf_counter_ns()
            d = ce_draw(sc, mu, v, BUDGET, r)
            vals = d["f"][:, j] * d["w"]
            rt.append((time.perf_counter_ns() - t1) / 1e6)
            est.append(vals.mean()); se.append(vals.std(ddof=1) / math.sqrt(BUDGET))
        out[SHORT[j]] = summarize(est, sc.truth[j], rt, ses=se, extra={
            "contract_evaluations": BUDGET, "offline_proposal_setup_ms": setup_ms,
            "offline_setup_draws_not_counted": CE_ORACLE_PILOT * (len(trace) + 2)})
    return out, checks


def run_sus(sc):
    lmap = LatentMap(sc)
    out = {}
    plan = [("L", "band", 0), ("J", "prob", 1), ("L", "prob", 2), ("L", "band", 3), ("X", "prob", 4)]
    grid = {}
    for level_n in SUS_N_GRID:
        runs = {"L": [], "J": [], "X": []}
        times = {"L": [], "J": [], "X": []}
        for key in runs:
            for seed in SEEDS:
                t0 = time.perf_counter_ns()
                runs[key].append(subset_simulation(lmap, key, seed * 7 + "LJX".index(key), level_n))
                times[key].append((time.perf_counter_ns() - t0) / 1e6)
        rows = {}
        for j, (key, kind, fidx) in enumerate(plan):
            rs = runs[key]
            est = [r["probability"] if kind == "prob" else r["band_expectations"][fidx] for r in rs]
            acc = [np.mean(r["acceptance"]) for r in rs if r["acceptance"]]
            rows[SHORT[j]] = summarize(est, sc.truth[j], times[key], extra={
                "contract_evaluations": BUDGET, "samples_per_level": level_n, "p0": SUS_P0, "score": key,
                "estimator": "level probability" if kind == "prob" else "level band expectation",
                "mean_levels": float(np.mean([r["levels"] for r in rs])),
                "mean_acceptance": float(np.mean(acc)) if acc else None,
                "shared_run_targets": [SHORT[i] for i, p in enumerate(plan) if p[0] == key]})
        grid[level_n] = rows
    for t in SHORT:
        best = min(SUS_N_GRID, key=lambda n: grid[n][t]["relative_rmse"])
        out[t] = dict(grid[best][t], grid_relative_rmse={str(n): grid[n][t]["relative_rmse"] for n in SUS_N_GRID})
    return out, lmap


def latent_exactness(sc, lmap, n=2 ** 20):
    """Nominal draws through both latent samplers reproduce the exact terminal law."""
    rng = np.random.default_rng(424242)
    chunk = 2 ** 16
    freq_ce, freq_sus, unit = np.zeros(100), np.zeros(100), True
    for _ in range(n // chunk):  # streamed to stay inside the 4 GiB process limit
        d = ce_draw(sc, sc.mu, sc.w, chunk, rng)
        freq_ce += np.bincount(d["state"], minlength=100)
        unit &= bool(np.allclose(d["w"], 1., atol=1e-14))
        _, _, st = lmap(rng.standard_normal((chunk, lmap.DIM)))
        freq_sus += np.bincount(st, minlength=100)
    freq_ce, freq_sus = freq_ce / n, freq_sus / n
    sd = np.sqrt(sc.term * (1 - sc.term) / n) + 1e-12
    return {"draws": n, "ce_sampler_max_z": float(np.max(np.abs(freq_ce - sc.term) / sd)),
            "sus_map_max_z": float(np.max(np.abs(freq_sus - sc.term) / sd)),
            "nominal_likelihood_ratio_exactly_one": unit}


def run_iqae(sc, qs, ledger):
    theta = math.asin(math.sqrt(qs["p_good"]))
    grid = {}
    for method in ["clopper_pearson", "chernoff"]:
        for shots in IQAE_SHOT_GRID:
            res, rt = [], []
            for seed in SEEDS:
                t0 = time.perf_counter_ns()
                res.append(iqae(theta, 1e-9, IQAE_ALPHA, shots, np.random.default_rng(seed), method,
                                oracle_cap=BUDGET, shot_cap=BUDGET))
                rt.append((time.perf_counter_ns() - t0) / 1e6)
            grid[f"{method}_shots{shots}"] = iqae_row(res, qs["truth"], rt, ledger)
    best = min(grid, key=lambda k: grid[k]["relative_rmse"])
    return {"matched_budget_grid": grid, "matched_budget_best": best, "matched_budget": grid[best]}


def iqae_row(res, truth, rt, ledger):
    no = np.array([r["oracle_calls"] for r in res])
    ns = np.array([r["shots"] for r in res])
    cnot = ns * ledger["cnot_per_A"] + no * ledger["cnot_per_Q"]
    depth = ns * ledger["serial_depth_per_A"] + no * ledger["serial_depth_per_Q"]
    if not np.all(cnot == (2 * no + ns) * ledger["cnot_per_A"] + no * ledger["cnot_per_S0"]):
        raise AssertionError("CNOT ledger identity failed")
    return summarize([r["estimate"] for r in res], truth, rt, cis=[r["ci"] for r in res], extra={
        "mean_oracle_calls": float(no.mean()), "max_oracle_calls": int(no.max()),
        "stalled_runs": int(sum(r["stalled_at_shot_ceiling"] for r in res)),
        "mean_shots": float(ns.mean()), "max_shots": int(ns.max()),
        "mean_a_or_a_dagger_calls": float((2 * no + ns).mean()),
        "k_max": int(max(r["k_max"] for r in res)), "mean_rounds": float(np.mean([r["rounds"] for r in res])),
        "mean_total_cnots": float(cnot.mean()), "max_total_cnots": int(cnot.max()),
        "max_circuit_cnots": int(ledger["cnot_per_A"] + max(r["k_max"] for r in res) * ledger["cnot_per_Q"]),
        "mean_total_serial_depth": float(depth.mean())})


def iqae_accuracy(qs, ledger):
    theta = math.asin(math.sqrt(qs["p_good"]))
    rows = []
    plan = [("amplitude", m, rel) for m in ["clopper_pearson", "chernoff"] for rel in IQAE_ACCURACY_GRID]
    plan += [("angle", m, rel) for m in ["clopper_pearson", "chernoff"] for rel in [.05, .01]]
    for stop, method, rel in plan:
            res, rt = [], []
            for seed in SEEDS:
                t0 = time.perf_counter_ns()
                res.append(iqae(theta, rel * qs["truth"], IQAE_ALPHA, IQAE_ACCURACY_SHOTS, np.random.default_rng(seed),
                                method, stop=stop))
                rt.append((time.perf_counter_ns() - t0) / 1e6)
            row = iqae_row(res, qs["truth"], rt, ledger)
            row.update({"method": method, "stop_rule": stop, "target_relative_halfwidth": rel,
                        "runs_within_target": int(sum(abs(r["estimate"] - qs["truth"]) <= rel * qs["truth"] for r in res))})
            rows.append(row)
    return rows


def projection(rows, target_rmse, p):
    """Log log fit of measured oracle calls against measured relative RMSE."""
    out = {}
    for method in ["clopper_pearson", "chernoff"]:
        r = [x for x in rows if x["method"] == method and x["stop_rule"] == "amplitude" and x["stalled_runs"] == 0]
        x = np.log([v["relative_rmse"] for v in r]); y = np.log([v["mean_oracle_calls"] for v in r])
        slope, icept = np.polyfit(x, y, 1)
        xs = np.log([v["relative_rmse"] for v in r]); ys = np.log([v["mean_total_cnots"] for v in r])
        cs, ci = np.polyfit(xs, ys, 1)
        n_or = float(math.exp(icept + slope * math.log(target_rmse)))
        mc = (1 - p) / (p * target_rmse ** 2)
        out[method] = {"fit_slope_log_oracle_vs_log_rmse": float(slope),
                       "projected_oracle_calls": n_or,
                       "projected_total_cnots": float(math.exp(ci + cs * math.log(target_rmse))),
                       "naive_mc_draws_for_same_rmse": mc,
                       "query_ratio_naive_mc_over_iqae": mc / n_or,
                       "closest_measured": min(r, key=lambda v: abs(math.log(v["relative_rmse"] / target_rmse)))["target_relative_halfwidth"]}
    return out


def crossover(n_oracle_1pct, mps_ms):
    """Classical exact contraction against IQAE for n=15 at growing bond dimension."""
    rows = []
    rng = np.random.default_rng(7)
    for chi in [10, 32, 100, 316, 1000, 3162, 10000, 100000]:
        b = int(math.ceil(math.log2(chi)))
        mem = 15 * chi * chi * 8
        measured = None
        if chi <= 1000:
            cores = [rng.random((chi, chi)) for _ in range(13)]
            vec = rng.random(chi)
            t0 = time.perf_counter_ns()
            for _ in range(3):
                x = vec.copy()
                for c in cores:
                    x = x @ c
                    x /= x.sum()
            measured = (time.perf_counter_ns() - t0) / 3e6
        a_cx = 15 * qr.qsd_cnot(4 + b) + qr.mcx_cost(60)["cnot"]
        # The purified product mixture loader applies when the model is a latent mixture (chi <= 256 in the ledger).
        purified = qr.estimate(15, chi)["end_to_end"]["cnot_per_Q"] if chi <= 256 else None
        q_cx = 2 * a_cx + qr.mcx_cost(60 + b)["cnot"]
        rows.append({"chi": chi, "classical_exact_flops": 2 * 15 * chi * chi, "classical_memory_bytes": mem,
                     "classical_measured_ms": measured, "fits_4gb": mem <= 4 * 1024 ** 3,
                     "generic_amplitude_loader_cnot_per_A_qsd_bound": a_cx,
                     "iqae_total_cnots_at_1pct": n_oracle_1pct * q_cx,
                     "iqae_serial_seconds_at_1us_per_cnot_layer": n_oracle_1pct * q_cx * 1e-6,
                     "purified_mixture_loader_cnot_per_Q": purified,
                     "iqae_total_cnots_at_1pct_purified": None if purified is None else n_oracle_1pct * purified})
    return rows


def ledger_from_resources():
    est = qr.estimate(2, 10, .01, qr.P_BASE)
    e = est["end_to_end"]
    load = est["purified_mps_loader"]
    return {"loader_cnot": load["cnot_count"], "loader_ry": load["ry_count"],
            "flag_mcx_cnot": e["cnot_per_A"] - load["cnot_count"],
            "cnot_per_A": e["cnot_per_A"], "cnot_per_S0": e["cnot_per_Q"] - 2 * e["cnot_per_A"],
            "cnot_per_Q": e["cnot_per_Q"], "serial_depth_per_A": e["cnot_per_A"] + e["one_qubit_gates_per_A"],
            "serial_depth_per_Q": e["serial_depth_per_Q_upper"], "oracle_qubits": e["qubits"]}


def rss(stage):
    """Print the process peak resident set size after a stage (macOS reports bytes)."""
    print(json.dumps({"stage": stage, "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30}), flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protected = ["team_solver.py", "src/probe_s2_allwin.py", "src/quantum_resources_s2.py",
                 "src/verify_s4_qjlb.py", "src/verify_adversarial_s2.py", "src/s2_contract.py"]
    before = {p: sha256(ROOT / p) for p in protected}
    t_all = time.perf_counter()
    ledger = ledger_from_resources()
    result = {"schema": "adb.s4.sota_baselines.v1", "budget_per_run": BUDGET, "replicates": len(SEEDS),
              "seeds": [SEEDS[0], SEEDS[-1]], "miss_definition": "absolute relative error above 10 percent",
              "reference": "exact contract targets from src/s2_contract.py exact_contract(parameters(scenario))",
              "quantum_ledger": ledger, "scenarios": {}, "verification": {}}
    for name in SCENARIOS:
        sc = Scenario(name)
        print(json.dumps({"stage": "scenario", "scenario": name}), flush=True)
        mps = run_mps(sc); rss("mps")
        ce = run_ce(sc); rss("ce")
        oracle, rn = run_oracle(sc); rss("oracle")
        sus, lmap = run_sus(sc); rss("sus")
        exact_map = latent_exactness(sc, lmap); rss("latent")
        qs = build_quantum(name); rss("loader")
        gcheck = grover_check(qs); rss("grover")
        q = run_iqae(sc, qs, ledger); rss("iqae")
        result["scenarios"][name] = {
            "exact_targets": dict(zip(SHORT, sc.truth.tolist())),
            "methods": {"MPS-QMC-IS": mps, "CE-IS": ce, "Oracle-Tilted Rate IS": oracle, "Subset Simulation": sus,
                        "IQAE": {"dual_exhaustion": q["matched_budget"]}},
            "iqae_matched_budget_grid": q["matched_budget_grid"], "iqae_matched_budget_selected": q["matched_budget_best"],
            "quantum_loader": {"qubits": qs["qubits"], "cnot": qs["cnot"], "ry": qs["ry"], "gates": qs["cnot"] + qs["ry"],
                               "terminal_distribution_max_error": qs["terminal_error"],
                               "statevector_norm_error": qs["norm_error"],
                               "crosscheck_against_reference_simulator": qs["simulator_crosscheck"],
                               "good_basis_states": len(qs["good"]), "p_good": qs["p_good"],
                               "p_good_minus_contract": qs["p_good"] - qs["truth"], "grover": gcheck},
            "radon_nikodym_checks": rn, "latent_exactness": exact_map}
        result["verification"][name] = {
            "ce_budget": BUDGET, "sus_budget": BUDGET,
            "max_abs_lr_z": max(abs(v["likelihood_ratio_z"]) for v in rn.values()),
            "grover_angle_error_k_le_64": gcheck["max_angle_error_k_le_64"],
            "grover_angle_error_k_le_128": gcheck["max_angle_error_k_le_128"],
            "grover_norm_error_k_le_128": gcheck["max_norm_error_k_le_128"]}
        assert qs["cnot"] == 494 and qs["ry"] == 495 and qs["qubits"] == 12
        assert qs["terminal_error"] < 1e-12 and qs["norm_error"] < 1e-12 and abs(qs["p_good"] - qs["truth"]) < 1e-12
        # Rounding grows with the number of gate level Q powers; 1e-12 holds through k=64.
        assert gcheck["max_angle_error_k_le_64"] < 1e-12
        assert gcheck["max_angle_error_k_le_128"] < 1e-11 and gcheck["max_norm_error_k_le_128"] < 1e-11
        assert all(abs(v["likelihood_ratio_z"]) < 5 for v in rn.values())
        assert exact_map["ce_sampler_max_z"] < 6 and exact_map["sus_map_max_z"] < 6 and exact_map["nominal_likelihood_ratio_exactly_one"]
        print(json.dumps({"scenario": name, "dual_exhaustion_rel_rmse": {m: result["scenarios"][name]["methods"][m]["dual_exhaustion"]["relative_rmse"]
                                                                         for m in result["scenarios"][name]["methods"]}}), flush=True)
    base_q = build_quantum("adb-base")
    acc = iqae_accuracy(base_q, ledger); rss("iqae_accuracy")
    mps_base = result["scenarios"]["adb-base"]["methods"]["MPS-QMC-IS"]["dual_exhaustion"]
    proj = projection(acc, mps_base["relative_rmse"], base_q["truth"])
    one_pct = [r for r in acc if r["method"] == "clopper_pearson" and r["stop_rule"] == "amplitude" and r["target_relative_halfwidth"] == .01][0]
    result["iqae_target_accuracy_adb_base"] = {"p": base_q["truth"], "shots_per_round": IQAE_ACCURACY_SHOTS,
                                               "alpha": IQAE_ALPHA, "rows": acc,
                                               "projection_to_mps_qmc_is_rmse": {"target_relative_rmse": mps_base["relative_rmse"], **proj}}
    result["crossover_n15"] = crossover(one_pct["mean_oracle_calls"], mps_base["runtime_ms"]); rss("crossover")
    after = {p: sha256(ROOT / p) for p in protected}
    if before != after:
        raise AssertionError("Protected files changed")
    result["protected_sha256_unchanged"] = True
    result["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                             "platform": platform.platform()}
    result["implementation_sha256"] = sha256(Path(__file__))
    result["runtime_seconds"] = time.perf_counter() - t_all
    text = json.dumps(result, indent=2, allow_nan=False)
    src = Path(__file__).read_text()
    for bad in [chr(0x2014), "Puf" + "fy", "nx" + "sio", "/Us" + "ers/"]:
        assert bad not in src and bad not in text, bad
    (ROOT / "data/s4_sota_baselines.json").write_text(text + "\n")
    print(json.dumps({"output": "data/s4_sota_baselines.json", "runtime_seconds": result["runtime_seconds"]}), flush=True)
    return result


def pct(x, digits=3):
    return f"{100 * x:.{digits}f}%"


def render(result):
    """Markdown report built only from the JSON written by main()."""
    methods = ["MPS-QMC-IS", "Oracle-Tilted Rate IS", "CE-IS", "Subset Simulation"]
    names = {"expected_loss": "Expected loss", "joint_attachment": "Joint attachment",
             "loss_exceedance": "P(L>T)", "stop_loss": "Stop loss", "dual_exhaustion": "Dual exhaustion"}
    sc = result["scenarios"]
    led = result["quantum_ledger"]
    L = ["# S4 rare event and quantum baselines on the ADB DRB contract", ""]
    L += ["All methods estimate the same five term targets of the official two country contract "
          "(adb-base, adb-independent, adb-common-shock, adb-rain-stress, adb-rain-cover). "
          "Reference values are the exact capped CTMC contraction in `src/s2_contract.py`. "
          f"Each classical run spends exactly {result['budget_per_run']:,} contract evaluations; "
          f"each configuration is repeated over {result['replicates']} independent seeds. "
          "A miss is a replicate with absolute relative error above 10%.", ""]
    L += ["## Dual exhaustion P(A and B)", "",
          "Relative RMSE over 32 seeds, misses in parentheses. IQAE uses the 12 qubit loader with at most "
          "4,096 Grover calls and 4,096 shots per run.", "",
          "| Scenario | Exact | MPS-QMC-IS | Oracle-Tilted IS | CE-IS | Subset Simulation | IQAE |",
          "|---|---|---|---|---|---|---|"]
    for name, s in sc.items():
        cells = [f"{s['methods'][m]['dual_exhaustion']['relative_rmse'] * 100:.3g}% ({s['methods'][m]['dual_exhaustion']['misses']})"
                 for m in methods + ["IQAE"]]
        L.append(f"| {name} | {s['exact_targets']['dual_exhaustion']:.4e} | " + " | ".join(cells) + " |")
    L += ["", "## All targets", "",
          "Relative RMSE. MPS-QMC-IS answers all five targets from one 4,096 path run; every baseline receives a "
          "separate 4,096 evaluation run per target (Subset Simulation shares one run across the three loss targets).", ""]
    for name, s in sc.items():
        L += [f"### {name}", "", "| Method | " + " | ".join(names.values()) + " | ms per run |",
              "|---|" + "---|" * 6]
        for m in methods:
            row = s["methods"][m]
            ms = np.mean([row[t]["runtime_ms"] for t in SHORT])
            L.append(f"| {m} | " + " | ".join(f"{row[t]['relative_rmse'] * 100:.3g}% ({row[t]['misses']})" for t in SHORT)
                     + f" | {ms:.1f} |")
        L.append("")
    L += ["### Calibration of the baselines", "",
          "Mean relative bias with its standard error (32 seeds). Values near zero within two standard errors "
          "indicate unbiased behavior; CI noncoverage counts nominal 95% CLT intervals that miss the reference.", "",
          "| Scenario | Method | Joint attachment bias | P(L>T) bias | Dual exhaustion bias | CI misses (DE) |",
          "|---|---|---|---|---|---|"]
    for name, s in sc.items():
        for m in methods[1:]:
            row = s["methods"][m]
            cells = [f"{row[t]['relative_bias'] * 100:+.2f}% ± {row[t]['relative_bias_standard_error'] * 100:.2f}%"
                     for t in ["joint_attachment", "loss_exceedance", "dual_exhaustion"]]
            ci = row["dual_exhaustion"].get("ci95_noncoverage")
            L.append(f"| {name} | {m} | " + " | ".join(cells) + f" | {'no CI' if ci is None else f'{ci}/32'} |")
    L += ["", "## IQAE at a matched budget", "",
          f"Grover operator Q = -A S0 A^dagger S_chi with A the {led['loader_cnot']} CNOT plus {led['loader_ry']} RY purified "
          f"loader and a {led['flag_mcx_cnot']} CNOT flag; {led['cnot_per_A']} CNOT per A, {led['cnot_per_Q']} CNOT per Q. "
          "Configuration chosen per scenario from a declared grid (Clopper Pearson or Chernoff Hoeffding, 32 to 256 shots per round).", "",
          "| Scenario | Config | Rel RMSE | Misses | CI misses | Oracle calls | Shots | k max | Total CNOT |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, s in sc.items():
        r = s["methods"]["IQAE"]["dual_exhaustion"]
        L.append(f"| {name} | {s['iqae_matched_budget_selected']} | {pct(r['relative_rmse'], 1)} | {r['misses']} | "
                 f"{r['ci_noncoverage']} | {r['mean_oracle_calls']:,.0f} | {r['mean_shots']:,.0f} | {r['k_max']} | "
                 f"{r['mean_total_cnots']:.3e} |")
    acc = result["iqae_target_accuracy_adb_base"]
    L += ["", "## IQAE target accuracy on adb-base dual exhaustion", "",
          f"p = {acc['p']:.6e}, {acc['shots_per_round']} shots per round, alpha = {acc['alpha']}. Means over 32 seeds. Stop rule angle is the Grinko et al. criterion (angle interval width at most epsilon over pi), which is a worst case bound near amplitude 0.5 and overshoots small targets; stop rule amplitude halts once the amplitude CI half width is at most the target. Projection uses the amplitude rule. "
          "total CNOT = shots x CNOT(A) + oracle calls x CNOT(Q).", "",
          "| Stop | CI | Target half width | Rel RMSE | Within target | Oracle calls | Shots | k max | Max circuit CNOT | Total CNOT |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in acc["rows"]:
        L.append(f"| {r['stop_rule']} | {r['method']} | {pct(r['target_relative_halfwidth'], 2)} | {pct(r['relative_rmse'])} | {r['runs_within_target']}/32 ({r['stalled_runs']} stalled) | "
                 f"{r['mean_oracle_calls']:,.0f} | {r['mean_shots']:,.0f} | {r['k_max']:,} | {r['max_circuit_cnots']:,} | "
                 f"{r['mean_total_cnots']:.3e} |")
    pr = acc["projection_to_mps_qmc_is_rmse"]
    L += ["", f"Matching the MPS-QMC-IS relative RMSE of {pct(pr['target_relative_rmse'], 4)} (log log fit of measured rows):", "",
          "| CI | Slope | Oracle calls | Total CNOT | Naive MC draws | MC draws / IQAE calls |", "|---|---|---|---|---|---|"]
    for m in ["clopper_pearson", "chernoff"]:
        p = pr[m]
        L.append(f"| {m} | {p['fit_slope_log_oracle_vs_log_rmse']:.3f} | {p['projected_oracle_calls']:.3e} | "
                 f"{p['projected_total_cnots']:.3e} | {p['naive_mc_draws_for_same_rmse']:.3e} | {p['query_ratio_naive_mc_over_iqae']:.1f} |")
    mps = sc["adb-base"]["methods"]["MPS-QMC-IS"]["dual_exhaustion"]
    cp = pr["clopper_pearson"]
    L += ["", "## Reading the comparison", "",
          f"Query scaling. The measured IQAE slope of about {cp['fit_slope_log_oracle_vs_log_rmse']:.2f} against relative RMSE "
          "confirms the quadratic improvement over naive Monte Carlo, whose draw count scales with slope -2. "
          f"At the MPS-QMC-IS accuracy, naive Monte Carlo needs {cp['naive_mc_draws_for_same_rmse']:.2e} draws and IQAE "
          f"about {cp['projected_oracle_calls']:.2e} Grover calls.", "",
          f"Gate and time cost. The same accuracy requires about {cp['projected_total_cnots']:.2e} ideal CNOT executions, "
          f"about {cp['projected_total_cnots'] * 1e-6:,.0f} s of serial two qubit time at an assumed 1 us per CNOT, before "
          "error correction. "
          f"MPS-QMC-IS reaches it with 4,096 classical paths in {mps['runtime_ms']:.1f} ms per run, plus a one time "
          f"proposal setup of {mps['setup_ms']:.0f} ms, on one CPU core. At a matched budget of at most 4,096 Grover calls, "
          f"IQAE reaches {pct(sc['adb-base']['methods']['IQAE']['dual_exhaustion']['relative_rmse'], 1)} relative RMSE on "
          "adb-base.", "",
          "Ablation. Oracle-Tilted Rate IS uses the best product rate and mark tilt found offline with 2^17 draws per CE "
          "iteration, and spends all 4,096 draws on it. The gap between it and MPS-QMC-IS isolates the contribution of the "
          "terminal diagonal MPO tilt, the exact Markov bridge and the scrambled Sobol stratification. CE-IS spends part of "
          "the same 4,096 on its own adaptation and shows the cost of learning the tilt in budget.", "",
          "Where amplitude estimation becomes competitive. MPS-QMC-IS depends on an exact probability MPS of low bond "
          "dimension (chi = 10 here). The table below holds n = 15 countries and the 1% IQAE query count fixed and grows chi. "
          "Classical exact contraction scales as n chi^2 in time and memory; a generic amplitude MPS loader also scales as "
          "chi^2 in CNOT count, so amplitude estimation does not remove the bond dimension cost. It becomes the better route "
          "only for dependence graphs with no low rank classical contraction (high treewidth, non separable) that still admit "
          "a compact state preparation circuit.", "",
          "Two loaders are listed. The purified mixture loader covers latent mixture models such as this contract "
          "(ledger range chi <= 256); the generic amplitude MPS loader uses the QSD upper bound for arbitrary MPS.", "",
          "| chi | Classical flops | Memory | Measured ms | Fits 4 GB | IQAE CNOT at 1%, purified | CNOT per A, generic QSD | IQAE CNOT at 1%, generic | Serial s at 1 us, generic |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in result["crossover_n15"]:
        ms = "not run" if r["classical_measured_ms"] is None else f"{r['classical_measured_ms']:.3f}"
        pur = "not applicable" if r["iqae_total_cnots_at_1pct_purified"] is None else f"{r['iqae_total_cnots_at_1pct_purified']:.2e}"
        L.append(f"| {r['chi']:,} | {r['classical_exact_flops']:.2e} | {r['classical_memory_bytes'] / 2 ** 20:,.1f} MiB | {ms} | "
                 f"{'yes' if r['fits_4gb'] else 'no'} | {pur} | {r['generic_amplitude_loader_cnot_per_A_qsd_bound']:.2e} | "
                 f"{r['iqae_total_cnots_at_1pct']:.2e} | {r['iqae_serial_seconds_at_1us_per_cnot_layer']:.2e} |")
    v = result["verification"]
    L += ["", "The 1 us per serial CNOT layer is an assumption for scale, not a measured device figure.", "",
          "## Built in verification", "",
          "| Scenario | Max abs LR z | Grover angle error k<=64 | Grover angle error k<=128 | Norm drift k<=128 | Loader terminal error |",
          "|---|---|---|---|---|---|"]
    for name in sc:
        q = sc[name]["quantum_loader"]
        L.append(f"| {name} | {v[name]['max_abs_lr_z']:.2f} | {v[name]['grover_angle_error_k_le_64']:.1e} | "
                 f"{v[name]['grover_angle_error_k_le_128']:.1e} | {v[name]['grover_norm_error_k_le_128']:.1e} | "
                 f"{q['terminal_distribution_max_error']:.1e} |")
    L += ["", "Every classical run spends exactly 4,096 contract evaluations (asserted). Likelihood ratios of each oracle "
          "proposal average to one within five standard errors over 2^18 draws. Both latent samplers reproduce the exact "
          "100 state terminal law within six standard errors over 2^20 nominal draws. The 12 qubit loader has 494 CNOT and "
          "495 RY gates, ||A|0>|| = 1 to 1e-12, and its good state mass equals the contract exhaustion probability to 1e-12. "
          "Gate by gate Grover powers match sin^2((2k+1)theta) to 1e-12 through k = 64. All outputs are finite.", "",
          "## Method notes", "",
          "CE-IS (Rubinstein and Kroese 2004). Product proposal over the six Poisson means (own and shared earthquake, own "
          "and shared rain) and the four mark distributions. Multilevel elite quantile rho = 0.1 on a plateau free score, "
          "adaptive rho down to 0.01 when the event is already hit, smoothing 0.8, 512 draws per pilot iteration, at most "
          "five iterations, remaining draws for the final estimate with exact likelihood ratios.", "",
          "Subset Simulation (Au and Beck 2001). 22 dimensional standard normal latent space mapped exactly to the capped "
          "contract; scores add a fractional cell position below the smallest payout gap, so every final threshold equals "
          "the contract event. p0 = 0.1, preconditioned Crank Nicolson conditional sampling with acceptance adapted toward "
          "0.44, level size chosen per target from {512, 1024}; leftover budget extends the final level.", "",
          "IQAE (Grinko et al. 2021). Algorithm 1 with next power selection, Clopper Pearson or Chernoff Hoeffding intervals "
          "at alpha / T, and binomial shot sampling at each power. Success probabilities come from the verified Grover "
          "rotation of the gate level statevector.", "",
          "MPS-QMC-IS. `TerminalTilt` from `src/probe_official_s2_scaling.py`, seeds 1729 to 1760, unchanged.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="only rebuild docs/s4_sota_baselines.md from the JSON")
    args = parser.parse_args()
    res = json.loads((ROOT / "data/s4_sota_baselines.json").read_text()) if args.render else main()
    md = render(res)
    for bad in [chr(0x2014), "Puf" + "fy", "nx" + "sio", "/Us" + "ers/"]:
        assert bad not in md, bad
    (ROOT / "docs/s4_sota_baselines.md").write_text(md)
