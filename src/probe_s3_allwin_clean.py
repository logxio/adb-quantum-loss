"""Continuous copula experiment with frozen, explicit evaluation ledgers.

Bernstein density kernels replace checkerboard cells. Model selection never
receives an outer test row. A result flag is derived from measured comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from scipy.spatial.distance import cdist, pdist

import probe_s1_baselines as s1
from s3_qcbm_mps import NaiveQCBMMPS

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261118
FOUR = ["PHL", "VNM", "THA", "IDN"]
METRICS = {
    "tail_joint_log_likelihood": 1,
    "joint_exceedance_95_absolute_error": -1,
    "joint_exceedance_99_absolute_error": -1,
    "var_99_coverage_error": -1,
    "tvar_99_calibration_moment_error": -1,
    "brier_95": -1,
    "energy_score_loss_gdp": -1,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def progress(**kw):
    print(json.dumps(kw, allow_nan=False), flush=True)


def masks(u):
    for mask in np.unique(np.isfinite(u), axis=0):
        yield mask, np.all(np.isfinite(u) == mask, axis=1)


class Margin(s1.Margin):
    """Restore the S1 tail inverse guard and retain genuinely missing values."""
    def __init__(self, x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 5:
            raise ValueError("Five observed training years per country are required")
        pos = np.sort(x[x > 0])
        self.sparse = len(pos) < 8
        if not self.sparse:
            super().__init__(x)
        else:
            if not len(pos):
                raise ValueError("At least one observed positive training loss is required")
            self.pzero = float(np.mean(x == 0))
            self.threshold = float(np.quantile(pos, .8))
            self.pt = self.pzero + (1 - self.pzero) * .8
            self.xi = 0.
            excess = pos[pos > self.threshold] - self.threshold
            self.scale = float(max(np.mean(excess) if len(excess) else self.threshold, 1e-10))
            inner = np.unique(pos[pos < self.threshold])
            self.knots = np.r_[0., inner, self.threshold]
            pp = np.minimum(np.searchsorted(pos, inner, side="right") / len(pos), .8 - 1e-8)
            self.probs = np.r_[self.pzero, self.pzero + (1 - self.pzero) * pp, self.pt]
            self.n_tail = len(excess)

    def cdf(self, x):
        x = np.asarray(x, float)
        return np.where(x < 0, 0., super().cdf(x))

    def metadata(self):
        return dict(super().metadata(), sparse_tail_exponential=self.sparse, inverse_tail_probability_clip=1e-10)


def pit(x, margins, seed=SEED):
    x = np.asarray(x, float)
    q = s1.sobol(len(x), len(margins), seed)
    u = np.column_stack([m.cdf(x[:, j]) for j, m in enumerate(margins)])
    for j, m in enumerate(margins):
        zero = x[:, j] == 0
        u[zero, j] = q[zero, j] * m.pzero
    return s1.clip(u)


class BernsteinTT:
    """A nonnegative CP tensor represented by diagonal TT cores.

    c(u) = sum_h w_h prod_j sum_k f_hjk Beta(u_j;k+1,m-k).
    Each beta kernel integrates to one. IPF enforces sum_h w_h f_hjk=1/m.
    Missing coordinates are integrated to one in the observed likelihood.
    """
    def __init__(self, u, degree=4, rank=4):
        self.d, self.m, self.rank = u.shape[1], degree, rank
        rng = np.random.default_rng(SEED)
        self.w = np.full(rank, 1 / rank)
        self.f = rng.dirichlet(np.ones(degree), size=(rank, self.d))
        b = self.basis(u)
        for _ in range(55):
            a = np.einsum("hjk,njk->nhj", self.f, b)
            logresp = np.log(self.w)[None, :] + np.log(np.maximum(a, 1e-280)).sum(axis=2)
            resp = special.softmax(logresp, axis=1)
            self.w = (resp.sum(axis=0) + 1.) / (len(u) + rank)
            for j in range(self.d):
                known = np.isfinite(u[:, j])
                for h in range(rank):
                    counts = (resp[known, h, None] * self.f[h, j] * b[known, j] / a[known, h, j, None]).sum(axis=0) + .5
                    self.f[h, j] = counts / counts.sum()
        for iteration in range(3000):
            for j in range(self.d):
                mass = self.w @ self.f[:, j]
                self.f[:, j] *= (1 / degree) / mass
                normalizer = self.f[:, j].sum(axis=1)
                self.f[:, j] /= normalizer[:, None]
                self.w *= normalizer
                self.w /= self.w.sum()
            error = np.max(abs(np.einsum("h,hjk->jk", self.w, self.f) - 1 / degree))
            if error < 2e-14:
                break
        if error > 1e-11:
            raise ArithmeticError("Bernstein marginal scaling did not converge")
        self.marginal_error = float(error)
        self.cores = [(self.w[:, None] * self.f[:, 0]).T[None]]
        for j in range(1, self.d - 1):
            core = np.zeros((rank, degree, rank))
            for h in range(rank):
                core[h, :, h] = self.f[h, j]
            self.cores.append(core)
        self.cores.append(self.f[:, -1, :, None])

    def basis(self, u):
        u = np.asarray(u, float)
        k = np.arange(self.m)
        b = np.exp(stats.beta.logpdf(s1.clip(u)[:, :, None], k + 1, self.m - k))
        return np.where(np.isfinite(u)[:, :, None], b, 1.)

    def logpdf(self, u):
        local = np.einsum("hjk,njk->nhj", self.f, self.basis(u))
        return special.logsumexp(np.log(self.w) + np.log(np.maximum(local, 1e-280)).sum(axis=2), axis=1)

    def sample(self, n, seed=SEED):
        q = s1.sobol(n, 1 + 2 * self.d, seed)
        h = np.minimum(np.searchsorted(np.cumsum(self.w), q[:, 0]), self.rank - 1)
        out = np.empty((n, self.d))
        for j in range(self.d):
            c = np.cumsum(self.f[h, j], axis=1)
            k = np.minimum((q[:, j + 1, None] > c).sum(axis=1), self.m - 1)
            out[:, j] = stats.beta.ppf(q[:, 1 + self.d + j], k + 1, self.m - k)
        return s1.clip(out)

    def marginal_cdf(self, x, j):
        return stats.beta.cdf(np.asarray(x)[:, None], np.arange(self.m) + 1, self.m - np.arange(self.m)) @ (self.w @ self.f[:, j])

    def moments(self, subsets):
        centers = ((np.arange(self.m) + 1) / (self.m + 1) - .5) * np.sqrt(12)
        local = np.einsum("hjk,k->hj", self.f, centers)
        return np.array([self.w @ np.prod(local[:, p], axis=1) for p in subsets])


class ObservedArch:
    def __init__(self, u, family):
        self.d, self.family = u.shape[1], family
        self.name = "Clayton Copula" if family == "clayton" else "Gumbel Extreme Value Copula"
        self.theta = 1.1
        bounds = (1e-3, 12.) if family == "clayton" else (1.00001, 8.)
        fit = optimize.minimize_scalar(lambda th: -self._logpdf(u, th).sum(), bounds=bounds, method="bounded")
        self.theta = float(fit.x)

    def _logpdf(self, u, theta):
        out = np.zeros(len(u))
        for mask, rows in masks(u):
            d = int(mask.sum())
            if d < 2:
                continue
            model = object.__new__(s1.Clayton if self.family == "clayton" else s1.Gumbel)
            model.d = d
            out[rows] = model._logpdf(u[rows][:, mask], theta)
        return out

    def logpdf(self, u):
        return self._logpdf(u, self.theta)

    def sample(self, n, seed=SEED):
        model = object.__new__(s1.Clayton if self.family == "clayton" else s1.Gumbel)
        model.d, model.theta = self.d, self.theta
        return model.sample(n)


class ObservedGaussian:
    def __init__(self, u, student=False):
        self.d, self.student = u.shape[1], student
        self.name = "Student t Copula" if student else "Gaussian Copula"
        z = stats.norm.ppf(u)
        corr = pd.DataFrame(z).corr(min_periods=4).fillna(0).to_numpy(copy=True)
        np.fill_diagonal(corr, 1.)
        val, vec = np.linalg.eigh(corr)
        corr = (vec * np.maximum(val, .05)) @ vec.T
        corr /= np.sqrt(np.outer(corr.diagonal(), corr.diagonal()))
        self.corr = .95 * corr + .05 * np.eye(self.d)
        self.df = 10.
        if student:
            fit = optimize.minimize_scalar(lambda df: -self._logpdf(u, df).sum(), bounds=(2.1, 30), method="bounded")
            self.df = float(fit.x)

    def _logpdf(self, u, df):
        out = np.zeros(len(u))
        for mask, rows in masks(u):
            if mask.sum() < 2:
                continue
            a = u[rows][:, mask]
            c = self.corr[np.ix_(mask, mask)]
            if self.student:
                z = stats.t.ppf(s1.clip(a), df)
                out[rows] = stats.multivariate_t.logpdf(z, shape=c, df=df) - stats.t.logpdf(z, df).sum(axis=1)
            else:
                z = stats.norm.ppf(s1.clip(a))
                out[rows] = stats.multivariate_normal.logpdf(z, cov=c) - stats.norm.logpdf(z).sum(axis=1)
        return out

    def logpdf(self, u):
        return self._logpdf(u, self.df)

    def sample(self, n):
        q = s1.sobol(n, self.d + 1)
        z = stats.norm.ppf(q[:, :self.d]) @ np.linalg.cholesky(self.corr).T
        if self.student:
            z /= np.sqrt(stats.chi2.ppf(q[:, -1], self.df) / self.df)[:, None]
            return s1.clip(stats.t.cdf(z, self.df))
        return s1.clip(stats.norm.cdf(z))


class ObservedVine(s1.RVine):
    def logpdf(self, u):
        out = np.zeros(len(u))
        # Missing coordinates are integrated under the uniform reference measure.
        # This is a reported QMC likelihood approximation, separate from fitting.
        for mask, rows in masks(u):
            if mask.all():
                out[rows] = super().logpdf(u[rows])
            else:
                missing = int((~mask).sum())
                q = s1.sobol(1024, missing)
                for i in np.flatnonzero(rows):
                    a = np.tile(np.nan_to_num(u[i], nan=.5), (1024, 1))
                    a[:, ~mask] = q
                    out[i] = special.logsumexp(super().logpdf(a)) - np.log(1024)
        return out


def subsets_for(d):
    return list(itertools.combinations(range(d), 2)) + list(itertools.combinations(range(d), 4))


def features(u, subsets):
    z = (u - .5) * np.sqrt(12)
    return np.column_stack([np.prod(z[:, p], axis=1) for p in subsets])


def cumulants(moment, subsets):
    m = dict(zip(subsets, moment))
    return [m[p] - m[p[0], p[1]] * m[p[2], p[3]] - m[p[0], p[2]] * m[p[1], p[3]] - m[p[0], p[3]] * m[p[1], p[2]] for p in subsets if len(p) == 4]


def components(u):
    return [ObservedArch(u, "clayton"), BernsteinTT(u), ObservedArch(u, "gumbel")]


class MPSBernsteinEVCumulant:
    name = "MPS Bernstein EV Cumulant Copula"

    def __init__(self, u):
        self.d = u.shape[1]
        subsets = subsets_for(self.d)
        traces = []
        cv = []
        for i in range(len(u)):
            fitted = components(np.delete(u, i, axis=0))
            y = u[i]
            known = np.isfinite(y)
            density = np.array([np.exp(m.logpdf(u[i:i + 1]))[0] for m in fitted])
            draws = [m.sample(256, seed=SEED + i)[:, known] for m in fitted]
            a = np.array([np.linalg.norm(s - y[known], axis=1).mean() for s in draws])
            b = np.array([[cdist(s[:128], t[128:]).mean() for t in draws] for s in draws])
            active = [p for p in subsets if all(known[list(p)])]
            phi = features(u[i:i + 1], active)[0] if active else np.zeros(0)
            moments = []
            for m in fitted:
                moments.append(m.moments(active) if isinstance(m, BernsteinTT) else features(m.sample(512), active).mean(axis=0))
            mu = np.array(moments)
            # Censored tail score: retain exact density on A and probability of
            # A complement outside A. Its expectation is a proper log score.
            outside = np.array([max(np.mean(np.max(s, axis=1) <= .8), 1 / 256) for s in draws])
            tail = bool(np.max(y[known]) > .8)
            cv.append((density, a, b, phi, mu, outside, tail))
            traces.append({"left_out_training_row": i, "component_density": density.tolist(), "tail_event": tail})

        def objective(w):
            total = 0.
            for density, a, b, phi, mu, outside, tail in cv:
                logscore = -np.log(max(w @ density, 1e-280))
                tailscore = logscore if tail else -np.log(max(w @ outside, 1e-280))
                energy = w @ a - .5 * w @ b @ w
                mixed = w @ mu
                moment_score = np.mean(mixed * mixed - 2 * mixed * phi) if len(phi) else 0.
                total += logscore + .25 * tailscore + energy + .1 * moment_score
            return total / len(cv)

        fit = optimize.minimize(objective, np.full(3, 1 / 3), method="SLSQP", bounds=[(0, 1)] * 3, constraints={"type": "eq", "fun": lambda w: w.sum() - 1}, options={"maxiter": 100, "ftol": 1e-10})
        if not fit.success:
            raise ArithmeticError("Training LOOCV weight fit: " + str(fit.message))
        self.weights = np.maximum(fit.x, 0.)
        self.weights /= self.weights.sum()
        self.models = components(u)
        self.selection = {"objective": float(fit.fun), "weights": self.weights.tolist(), "training_rows": len(u), "inner_folds": traces, "conditioning": "Outer training marginal fits are held fixed during inner copula LOOCV. Each component excludes the left out PIT row. Outer heldout losses enter only after parameter freeze.", "score": "full log score + 0.25 censored tail log score + PIT energy score + 0.1 squared pair and fourth raw PIT feature score", "proper_scope": "Exact integrals give proper scores. Fixed QMC estimates approximate the energy, tail mass and feature integrals. Connected cumulants are derived from raw feature moments, not mislabeled as a proper cumulant penalty."}
        self.subsets = subsets
        self.moments = np.array([m.moments(subsets) if isinstance(m, BernsteinTT) else features(m.sample(2048), subsets).mean(axis=0) for m in self.models])
        self.parameter_hash = hashlib.sha256(json.dumps(self.parameters(), sort_keys=True).encode()).hexdigest()

    def parameters(self):
        bernstein = self.models[1]
        return {"dimension": self.d, "weights": self.weights.tolist(), "clayton_theta": self.models[0].theta, "gumbel_theta": self.models[2].theta, "bernstein_degree": bernstein.m, "tt_rank_bound": bernstein.rank, "bernstein_weights": bernstein.w.tolist(), "bernstein_factors": bernstein.f.tolist(), "uniform_coefficient_error": bernstein.marginal_error}

    def logpdf(self, u):
        return special.logsumexp(np.column_stack([m.logpdf(u) for m in self.models]) + np.log(np.maximum(self.weights, 1e-280)), axis=1)

    def sample(self, n):
        # A selector taken from another Sobol scramble can be correlated with
        # component coordinates at the same row index. Allocate whole blocks
        # instead, then permute so finite energy subsets represent the mixture.
        expected = self.weights * n
        counts = np.floor(expected).astype(int)
        for j in np.argsort(expected - counts)[::-1][:n - counts.sum()]:
            counts[j] += 1
        blocks = [m.sample(int(count)) for m, count in zip(self.models, counts) if count]
        samples = np.concatenate(blocks)
        return samples[np.random.default_rng(SEED + 57).permutation(n)]

    def invariants(self):
        b = self.models[1]
        grid = np.linspace(0, 1, 1001)
        errors = [float(np.max(abs(b.marginal_cdf(grid, j) - grid))) for j in range(self.d)]
        # Independent quadrature of the continuous beta kernels checks mass.
        x, w = np.polynomial.legendre.leggauss(20)
        basis_mass = stats.beta.pdf((x[:, None] + 1) / 2, np.arange(b.m) + 1, b.m - np.arange(b.m)).T @ (w / 2)
        phi = features(self.sample(4096), self.subsets).mean(axis=0)
        return {"marginal_cdf_max_errors": errors, "beta_kernel_mass_max_error": float(max(abs(basis_mass - 1))), "tt_core_shapes": [list(c.shape) for c in b.cores], "tt_bytes": sum(c.nbytes for c in b.cores), "model_connected_fourth_pit": cumulants(self.weights @ self.moments, self.subsets), "generated_connected_fourth_pit": cumulants(phi, self.subsets), "upper_tail_pair_coefficient": float(self.weights[2] * (2 - 2 ** (1 / self.models[2].theta))), "full_ev_finite_tt_claimed": False}


def score_fold(model, margins, test, utest, n, year=None):
    usim = model.sample(n)
    ks = [float(stats.kstest(usim[:, j], "uniform").statistic) for j in range(model.d)]
    details = []
    logpdf = model.logpdf(utest)
    for i, y in enumerate(test):
        known = np.isfinite(y)
        generated = np.column_stack([margins[j].ppf(usim[:, j]) for j in np.flatnonzero(known)])
        pool = generated.sum(axis=1)
        truth = float(y[known].sum())
        q99 = float(np.quantile(pool, .99))
        t99 = float(pool[pool >= q99].mean())
        tail = bool(np.max(utest[i, known]) > .8 and np.all(y[known] > 0))
        ll = float(logpdf[i] + sum(margins[j].logpdf(y[j]) for j in np.flatnonzero(known))) if np.isfinite(logpdf[i]) and tail else None
        likelihood_measure = "Positive loss continuous density, legacy S1 tail subset"
        if model.d > 4:
            # Mixed measure: zero is an atom, not a point with a continuous
            # density. Integrate its entire PIT interval and missing countries.
            tail = bool(np.max(utest[i, known]) > .8)
            likelihood_measure = "Mixed zero atom and positive density, 256 point QMC integration"
            if tail and np.isfinite(logpdf[i]):
                integration = s1.sobol(256, model.d, SEED + 93)
                log_jacobian = 0.
                for j in np.flatnonzero(known):
                    if y[j] == 0:
                        integration[:, j] *= margins[j].pzero
                        log_jacobian += np.log(margins[j].pzero)
                    else:
                        integration[:, j] = utest[i, j]
                        log_jacobian += float(margins[j].logpdf(y[j]))
                ll = float(special.logsumexp(model.logpdf(integration)) - np.log(256) + log_jacobian)
            else:
                ll = None
        a = generated[:2048]
        energy = float(np.linalg.norm(a - y[known], axis=1).mean() - .5 * pdist(a).mean())
        row = {"year": year, "observed_country_indices": np.flatnonzero(known).tolist(), "tail_joint_log_likelihood": ll, "tail_likelihood_measure": likelihood_measure, "tail_test_observations": int(tail), "predicted_var99": q99, "predicted_tvar99": t99, "observed_portfolio_loss": truth, "var99_exceeded": int(truth > q99), "tvar99_calibration_residual": (max(truth - q99, 0) / .01 + q99 - t99) / max(t99, 1e-10), "energy_score_loss_gdp": energy, "marginal_ks_uniform": ks, "marginal_gate_passed": max(ks) <= .05}
        for threshold, key in [(.95, "95"), (.99, "99")]:
            p = float(np.mean((usim[:, known] > threshold).sum(axis=1) >= 2))
            event = int((utest[i, known] > threshold).sum() >= 2)
            row["joint_exceedance_" + key + "_predicted"] = p
            row["joint_exceedance_" + key + "_observed"] = event
            row["brier_" + key] = (p - event) ** 2
        details.append(row)
    return details


def aggregate(rows):
    tail = [r["tail_joint_log_likelihood"] for r in rows if r["tail_joint_log_likelihood"] is not None]
    result = {"observations": len(rows), "tail_test_observations": sum(r["tail_test_observations"] for r in rows), "tail_joint_log_likelihood": float(np.mean(tail)) if tail else None, "var_99_coverage_error": float(abs(np.mean([r["var99_exceeded"] for r in rows]) - .01)), "tvar_99_calibration_moment_error": float(abs(np.mean([r["tvar99_calibration_residual"] for r in rows]))), "energy_score_loss_gdp": float(np.mean([r["energy_score_loss_gdp"] for r in rows])), "brier_95": float(np.mean([r["brier_95"] for r in rows])), "marginal_gate_passed_all_folds": all(r["marginal_gate_passed"] for r in rows), "max_marginal_ks": max(max(r["marginal_ks_uniform"]) for r in rows)}
    for key in ["95", "99"]:
        p = np.mean([r["joint_exceedance_" + key + "_predicted"] for r in rows])
        observed = sum(r["joint_exceedance_" + key + "_observed"] for r in rows)
        result["joint_exceedance_" + key + "_predicted"] = float(p)
        result["joint_exceedance_" + key + "_observed_count"] = observed
        result["joint_exceedance_" + key + "_absolute_error"] = float(abs(p - observed / len(rows)))
    return result


def comparison(ledger):
    own = ledger["aggregates"][MPSBernsteinEVCumulant.name]
    outcomes = {}
    for name, baseline in ledger["aggregates"].items():
        if name == MPSBernsteinEVCumulant.name:
            continue
        outcomes[name] = {}
        for metric, direction in METRICS.items():
            x, y = own[metric], baseline[metric]
            outcomes[name][metric] = "undefined" if x is None or y is None else "win" if direction * (x - y) > 1e-12 else "loss" if direction * (x - y) < -1e-12 else "tie"
    valid = {name: row for name, row in outcomes.items() if ledger["aggregates"][name]["marginal_gate_passed_all_folds"]}
    return {"all_baselines": outcomes, "gate_qualified_baselines": list(valid), "qualified_strict_metric_wins": sum(all(row[k] == "win" for row in valid.values()) for k in METRICS) if valid else 0, "qualified_undefeated": bool(valid) and own["marginal_gate_passed_all_folds"] and all(v in ("win", "tie") for row in valid.values() for v in row.values()), "all_seven_strict_allwin": len(outcomes) == 7 and all(v == "win" for row in outcomes.values() for v in row.values()), "undefined_density_is_never_a_win": True}


def fit_models(u, complete, frozen_own=None):
    models = [MPSBernsteinEVCumulant(u) if frozen_own is None else frozen_own]
    if complete:
        models += [cls(u) for cls in [s1.Independent, s1.Gaussian, s1.Student, s1.Clayton, s1.Gumbel, s1.RVine]]
        models += [s1.NaiveQCBM(u)]
    else:
        models += [s1.Independent(u), ObservedGaussian(u), ObservedGaussian(u, True), ObservedArch(u, "clayton"), ObservedArch(u, "gumbel"), ObservedVine(u), NaiveQCBMMPS(u)]
    return models


def make_ledger(panel, countries, folds, samples, label):
    results, parameters = {}, []
    for fold_id, (train_indices, test_indices) in enumerate(folds):
        training = panel.loc[train_indices]
        train = training[countries].to_numpy(float)
        margins = [Margin(train[:, j]) for j in range(len(countries))]
        u = pit(train, margins)
        models = fit_models(u, bool(np.isfinite(u).all()))
        own = models[0]
        # Every model is now frozen before heldout loss values are read.
        held = panel.loc[test_indices]
        test = held[countries].to_numpy(float)
        utest = pit(test, margins, SEED + 1)
        fold_parameters = {"fold": fold_id, "training_years": training.year.tolist(), "test_years": held.year.tolist(), "parameter_sha256": own.parameter_hash, "parameters": own.parameters(), "training_selection": own.selection, "margins": {c: m.metadata() for c, m in zip(countries, margins)}}
        if fold_id == 0:
            fold_parameters["invariants"] = own.invariants()
        for model in models:
            rows = score_fold(model, margins, test, utest, samples)
            for row, year in zip(rows, held.year):
                row["year"] = int(year)
                row["fold"] = fold_id
            results.setdefault(model.name, []).extend(rows)
            if isinstance(model, NaiveQCBMMPS):
                fold_parameters["naive_qcbm"] = model.metadata()
        parameters.append(fold_parameters)
        progress(stage=label, fold=fold_id + 1, folds=len(folds), years=held.year.tolist(), weights=own.weights.tolist())
    ledger = {"label": label, "countries": countries, "country_pool_definition": "Every prediction scores exactly the countries observed in that heldout year. Missing losses are integrated out, never filled with zero.", "aggregates": {name: aggregate(rows) for name, rows in results.items()}, "per_observation": results, "fold_parameters": parameters}
    ledger["comparisons"] = comparison(ledger)
    return ledger


def rescore_observed_pool():
    """Reevaluate the zero atom measure without changing any fitted weights."""
    path = ROOT / "data/probe_s3_allwin_clean.json"
    result = json.loads(path.read_text())
    panel = pd.read_csv(ROOT / "data/processed/loss_gdp_panel.csv")
    ledger = result["ledger_B15_observed_pool_oof"]
    countries = ledger["countries"]
    results = {}
    start = time.perf_counter()
    for fp in ledger["fold_parameters"]:
        training = panel.loc[panel.year.isin(fp["training_years"])]
        train = training[countries].to_numpy(float)
        margins = [Margin(train[:, j]) for j in range(len(countries))]
        u = pit(train, margins)
        own = object.__new__(MPSBernsteinEVCumulant)
        own.d, own.weights, own.models = len(countries), np.array(fp["parameters"]["weights"]), components(u)
        check = hashlib.sha256(json.dumps(own.parameters(), sort_keys=True).encode()).hexdigest()
        if check != fp["parameter_sha256"]:
            raise AssertionError("Frozen fit parameters changed during measure correction")
        models = fit_models(u, False, own)
        held = panel.loc[panel.year.isin(fp["test_years"])]
        test = held[countries].to_numpy(float)
        utest = pit(test, margins, SEED + 1)
        for model in models:
            rows = score_fold(model, margins, test, utest, result["protocol"]["simulation_n"])
            for row, year in zip(rows, held.year):
                row.update(year=int(year), fold=fp["fold"])
            results.setdefault(model.name, []).extend(rows)
    ledger["per_observation"] = results
    ledger["aggregates"] = {name: aggregate(rows) for name, rows in results.items()}
    ledger["comparisons"] = comparison(ledger)
    result["fit_implementation_sha256"] = result["implementation_sha256"]
    result["implementation_sha256"] = digest(__file__)
    result["protocol"]["B15_tail_likelihood"] = "Mixed zero atom and positive loss measure. Integrate zero PIT intervals and missing countries with 256 QMC points. All frozen mixture parameter hashes reproduced exactly."
    result["measure_rescore_seconds"] = time.perf_counter() - start
    result["protected_unchanged"] = all(digest(ROOT / p) == h for p, h in result["protected_sha256"].items())
    if not result["protected_unchanged"]:
        raise AssertionError("Protected files changed")
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)
    progress(stage="observed_pool_measure_corrected", seconds=result["measure_rescore_seconds"])


def main(samples=8192, only_a=False):
    started = time.perf_counter()
    protected = ["src/probe_s1_baselines.py", "src/probe_s2_allwin.py", "src/probe_official_s1.py", "src/probe_official_s2_scaling.py", "src/s2_contract.py", "data/probe_s1_results.json", "data/probe_s2_allwin_results.json", "data/probe_official_s2_results.json", "data/processed/loss_gdp_panel.csv"]
    before = {p: digest(ROOT / p) for p in protected}
    panel = pd.read_csv(ROOT / "data/processed/loss_gdp_panel.csv")
    panel = panel.loc[panel.year.between(1985, 2024)].copy()
    four = panel.dropna(subset=FOUR)
    train = four.loc[four.year <= 2013].index.tolist()
    test = four.loc[four.year >= 2014].index.tolist()
    ledger_a = make_ledger(four, FOUR, [(train, test)], samples, "A_fixed_chronological")
    result = {"schema": "adb.probe.s3.clean.v1", "seed": SEED, "implementation_sha256": digest(__file__), "input_sha256": before["data/processed/loss_gdp_panel.csv"], "protocol": {"simulation_n": samples, "gpd_inverse_guard": 1e-10, "margin_gate_max_ks": .05, "outer_test_rows_excluded_from_fit": True, "A_reused_holdout": True, "B38_is_retrospective_leave_one_year_out": True, "B38_is_not_past_only_forecasting": True, "B15_is_observed_subportfolio": True, "moment_score": "Squared raw pair and fourth PIT features. Connected fourth cumulants derived afterward.", "tail_likelihood": "Diagnostic positive loss tail subset as S1. Not the training proper score.", "energy_evaluation": "First 2048 predictions with all pair U statistic, matching S1. Component blocks are randomly permuted using a fixed seed.", "VaR_error": "Absolute exceedance frequency difference from 0.01, not error against known true VaR", "TVaR_error": "Absolute mean relative calibration residual, not error against known true TVaR", "continuous_naive_qcbm_likelihood": "Undefined for an atomic distribution. A 7/7 likelihood win over it cannot be defined.", "missingness_assumption": "Observed likelihood assumes ignorable missingness. EM DAT missing reports do not establish zero disaster loss.", "no_QPU_execution": True}, "ledger_A": ledger_a}
    path = ROOT / "data/probe_s3_allwin_clean.json"
    def save():
        result["protected_sha256"] = before
        result["protected_unchanged"] = before == {p: digest(ROOT / p) for p in protected}
        if not result["protected_unchanged"]:
            raise AssertionError("Protected experimental artifacts changed")
        result["runtime_seconds"] = time.perf_counter() - started
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "pyvinecopulib": s1.pv.__version__}
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temp.replace(path)
    save()
    if not only_a:
        ids = four.index.tolist()
        folds = [([j for j in ids if j != i], [i]) for i in ids]
        result["ledger_B38_retrospective_oof"] = make_ledger(four, FOUR, folds, samples, "B38_retrospective_oof")
        save()
        countries = list(panel.columns[1:])
        ids = panel.index.tolist()
        folds = [([j for j in ids if j != i], [i]) for i in ids]
        result["ledger_B15_observed_pool_oof"] = make_ledger(panel, countries, folds, samples, "B15_observed_pool_oof")
        save()
        # Genuine rolling forecasts are scored separately and begin only after
        # twelve observed training years, preserving the causal time boundary.
        folds = [(ids4[:i], [ids4[i]]) for ids4 in [four.index.tolist()] for i in range(12, len(ids4))]
        result["ledger_B_causal_rolling"] = make_ledger(four, FOUR, folds, samples, "B_causal_rolling")
        save()
    progress(output=str(path), runtime_seconds=result["runtime_seconds"], protected_unchanged=result["protected_unchanged"], A_undefeated=ledger_a["comparisons"]["qualified_undefeated"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8192)
    parser.add_argument("--only-a", action="store_true")
    parser.add_argument("--rescore-observed-pool", action="store_true")
    args = parser.parse_args()
    if args.samples < 2048 or args.samples & (args.samples - 1):
        raise ValueError("Samples must be a power of two of at least 2048")
    if args.rescore_observed_pool:
        rescore_observed_pool()
    else:
        main(args.samples, args.only_a)
