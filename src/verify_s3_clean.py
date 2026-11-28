"""Independent continuous kernel checks and a rendered results ledger."""
from pathlib import Path
import hashlib
import json
import itertools

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import probe_s3_allwin_clean as s3

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "data/probe_s3_allwin_clean.json").read_text())
    checks = []
    def check(name, condition, value=None):
        if not condition:
            raise AssertionError(name + ": " + str(value))
        checks.append({"name": name, "value": value, "passed": True})

    rng = np.random.default_rng(20261118)
    u = rng.uniform(.001, .999, (70, 4))
    u[::5, 2] = np.nan
    model = s3.BernsteinTT(u)
    check("coefficient uniform margins", model.marginal_error < 1e-11, model.marginal_error)
    x = rng.uniform(.01, .99, (20, 4))
    b = model.basis(x)
    state = np.ones((len(x), 1))
    for j, core in enumerate(model.cores):
        state = np.einsum("na,akb,nk->nb", state, core, b[:, j])
    error = float(max(abs(state[:, 0] - np.exp(model.logpdf(x)))))
    check("TT contraction matches continuous density", error < 1e-12, error)
    gx, gw = np.polynomial.legendre.leggauss(20)
    gx, gw = (gx + 1) / 2, gw / 2
    observed = x[:1].copy()
    observed[:, 2:] = np.nan
    cube = np.repeat(x[:1], 400, axis=0)
    cube[:, 2:] = np.array(list(itertools.product(gx, gx)))
    exact = float(np.exp(model.logpdf(observed))[0])
    integral = float(np.exp(model.logpdf(cube)) @ np.outer(gw, gw).ravel())
    check("missing dimensions integrated exactly", abs(exact - integral) < 1e-12, abs(exact - integral))
    boundary = np.full((2, 4), .5)
    boundary[:, 0] = [.25 - 1e-8, .25 + 1e-8]
    jump = float(np.ptp(np.exp(model.logpdf(boundary))))
    check("former cell boundary is smooth", jump < 1e-6, jump)
    sample = model.sample(65536)
    ks = max(stats.kstest(sample[:, j], "uniform").statistic for j in range(4))
    check("continuous sampler preserves uniform margins", ks < .01, float(ks))

    own_name = s3.MPSBernsteinEVCumulant.name
    for key, ledger in data.items():
        if not key.startswith("ledger_"):
            continue
        for fp in ledger["fold_parameters"]:
            check(key + " excludes outer test years fold " + str(fp["fold"]), not set(fp["training_years"]) & set(fp["test_years"]))
        if key == "ledger_B_causal_rolling":
            check("causal folds contain only past years", all(max(fp["training_years"]) < min(fp["test_years"]) for fp in ledger["fold_parameters"]))
        for name, rows in ledger["per_observation"].items():
            actual = s3.aggregate(rows)
            check(key + " recomputed aggregate " + name, actual == ledger["aggregates"][name])
        check(key + " mixture passes sampled marginal gate", ledger["aggregates"][own_name]["marginal_gate_passed_all_folds"], ledger["aggregates"][own_name]["max_marginal_ks"])
        check(key + " comparisons derived from raw scores", s3.comparison(ledger) == ledger["comparisons"])

    check("public experiment files unchanged", all(s3.digest(ROOT / p) == h for p, h in data["public_package_sha256"].items()))
    check("current implementation recorded", s3.digest(ROOT / "src/probe_s3_allwin_clean.py") == data["implementation_sha256"])
    check("QCBM support alone violates continuous KS gate", 1 / 16 > data["protocol"]["margin_gate_max_ks"], 1 / 16)
    m = s3.Margin(np.geomspace(1e-5, 1., 30))
    finite = m.ppf(np.array([0., .5, 1 - 1e-10, 1.]))
    check("guarded GPD inverse has finite monotone endpoints", bool(np.isfinite(finite).all() and np.all(np.diff(finite) >= 0)), finite.tolist())
    sparse = s3.Margin(np.r_[np.zeros(12), [.1, .2, .3]])
    check("sparse tail preserves zero atom", sparse.ppf(np.array([sparse.pzero / 2]))[0] == 0.)

    names = ["Independent", "Gaussian", "Student t", "Clayton", "Gumbel", "Regular Vine", "QCBM, KS > .65"]
    metric_names = ["Tail LL", "Joint95", "Joint99", "VaR99", "TVaR99", "Brier95", "Energy"]
    # Colorblind-safe outcomes: terracotta loss, slate tie, Prussian blue win.
    colors = ["#C2410C", "#CBD5E1", "#0072B2", "#373a3d"]
    codes = {"loss": 0, "tie": 1, "win": 2, "undefined": 3}
    labels = [("ledger_A", "A: chronological holdout, 10 years"), ("ledger_B38_retrospective_oof", "B: retrospective OOF, 38 years"), ("ledger_B15_observed_pool_oof", "B: 15 country observed pool, 40 years"), ("ledger_B_causal_rolling", "Past only rolling forecast, 26 years")]
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 10), layout="constrained")
    for ax, (key, title) in zip(axes.ravel(), labels):
        ledger = data[key]
        outcomes = ledger["comparisons"]["all_baselines"]
        matrix = np.array([[codes[v] for v in row.values()] for row in outcomes.values()])
        ax.imshow(matrix, cmap=ListedColormap(colors), vmin=0, vmax=3, aspect="auto")
        ax.set_xticks(range(7), metric_names, fontsize=9)
        ax.set_yticks(range(7), names, fontsize=9)
        for i in range(7):
            for j in range(7):
                ax.text(j, i, ["L", "=", "W", "NA"][matrix[i, j]], ha="center", va="center", fontsize=11, color="white" if matrix[i, j] in [0, 2, 3] else "#333333")
        own = ledger["aggregates"][own_name]
        ax.set_title(title + "\nMixture max KS = " + f"{own['max_marginal_ks']:.5f}", fontsize=12, loc="left", pad=12)
        ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Historical loss diagnostics: continuous Bernstein mixture versus seven baselines", fontsize=16, x=.01, ha="left")
    fig.legend(handles=[Patch(facecolor=colors[i], label=label) for i, label in enumerate(["L: baseline better", "=: tie", "W: mixture better", "NA: undefined continuous density"])], loc="outside lower center", ncols=4, frameon=False, fontsize=10)
    png = ROOT / "docs/probe_s3_allwin_clean.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    result = {"checks": checks, "checks_passed": len(checks), "all_checks_passed": True, "figure": str(png), "figure_sha256": s3.digest(png), "result_sha256": s3.digest(ROOT / "data/probe_s3_allwin_clean.json")}
    (ROOT / "data/probe_s3_clean_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}))


if __name__ == "__main__":
    main()
