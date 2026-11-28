"""Stress physical invariants independently of any desired leaderboard outcome."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy import stats
from scipy.linalg import expm

try:
    import pandas as pd
    import probe_official_s1 as s1
    import probe_official_s2_scaling as scale
    import probe_s2_allwin as hist
    import quantum_resources_s2 as quantum
    from s2_contract import BASE_RAIN, RATES, TOTAL_RATES, exact_contract, parameters
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing optional analysis dependency '{exc.name}'. "
        "Run 'python -m pip install -r requirements-analysis.txt' before executing verify_adversarial_s2.py."
    ) from None

ROOT = Path(__file__).resolve().parents[1]


class Checks:
    def __init__(self):
        self.rows = []

    def check(self, name, passed, value=None, bound=None):
        row = {"name": name, "passed": bool(passed)}
        if value is not None:
            row["value"] = value
        if bound is not None:
            row["bound"] = bound
        self.rows.append(row)

    def error(self, name, func):
        try:
            func()
        except (ValueError, TypeError, ArithmeticError) as exc:
            self.check(name, True, type(exc).__name__)
        except Exception as exc:
            self.check(name, False, "Unexpected " + type(exc).__name__)
        else:
            self.check(name, False, "Invalid input was accepted")


def main():
    start = time.perf_counter()
    c = Checks()
    s2 = json.loads((ROOT / "data/probe_official_s2_results.json").read_text())
    b = json.loads((ROOT / "data/probe_s2_allwin_results.json").read_text())
    q = json.loads((ROOT / "data/quantum_resource_estimation_s2.json").read_text())
    sweep = []
    web_cases = []
    for m in [0., 1., 2., 5., 10.]:
        result = exact_contract(multiplier=m)
        sweep.append(result)
        label = "m" + str(int(m))
        c.check(label + ".term_normalization", abs(result["term"].sum() - 1) <= 1e-14,
                float(result["term"].sum()), 1e-14)
        c.check(label + ".term_nonnegative", result["term"].min() >= 0,
                float(result["term"].min()))
        c.check(label + ".generator_rows_sum_zero", np.max(abs(result["generator"].sum(axis=1))) < 1e-14)
        off = result["generator"].copy()
        np.fill_diagonal(off, 0)
        c.check(label + ".generator_offdiagonals_nonnegative", off.min() >= 0)
        for j in range(2):
            actual = np.bincount(s1.STATES[:, j], weights=result["annual"], minlength=5)
            expected = scale.compound_capped(TOTAL_RATES[j], RATES[j] / TOTAL_RATES[j], 1.)
            c.check(label + ".annual_EQ_margin_" + str(j), np.max(abs(actual-expected)) < 1e-14,
                    float(np.max(abs(actual-expected))), 1e-14)
        web_cases.append({"parameters": {"m": m, "rain": BASE_RAIN, "sharedRain": BASE_RAIN*.25,
                                          "eqcap": 70., "threshold": 80.},
                          "term": result["term"].tolist(), "targets": result["targets"].tolist()})
    for index in [1, 3, 4]:
        values = [r["targets"][index] for r in sweep]
        c.check("monotonic_shared_EQ_target_" + str(index), np.all(np.diff(values) >= -1e-9), values)
    el = [r["targets"][0] for r in sweep]
    c.check("m0_1_2_5_10_expected_loss_conserved", max(el)-min(el) < 1e-7, max(el)-min(el), 1e-7)
    for scenario in s1.PUBLISHED:
        result = exact_contract(**parameters(scenario))
        original = s1.exact_term(s1.annual_model(scenario)[0])
        c.check(scenario + ".exact_100_state_term_preserved", np.max(abs(original-result["term"])) < 1e-14)
        truth = np.array(list(s2["scenarios"][scenario]["exact_targets"].values()))
        c.check(scenario + ".all_five_targets_preserved", np.allclose(result["targets"], truth, atol=1e-8, rtol=1e-14))
        p = parameters(scenario)
        web_cases.append({"scenario": scenario, "parameters": {"m":p["multiplier"], "rain":p["rain_rate"],
                         "sharedRain":p["shared_rain"], "eqcap":p["eq_cap_m"], "threshold":p["threshold_m"]},
                         "term":result["term"].tolist(), "targets":result["targets"].tolist()})
        tilt = scale.TerminalTilt(s1.annual_model(scenario)[0], scenario)
        analytic = np.sum(tilt.proposal[:,None]*tilt.weight[:,None]*tilt.features, axis=0)
        c.check(scenario + ".IS_likelihood_identity", np.allclose(analytic, truth, atol=1e-8, rtol=1e-14))
        c.check(scenario + ".proposal_support_complete", np.all(tilt.proposal[tilt.term > 0] > 0))
        x, paths = tilt.sample(4096, 1729, True, True)
        c.check(scenario + ".deterministic_QMC_replay", np.array_equal(x, tilt.sample(4096, 1729)))
        c.check(scenario + ".three_year_paths_valid", paths.shape == (4096, 3) and paths.min() >= 0 and paths.max() < 100)
        c.check(scenario + ".raw_path_payoff_likelihood_replay", np.all(np.isfinite(x)) and x[4] > 0)
        for method, row in s2["scenarios"][scenario]["sampling_at_4096"].items():
            estimates = np.array(row["replicate_estimates"])
            rmse = np.sqrt(np.mean((estimates-truth)**2, axis=0))/truth
            saved = np.array([row["relative_rmse"][k] for k in s1.TARGETS])
            c.check(scenario + "." + method + ".recomputed_RMSE", np.allclose(rmse, saved, atol=1e-12, rtol=1e-10))
            c.check(scenario + "." + method + ".full_32_replicates", estimates.shape == (32,5))
        own = s2["scenarios"][scenario]["sampling_at_4096"]["MPS-QMC-IS"]
        c.check(scenario + ".zero_exhaustion_replicates", own["zero_exhaustion_estimate_replicates"] == 0)
        for key in s1.TARGETS:
            for baseline in scale.BASELINES:
                c.check(scenario + "." + key + ".beats_" + baseline,
                        own["relative_rmse"][key] < s2["scenarios"][scenario]["sampling_at_4096"][baseline]["relative_rmse"][key])
    for cap in [0., 60., 70., 80.]:
        for threshold in [0., 80., 160.]:
            r = exact_contract(eq_cap_m=cap, threshold_m=threshold)
            c.check(f"contract.cap{cap}.T{threshold}.valid_mass", abs(r["term"].sum()-1) < 1e-14)
            c.check(f"contract.cap{cap}.T{threshold}.payout_bound", np.max(r["loss_m"]) <= 160 and np.min(r["loss_m"]) >= 0)
            c.check(f"contract.cap{cap}.T{threshold}.stop_loss_bound", 0 <= r["targets"][3] <= r["targets"][0]+1e-8)
            if threshold == 160:
                c.check(f"contract.cap{cap}.zero_top_layer", r["targets"][3] == r["targets"][2] == 0)
    for rate in [0., BASE_RAIN, 2*BASE_RAIN]:
        r = exact_contract(rain_rate=rate, shared_rain=rate)
        c.check(f"rain{rate}.complete_shared_stochastic", abs(r["annual"].sum()-1) < 1e-14 and r["annual"].min() >= 0)
        c.check(f"rain{rate}.no_unilateral_rain", abs(r["annual"][(s1.STATES[:,2] != s1.STATES[:,3])].sum()) < 1e-14)
    c.check("zero_CTMC_generator_identity", np.array_equal(expm(np.zeros((25,25))), np.eye(25)))
    for graph in scale.GRAPHS:
        for n in range(2,16):
            r = scale.model(n,graph)
            p = scale.contract(r["cores"])
            prefix = f"n{n}.{graph}."
            c.check(prefix + "probability_sum", abs(p.sum()-1) <= 1e-14, float(abs(p.sum()-1)), 1e-14)
            c.check(prefix + "nonnegative_pmf", p.min() >= 0)
            c.check(prefix + "nonnegative_cores", all(x.min() >= 0 for x in r["cores"]))
            c.check(prefix + "grid_support", len(p) == 32*n+1)
            for j in {0,n-1}:
                k = j%2
                eq = scale.compound_capped(TOTAL_RATES[k],RATES[k]/TOTAL_RATES[k])
                expected = np.outer(eq,[.98**3,1-.98**3]).ravel()
                c.check(prefix + f"country{j}.marginal_conserved", np.max(abs(scale.marginal(r["cores"],j)-expected)) < 1e-14)
            empty = scale.contract(scale.model(n,graph,horizon=0.)["cores"])
            c.check(prefix + "zero_horizon_delta", empty[0] == 1 and np.count_nonzero(empty) == 1)
            if n <= 3:
                tensor = scale.dense(r["cores"])
                c.check(prefix + "dense_independent_normalization", abs(tensor.sum()-1) < 1e-14)
            if graph == "global":
                web_cases.append({"n":n,"portfolio_pmf":p.tolist(),"all_exhaustion_probability":float(p[-1])})
    for rate in [0., .01, 1., 10.]:
        cp = scale.compound_capped(rate, [.25]*4)
        c.check(f"compound{rate}.mass", abs(cp.sum()-1) < 1e-14)
        c.check(f"compound{rate}.nonnegative", cp.min() >= 0)
        if rate == 0:
            c.check("zero_occurrence_compound_delta", np.array_equal(cp,[1,0,0,0,0]))
    for n in [2,5,15]:
        cores = []
        weights = np.ones(10)/10
        for j in range(n):
            if j == 0:
                core = np.diag(weights)[None,:,:]
            elif j == n-1:
                core = np.eye(10)[:,:,None]
            else:
                core = np.zeros((10,10,10))
                for h in range(10): core[h,h,h] = 1
            cores.append(core)
        p = scale.contract(cores)
        expected = np.zeros(32*n+1)
        expected[scale.LOSS_UNITS*n] = .1
        c.check(f"perfect_correlation_n{n}.exact_distribution", np.max(abs(p-expected)) < 1e-14)
    training = pd.read_csv(ROOT / "data/processed/loss_gdp_panel.csv")
    training = training.loc[(training.year>=1985)&(training.year<=2013),hist.COUNTRIES].dropna().to_numpy(float)
    margins = [hist.Margin(training[:,j]) for j in range(4)]
    u = hist.pit(training,margins,hist.SEED)
    cfg = b["training_selection"]["candidate_grid_declared_before_test"][b["training_selection"]["selected_candidate"]]
    ev = hist.MPSEVCumulant(u,**cfg)
    c.check("historical_frozen_parameter_hash_replayed", ev.parameter_hash == b["frozen_parameter_sha256"])
    c.check("historical_uniform_sample_replay", np.array_equal(ev.sample(4096),ev.sample(4096)))
    sample = ev.sample(16384)
    c.check("historical_samples_finite_interior", np.all(np.isfinite(sample)) and np.all((sample>0)&(sample<1)))
    c.check("continuous_EV_strictly_positive_lambda", ev.lambda_upper>0, ev.lambda_upper)
    c.check("mixture_weights_conserved", abs(ev.weights.sum()-1) <= 1e-14 and ev.weights.min()>=0)
    for j,margin in enumerate(margins):
        grid = np.linspace(0,1,5001)
        err = float(np.max(abs(ev.marginal_cdf(grid,j)-grid)))
        c.check(f"EV_margin{j}.full_CDF", err<=1e-12,err,1e-12)
        probs = np.r_[np.linspace(margin.pt+1e-9,.999999,1000),np.nextafter(1.,0.)]
        c.check(f"GPD{j}.tail_roundtrip", np.max(abs(margin.cdf(margin.ppf(probs))-probs)) < 1e-12)
        c.check(f"GPD{j}.u0_is_zero", margin.ppf(0.) == 0.)
        c.check(f"GPD{j}.u1_is_infinite", np.isinf(margin.ppf(1.)))
        c.check(f"GPD{j}.no_loss_support_below_zero", margin.cdf(-1.) == 0.)
        c.error(f"GPD{j}.reject_nan",lambda margin=margin:margin.ppf(np.nan))
        c.error(f"GPD{j}.reject_u_above_one",lambda margin=margin:margin.ppf(1.01))
    for shape in [-.25,0.,.2,.95]:
        ps = np.array([0.,.25,.5,.9,.999999])
        inverse = -2*np.log1p(-ps) if shape==0 else 2*np.expm1(-shape*np.log1p(-ps))/shape
        c.check(f"GPD_shape{shape}.independent_inverse_formula", np.max(abs(inverse-stats.genpareto.ppf(ps,shape,scale=2.))) < 1e-7)
        c.check(f"GPD_shape{shape}.quantiles_nondecreasing", np.all(np.diff(inverse)>=0))
        if shape<0:
            c.check("bounded_GPD_extreme_upper_endpoint", stats.genpareto.ppf(1.,shape,scale=2.) == -2/shape)
    for n in [0,1,16,2.5]:c.error(f"reject_country_count_{n}",lambda n=n:scale.model(n))
    for m in [-1.,float('nan'),float('inf'),100.]:c.error("reject_EQ_multiplier_"+str(m),lambda m=m:scale.model(2,m=m))
    for horizon in [-1.,float('nan')]:c.error("reject_horizon_"+str(horizon),lambda horizon=horizon:scale.model(2,horizon=horizon))
    c.error("reject_empty_training",lambda:hist.MPSEVCumulant(np.empty((0,4))))
    c.error("reject_NaN_training",lambda:hist.MPSEVCumulant(np.full((20,4),np.nan)))
    c.error("reject_zero_EV_sampling_count",lambda:ev.sample(0))
    c.error("reject_negative_compound_rate",lambda:scale.compound_capped(-1,[.25]*4))
    c.error("reject_unnormalized_severity_weights",lambda:scale.compound_capped(.1,[1,1,1,1]))
    c.error("reject_zero_rank",lambda:scale.round_mps(scale.model(2)["cores"],0))
    c.error("reject_empty_core_list",lambda:scale.round_mps([],1))
    c.error("reject_nonphysical_shared_rain",lambda:exact_contract(shared_rain=.2))
    c.error("reject_negative_sublimit",lambda:exact_contract(eq_cap_m=-1))
    c.error("reject_stoploss_above_support",lambda:exact_contract(threshold_m=200))
    resource_cases = []
    for n in [2,5,10,15]:
        true_p = float(scale.contract(scale.model(n)["cores"])[-1])
        r = quantum.estimate(n,10,.01,true_p)
        c.check(f"quantum_n{n}.qubits_include_work", r["end_to_end"]["qubits"]>r["purified_mps_loader"]["qubits"])
        c.check(f"quantum_n{n}.inverse_loading_in_Q",r["end_to_end"]["cnot_per_Q"]>2*r["purified_mps_loader"]["cnot_count"])
        c.check(f"quantum_n{n}.epsilon_relative_not_absolute",r["iqae"]["epsilon_absolute"]==.01*true_p)
        c.check(f"quantum_n{n}.hardware_measurements_explicit",r["end_to_end"]["native_mapped_counts"] is None)
        c.check(f"quantum_n{n}.precision_cost_monotonic",quantum.iqae_bound(.001,.005)["grover_q_calls_upper_bound"]>quantum.iqae_bound(.001,.01)["grover_q_calls_upper_bound"])
        resource_cases.append({"n":n,"chi":10,"relative":.01,"p":r["iqae"]["epsilon_absolute"]/.01,"expected":r})
    for p in [0.,-1.,2.,float('nan')]:c.error("reject_IQAE_probability_"+str(p),lambda p=p:quantum.iqae_bound(p))
    for chi in [0,1.5,257]:c.error("reject_quantum_rank_"+str(chi),lambda chi=chi:quantum.estimate(2,chi))
    loader = quantum.loader_verification()
    c.check("actual_RY_CNOT_loader_norm",loader["statevector_norm_error"]<1e-12)
    c.check("actual_RY_CNOT_loader_official_pmf",loader["marginal_term_pmf_max_abs_error"]<1e-12)
    c.check("actual_RY_CNOT_loader_amplitude",loader["amplitude_max_abs_error"]<1e-12)
    c.check("compiled_loader_CNOTs_counted",loader["cnot_count"]==494)
    c.check("TN1_retirement_explicit",q["rows"][0]["hardware_mapping"]["TN1"]["status"]=="retired")
    c.check("Aquila_analog_not_gate_mapped",not q["rows"][0]["hardware_mapping"]["QuEra Aquila"]["loader_width_fits"])
    c.check("historical_metrics_keep_nonwins",b["allwin"]["all_seven_metrics_against_all_seven_baselines"] is False if "all_seven_metrics_against_all_seven_baselines" in b["allwin"] else not all(x=="win" for row in b["mps_vs_baselines"].values() for x in row.values()))
    for r in b["results"][:-1]:
        for key,direction in hist.METRICS.items():
            own=b["results"][-1][key]
            expected="not_comparable" if own is None or r[key] is None else "win" if direction*(own-r[key])>1e-12 else "loss" if direction*(own-r[key]) < -1e-12 else "tie"
            c.check("historical_WLT."+r["model"]+"."+key,b["mps_vs_baselines"][r["model"]][key]==expected)
    old = json.loads((ROOT / "data/s2_s1_preservation_manifest.json").read_text())
    for name,expected in old["files"].items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        c.check("S1_immutable."+name,actual==expected)
    owned=[p for p in (ROOT/"src").glob("*s2*.py")]
    for p in owned:
        c.check("zero_dash_punctuation."+p.name,not any(ch in p.read_text() for ch in "\u2013\u2014\u2015"))
    reference={"contract_cases":[r for r in web_cases if "parameters" in r],"portfolio_cases":[r for r in web_cases if "n" in r],"resource_cases":resource_cases}
    (ROOT/"data/s2_web_reference.json").write_text(json.dumps(reference,allow_nan=False,separators=(",",":"))+"\n")
    failures=[r for r in c.rows if not r["passed"]]
    result={"schema":"adb.verify.adversarial.s2.v1","count":len(c.rows),"passed_count":len(c.rows)-len(failures),
            "passed":not failures,"checks":c.rows,"failures":failures,"runtime_seconds":time.perf_counter()-start,
            "peak_resident_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope":"Physical invariants, exact replay, independent formulas, saved outcome consistency and unchanged S1 inputs. Statistical wins remain a separate measured result."}
    (ROOT/"data/verify_adversarial_s2.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ["count","passed_count","passed","failures","runtime_seconds","peak_resident_bytes"]},indent=2))
    return 0 if not failures else 1


if __name__=="__main__":
    raise SystemExit(main())
