# S4: verified dual-track qjlb results

The standalone solver passes the official dual-track benchmark in all three modes. Each primary case uses 10 fresh processes and an 8,192-sample budget. The suite contains seven unchanged public scenarios and six cases generated with seed 20261122.

| Mode | Passed cases | ADB geometric time ratio | World Bank geometric time ratio | Worst normalized RMSE |
|---|---:|---:|---:|---:|
| exact | 13/13 | 0.067833 | 0.128084 | 3.26104e-12 |
| sample | 13/13 | 1.034221 | 0.237206 | 0.290554 |
| hybrid | 13/13 | 0.240150 | 0.226789 | 1.64849e-10 |

Ratios divide full fresh-process median time by the fastest accuracy-qualified official baseline for each case, then take the geometric mean within each track. Accuracy, process startup, loading, preparation, execution and output writing are included. Official ranking remains provisional pending independent code or execution-evidence review.

| Official baseline | Passed cases |
|---|---:|
| exact | 13/13 |
| mc | 0/13 |
| qmc | 0/13 |
| stratified | 1/13 |
| importance | 0/13 |

Both additional holdout seeds, 20261123 and 20261124, pass 6/6 cases in each mode with 10 repeats through the unchanged official invocation and scoring functions. These additional suites establish measured generalization across 12 new contracts.

The solver is a single 21536-byte file. ADB exact execution imports only the standard library. World Bank exact execution imports NumPy and reads the rate and loss arrays once per peril. Existing S2 implementations, submission documents and the website remain unchanged.

Sampling uses genuine seed-dependent estimates and exactly 8,192 sampled paths or event rows. ADB uses exponential terminal tilting, scrambled Sobol draws and complete backward-bridge paths (structural variance reduction with full-model access). World Bank uses covariate-only rate-weighted stratification over contract trigger regimes and fixed geometric loss bands (zero target expectations enter stratum weights or budget allocation), with randomized stratified inverse-CDF draws per stratum.

Hybrid execution emits and executes OpenQASM 3.0 RY/CX circuits. ADB uses 12 qubits and contracts the full simulated term-state marginal. World Bank uses a 3-, 4- or 6-qubit latent-peril/attachment-signature circuit and weights per-signature conditional expectations directly by the simulated statevector probabilities. Circuit source, SHA-256, actual seeded shot counts (recorded as execution evidence rather than used to compute the deterministic statevector estimates) and unique local job IDs are retained. The backend is the verified deterministic statevector implementation; Qiskit and PennyLane were not installed.

## Horizon and common-shock findings

Only the earthquake common-rate multiplier varies below. Rain common share stays fixed at 0.25. Each horizon has invariant expected loss across earthquake coupling, while joint exhaustion and stop loss rise. This is smooth nonlinear horizon dependence.

| Years | EQ multiplier | Expected principal loss USD | Both exhausted probability | Stop loss USD |
|---:|---:|---:|---:|---:|
| 1 | 0 | 3,415,425.000000 | 6.49926808144e-07 | 19,529.234545 |
| 1 | 1 | 3,415,425.000000 | 1.8794874246e-06 | 53,691.487259 |
| 1 | 2 | 3,415,425.000000 | 3.1092713729e-06 | 87,843.043326 |
| 3 | 0 | 10,063,513.348954 | 1.92689839677e-05 | 179,587.682571 |
| 3 | 1 | 10,063,513.348954 | 3.16330463092e-05 | 282,354.698566 |
| 3 | 2 | 10,063,513.348954 | 4.40015421761e-05 | 385,043.370223 |
| 10 | 0 | 31,486,318.788892 | 0.00090172828851 | 2,062,243.304290 |
| 10 | 1 | 31,486,318.788892 | 0.00107886405291 | 2,382,658.869162 |
| 10 | 2 | 31,486,318.788892 | 0.00125604183552 | 2,702,995.623019 |

## Central Asia regional pool

The catalogue contains 419,968 seismic rows and 2,000 flood rows, totaling 421,968. Nested portfolios use identical country layers and a portfolio attachment equal to half total limits. Active means at least one selected country loss exceeds its peril-specific attachment.

| Countries | Active rows | Active percent | All-country joint rate/year | Stop loss USD/year |
|---|---:|---:|---:|---:|
| KAZ, KGZ | 5,795 | 1.373327% | 0.125000275 | 1,236,767.106573 |
| KAZ, KGZ, TJK | 7,668 | 1.817199% | 0.0725 | 2,465,085.780134 |
| KAZ, KGZ, TJK, TKM, UZB | 15,135 | 3.586765% | 0.024 | 5,718,288.360000 |

The five-country all-country event produces only 4.572 expected positive observations in 8,192 direct IID catalogue draws. Under uniform Monte Carlo its relative standard error is approximately 47%, against a 5% probability-target tolerance. Across these nested contracts the all-country joint rate declines, while absolute portfolio stop loss increases. The full report separates seismic and flood contributions and includes every case-specific baseline error.

## Reproduction and verification

```sh
.venv/bin/python src/run_s4_qjlb.py
.venv/bin/python src/run_s4_qjlb.py suite-13 exact
.venv/bin/python src/run_s4_qjlb.py suite-13 sample
.venv/bin/python src/run_s4_qjlb.py suite-13 hybrid
.venv/bin/python src/verify_s4_qjlb.py --additional-processes
.venv/bin/python src/s4_scientific_findings.py
.venv/bin/python src/verify_s4_qjlb.py
```

The official runner preserves output folders. Move a previous evaluation directory before repeating its command. All raw runs are in `data/s4/`; the consolidated machine-readable report is `data/s4_qjlb_monopoly_report.json`. Precision checks use absolute probability error and scale currency errors by max(1, absolute reference).

Independent verification: **3684/3684 assertions pass**, including 200 randomized contracts, all target comparisons, official score recomputation, seeded sampling, circuit hashes, shot budgets and an independent tensor-axis statevector implementation.
