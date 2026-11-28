# S4 rare event and quantum baselines on the ADB DRB contract

All methods estimate the same five term targets of the official two country contract (adb-base, adb-independent, adb-common-shock, adb-rain-stress, adb-rain-cover). Reference values are the exact capped CTMC contraction in `src/s2_contract.py`. Each classical run spends exactly 4,096 contract evaluations; each configuration is repeated over 32 independent seeds. A miss is a replicate with absolute relative error above 10%.

## Dual exhaustion P(A and B)

Relative RMSE over 32 seeds, misses in parentheses. IQAE uses the 12 qubit loader with at most 4,096 Grover calls and 4,096 shots per run.

| Scenario | Exact | MPS-QMC-IS | Oracle-Tilted IS | CE-IS | Subset Simulation | IQAE |
|---|---|---|---|---|---|---|
| adb-base | 3.1633e-05 | 0.0504% (0) | 7.89% (6) | 62.1% (30) | 95% (30) | 33.3% (26) |
| adb-independent | 3.9239e-06 | 0.0613% (0) | 12.1% (12) | 84.6% (29) | 102% (31) | 307% (32) |
| adb-common-shock | 4.4002e-05 | 0.044% (0) | 11.8% (13) | 344% (26) | 82.4% (29) | 22.2% (20) |
| adb-rain-stress | 6.9246e-05 | 0.0481% (0) | 8.24% (8) | 42% (25) | 68.1% (28) | 20.5% (20) |
| adb-rain-cover | 3.1633e-05 | 0.000429% (0) | 8.54% (9) | 62.3% (22) | 62.6% (27) | 33.3% (26) |

## All targets

Relative RMSE. MPS-QMC-IS answers all five targets from one 4,096 path run; every baseline receives a separate 4,096 evaluation run per target (Subset Simulation shares one run across the three loss targets).

### adb-base

| Method | Expected loss | Joint attachment | P(L>T) | Stop loss | Dual exhaustion | ms per run |
|---|---|---|---|---|---|---|
| MPS-QMC-IS | 0.0384% (0) | 0.884% (0) | 0.146% (0) | 0.0531% (0) | 0.0504% (0) | 1.7 |
| Oracle-Tilted Rate IS | 1.19% (0) | 2.65% (0) | 3.58% (0) | 2.87% (0) | 7.89% (6) | 3.1 |
| CE-IS | 2.91% (1) | 6.32% (4) | 14.5% (15) | 13.7% (9) | 62.1% (30) | 28.2 |
| Subset Simulation | 5.27% (3) | 11.8% (11) | 19.5% (19) | 21.9% (20) | 95% (30) | 10.4 |

### adb-independent

| Method | Expected loss | Joint attachment | P(L>T) | Stop loss | Dual exhaustion | ms per run |
|---|---|---|---|---|---|---|
| MPS-QMC-IS | 0.0381% (0) | 1.77% (0) | 0.091% (0) | 0.0394% (0) | 0.0613% (0) | 1.4 |
| Oracle-Tilted Rate IS | 1.29% (0) | 3.55% (0) | 3.7% (0) | 3.1% (0) | 12.1% (12) | 3.0 |
| CE-IS | 1.52% (0) | 4.63% (1) | 9.11% (6) | 5.14% (3) | 84.6% (29) | 4.0 |
| Subset Simulation | 5.63% (3) | 13.4% (16) | 17.8% (21) | 17.2% (17) | 102% (31) | 8.1 |

### adb-common-shock

| Method | Expected loss | Joint attachment | P(L>T) | Stop loss | Dual exhaustion | ms per run |
|---|---|---|---|---|---|---|
| MPS-QMC-IS | 0.0362% (0) | 1.21% (0) | 0.0895% (0) | 0.0335% (0) | 0.044% (0) | 1.4 |
| Oracle-Tilted Rate IS | 1.37% (0) | 2.63% (0) | 3.88% (1) | 2.68% (0) | 11.8% (13) | 3.0 |
| CE-IS | 3.73% (1) | 4.22% (0) | 15.6% (12) | 8.89% (7) | 344% (26) | 3.9 |
| Subset Simulation | 4.91% (1) | 11% (16) | 14.7% (15) | 16.9% (14) | 82.4% (29) | 8.9 |

### adb-rain-stress

| Method | Expected loss | Joint attachment | P(L>T) | Stop loss | Dual exhaustion | ms per run |
|---|---|---|---|---|---|---|
| MPS-QMC-IS | 0.0292% (0) | 0.644% (0) | 0.152% (0) | 0.0648% (0) | 0.0481% (0) | 1.6 |
| Oracle-Tilted Rate IS | 1.4% (0) | 2.12% (0) | 3.9% (1) | 2.91% (0) | 8.24% (8) | 3.1 |
| CE-IS | 2.61% (0) | 3.13% (0) | 15.5% (11) | 10.6% (9) | 42% (25) | 4.3 |
| Subset Simulation | 4.78% (3) | 11.5% (12) | 15% (15) | 15.8% (18) | 68.1% (28) | 9.2 |

### adb-rain-cover

| Method | Expected loss | Joint attachment | P(L>T) | Stop loss | Dual exhaustion | ms per run |
|---|---|---|---|---|---|---|
| MPS-QMC-IS | 0.0459% (0) | 0.72% (0) | 0.179% (0) | 0.0485% (0) | 0.000429% (0) | 1.5 |
| Oracle-Tilted Rate IS | 1.33% (0) | 2.69% (0) | 3.86% (0) | 2.79% (0) | 8.54% (9) | 3.0 |
| CE-IS | 2.17% (0) | 6.32% (4) | 46.2% (15) | 13.2% (12) | 62.3% (22) | 3.9 |
| Subset Simulation | 4.35% (1) | 11.6% (11) | 19.2% (18) | 19.6% (20) | 62.6% (27) | 9.3 |

### Calibration of the baselines

Mean relative bias with its standard error (32 seeds). Values near zero within two standard errors indicate unbiased behavior; CI noncoverage counts nominal 95% CLT intervals that miss the reference.

| Scenario | Method | Joint attachment bias | P(L>T) bias | Dual exhaustion bias | CI misses (DE) |
|---|---|---|---|---|---|
| adb-base | Oracle-Tilted Rate IS | +0.16% ± 0.48% | +0.52% ± 0.64% | -0.22% ± 1.42% | 1/32 |
| adb-base | CE-IS | +0.17% ± 1.13% | +0.51% ± 2.61% | -22.29% ± 10.40% | 13/32 |
| adb-base | Subset Simulation | -3.14% ± 2.03% | -3.65% ± 3.43% | -3.74% ± 17.05% | no CI |
| adb-independent | Oracle-Tilted Rate IS | -0.17% ± 0.64% | -1.61% ± 0.60% | +0.48% ± 2.17% | 0/32 |
| adb-independent | CE-IS | -1.13% ± 0.81% | +0.24% ± 1.64% | -25.69% ± 14.48% | 16/32 |
| adb-independent | Subset Simulation | -3.43% ± 2.32% | -5.63% ± 3.03% | -25.81% ± 17.72% | no CI |
| adb-common-shock | Oracle-Tilted Rate IS | +0.14% ± 0.47% | +0.38% ± 0.69% | +3.59% ± 2.01% | 3/32 |
| adb-common-shock | CE-IS | -0.30% ± 0.76% | -1.01% ± 2.79% | +61.25% ± 60.84% | 9/32 |
| adb-common-shock | Subset Simulation | -1.70% ± 1.95% | -1.39% ± 2.63% | +2.98% ± 14.79% | no CI |
| adb-rain-stress | Oracle-Tilted Rate IS | -0.24% ± 0.38% | +0.64% ± 0.69% | +0.21% ± 1.48% | 0/32 |
| adb-rain-stress | CE-IS | -0.42% ± 0.56% | +2.36% ± 2.75% | -7.85% ± 7.41% | 11/32 |
| adb-rain-stress | Subset Simulation | -2.55% ± 2.02% | -1.98% ± 2.67% | -20.93% ± 11.64% | no CI |
| adb-rain-cover | Oracle-Tilted Rate IS | +0.22% ± 0.48% | +0.34% ± 0.69% | +0.20% ± 1.53% | 1/32 |
| adb-rain-cover | CE-IS | +0.17% ± 1.13% | +6.33% ± 8.22% | -1.79% ± 11.19% | 10/32 |
| adb-rain-cover | Subset Simulation | -4.60% ± 1.90% | -1.76% ± 3.43% | +10.12% ± 11.10% | no CI |

## IQAE at a matched budget

Grover operator Q = -A S0 A^dagger S_chi with A the 494 CNOT plus 495 RY purified loader and a 78 CNOT flag; 572 CNOT per A, 1270 CNOT per Q. Configuration chosen per scenario from a declared grid (Clopper Pearson or Chernoff Hoeffding, 32 to 256 shots per round).

| Scenario | Config | Rel RMSE | Misses | CI misses | Oracle calls | Shots | k max | Total CNOT |
|---|---|---|---|---|---|---|---|---|
| adb-base | clopper_pearson_shots32 | 33.3% | 26 | 0 | 2,783 | 178 | 75 | 3.636e+06 |
| adb-independent | clopper_pearson_shots32 | 307.1% | 32 | 0 | 2,132 | 160 | 48 | 2.799e+06 |
| adb-common-shock | clopper_pearson_shots32 | 22.2% | 20 | 0 | 3,049 | 185 | 75 | 3.978e+06 |
| adb-rain-stress | clopper_pearson_shots32 | 20.5% | 20 | 0 | 3,092 | 193 | 74 | 4.037e+06 |
| adb-rain-cover | clopper_pearson_shots32 | 33.3% | 26 | 0 | 2,783 | 178 | 75 | 3.636e+06 |

## IQAE target accuracy on adb-base dual exhaustion

p = 3.163305e-05, 100 shots per round, alpha = 0.05. Means over 32 seeds. Stop rule angle is the Grinko et al. criterion (angle interval width at most epsilon over pi), which is a worst case bound near amplitude 0.5 and overshoots small targets; stop rule amplitude halts once the amplitude CI half width is at most the target. Projection uses the amplitude rule. total CNOT = shots x CNOT(A) + oracle calls x CNOT(Q).

| Stop | CI | Target half width | Rel RMSE | Within target | Oracle calls | Shots | k max | Max circuit CNOT | Total CNOT |
|---|---|---|---|---|---|---|---|---|---|
| amplitude | clopper_pearson | 5.00% | 0.839% | 32/32 (0 stalled) | 168,506 | 600 | 1,903 | 2,417,382 | 2.143e+08 |
| amplitude | clopper_pearson | 1.00% | 0.137% | 32/32 (0 stalled) | 761,428 | 700 | 8,726 | 11,082,592 | 9.674e+08 |
| amplitude | clopper_pearson | 0.50% | 0.048% | 32/32 (0 stalled) | 2,370,747 | 762 | 27,414 | 34,816,352 | 3.011e+09 |
| amplitude | clopper_pearson | 0.20% | 0.030% | 32/32 (0 stalled) | 3,551,488 | 800 | 42,653 | 54,169,882 | 4.511e+09 |
| amplitude | clopper_pearson | 0.10% | 0.015% | 32/32 (0 stalled) | 13,020,878 | 875 | 139,161 | 176,735,042 | 1.654e+10 |
| amplitude | clopper_pearson | 0.05% | 0.008% | 32/32 (0 stalled) | 17,243,128 | 900 | 205,739 | 261,289,102 | 2.190e+10 |
| amplitude | clopper_pearson | 0.02% | 0.003% | 32/32 (0 stalled) | 66,652,672 | 981 | 693,527 | 880,779,862 | 8.465e+10 |
| amplitude | chernoff | 5.00% | 33.803% | 31/32 (1 stalled) | 14,752,766 | 63,344 | 2,734 | 3,472,752 | 1.877e+10 |
| amplitude | chernoff | 1.00% | 0.169% | 32/32 (0 stalled) | 1,092,819 | 1,028 | 13,568 | 17,231,932 | 1.388e+09 |
| amplitude | chernoff | 0.50% | 0.075% | 32/32 (0 stalled) | 2,277,416 | 1,088 | 25,046 | 31,808,992 | 2.893e+09 |
| amplitude | chernoff | 0.20% | 0.046% | 32/32 (0 stalled) | 5,178,466 | 1,150 | 66,264 | 84,155,852 | 6.577e+09 |
| amplitude | chernoff | 0.10% | 0.017% | 32/32 (0 stalled) | 10,121,575 | 1,206 | 124,641 | 158,294,642 | 1.286e+10 |
| amplitude | chernoff | 0.05% | 0.010% | 32/32 (0 stalled) | 21,676,659 | 1,275 | 273,327 | 347,125,862 | 2.753e+10 |
| amplitude | chernoff | 0.02% | 0.003% | 32/32 (0 stalled) | 61,380,162 | 1,359 | 678,441 | 861,620,642 | 7.795e+10 |
| angle | clopper_pearson | 5.00% | 0.053% | 32/32 (0 stalled) | 21,555,966 | 900 | 247,486 | 314,307,792 | 2.738e+10 |
| angle | clopper_pearson | 1.00% | 0.001% | 32/32 (0 stalled) | 95,138,106 | 1,000 | 1,102,144 | 1,399,723,452 | 1.208e+11 |
| angle | chernoff | 5.00% | 33.792% | 31/32 (1 stalled) | 32,670,566 | 63,688 | 239,858 | 304,620,232 | 4.153e+10 |
| angle | chernoff | 1.00% | 0.002% | 32/32 (0 stalled) | 93,107,606 | 1,375 | 1,232,931 | 1,565,822,942 | 1.182e+11 |

Matching the MPS-QMC-IS relative RMSE of 0.0504% (log log fit of measured rows):

| CI | Slope | Oracle calls | Total CNOT | Naive MC draws | MC draws / IQAE calls |
|---|---|---|---|---|---|
| clopper_pearson | -1.069 | 2.637e+06 | 3.351e+09 | 1.244e+11 | 47165.7 |
| chernoff | -1.028 | 3.803e+06 | 4.831e+09 | 1.244e+11 | 32706.9 |

## Reading the comparison

Query scaling. The measured IQAE slope of about -1.07 against relative RMSE confirms the quadratic improvement over naive Monte Carlo, whose draw count scales with slope -2. At the MPS-QMC-IS accuracy, naive Monte Carlo needs 1.24e+11 draws and IQAE about 2.64e+06 Grover calls.

Gate and time cost. The same accuracy requires about 3.35e+09 ideal CNOT executions, about 3,351 s of serial two qubit time at an assumed 1 us per CNOT, before error correction. MPS-QMC-IS reaches it with 4,096 classical paths in 1.7 ms per run, plus a one time proposal setup of 14 ms, on one CPU core. At a matched budget of at most 4,096 Grover calls, IQAE reaches 33.3% relative RMSE on adb-base.

Ablation. Oracle-Tilted Rate IS uses the best product rate and mark tilt found offline with 2^17 draws per CE iteration, and spends all 4,096 draws on it. The gap between it and MPS-QMC-IS isolates the contribution of the terminal diagonal MPO tilt, the exact Markov bridge and the scrambled Sobol stratification. CE-IS spends part of the same 4,096 on its own adaptation and shows the cost of learning the tilt in budget.

Where amplitude estimation becomes competitive. MPS-QMC-IS depends on an exact probability MPS of low bond dimension (chi = 10 here). The table below holds n = 15 countries and the 1% IQAE query count fixed and grows chi. Classical exact contraction scales as n chi^2 in time and memory; a generic amplitude MPS loader also scales as chi^2 in CNOT count, so amplitude estimation does not remove the bond dimension cost. It becomes the better route only for dependence graphs with no low rank classical contraction (high treewidth, non separable) that still admit a compact state preparation circuit.

Two loaders are listed. The purified mixture loader covers latent mixture models such as this contract (ledger range chi <= 256); the generic amplitude MPS loader uses the QSD upper bound for arbitrary MPS.

| chi | Classical flops | Memory | Measured ms | Fits 4 GB | IQAE CNOT at 1%, purified | CNOT per A, generic QSD | IQAE CNOT at 1%, generic | Serial s at 1 us, generic |
|---|---|---|---|---|---|---|---|---|
| 10 | 3.00e+03 | 0.0 MiB | 0.014 | yes | 7.14e+09 | 4.66e+05 | 7.10e+11 | 7.10e+05 |
| 32 | 3.07e+04 | 0.1 MiB | 0.020 | yes | 1.27e+10 | 1.87e+06 | 2.85e+12 | 2.85e+06 |
| 100 | 3.00e+05 | 1.1 MiB | 0.026 | yes | 4.57e+10 | 3.01e+07 | 4.58e+13 | 4.58e+07 |
| 316 | 3.00e+06 | 11.4 MiB | 0.093 | yes | not applicable | 4.82e+08 | 7.34e+14 | 7.34e+08 |
| 1,000 | 3.00e+07 | 114.4 MiB | 1.384 | yes | not applicable | 1.93e+09 | 2.94e+15 | 2.94e+09 |
| 3,162 | 3.00e+08 | 1,144.2 MiB | not run | yes | not applicable | 3.09e+10 | 4.70e+16 | 4.70e+10 |
| 10,000 | 3.00e+09 | 11,444.1 MiB | not run | no | not applicable | 4.94e+11 | 7.52e+17 | 7.52e+11 |
| 100,000 | 3.00e+11 | 1,144,409.2 MiB | not run | no | not applicable | 3.16e+13 | 4.81e+19 | 4.81e+13 |

The 1 us per serial CNOT layer is an assumption for scale, not a measured device figure.

## Built in verification

| Scenario | Max abs LR z | Grover angle error k<=64 | Grover angle error k<=128 | Norm drift k<=128 | Loader terminal error |
|---|---|---|---|---|---|
| adb-base | 2.74 | 1.7e-13 | 8.4e-13 | 8.6e-13 | 2.2e-16 |
| adb-independent | 1.28 | 8.8e-14 | 6.5e-13 | 2.7e-12 | 6.0e-15 |
| adb-common-shock | 2.41 | 4.2e-13 | 1.4e-12 | 1.4e-12 | 3.0e-15 |
| adb-rain-stress | 1.12 | 2.4e-13 | 5.2e-13 | 6.4e-13 | 5.6e-16 |
| adb-rain-cover | 0.98 | 1.7e-13 | 8.4e-13 | 8.6e-13 | 2.2e-16 |

Every classical run spends exactly 4,096 contract evaluations (asserted). Likelihood ratios of each oracle proposal average to one within five standard errors over 2^18 draws. Both latent samplers reproduce the exact 100 state terminal law within six standard errors over 2^20 nominal draws. The 12 qubit loader has 494 CNOT and 495 RY gates, ||A|0>|| = 1 to 1e-12, and its good state mass equals the contract exhaustion probability to 1e-12. Gate by gate Grover powers match sin^2((2k+1)theta) to 1e-12 through k = 64. All outputs are finite.

## Method notes

CE-IS (Rubinstein and Kroese 2004). Product proposal over the six Poisson means (own and shared earthquake, own and shared rain) and the four mark distributions. Multilevel elite quantile rho = 0.1 on a plateau free score, adaptive rho down to 0.01 when the event is already hit, smoothing 0.8, 512 draws per pilot iteration, at most five iterations, remaining draws for the final estimate with exact likelihood ratios.

Subset Simulation (Au and Beck 2001). 22 dimensional standard normal latent space mapped exactly to the capped contract; scores add a fractional cell position below the smallest payout gap, so every final threshold equals the contract event. p0 = 0.1, preconditioned Crank Nicolson conditional sampling with acceptance adapted toward 0.44, level size chosen per target from {512, 1024}; leftover budget extends the final level.

IQAE (Grinko et al. 2021). Algorithm 1 with next power selection, Clopper Pearson or Chernoff Hoeffding intervals at alpha / T, and binomial shot sampling at each power. Success probabilities come from the verified Grover rotation of the gate level statevector.

MPS-QMC-IS. `TerminalTilt` from `src/probe_official_s2_scaling.py`, seeds 1729 to 1760, unchanged.
