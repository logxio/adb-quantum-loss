"""Parameterized contractual CTMC with a conserved compound Poisson law."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.linalg import expm

from probe_official_s1 import CAP, COMBINE, PUBLISHED, STATES, compound_rates

RATES = np.array([compound_rates(t)[0] for t in (
    [.03883, .02331, .01790, .01184],
    [.03925, .01743, .01332, .01043],
)])
TOTAL_RATES = RATES.sum(axis=1)
BASE_SHARED_EQ = (.075 * TOTAL_RATES[0] + .074 * TOTAL_RATES[1]) / 2
BASE_RAIN = -np.log(.98)


def parameters(scenario="adb-base"):
    if scenario not in PUBLISHED:
        raise ValueError("Unknown contractual scenario")
    rain_rate = BASE_RAIN * (2 if scenario == "adb-rain-stress" else 1)
    return {"multiplier": {"adb-independent": 0., "adb-common-shock": 2.}.get(scenario, 1.),
            "rain_rate": float(rain_rate),
            "shared_rain": float(0 if scenario == "adb-independent" else .25 * rain_rate),
            "eq_cap_m": 60. if scenario == "adb-rain-cover" else 70., "threshold_m": 80.}


def annual_distribution(multiplier=1., rain_rate=BASE_RAIN, shared_rain=BASE_RAIN*.25,
                        eq_cap_m=70., threshold_m=80.):
    values = [multiplier, rain_rate, shared_rain, eq_cap_m, threshold_m]
    if not np.all(np.isfinite(values)):
        raise ValueError("Parameters must be finite")
    common = multiplier * BASE_SHARED_EQ
    if multiplier < 0 or common > TOTAL_RATES.min():
        raise ValueError("Shared earthquake intensity exceeds available marginal intensity")
    if rain_rate < 0 or not 0 <= shared_rain <= rain_rate:
        raise ValueError("Shared rain intensity must lie inside the marginal intensity")
    if not 0 <= eq_cap_m <= 80 or not 0 <= threshold_m <= 160:
        raise ValueError("Contractual sublimit or threshold outside the bond support")
    w = RATES / TOTAL_RATES[:, None]
    generator = np.zeros((25, 25))
    for a in range(5):
        for b in range(5):
            row = a*5+b
            for k in range(1, 5):
                generator[row, min(a+k, 4)*5+b] += (TOTAL_RATES[0]-common)*w[0, k-1]
                generator[row, a*5+min(b+k, 4)] += (TOTAL_RATES[1]-common)*w[1, k-1]
                for j in range(1, 5):
                    generator[row, min(a+k, 4)*5+min(b+j, 4)] += common*w[0,k-1]*w[1,j-1]
            generator[row, row] -= generator[row].sum()
    eq = expm(generator)[0].reshape(5,5)
    p00 = np.exp(-2*rain_rate+shared_rain)
    p10 = np.exp(-rain_rate)-p00
    rain = np.array([p00,p10,p10,1-p00-2*p10])
    annual = (eq[:,:,None]*rain).ravel()
    return annual, generator


def exact_contract(**kwargs):
    params = parameters()
    params.update(kwargs)
    annual, generator = annual_distribution(**params)
    term = np.zeros(100)
    term[0] = 1.
    for _ in range(3):
        term = np.bincount(COMBINE.ravel(), weights=(term[:,None]*annual).ravel(), minlength=100)
    eq_cap = params["eq_cap_m"]
    rain_cap = 80-eq_cap
    a = STATES[:,0]*eq_cap/4+STATES[:,2]*rain_cap
    b = STATES[:,1]*eq_cap/4+STATES[:,3]*rain_cap
    loss = a+b
    features = np.stack([loss*1e6,(a>0)&(b>0),loss>params["threshold_m"],
                         np.maximum(loss-params["threshold_m"],0)*1e6,
                         (a>=80-1e-12)&(b>=80-1e-12)],axis=1)
    return {"annual":annual,"generator":generator,"term":term,"targets":term@features,
            "loss_m":loss,"parameters":params}
