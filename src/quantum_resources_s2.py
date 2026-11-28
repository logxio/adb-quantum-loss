"""Auditable ideal gate ledgers and IQAE bounds for the S2 catastrophe model."""
from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

from probe_official_s1 import annual_model, compound_rates, exact_term, TARGETS

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/quantum_sources_s2"
P_BASE = 0.000031633046


def iqae_bound(p: float, rel_error: float = .01, alpha: float = .05):
    """Grinko et al., Theorem 1, Eq. 8; Q applications include repeated powers."""
    if not (0 < p <= 1 and 0 < rel_error < 1 and 0 < alpha < 1):
        raise ValueError("Require 0<p<=1, 0<relative error<1 and 0<alpha<1")
    epsilon = p * rel_error
    if epsilon >= math.pi / 8:
        raise ValueError("This ledger uses the small error regime epsilon<pi/8")
    logterm = math.log((2 / alpha) * math.log2(math.pi / (4 * epsilon)))
    shots_per_round = 32 / (1 - 2 * math.sin(math.pi / 14)) ** 2 * logterm
    rounds = math.ceil(math.log2(math.pi / (8 * epsilon)))
    nq = math.ceil(50 / epsilon * logterm)
    shots = rounds * math.ceil(shots_per_round)
    # Eq. 2 is a separate canonical QAE comparison, not an IQAE theorem count.
    a = 2 * math.pi * math.sqrt(p * (1-p))
    canonical_m = math.ceil((a + math.sqrt(a*a + 4*epsilon*math.pi**2)) / (2*epsilon))
    return {
        "epsilon_absolute": epsilon,
        "relative_error": rel_error,
        "alpha": alpha,
        "confidence": 1-alpha,
        "grover_q_calls_upper_bound": nq,
        "max_rounds": rounds,
        "shots_per_round_upper_bound": math.ceil(shots_per_round),
        "shots_upper_bound": shots,
        "a_or_inverse_a_calls_upper_bound": 2*nq+shots,
        "oracle_unit": "One Q=A S0 A_dagger Sgood application, weighted by its power and shot count",
        "actual_adaptive_calls": None,
        "count_status": "Theorem upper bound, not an observed hardware runtime or an IQAE run",
        "canonical_qae_eq2_M_at_8_over_pi2_success": canonical_m,
        "canonical_note": "Variance aware canonical QAE comparison only; 8/pi^2 confidence needs repetition to reach 95 percent",
    }


def mcx_cost(controls: int):
    """Exact clean v-chain template, six CNOT and nine 1q gates per Toffoli."""
    if controls < 0:
        raise ValueError("controls must be nonnegative")
    if controls == 0:
        return {"cnot": 0, "one_qubit": 1, "ancillas": 0}
    if controls == 1:
        return {"cnot": 1, "one_qubit": 0, "ancillas": 0}
    toffoli = 2*controls-3
    return {"cnot": 6*toffoli, "one_qubit": 9*toffoli,
            "ancillas": controls-2}


def qsd_cnot(q: int):
    """Shende, Bullock and Markov generic unitary upper bound."""
    if q <= 1:
        return 0
    return (23*4**q-72*2**q+64)//48


def estimate(n: int, chi: int = 10, rel_error: float = .01,
             p: float = P_BASE, alpha: float = .05):
    """Return a parameterized resource ledger; chi denotes latent component count."""
    if not isinstance(n, (int, np.integer)) or not 2 <= n <= 15:
        raise ValueError("n must be an integer from 2 through 15")
    if not isinstance(chi, (int, np.integer)) or not 1 <= chi <= 256:
        raise ValueError("chi must be an integer from 1 through 256")
    b = int(math.ceil(math.log2(chi)))
    r = 2**b
    data = 4*n
    acc = int(math.ceil(math.log2(32*n+1)))
    load_cx = max(r-2, 0) + n*r*15 if b else 14*n
    load_ry = r-1+n*r*15
    load = {"qubits": data+b, "data_qubits": data, "latent_qubits": b,
            "latent_components": int(chi), "padded_latent_dimension": r,
            "local_physical_dimension": 10, "padded_local_dimension": 16,
            "cnot_count": load_cx, "ry_count": load_ry,
            "serial_depth_upper": load_cx+load_ry,
            "status": "Exact ideal RY plus CNOT gate template; native transpilation and rotation synthesis remain separate"}
    iqae = iqae_bound(p, rel_error, alpha)
    event = mcx_cost(data)
    reflection = mcx_cost(data+b)
    work = max(event["ancillas"], reflection["ancillas"])
    qubits = data+b+1+work
    a_cx = load_cx+event["cnot"]
    # Negative controls select local state 9=1001, two zero bits per country.
    a_1q = load_ry+event["one_qubit"]+4*n
    s0_cx = reflection["cnot"]
    s0_1q = reflection["one_qubit"]+2*(data+b+1)+2
    q_cx = 2*a_cx+s0_cx
    q_depth = 2*(a_cx+a_1q)+s0_cx+s0_1q+1
    total_cx = iqae["grover_q_calls_upper_bound"]*q_cx+iqae["shots_upper_bound"]*a_cx
    primary = {
        "payoff": "All n bonds exhausted; every local eq/rain state equals 9",
        "qubits": qubits, "data_qubits": data, "latent_qubits": b,
        "objective_qubits": 1, "reusable_clean_work_qubits": work,
        "cnot_per_A": a_cx, "one_qubit_gates_per_A": a_1q,
        "cnot_per_Q": q_cx, "serial_depth_per_Q_upper": q_depth,
        "iqae_total_cnot_upper_estimate": total_cx,
        "iqae_total_serial_depth_upper_estimate": iqae["grover_q_calls_upper_bound"]*q_depth+iqae["shots_upper_bound"]*(a_cx+a_1q),
        "native_mapped_counts": None,
        "fault_tolerant_physical_qubits": None,
        "clifford_t_synthesis_counts": None,
        "error_mitigation_overhead": None,
        "status": "Concrete ideal logical gate ledger; width fit alone does not establish executable IQAE",
    }
    # Generic portfolio stop loss uses reversible local lookup, controlled increment,
    # a sum-controlled RY, then uncomputes the entire sum calculation.
    pay = [7*(x//2)+4*(x%2) for x in range(10)]
    lookup_terms = sum(v.bit_count() for v in pay)
    lookup_cx = lookup_terms*mcx_cost(4)["cnot"]
    lookup_1q = sum(v.bit_count()*(mcx_cost(4)["one_qubit"]+2*(4-x.bit_count())) for x,v in enumerate(pay))
    inc_cx, inc_1q, inc_work = 0, 0, 0
    for k in range(5):
        for j in range(acc-1, k, -1):
            c = mcx_cost(j-k+1)
            inc_cx += c["cnot"]
            inc_1q += c["one_qubit"]
            inc_work = max(inc_work, c["ancillas"])
        inc_cx += 1
    sum_forward_cx = n*(2*lookup_cx+inc_cx)
    sum_forward_1q = n*(2*lookup_1q+inc_1q)
    rotation_cx = 2**acc
    rotation_ry = 2**acc
    generic_a_cx = load_cx+2*sum_forward_cx+rotation_cx
    generic_a_1q = load_ry+2*sum_forward_1q+rotation_ry
    generic_q_cx = 2*generic_a_cx+s0_cx
    generic_q_depth = 2*(generic_a_cx+generic_a_1q)+s0_cx+s0_1q+1
    generic = {
        "payoff": "Normalized portfolio stop loss, grid unit USD2500000",
        "qubits": data+b+acc+5+1+max(work,inc_work,2),
        "sum_register_qubits": acc, "local_lookup_qubits": 5,
        "local_lookup_table": pay, "local_lookup_mcx_terms": lookup_terms,
        "cnot_sum_compute": sum_forward_cx,
        "cnot_sum_uncompute": sum_forward_cx,
        "cnot_payoff_rotation": rotation_cx,
        "cnot_per_A": generic_a_cx, "one_qubit_gates_per_A": generic_a_1q,
        "cnot_per_Q": generic_q_cx,
        "serial_depth_per_Q_upper": generic_q_depth,
        "status": "Exact ideal gate template for grid payoffs; arithmetic and uncomputation are fully included",
    }
    compact_q = int(math.ceil(n*math.log2(10)))
    naive = {
        "classical_probability_entries": 10**n,
        "classical_float64_bytes": 8*10**n,
        "nontrivial_branch_rotations_support_aware": 10**n-1,
        "branch_mass_access_assumption": "All required prefix masses are supplied coherently or precomputed; cost cannot be ignored",
        "compact_binary_data_qubits": compact_q,
        "compact_generic_real_loader_cnot": 2**compact_q-2,
        "compact_generic_real_loader_ry": 2**compact_q-1,
        "site_padded_data_qubits": data,
        "site_padded_generic_real_loader_cnot": 2**data-2,
        "site_padded_generic_real_loader_ry": 2**data-1,
        "site_padded_serial_depth_upper": 2**(data+1)-3,
        "status": "Chosen dense UCR implementation, not a lower bound on structured Grover Rudolph loading",
    }
    amplitude = {
        "assumed_amplitude_bond_dimension": int(chi),
        "padded_amplitude_bond_dimension": r,
        "local_unitary_qubits": 4+b,
        "qubits": data+b,
        "cnot_upper_bound": n*qsd_cnot(4+b),
        "two_qubit_serial_depth_upper": n*qsd_cnot(4+b),
        "status": "Conditional generic QSD upper bound for sequential canonical amplitude MPS isometries; not an observed compilation",
        "amplitude_rank_certificate_for_unpurified_probability": None,
        "note": "Probability MPS rank does not bound the rank of its entrywise square root; purification is the certified loader above",
    }
    maps = {
        "SV1": {"qubit_limit":34, "loader_width_fits":data+b<=34,
                "exhaustion_oracle_width_fits":qubits<=34,
                "end_to_end_iqae_demonstrated":False,
                "boundary":"State vector simulator, maximum runtime six hours; exponential memory and long IQAE depth still apply"},
        "TN1": {"status":"retired", "retired_date":"2026-06-23",
                "loader_width_fits":False,"exhaustion_oracle_width_fits":False,
                "end_to_end_iqae_demonstrated":False,
                "boundary":"Retired device; historical 50 qubit descriptions do not establish current availability"},
        "IonQ Forte": {"qubit_limit":36,"loader_width_fits":data+b<=36,
                       "exhaustion_oracle_width_fits":qubits<=36,
                       "native_two_qubit_gate":"XX family after transpilation",
                       "on_demand_gates_per_circuit_limit":2000,
                       "reservation_gates_per_circuit_limit":5000,
                       "end_to_end_iqae_demonstrated":False,
                       "boundary":"All to all connectivity; native gate count, calibration and Grover powers require a fresh device check"},
        "IQM Garnet": {"qubit_limit":20,"loader_width_fits":data+b<=20,
                       "exhaustion_oracle_width_fits":qubits<=20,
                       "native_two_qubit_gate":"CZ",
                       "end_to_end_iqae_demonstrated":False,
                       "boundary":"Square lattice routing adds SWAPs; native count and noise remain unmeasured"},
        "QuEra Aquila": {"atom_limit":256,"paradigm":"Analog Hamiltonian Simulation",
                         "loader_width_fits":False,"exhaustion_oracle_width_fits":False,
                         "end_to_end_iqae_demonstrated":False,
                         "boundary":"Analog Rydberg device does not execute this arbitrary RY/CNOT plus IQAE gate circuit"},
        "QCentroid Phoenix": {"status":"Phase2 sandbox eligibility is described by ADB",
                              "public_generic_launchpad_cpu_cores":16,
                              "public_generic_launchpad_ram_gb":32,
                              "phoenix_gpu_model":None,"phoenix_gpu_count":None,
                              "assigned_credits":None,"assigned_walltime_limit":None,
                              "end_to_end_iqae_demonstrated":False,
                              "boundary":"Generic LaunchPad specification is not a Phoenix allocation or a quantum device capacity guarantee"},
    }
    return {"n":int(n),"single_year_states":10**n,"chi":int(chi),
            "purified_mps_loader":load,"naive_grover_rudolph":naive,
            "unpurified_amplitude_mps":amplitude,"iqae":iqae,
            "end_to_end":primary,"generic_stop_loss_oracle":generic,
            "hardware_mapping":maps}


def global_factors(n: int):
    """Exact term model conditional on capped shared counts, for loader verification."""
    rates = [compound_rates([.03883,.02331,.01790,.01184])[0],
             compound_rates([.03925,.01743,.01332,.01043])[0]]
    totals = [r.sum() for r in rates]
    common = (.075*totals[0]+.074*totals[1])/2
    rain = -math.log(.98)
    shared_rain = .25*rain
    pk = np.r_[poisson.pmf(np.arange(4),3*common),poisson.sf(3,3*common)]
    pr = np.array([math.exp(-3*shared_rain),-math.expm1(-3*shared_rain)])
    weights = np.outer(pk,pr).ravel()
    f = np.empty((10,n,10))
    for j in range(n):
        z = j%2
        severity = rates[z]/totals[z]
        generator = np.zeros((5,5))
        for x in range(5):
            for a in range(1,5):
                generator[x,min(x+a,4)] += 3*(totals[z]-common)*severity[a-1]
            generator[x,x] -= generator[x].sum()
        base = expm(generator)[0]
        common_severity = np.r_[0.,severity]
        kr = base.copy()
        for k in range(5):
            eq = kr if k<4 else np.array([0.,0.,0.,0.,1.])
            for r in range(2):
                p = 1. if r else -math.expm1(-3*(rain-shared_rain))
                f[2*k+r,j] = np.outer(eq,[1-p,p]).ravel()
            if k<3:
                kr = np.bincount(np.minimum(np.arange(5)[:,None]+np.arange(5),4).ravel(),
                                 weights=np.outer(kr,common_severity).ravel(),minlength=5)
    return weights,f


def uniformly_controlled_ry(angles, controls, target):
    """Gray code synthesis with an explicit final CNOT cycle."""
    angles = np.asarray(angles,dtype=float)
    k = len(controls)
    if len(angles) != 2**k:
        raise ValueError("One angle is required for every control bitstring")
    if k == 0:
        return [("ry",target,float(angles[0]))]
    gray = np.arange(2**k,dtype=np.int64)
    gray ^= gray>>1
    signs = np.array([[1 if (int(i)&int(j)).bit_count()%2==0 else -1
                       for j in range(2**k)] for i in gray])
    transformed = signs@angles/(2**k)
    gates = []
    for i,a in enumerate(transformed):
        gates.append(("ry",target,float(a)))
        change = int(gray[i]^gray[(i+1)%(2**k)])
        gates.append(("cx",controls[k-1-(change.bit_length()-1)],target))
    return gates


def conditional_real_preparation(probabilities, controls, targets):
    """Prepare nonnegative amplitudes conditional on an external binary register."""
    probabilities = np.asarray(probabilities,dtype=float)
    branches = 2**len(controls)
    q = len(targets)
    if probabilities.shape != (branches,2**q):
        raise ValueError("Conditional probability table dimensions do not match registers")
    if probabilities.min() < -1e-15 or not np.allclose(probabilities.sum(axis=1),1,atol=1e-14):
        raise ValueError("Every conditional branch must be a probability vector")
    gates = []
    for k in range(q):
        table = probabilities.reshape(branches,2**k,2,-1).sum(axis=-1)
        denom = table.sum(axis=-1)
        ratio = np.divide(table[:,:,1],denom,out=np.zeros_like(denom),where=denom>0)
        angles = 2*np.arcsin(np.sqrt(np.clip(ratio,0,1)))
        gates += uniformly_controlled_ry(angles.ravel(),list(controls)+list(targets[:k]),targets[k])
    return gates


def compile_purified_loader(weights, local):
    """Return an actual RY/CNOT gate list for a nonnegative product mixture."""
    weights,local = np.asarray(weights),np.asarray(local)
    chi,n,d = local.shape
    if d != 10 or weights.shape != (chi,):
        raise ValueError("The S2 loader requires d=10 and one weight per component")
    b = math.ceil(math.log2(chi))
    r = 2**b
    wp = np.pad(weights,(0,r-chi))
    gates = conditional_real_preparation(wp[None,:],[],list(range(b)))
    for j in range(n):
        pp = np.zeros((r,16))
        pp[:chi,:10] = local[:,j]
        pp[chi:,0] = 1
        gates += conditional_real_preparation(pp,list(range(b)),list(range(b+4*j,b+4*j+4)))
    return gates,b+4*n


def simulate_real_gates(gates,qubits):
    """Small exact CPU statevector check; bounded to sixteen qubits."""
    if qubits>16:
        raise ValueError("Verification is limited to sixteen qubits")
    state = np.zeros(2**qubits)
    state[0] = 1.
    indices = np.arange(2**qubits,dtype=np.int64)
    for gate in gates:
        if gate[0] == "ry":
            _,target,angle = gate
            mask = 1<<target
            ix = indices[(indices&mask)==0]
            other = ix|mask
            c,s = math.cos(angle/2),math.sin(angle/2)
            a,z = state[ix].copy(),state[other].copy()
            state[ix],state[other] = c*a-s*z,s*a+c*z
        else:
            _,control,target = gate
            ix = indices[((indices&(1<<control))!=0)&((indices&(1<<target))==0)]
            other = ix|(1<<target)
            state[ix],state[other] = state[other].copy(),state[ix].copy()
    return state


def loader_verification():
    weights,local = global_factors(2)
    gates,q = compile_purified_loader(weights,local)
    state = simulate_real_gates(gates,q)
    marginal = np.zeros((10,10))
    expected_amp = np.zeros_like(state)
    for h in range(10):
        for x in range(10):
            for y in range(10):
                seq = f"{h:04b}{x:04b}{y:04b}"
                ix = sum(int(bit)<<i for i,bit in enumerate(seq))
                expected_amp[ix] = math.sqrt(weights[h]*local[h,0,x]*local[h,1,y])
                marginal[x,y] += state[ix]**2
    annual,_ = annual_model("adb-base")
    official = exact_term(annual).reshape(5,5,2,2).transpose(0,2,1,3).reshape(10,10)
    matrix_error = float(np.max(abs(marginal-official)))
    amplitude_error = float(np.max(abs(state-expected_amp)))
    if matrix_error>1e-12 or amplitude_error>1e-12:
        raise AssertionError("The synthesized loader does not reproduce the official term model")
    unpurified_singular = np.linalg.svd(np.sqrt(official),compute_uv=False)
    return {"n":2,"qubits":q,"gate_count":len(gates),
            "cnot_count":sum(g[0]=="cx" for g in gates),
            "ry_count":sum(g[0]=="ry" for g in gates),
            "statevector_norm_error":float(abs(state@state-1)),
            "amplitude_max_abs_error":amplitude_error,
            "marginal_term_pmf_max_abs_error":matrix_error,
            "unpurified_sqrt_probability_matrix_singular_values":unpurified_singular.tolist(),
            "unpurified_sqrt_probability_matrix_rank_at_1e_12":int(np.sum(unpurified_singular>1e-12)),
            "hardware_used":False,"engine":"numpy real statevector over the actual synthesized gate list"}


def source_manifest():
    """Reference URLs accompany the public resource estimates."""
    result = {"checked_at_jst": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(),
            "scope": "Source documents are obtained separately.",
            "criteria_url": "https://challenges.adb.org/en/challenges/quantum-joint-loss-challenge/pages/timeline-process-and-selection-criteria?lang=en"}
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    (SOURCE_DIR/"manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def run():
    s1 = json.loads((ROOT/"data/probe_official_s1_results.json").read_text())
    p = s1["scenarios"]["adb-base"]["exact_targets"][TARGETS[4]]
    rows = [estimate(n,10,.01,p) for n in (2,5,10,15)]
    verification = loader_verification()
    for row in rows:
        weights,local = global_factors(row["n"])
        gates,q = compile_purified_loader(weights,local)
        actual_cx = sum(g[0]=="cx" for g in gates)
        actual_ry = sum(g[0]=="ry" for g in gates)
        assert actual_cx == row["purified_mps_loader"]["cnot_count"]
        assert actual_ry == row["purified_mps_loader"]["ry_count"]
        row["purified_mps_loader"]["actual_generated_gate_count"] = len(gates)
        row["purified_mps_loader"]["gate_list_sha256"] = hashlib.sha256(json.dumps(gates,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        model_p = float(weights@np.prod(local[:,:,9],axis=1))
        row["model_specific_exhaustion_probability"] = model_p
        row["model_specific_exhaustion_iqae"] = iqae_bound(model_p)
        model_nq = row["model_specific_exhaustion_iqae"]["grover_q_calls_upper_bound"]
        model_ns = row["model_specific_exhaustion_iqae"]["shots_upper_bound"]
        row["model_specific_iqae_total_cnot_upper_estimate"] = model_nq*row["end_to_end"]["cnot_per_Q"]+model_ns*row["end_to_end"]["cnot_per_A"]
    manifest = source_manifest()
    result = {
        "schema":"adb.quantum.resource.s2.v1","checked_at_jst":manifest["checked_at_jst"],
        "hardware_used":False,"paid_tasks_submitted":0,
        "environment":{"python":platform.python_version(),"numpy":np.__version__},
        "rows":rows,"loader_statevector_verification":verification,
        "tail_probability_scenario":{"p":p,"source":"Official two country adb-base term exhaustion probability",
                                     "n_greater_than_2_interpretation":"Common probability sensitivity scenario; not a measured n-country exhaustion probability"},
        "stop_loss_normalization":{"loss_unit_usd":2500000,"n2_threshold_usd":80000000,
                                   "n2_max_payoff_usd":80000000,
                                   "n2_expected_stop_loss_usd":s1["scenarios"]["adb-base"]["exact_targets"][TARGETS[3]],
                                   "n2_amplitude":s1["scenarios"]["adb-base"]["exact_targets"][TARGETS[3]]/80000000,
                                   "n2_iqae":iqae_bound(s1["scenarios"]["adb-base"]["exact_targets"][TARGETS[3]]/80000000)},
        "cost_accounting":{
            "classical":"Calibration, TT construction, canonicalization and gate angle synthesis precede coherent oracle execution",
            "loading":"Purification prepares shared latent and conditional local states; latent is retained, never postselected",
            "payoff":"Specialized exhaustion MCX and generic loss lookup plus addition plus rotation are separately listed",
            "inverse":"Every Q includes A and its inverse, plus zero and good state reflections",
            "sampling":"IQAE counts Q applications across all shots and powers, plus initial A per circuit shot",
            "precision":"RY angles are ideal real rotations; fault tolerant synthesis, code distance and physical overhead require a chosen gate error budget",
            "bias":"Allocate separate loading, arithmetic and statistical error budgets before a 1 percent total relative error claim",
        },
        "references":[
            {"id":"chakrabarti2021","url":"https://quantum-journal.org/papers/q-2021-06-01-463/","doi":"10.22331/q-2021-06-01-463","used_for":"Full loading, payoff, inverse, oracle repetition, precision and fault tolerant cost accounting"},
            {"id":"egger2021","url":"https://arxiv.org/abs/1907.03044","doi":"10.1109/TC.2020.3038063","used_for":"Correlated latent loading U, summation S and comparator C; A=C S U"},
            {"id":"grinko2021","url":"https://www.nature.com/articles/s41534-021-00379-1","doi":"10.1038/s41534-021-00379-1","used_for":"Theorem1 Eqs6 through8, epsilon=relative_error times probability"},
            {"id":"shende2006","url":"https://arxiv.org/abs/quant-ph/0406176","used_for":"Uniformly controlled rotations and generic QSD unitary synthesis bound"},
            {"id":"mottonen2005","url":"https://arxiv.org/abs/quant-ph/0407010","used_for":"Explicit Gray code controlled rotation angles and ideal CNOT gate template"},
            {"id":"grover_rudolph2002","url":"https://arxiv.org/abs/quant-ph/0208112","used_for":"Conditional prefix mass loading; efficient integrability is an assumption, not free arbitrary data access"},
            {"id":"braket_devices","url":"https://docs.aws.amazon.com/braket/latest/developerguide/braket-devices.html"},
            {"id":"braket_simulators","url":"https://docs.aws.amazon.com/braket/latest/developerguide/braket-submit-tasks-simulators.html"},
            {"id":"braket_retirement","url":"https://docs.aws.amazon.com/braket/latest/developerguide/doc-history.html","fact":"TN1 retired June23,2026"},
            {"id":"braket_quotas","url":"https://docs.aws.amazon.com/braket/latest/developerguide/braket-quotas.html"},
            {"id":"ionq","url":"https://www.ionq.com/quantum-systems/forte"},
            {"id":"iqm","url":"https://aws.amazon.com/braket/quantum-computers/iqm/"},
            {"id":"quera","url":"https://aws.amazon.com/braket/quantum-computers/quera/"},
            {"id":"qcentroid","url":"https://docs.qcentroid.com/launchpad/getting-started/"},
            {"id":"adb_sandbox","url":"https://challenges.adb.org/en/challenges/quantum-joint-loss-challenge/pages/timeline-process-and-selection-criteria?lang=en"},
        ],
        "source_manifest":"data/quantum_sources_s2/manifest.json",
        "interpretation":"Classical low rank contraction and an ideal coherent loader are demonstrated separately; an end to end quantum speedup has not been measured",
    }
    path = ROOT/"data/quantum_resource_estimation_s2.json"
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    result = run()
    print(json.dumps({"output":"data/quantum_resource_estimation_s2.json",
                      "loader_verification":result["loader_statevector_verification"],
                      "rows":[{"n":r["n"],"loader_cnot":r["purified_mps_loader"]["cnot_count"],
                               "oracle_qubits":r["end_to_end"]["qubits"],
                               "cnot_per_Q":r["end_to_end"]["cnot_per_Q"],
                               "iqae_Q_bound":r["iqae"]["grover_q_calls_upper_bound"]} for r in result["rows"]]},indent=2))
