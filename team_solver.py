"""Portable joint loss solver with timed structural preparation and simulation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


def capped(a, b):
    out = [0.0] * len(a)
    cap = len(a) - 1
    for i, p in enumerate(a):
        for j, q in enumerate(b):
            out[min(i + j, cap)] += p * q
    return out


def power(a, n):
    out = [1.0] + [0.0] * (len(a) - 1)
    while n:
        if n & 1:
            out = capped(out, a)
        n >>= 1
        if n:
            a = capped(a, a)
    return out


def poisson4(lam):
    p = [math.exp(-lam)]
    for k in range(1, 4):
        p.append(p[-1] * lam / k)
    t = p[-1] * lam / 4
    tail = t
    for k in range(5, 1000):
        t *= lam / k
        tail += t
        if t <= tail * 1e-17:
            break
    return p + [tail]


def compound(rates, duration):
    r = [x * duration for x in rates]
    p = [math.exp(-sum(r))]
    for k in range(1, 4):
        p.append(sum(j * r[j - 1] * p[k - j] for j in range(1, k + 1)) / k)
    # The tail includes every path with four or more positive jumps.
    pk = poisson4(sum(r))
    marks = [0.0] + [x / sum(r) for x in r] if sum(r) else [1., 0., 0., 0., 0.]
    tail = pk[4]
    for k in range(1, 4):
        tail += pk[k] * power(marks, k)[4]
    return p + [tail]


def bond_factors(c, terms, years=None):
    years = c['term_years'] if years is None else years
    rates = []
    for country in c['countries']:
        tail = terms['annual_earthquake_exceedance_probabilities'][country]
        p0 = 1 - tail[0]
        r1 = (tail[0] - tail[1]) / p0
        r2 = (tail[1] - tail[2]) / p0 - r1*r1/2
        r3 = (tail[2] - tail[3]) / p0 - r1*r2 - r1**3/6
        rates.append([r1, r2, r3, -math.log(p0)-r1-r2-r3])
    totals = list(map(sum, rates))
    cond = terms['same_event_conditional_trigger_probabilities']
    common = (totals[0]*cond['TJK_given_KGZ'] + totals[1]*cond['KGZ_given_TJK'])/2
    common *= c['earthquake_common_rate_multiplier']
    rain = [-math.log1p(-x)*c['precipitation_rate_multiplier'] for x in c['precipitation_annual_trigger_probability']]
    shared_rain = min(rain)*c['precipitation_common_rate_share']
    pk = poisson4(years*common)
    pr = [math.exp(-years*shared_rain), -math.expm1(-years*shared_rain)]
    weights = [p*q for p in pk for q in pr]
    local = [[None, None] for _ in weights]
    for j in range(2):
        marks = [0.] + [r/totals[j] for r in rates[j]]
        base = compound([r*(1-common/totals[j]) for r in rates[j]], years)
        for k in range(5):
            eq = capped(base, power(marks, k)) if k < 4 else [0., 0., 0., 0., 1.]
            for bit in range(2):
                rp = [0., 1.] if bit else [math.exp(-years*(rain[j]-shared_rain)), -math.expm1(-years*(rain[j]-shared_rain))]
                local[2*k+bit][j] = [a*b for a in eq for b in rp]
    return weights, local


def bond_distribution(weights, local):
    return [math.fsum(weights[h]*local[h][0][2*a+x]*local[h][1][2*b+y] for h in range(10))
            for a in range(5) for b in range(5) for x in range(2) for y in range(2)]


def bond_features(c):
    eq, rain = c['earthquake_sublimit_usd'], c['precipitation_sublimit_usd']
    out = []
    for a in range(5):
        for b in range(5):
            for x in range(2):
                for y in range(2):
                    la, lb = a*eq[0]/4+x*rain[0], b*eq[1]/4+y*rain[1]
                    loss = la+lb
                    out.append([loss, float(la>0 and lb>0), float(loss>c['portfolio_threshold_usd']),
                                max(0., loss-c['portfolio_threshold_usd']), float(a==4 and b==4 and x==1 and y==1)])
    return out


def bond_sample(c, terms, p, features, samples, seed):
    import numpy as np
    from scipy.stats import qmc
    from scipy.optimize import minimize_scalar
    from scipy.special import logsumexp
    p, f = np.asarray(p), np.asarray(features)
    loss = f[:, 0]/sum(c['earthquake_sublimit_usd']+c['precipitation_sublimit_usd'])
    logp = np.log(np.maximum(p, 1e-300))
    proposals, tilts = [], []
    for j in range(5):
        active = (p>0) & (f[:, j]>0)
        if not active.any():
            proposals.append(p/p.sum()); tilts.append(0.)
            continue
        logf = np.log(f[active, j]/f[:, j].max())
        def objective(theta):
            return logsumexp(logp+theta*loss)+logsumexp(logp[active]+2*logf-theta*loss[active])
        theta = float(minimize_scalar(objective, bounds=(-20, 80), method='bounded').x)
        q = np.exp(logp+theta*loss-logsumexp(logp+theta*loss))
        proposals.append(q); tilts.append(theta)
    proposal = np.mean(proposals, axis=0)
    order = np.argsort(loss, kind='stable')
    cdf = np.cumsum(proposal[order]); cdf[-1] = 1.
    years = c['term_years']
    engine = qmc.Sobol(years, scramble=True, seed=seed)
    u = engine.random_base2((samples-1).bit_length())[:samples]
    terminal = order[np.searchsorted(cdf, u[:, 0], side='right')]
    # Exact backward bridges generate all requested complete contract paths.
    states = np.array(list(__import__('itertools').product(range(5), range(5), range(2), range(2))))
    dest = np.minimum(states[:, None, :]+states[None, :, :], [4, 4, 1, 1])
    dest = ((dest[..., 0]*5+dest[..., 1])*2+dest[..., 2])*2+dest[..., 3]
    annual = np.asarray(bond_distribution(*bond_factors(c, terms, 1)))
    alphas = [np.eye(1, 100, 0).ravel()]
    for _ in range(years):
        alphas.append(np.bincount(dest.ravel(), weights=(alphas[-1][:, None]*annual).ravel(), minlength=100))
    pair_indices = [np.flatnonzero(dest.ravel()==end) for end in range(100)]
    paths = np.empty((samples, years), dtype=int)
    current = terminal.copy()
    for step, horizon in enumerate(range(years, 1, -1), 1):
        joint = (alphas[horizon-1][:, None]*annual).ravel()
        for end in np.unique(current):
            rows = np.flatnonzero(current==end)
            pairs = pair_indices[end]
            cc = np.cumsum(joint[pairs]); cc /= cc[-1]; cc[-1] = 1.
            selected = pairs[np.searchsorted(cc, u[rows, step], side='right')]
            paths[rows, horizon-1] = selected%100
            current[rows] = selected//100
    paths[:, 0] = current
    if not np.array_equal(np.minimum(states[paths].sum(axis=1), [4,4,1,1]), states[terminal]):
        raise ArithmeticError('Bridge endpoint mismatch')
    values = f[terminal]*(p[terminal]/proposal[terminal])[:, None]
    return values.mean(axis=0).tolist(), {'samples': samples, 'complete_paths': samples,
        'annual_state_draws': samples*years, 'tilts': tilts, 'terminal_states_scanned':100,
        'path_sha256': hashlib.sha256(paths.tobytes()).hexdigest()}


def elt_load(c, base):
    import numpy as np
    directory = base/c['dataset_directory']
    blocks = []
    details = []
    for h, peril in enumerate(('seismic', 'flood')):
        with np.load(directory/(peril+'.npz'), allow_pickle=False) as z:
            cols = [z['countries'].tolist().index(country) for country in c['countries']]
            rates = z['annual_rate']
            loss = z['loss_usd'][:, cols]
        attachment = np.asarray(c['attachment_usd'][h])
        active = np.any(loss>attachment, axis=1)
        pay = np.zeros_like(loss)
        pay[active] = np.asarray(c['limit_usd'])*np.clip((loss[active]-attachment)/(np.asarray(c['exhaustion_usd'][h])-attachment), 0, 1)
        total = pay.sum(axis=1)
        features = np.column_stack((loss.sum(axis=1), total, np.all(pay>0, axis=1), total>c['portfolio_threshold_usd'], np.maximum(total-c['portfolio_threshold_usd'], 0)))
        signature = ((loss>attachment)* (1 << np.arange(len(cols)))).sum(axis=1)
        blocks.append((rates, features, signature))
        details.append({'peril':peril, 'rows':len(rates), 'active_rows':int(active.sum()),
                        'active_percent':float(100*active.mean()), 'all_country_active_rows':int(np.count_nonzero(features[:,2]))})
    return blocks, details


def elt_sample(blocks, samples, seed):
    import numpy as np
    # Stratify over disjoint contract regimes and fixed geometric loss bands using catalogue rates only.
    rng = np.random.default_rng(seed)
    loss_edges = np.geomspace(1e5, 1e11, 25)
    strata = []
    regime_counts = {}
    offset = 0
    for h, (rates, f, _) in enumerate(blocks):
        n_rows = len(rates)
        pos = np.flatnonzero(rates > 0)
        if len(pos):
            r, fp = rates[pos], f[pos]
            gix = pos + offset
            active = fp[:, 1] > 0
            all_act = fp[:, 2] > 0
            exc = fp[:, 3] > 0
            reg = np.zeros(len(r), dtype=np.int8)
            reg[active & (~all_act) & (~exc)] = 1
            reg[active & (~all_act) & exc] = 2
            reg[all_act & (~exc)] = 3
            reg[all_act & exc] = 4
            bins = np.digitize(fp[:, 0], loss_edges)
            for code in range(5):
                rix = np.flatnonzero(reg == code)
                if not len(rix):
                    continue
                r_reg, f_reg, b_reg, g_reg = r[rix], fp[rix], bins[rix], gix[rix]
                reg_key = (h, code)
                order_b = np.argsort(b_reg, kind='mergesort')
                b_sorted = b_reg[order_b]
                splits = np.flatnonzero(np.diff(b_sorted)) + 1
                starts = np.concatenate(([0], splits))
                ends = np.concatenate((splits, [len(b_sorted)]))
                regime_counts[reg_key] = len(starts)
                for st, en in zip(starts, ends):
                    sub = order_b[st:en]
                    b = int(b_sorted[st])
                    r_sub = r_reg[sub]
                    w_sub = float(r_sub.sum())
                    hi = float(loss_edges[b]) if b < len(loss_edges) else float(f_reg[sub, 0].max())
                    lo = float(loss_edges[b - 1]) if b > 0 else 0.0
                    spread = max(hi - lo, 1.0)
                    strata.append((code > 0, r_sub, f_reg[sub], g_reg[sub], w_sub, w_sub * spread, reg_key))
        offset += n_rows
    k_strata = len(strata)
    if samples < k_strata:
        r_all = np.concatenate([s[1] for s in strata])
        f_all = np.concatenate([s[2] for s in strata])
        g_all = np.concatenate([s[3] for s in strata])
        w_all = float(r_all.sum())
        cdf = np.cumsum(r_all); cdf /= cdf[-1]; cdf[-1] = 1.0
        u = (np.arange(samples) + rng.random(samples)) / samples
        sel = np.searchsorted(cdf, u, side='right')
        return (w_all * f_all[sel].mean(axis=0)).tolist(), {'samples': samples, 'catalogue_rows_scanned': offset,
            'strata_count': 1, 'sample_indices_sha256': hashlib.sha256(g_all[sel].tobytes()).hexdigest(),
            'preparation': 'Rate-weighted catalogue stratification',
            'sampling': 'Seeded randomized stratified inverse CDF over catalogue rates'}
    nw = np.array([s[5] for s in strata], dtype=float)
    nw = nw / nw.sum() if nw.sum() > 0 else np.full(k_strata, 1.0 / k_strata)
    act_regs = sorted({s[6] for s in strata if s[0]})
    if act_regs:
        aw = np.zeros(k_strata, dtype=float)
        for i, s in enumerate(strata):
            if s[0]:
                aw[i] = 1.0 / (len(act_regs) * regime_counts[s[6]])
        aw /= aw.sum()
        alloc_w = 0.65 * nw + 0.35 * aw
    else:
        alloc_w = nw
    raw = alloc_w * (samples - k_strata)
    counts = 1 + np.floor(raw).astype(int)
    rem = samples - int(counts.sum())
    if rem > 0:
        counts[np.argsort(-(raw - np.floor(raw)))[:rem]] += 1
    est = np.zeros(5, dtype=float)
    selected_parts = []
    for (is_act, r_sub, f_sub, g_sub, w_sub, _, _), m in zip(strata, counts):
        if is_act and len(r_sub) > 1:
            order = np.argsort(f_sub[:, 1], kind='stable')
            r_sub = r_sub[order]; f_sub = f_sub[order]; g_sub = g_sub[order]
        cdf = np.cumsum(r_sub); cdf /= cdf[-1]; cdf[-1] = 1.0
        u = (np.arange(m) + rng.random(m)) / m
        sel = np.searchsorted(cdf, u, side='right')
        est += w_sub * f_sub[sel].mean(axis=0)
        selected_parts.append(g_sub[sel])
    selected = np.concatenate(selected_parts)
    return est.tolist(), {'samples': samples, 'catalogue_rows_scanned': offset,
        'strata_count': k_strata, 'sample_indices_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
        'preparation': 'Contract regime and geometric loss-band stratification over catalogue rates',
        'sampling': 'Seeded randomized stratified inverse CDF within rate-weighted strata'}


def ucry(angles, controls, target):
    import numpy as np
    k = len(controls)
    if not k:
        return [('ry', target, float(angles[0]))]
    gray = [i^(i>>1) for i in range(1<<k)]
    # Walsh transform, followed by Gray ordering.
    a = np.asarray(angles, dtype=float).copy()
    step = 1
    while step<len(a):
        v = a.reshape(-1, 2*step)
        lo, hi = v[:,:step].copy(), v[:,step:].copy()
        v[:,:step], v[:,step:] = lo+hi, lo-hi
        step *= 2
    a /= len(a)
    gates = []
    for i, g in enumerate(gray):
        gates.append(('ry', target, float(a[g])))
        change = g^gray[(i+1)%len(gray)]
        gates.append(('cx', controls[k-change.bit_length()], target))
    return gates


def prepare(probabilities, controls, targets):
    import numpy as np
    p = np.asarray(probabilities)
    gates = []
    for k, target in enumerate(targets):
        table = p.reshape(len(p), 1<<k, 2, -1).sum(axis=-1)
        denom = table.sum(axis=-1)
        ratio = np.divide(table[:,:,1], denom, out=np.zeros_like(denom), where=denom>0)
        gates += ucry((2*np.arcsin(np.sqrt(np.clip(ratio,0,1)))).ravel(), controls+targets[:k], target)
    return gates


def simulate_qasm(source):
    """Execute the emitted OpenQASM 3 RY/CX subset with little endian qubits."""
    import numpy as np
    import re
    qubits = int(re.search(r'qubit\[(\d+)\]', source)[1])
    state = np.zeros(1<<qubits); state[0] = 1.
    indices = np.arange(len(state))
    for line in source.splitlines()[3:]:
        if line.startswith('ry('):
            angle, target = re.fullmatch(r'ry\(([^)]+)\) q\[(\d+)\];', line).groups()
            target, angle = int(target), float(angle)
            ix = indices[(indices&(1<<target))==0]; other = ix|(1<<target)
            a, b = state[ix].copy(), state[other].copy()
            co, si = math.cos(angle/2), math.sin(angle/2)
            state[ix], state[other] = co*a-si*b, si*a+co*b
        elif line.startswith('cx '):
            control, target = map(int, re.fullmatch(r'cx q\[(\d+)\], q\[(\d+)\];', line).groups())
            ix = indices[((indices&(1<<control))!=0)&((indices&(1<<target))==0)]
            other = ix|(1<<target)
            state[ix], state[other] = state[other].copy(), state[ix].copy()
        else:
            raise ValueError('Unsupported circuit statement')
    return state


def quantum(c, factors, blocks, seed, shots, run_id, output):
    import numpy as np
    if c['track']=='adb_drb':
        weights, local = map(np.asarray, factors)
        gates = prepare(np.pad(weights,(0,6))[None,:], [], list(range(4)))
        for j in range(2):
            table = np.zeros((16,16)); table[:10,:10] = local[:,j]; table[10:,0] = 1.
            gates += prepare(table, list(range(4)), list(range(4+4*j,8+4*j)))
        qubits = 12
        expected = np.zeros(1<<qubits)
        for h in range(10):
            for a in range(10):
                for b in range(10):
                    index = int((f'{h:04b}{a:04b}{b:04b}')[::-1],2)
                    expected[index] = math.sqrt(weights[h]*local[h,0,a]*local[h,1,b])
        role = 'Exact latent shock purification of the full capped term distribution'
    else:
        n = len(c['countries']); qubits = n+1
        mass = np.array([np.bincount(b[2], weights=b[0], minlength=1<<n) for b in blocks])
        mass /= mass.sum()
        # The circuit retains peril and the full joint country attachment signature.
        gates = prepare(mass.sum(axis=1)[None,:], [], [0])
        gates += prepare(mass/mass.sum(axis=1)[:,None], [0], list(range(1,n+1)))
        expected = np.zeros(1<<qubits)
        for h in range(2):
            for signature in range(1<<n):
                index = int((f'{h:01b}'+format(signature,f'0{n}b'))[::-1],2)
                expected[index] = math.sqrt(mass[h,signature])
        role = 'Exact latent peril and joint attachment signature; conditional expectations weighted by statevector probabilities'
    lines = ['OPENQASM 3.0;', 'include "stdgates.inc";', f'qubit[{qubits}] q;']
    for gate in gates:
        if gate[0]=='ry':
            lines.append(f'ry({gate[2]:.17g}) q[{gate[1]}];')
        else:
            lines.append(f'cx q[{gate[1]}], q[{gate[2]}];')
    source = '\n'.join(lines)+'\n'
    Path(output).with_suffix('.qasm').write_text(source)
    state = simulate_qasm(source)
    err = float(np.max(np.abs(state-expected)))
    if err>1e-12:
        raise ArithmeticError('Circuit amplitude mismatch')
    probability = state*state
    counts = np.random.default_rng(seed).multinomial(shots, probability/probability.sum())
    evidence = {'provider':'local', 'backend':'openqasm3-ry-cx-statevector',
        'quantum_execution':'quantum_simulator', 'job_ids':[run_id+'-statevector-shots'],
        'circuit_sha256':hashlib.sha256(source.encode()).hexdigest(), 'circuit_openqasm':source,
        'gate_count':len(gates), 'amplitude_max_abs_error':err,
        'statevector_norm_error':float(abs(probability.sum()-1)), 'circuit_role':role,
        'shot_counts':{str(i):int(v) for i,v in enumerate(counts) if v},
        'shot_total_variation':float(np.abs(counts/shots-probability).sum()/2)}
    if c['track']=='adb_drb':
        pmf = np.zeros((5,5,2,2))
        for h in range(16):
            for a in range(10):
                for b in range(10):
                    index = int((f'{h:04b}{a:04b}{b:04b}')[::-1],2)
                    pmf[a//2,b//2,a%2,b%2] += probability[index]
        return evidence, qubits, pmf.ravel()
    q_mass = np.zeros((2, 1 << n))
    for h in range(2):
        for signature in range(1 << n):
            index = int((f'{h:01b}' + format(signature, f'0{n}b'))[::-1], 2)
            q_mass[h, signature] = probability[index]
    return evidence, qubits, q_mass


def solve(request, mode, output):
    start = time.perf_counter()
    path = Path(request['config_file'])
    c = json.loads(path.read_text())
    samples, seed = request['samples'], request['seed']
    extra, evidence = {}, None
    factors, blocks = None, None
    if c['track']=='adb_drb':
        terms = json.loads((path.parent/c['terms_file']).read_text())
        factors = bond_factors(c, terms)
        p = bond_distribution(*factors)
        f = bond_features(c)
        preparation = time.perf_counter()-start
        if mode=='sample':
            values, extra = bond_sample(c, terms, p, f, samples, seed)
        else:
            if mode=='hybrid':
                evidence, qubits, p = quantum(c, factors, None, seed, samples, request['run_id'], output)
            values = [math.fsum(float(p[i])*f[i][j] for i in range(100)) for j in range(5)]
        evaluations = samples if mode=='sample' else 100
    else:
        import numpy as np
        blocks, details = elt_load(c, path.parent)
        preparation = time.perf_counter()-start
        if mode=='sample':
            values, extra = elt_sample(blocks, samples, seed)
        elif mode=='hybrid':
            evidence, qubits, q_mass = quantum(c, None, blocks, seed, samples, request['run_id'], output)
            total_rate = math.fsum(float(r.sum()) for r, _, _ in blocks)
            acc = np.zeros(5, dtype=float)
            m_sig = 1 << len(c['countries'])
            for h, (r, f, sig) in enumerate(blocks):
                w_sig = np.bincount(sig, weights=r, minlength=m_sig)
                valid = (w_sig > 0.0) & (q_mass[h] > 0.0)
                if np.any(valid):
                    scale = total_rate * q_mass[h, valid] / w_sig[valid]
                    for j in range(5):
                        acc[j] += float(np.dot(scale, np.bincount(sig, weights=r * f[:, j], minlength=m_sig)[valid]))
            values = acc.tolist()
        else:
            values = np.sum([np.sum(r[:,None]*f,axis=0) for r,f,s in blocks], axis=0).tolist()
        extra['catalogue'] = details
        evaluations = samples if mode=='sample' else sum(len(b[0]) for b in blocks)
    result = {'problem_id':request['problem_id'], 'run_id':request['run_id'], 'seed':seed,
        'method':'structured-'+mode, 'execution':'hybrid' if mode=='hybrid' else 'classical_cpu',
        'estimates':dict(zip(request['targets'],values)),
        'resources':{'elapsed_seconds':time.perf_counter()-start, 'preparation_seconds':preparation,
            'shots':samples if mode=='hybrid' else 0, 'qubits':qubits if mode=='hybrid' else 0,
            'model_evaluations':evaluations, **extra}}
    if evidence is not None:
        result['evidence'] = evidence
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['exact','sample','hybrid'], default='exact')
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    result = solve(request, args.mode, args.output)
    Path(args.output).write_text(json.dumps(result, allow_nan=False)+'\n')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
