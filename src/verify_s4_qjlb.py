"""Independent numerical, process, circuit and report checks for S4."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
import tempfile
import numpy as np
from qjlb.cli import load_model, methods_for
from qjlb.bond import BondModel, bond_exact
from qjlb.evaluation import invoke, score, tolerances
from qjlb.holdout import load_suite, digest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import team_solver as solver
DATA = ROOT/'data/s4'


def additional_processes():
    rows=[]
    for seed in (20261123,20261124):
        suite=DATA/f'holdout-{seed}'
        manifest=load_suite(suite); rng=random.Random(seed)
        for case in manifest['cases']:
            config=suite/case; model,_=load_model(config)
            reference=methods_for(model)['exact'](model); bounds=tolerances(model,reference)
            seeds=[rng.randrange(2**32) for _ in range(10)]
            for mode in ('exact','sample','hybrid'):
                execution='hybrid' if mode=='hybrid' else 'classical_cpu'
                command=[sys.executable,str(ROOT/'team_solver.py'),'--mode',mode]
                runs=[invoke(command,suite,config,model,s,8192,60,execution) for s in seeds]
                result=score(runs,reference,bounds,model.metric_names)
                rows.append({'suite':suite.name,'case':case,'mode':mode,'track':model.config['track'],
                    'score':result,'runs':runs})
                print(suite.name,case,mode,result['accuracy_pass'],flush=True)
                (DATA/'additional_holdout_processes.json').write_text(json.dumps({'runner':'qjlb.evaluation.invoke and score',
                    'solver_sha256':digest(ROOT/'team_solver.py'),'repeats':10,'samples':8192,'cases':rows},indent=2)+'\n')
    return rows


def independent_statevector(source):
    q=int(re.search(r'qubit\[(\d+)\]',source)[1])
    state=np.zeros([2]*q); state[(0,)*q]=1
    for line in source.splitlines()[3:]:
        if line.startswith('ry'):
            angle,target=re.fullmatch(r'ry\(([^)]+)\) q\[(\d+)\];',line).groups()
            angle=float(angle); axis=q-1-int(target)
            matrix=np.array([[math.cos(angle/2),-math.sin(angle/2)],[math.sin(angle/2),math.cos(angle/2)]])
            state=np.moveaxis(np.tensordot(matrix,state,axes=(1,axis)),0,axis)
        else:
            control,target=map(int,re.fullmatch(r'cx q\[(\d+)\], q\[(\d+)\];',line).groups())
            sl=[slice(None)]*q; sl[q-1-control]=1
            block=state[tuple(sl)].copy()
            target_axis=q-1-target-(q-1-control<q-1-target)
            state[tuple(sl)]=np.flip(block,axis=target_axis)
    return state.ravel()


def verify():
    counts={}; errors=[]
    def check(condition,group,detail):
        counts[group]=counts.get(group,0)+1
        if not bool(condition):
            errors.append(group+': '+detail)
    reports={}
    for mode in ('exact','sample','hybrid'):
        report=json.loads((DATA/f'evaluation-suite-13-{mode}/report.json').read_text())
        reports[mode]=report
        check(report['protocol']=='evaluation-v1' and report['protocol_eligible'],'protocol',mode)
        check(report['repeats']==10 and report['baseline_samples']==8192,'protocol',mode+' budget')
        check(len(report['cases'])==13,'protocol',mode+' cases')
        check(next(v for k,v in report['command_file_sha256'].items() if Path(k).name=='team_solver.py')==digest(ROOT/'team_solver.py'),'hash',mode)
        for case in report['cases']:
            m,_=load_model(DATA/'suite-13/cases'/f"{case['case']}.json")
            reference=methods_for(m)['exact'](m)
            bounds=tolerances(m,reference)
            for method,result in case['methods'].items():
                recalculated=score(result['runs'],reference,bounds,m.metric_names)
                check(recalculated=={k:result[k] for k in recalculated},'official_score',case['case']+method)
            team=case['methods']['team']
            check(team['accuracy_pass'],'team_accuracy',case['case']+mode)
            for run in team['runs']:
                check(run['status']=='valid','process',case['case']+mode)
                sub=run['submission']; text=json.dumps(sub)
                check(not any(word in text.lower() for word in ('failed','degraded','fallback','oracle','leak','hack')) and '\u2014' not in text,'output_text',case['case'])
                if mode=='sample':
                    check(sub['resources']['samples']==8192,'budget',case['case'])
                if mode=='hybrid':
                    ev=sub['evidence']; source=ev['circuit_openqasm']
                    check(hashlib.sha256(source.encode()).hexdigest()==ev['circuit_sha256'],'circuit_digest',case['case'])
                    check(sum(ev['shot_counts'].values())==8192,'quantum_shots',case['case'])
                    if run is team['runs'][0]:
                        independent=independent_statevector(source)
                        direct=solver.simulate_qasm(source)
                        check(np.max(np.abs(independent-direct))<1e-12,'independent_gates',case['case'])
                        check(abs(independent@independent-1)<1e-12,'quantum_norm',case['case'])
                        if m.config['track']=='adb_drb':
                            marginal=np.zeros((5,5,2,2))
                            for h in range(16):
                                for a in range(10):
                                    for b in range(10):
                                        index=int((f'{h:04b}{a:04b}{b:04b}')[::-1],2)
                                        marginal[a//2,b//2,a%2,b%2]+=independent[index]**2
                            check(np.max(abs(marginal.ravel()-m.term_distribution()))<1e-12,'quantum_model',case['case'])
                        else:
                            n=len(m.countries)
                            signature=((m.loss>m.attachment[m.peril])*(1<<np.arange(n))).sum(axis=1)
                            mass=np.zeros((2,1<<n))
                            for h in range(2):
                                for group in range(1<<n):
                                    mass[h,group]=math.fsum(m.rates[(m.peril==h)&(signature==group)])/math.fsum(m.rates)
                            actual=np.zeros_like(mass)
                            for h in range(2):
                                for signature in range(1<<n):
                                    index=int((str(h)+format(signature,f'0{n}b'))[::-1],2)
                                    actual[h,signature]=independent[index]**2
                            check(np.max(abs(actual-mass))<1e-12,'quantum_model',case['case'])
                if mode=='exact':
                    for k,estimate in sub['estimates'].items():
                        target=reference['estimates'][k]['estimate']
                        check(abs(estimate-target)/max(1,abs(target))<1e-12,'exact_precision',case['case']+k)
            if mode=='sample':
                values=[tuple(run['submission']['estimates'].values()) for run in team['runs']]
                check(len(set(values))>1,'seed_dependence',case['case'])
        for track,summary in report['tracks'].items():
            cases=[c for c in report['cases'] if c['track']==track]
            ratios=[]
            for c in cases:
                fastest=min(v['median_seconds'] for k,v in c['methods'].items() if k!='team' and v['accuracy_pass'])
                ratios.append(c['methods']['team']['median_seconds']/fastest)
            expected=math.exp(sum(map(math.log,ratios))/len(ratios))
            check(abs(expected-summary['geometric_time_ratio'])<1e-14,'timing',mode+track)
    extra=json.loads((DATA/'additional_holdout_processes.json').read_text())
    check(extra['solver_sha256']==digest(ROOT/'team_solver.py'),'hash','extra')
    check(len(extra['cases'])==36,'holdout','case count')
    for row in extra['cases']:
        path=DATA/row['suite']/row['case']; m,_=load_model(path)
        reference=methods_for(m)['exact'](m); bounds=tolerances(m,reference)
        actual=score(row['runs'],reference,bounds,m.metric_names)
        check(actual==row['score'] and actual['accuracy_pass'],'holdout',str(path)+row['mode'])
        check(len(row['runs'])==10 and all(r['status']=='valid' for r in row['runs']),'holdout_process',str(path))
    # Independent randomized contract checks span all allowed horizons.
    base=json.loads((ROOT/'benchmark/scenarios/adb-base.json').read_text())
    terms=json.loads((ROOT/'benchmark/data/adb_terms.json').read_text())
    rng=random.Random(9112026); max_pmf=0.; max_relative=0.
    for i in range(200):
        c=copy.deepcopy(base); c['term_years']=1+i%10
        c['earthquake_sublimit_usd']=[rng.uniform(1e6,180e6) for _ in range(2)]
        c['precipitation_sublimit_usd']=[rng.uniform(1e5,60e6) for _ in range(2)]
        c['earthquake_common_rate_multiplier']=rng.uniform(0,2.5)
        c['precipitation_common_rate_share']=rng.random()
        c['precipitation_annual_trigger_probability']=[rng.uniform(.0001,.2) for _ in range(2)]
        c['precipitation_rate_multiplier']=rng.uniform(.1,3)
        c['portfolio_threshold_usd']=rng.random()*sum(c['earthquake_sublimit_usd']+c['precipitation_sublimit_usd'])
        model=BondModel(c,ROOT/'benchmark/data/adb_terms.json'); target=model.term_distribution()
        pmf=np.asarray(solver.bond_distribution(*solver.bond_factors(c,terms)))
        error=float(abs(pmf-target).max()); max_pmf=max(max_pmf,error)
        check(error<1e-12,'randomized_pmf',str(i))
        check(abs(pmf.sum()-1)<1e-12 and pmf.min()>=0,'probability',str(i))
        actual=pmf@np.asarray(solver.bond_features(c)); ref=bond_exact(model)
        for k,name in enumerate(model.metric_names):
            expected=ref['estimates'][name]['estimate']; e=abs(actual[k]-expected)/max(1,abs(expected)); max_relative=max(max_relative,e)
            check(e<1e-12,'randomized_metrics',str(i)+name)
    # Recompute scientific claims using official models.
    from s4_scientific_findings import measure
    measured=measure(); saved=json.loads((DATA/'scientific_findings.json').read_text())
    check(measured==saved,'findings','recomputed')
    for years in (1,3,10):
        rows=[r for r in saved['finding_a']['rows'] if r['years']==years]
        means=[r['values']['term_expected_principal_loss_usd'] for r in rows]
        check((max(means)-min(means))/max(means)<1e-12,'invariant',str(years))
    for pool in saved['finding_b']['pools']:
        check(pool['rows']==421968 and sum(p['rows'] for p in pool['perils'])==421968,'catalogue','rows')
        check(sum(p['active_rows'] for p in pool['perils'])==pool['active_rows'],'catalogue','active')
    report_path=DATA/'reproduced_summary.json'
    if not report_path.exists():
        report_path=ROOT/'data/s4_qjlb_monopoly_report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        for mode,r in reports.items():
            check(report['modes'][mode]['tracks']==r['tracks'],'report',mode)
    # Recheck arbitrary sample budgets and deterministic seeded replay.
    for relative in ('adb-base.json','kaz-kgz-base.json'):
        path=ROOT/'benchmark/scenarios'/relative
        model,_=load_model(path)
        for budget in (16,1000,8191):
            request={'config_file':str(path),'samples':budget,'seed':71,'run_id':'replay',
                'problem_id':model.fingerprint,'targets':list(model.metric_names)}
            a=solver.solve(request,'sample',str(DATA/'budget.json'))
            b=solver.solve(request,'sample',str(DATA/'budget.json'))
            check(a['resources']['samples']==budget and a['resources']['model_evaluations']==budget,'arbitrary_budget',relative+str(budget))
            check(a['estimates']==b['estimates'],'replay',relative+str(budget))
    resource_rows=json.loads((DATA/'resource_probe.json').read_text())
    for row in resource_rows:
        check(row['peak_resident_bytes']<4*1024**3 and row['under_4_gib'],'memory',row['case']+row['mode'])
    preserved=json.loads((ROOT/'data/public_code_sha256.json').read_text())
    for name,sha in preserved.items():
        check(digest(ROOT/name)==sha,'package_integrity',name)
    previous=json.loads((ROOT/'data/s4_qjlb_monopoly_report.json').read_text())['s2_preservation']
    check(previous['passed'] and previous['passed_count']==previous['count'],'s2_regression','existing invariants')
    result={'assertions':sum(counts.values()),'passed':sum(counts.values())-len(errors),'all_pass':not errors,
        'groups':counts,'issues':errors,'randomized_contracts':200,'randomized_pmf_max_abs_error':max_pmf,
        'randomized_metric_max_scaled_error':max_relative,'solver_sha256':digest(ROOT/'team_solver.py')}
    (DATA/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return 0 if not errors else 1


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--additional-processes',action='store_true'); args=p.parse_args()
    if args.additional_processes:
        additional_processes()
    else:
        raise SystemExit(verify())
