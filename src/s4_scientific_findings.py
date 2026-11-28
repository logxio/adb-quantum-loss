"""Measure horizon dependence and catalogue sparsity through official models."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
from qjlb.bond import BondModel, bond_exact
from qjlb.model import ELTModel, elt_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import team_solver


def measure():
    c = json.loads((ROOT/'benchmark/scenarios/adb-base.json').read_text())
    rows = []
    for years in (1,3,10):
        for multiplier in (0,1,2):
            config = copy.deepcopy(c)
            config.update(term_years=years, earthquake_common_rate_multiplier=multiplier)
            model = BondModel(config,ROOT/'benchmark/data/adb_terms.json')
            exact = bond_exact(model)
            rows.append({'years':years, 'earthquake_common_rate_multiplier':multiplier,
                'precipitation_common_rate_share':config['precipitation_common_rate_share'],
                'values':{k:v['estimate'] for k,v in exact['estimates'].items()}})
    pools = []
    base = json.loads((ROOT/'benchmark/scenarios/kaz-kgz-base.json').read_text())
    for countries in [['KAZ','KGZ'],['KAZ','KGZ','TJK'],['KAZ','KGZ','TJK','TKM','UZB']]:
        n = len(countries)
        config = copy.deepcopy(base)
        config.update(countries=countries,attachment_usd=[[250e6]*n,[150e6]*n],
            exhaustion_usd=[[1e9]*n,[400e6]*n],limit_usd=[80e6]*n,portfolio_threshold_usd=40e6*n)
        model = ELTModel(config,ROOT/'benchmark/data')
        result = elt_exact(model)
        active = np.any(model.loss>model.attachment[model.peril],axis=1)
        f = model.evaluate(np.arange(model.state_count)[:,None])['per_event_metrics']
        detail=[]
        for h, peril in enumerate(['seismic','flood']):
            ix = model.peril==h
            detail.append({'peril':peril,'rows':int(ix.sum()),'active_rows':int(np.count_nonzero(active & ix)),
                'active_percent':float(100*np.mean(active[ix])), 'joint_rows':int(np.count_nonzero(f[ix,2])),
                'portfolio_exceedance_rows':int(np.count_nonzero(f[ix,3])),
                'annual_rate':float(model.rates[ix].sum()),
                'values':dict(zip(model.metric_names,(model.rates[ix]@f[ix]).tolist()))})
        sample_probabilities = (model.probabilities[:,None]*(f>0)).sum(axis=0)
        pools.append({'countries':countries,'config':config,'rows':model.state_count,
            'active_rows':int(active.sum()),'active_percent':float(100*active.mean()),
            'values':{k:v['estimate'] for k,v in result['estimates'].items()},'perils':detail,
            'iid_expected_nonzero_counts_at_8192':dict(zip(model.metric_names,(8192*sample_probabilities).tolist())),
            'iid_zero_observation_probability_at_8192':dict(zip(model.metric_names,((1-sample_probabilities)**8192).tolist()))})
    return {'finding_a':{'coupling_scope':'Earthquake common-rate multiplier varies; rain common share fixed at 0.25.',
        'interpretation':'Mean loss is invariant to earthquake coupling at each horizon; caps create nonlinear horizon dependence in joint exhaustion and stop loss. This is a smooth finite-state transition, not a singular thermodynamic phase transition.', 'rows':rows},
        'finding_b':{'comparison_scope':'Nested country sets with identical per-country layers and threshold equal to half total limits.',
        'interpretation':'Joint means positive payment in every selected country. Finite catalogue zeros are actual reference values. They do not establish independence or diversified tail protection outside this catalogue.', 'pools':pools}}


if __name__=='__main__':
    path=ROOT/'data/s4/scientific_findings.json'
    path.write_text(json.dumps(measure(),indent=2,allow_nan=False)+'\n')
    print(path)
