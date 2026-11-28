"""Finite contract tilting and exact shared shock capped tensor contractions."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from scipy import optimize, special, stats
from scipy.linalg import expm
import probe_official_s1 as s1
ROOT=Path(__file__).resolve().parents[1]
TARGETS=s1.TARGETS
BUDGETS=[256,512,1024,2048,4096,8192,16384]
COUNTRY_COUNTS=[2,3,4,5,6,8,10,12,15]
COUNTRY_LABELS=['KGZ','TJK','PHL','VNM','THA','IDN','MMR','LAO','KHM','BGD','PAK','NPL','FJI','TON','VUT']
GRAPHS=['independent','blocks','chain','global']
BASELINES=['MonteCarlo','ScrambledSobol','StratifiedSampling','ImportanceSampling']
LOCAL_STATES=np.array(list(np.ndindex(5,2)),dtype=int)
LOSS_UNITS=LOCAL_STATES[:,0]*7+LOCAL_STATES[:,1]*4


def dict_targets(values):
    return dict(zip(TARGETS,np.asarray(values).tolist()))


def _finite_nonnegative(value,name):
    try:
        if np.ndim(value)!=0:
            raise ValueError(f'{name} must be a finite nonnegative scalar')
        scalar=float(value)
    except (TypeError,OverflowError,ValueError) as exc:
        raise ValueError(f'{name} must be a finite nonnegative scalar') from exc
    if not np.isfinite(scalar) or scalar<0:
        raise ValueError(f'{name} must be a finite nonnegative scalar')
    return scalar


class TerminalTilt:
    """A fully adapted terminal proposal followed by exact conditional path bridges."""
    def __init__(self,annual,scenario):
        start=time.perf_counter_ns()
        self.annual=np.asarray(annual)
        self.features=s1.payout_features(s1.STATES,scenario)
        graph='independent' if scenario=='adb-independent' else 'global'
        multiplier=2. if scenario=='adb-common-shock' else 1.
        tensor=model(2,graph,m=multiplier,rain_multiplier=2. if scenario=='adb-rain-stress' else 1.)
        self.cores=tensor['cores']
        self.term=dense(self.cores).reshape(5,2,5,2).transpose(0,2,1,3).ravel()
        self.mps_reference_error=float(np.max(abs(self.term-s1.exact_term(annual))))
        if self.mps_reference_error>1e-13:
            raise AssertionError('MPS contract and official annual convolution differ')
        loss=self.features[:,0]/160e6
        positive=self.term>0
        logp=np.log(np.maximum(self.term,1e-300))
        proposals=[]
        self.tilts=[]
        for j in range(5):
            f=self.features[:,j]/max(1.,self.features[:,j].max())
            nz=positive & (f>0)
            def objective(theta):
                return special.logsumexp(logp+theta*loss)+special.logsumexp(logp[nz]+2*np.log(f[nz])-theta*loss[nz])
            fit=optimize.minimize_scalar(objective,bounds=(-20.,80.),method='bounded',options={'xatol':1e-10})
            theta=float(fit.x)
            # Exponential payout tilt is a product diagonal MPO, so each
            # component retains the exact contract probability MPS bond rank.
            local_loss=(LOCAL_STATES[:,0]*(60e6/4)+LOCAL_STATES[:,1]*20e6
                        if scenario=='adb-rain-cover' else LOSS_UNITS*2.5e6)
            modified=[c*np.exp(theta*local_loss/160e6)[None,:,None] for c in self.cores]
            tilted=dense(modified).reshape(5,2,5,2).transpose(0,2,1,3).ravel()
            tilted/=tilted.sum()
            proposals.append(tilted)
            self.tilts.append(theta)
        # Fixed equal mixture supplies every focal layer, including the loss mean.
        self.proposal=np.mean(proposals,axis=0)
        self.proposal/=self.proposal.sum()
        self.order=np.argsort(loss,kind='stable')
        self.cdf=np.cumsum(self.proposal[self.order]); self.cdf[-1]=1.
        self.weight=self.term/self.proposal
        # Conditional pairs retain actual three year trajectories. They do not
        # change the Rao Blackwell terminal payoff or its one dimensional ordering.
        first=np.zeros(100); first[0]=1.
        alphas=[first]
        for _ in range(3):
            alphas.append(np.bincount(s1.COMBINE.ravel(),weights=(alphas[-1][:,None]*annual).ravel(),minlength=100))
        self.bridge=[]
        for horizon in [3,2]:
            joint=alphas[horizon-1][:,None]*annual
            cdfs=np.zeros((100,10000))
            for end in range(100):
                cond=np.where(s1.COMBINE==end,joint,0).ravel()
                if cond.sum()>0:
                    cdfs[end]=np.cumsum(cond/cond.sum()); cdfs[end,-1]=1.
            self.bridge.append(cdfs)
        self.setup_ms=(time.perf_counter_ns()-start)/1e6
        self.setup_entries=int(sum(c.size for c in self.bridge)+10000*3)

    def sample(self,n,seed,qmc=True,return_paths=False):
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n<1:
            raise ValueError('n must be a positive integer')
        if qmc and n & (n-1):
            raise ValueError('Sobol n must be a positive power of two')
        if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed<0:
            raise ValueError('seed must be a nonnegative integer')
        if qmc:
            u=stats.qmc.Sobol(3,scramble=True,seed=seed).random_base2(int(np.log2(n)))
        else:
            u=np.random.default_rng(seed).random((n,3))
        ix=self.order[np.minimum(np.searchsorted(self.cdf,u[:,0]),99)]
        payoff=self.features[ix]*self.weight[ix,None]
        current=ix.copy()
        paths=np.zeros((n,3),dtype=int)
        for k,cdfs in enumerate(self.bridge):
            pairs=np.empty(n,dtype=int)
            for end in np.unique(current):
                rows=np.flatnonzero(current==end)
                pairs[rows]=np.minimum(np.searchsorted(cdfs[end],u[rows,k+1]),9999)
            paths[:,2-k]=pairs%100
            current=pairs//100
        paths[:,0]=current
        final=np.minimum(s1.STATES[paths].sum(axis=1),s1.CAP)
        if not np.array_equal(final,s1.STATES[ix]):
            raise AssertionError('Conditional bridge end state mismatch')
        return (payoff.mean(axis=0),paths) if return_paths else payoff.mean(axis=0)


def compound_capped(rate,weights,horizon=3.):
    """Exact compound Poisson semigroup on the five capped severity states."""
    rate=_finite_nonnegative(rate,'rate')
    horizon=_finite_nonnegative(horizon,'horizon')
    try:
        weights=np.asarray(weights,dtype=float)
    except (TypeError,ValueError) as exc:
        raise ValueError('weights must contain four finite severity probabilities') from exc
    if (weights.shape!=(4,) or not np.all(np.isfinite(weights)) or np.any(weights<0)
            or abs(weights.sum()-1)>1e-14):
        raise ValueError('weights must contain four finite nonnegative probabilities summing to one')
    g=np.zeros((5,5))
    for a in range(5):
        for k in range(1,5):
            g[a,min(a+k,4)]+=rate*weights[k-1]
        g[a,a]-=g[a].sum()
    return expm(horizon*g)[0]


def capped_discrete(a,b):
    out=np.zeros(5)
    for i in range(5):
        for j in range(5):
            out[min(i+j,4)]+=a[i]*b[j]
    return out


def model(n,graph='global',m=1.,horizon=3.,rain_multiplier=1.):
    """Exact nonnegative probability MPS with physical order (EQ state, rain bit)."""
    if not isinstance(n,(int,np.integer)) or n<2 or n>15:
        raise ValueError('n must be an integer from 2 through 15')
    m=_finite_nonnegative(m,'m')
    horizon=_finite_nonnegative(horizon,'horizon')
    rain_multiplier=_finite_nonnegative(rain_multiplier,'rain_multiplier')
    if graph not in GRAPHS:
        raise ValueError('Invalid graph, intensity multiplier or horizon')
    rates=[s1.compound_rates(t) for t in [[.03883,.02331,.01790,.01184],[.03925,.01743,.01332,.01043]]]
    lam=np.array([rates[j%2][1] for j in range(n)])
    weights=np.array([rates[j%2][0]/rates[j%2][1] for j in range(n)])
    c=m*(.075*rates[0][1]+.074*rates[1][1])/2
    r=-np.log(.98)*rain_multiplier
    s=.25*r
    edges=[]
    if graph=='global':
        edges=[list(range(n))]
    elif graph=='chain':
        edges=[[j,j+1] for j in range(n-1)]
    elif graph=='blocks':
        edges=[[j,j+1] for j in range(0,n-1,2)]
    degree=np.array([sum(j in e for e in edges) for j in range(n)])
    if np.any(c*degree>lam+1e-15) or np.any(s*degree>r+1e-15):
        raise ValueError('Shared intensities exceed a country marginal intensity')
    eq_count=np.r_[stats.poisson.pmf(np.arange(4),horizon*c),stats.poisson.sf(3,horizon*c)]
    rain_bit=np.array([np.exp(-horizon*s),-np.expm1(-horizon*s)])
    latent=np.outer(eq_count,rain_bit).ravel()
    conditional=[]
    for j in range(n):
        independent_eq=compound_capped(max(0.,lam[j]-degree[j]*c),weights[j],horizon)
        eqs=[independent_eq]
        mark=np.r_[0.,weights[j]]
        for k in range(1,5):
            eqs.append(capped_discrete(eqs[-1],mark))
        p_r=-np.expm1(-horizon*max(0.,r-degree[j]*s))
        local=np.zeros((5,2,10))
        for k in range(5):
            for bit in range(2):
                rp=np.array([1-p_r,p_r]) if bit==0 else np.array([0.,1.])
                local[k,bit]=np.outer(eqs[k],rp).ravel()
        conditional.append(local)
    cores=[]
    if graph=='global':
        for j in range(n):
            f=conditional[j].reshape(10,10)
            if j==0:
                core=(latent[:,None]*f).T[None,:,:]
            elif j==n-1:
                core=f[:,:,None]
            else:
                core=np.zeros((10,10,10))
                for h in range(10):core[h,:,h]=f[h]
            cores.append(core)
    else:
        edge_set={e[0] for e in edges}
        for j in range(n):
            left=j-1 in edge_set; right=j in edge_set
            core=np.zeros((10 if left else 1,10,10 if right else 1))
            for a in range(core.shape[0]):
                ka,ra=divmod(a,2) if left else (0,0)
                for b in range(core.shape[2]):
                    kb,rb=divmod(b,2) if right else (0,0)
                    core[a,:,b]=conditional[j][min(ka+kb,4),int(ra or rb)]*(latent[b] if right else 1.)
            cores.append(core)
    cut_crossings=[sum(any(v<=j for v in e) and any(v>j for v in e) for e in edges) for j in range(n-1)]
    return {'n':n,'graph':graph,'m':m,'horizon_years':horizon,'cores':cores,
            'country_order':COUNTRY_LABELS[:n],
            'marginal_template_order':['KGZ' if j%2==0 else 'TJK' for j in range(n)],
            'latent_weights':latent,'local_conditionals':np.array(conditional),
            'country_marginal_eq_rate':lam,'common_eq_rate':c,'country_rain_rate':r,'common_rain_rate':s,
            'shared_groups':edges,'country_shared_degree':degree,'cut_crossing_group_counts':cut_crossings,
            'physical_states':LOCAL_STATES,'loss_grid_usd':2.5e6,'loss_units':LOSS_UNITS}


def contract(cores,loss_units=LOSS_UNITS):
    """Contract the exact portfolio probability generating polynomial."""
    out=np.ones((1,1))
    for core in cores:
        nxt=np.zeros((core.shape[2],out.shape[1]+int(max(loss_units))))
        for x,amount in enumerate(loss_units):
            nxt[:,amount:amount+out.shape[1]]+=core[:,x,:].T@out
        out=nxt
    if out.shape[0]!=1:raise ValueError('Open right bond')
    return out[0]


def marginal(cores,j):
    left=np.ones(1)
    for c in cores[:j]:left=left@c.sum(axis=1)
    right=np.ones(1)
    for c in cores[j+1:][::-1]:right=c.sum(axis=1)@right
    return np.einsum('a,axb,b->x',left,cores[j],right)


def dense(cores):
    out=cores[0][0]
    for core in cores[1:]:out=np.tensordot(out,core,axes=([-1],[0]))
    return out[...,0]


def round_mps(cores,chi):
    """Canonical TT SVD truncation with a Frobenius error bound."""
    if isinstance(chi,(bool,np.bool_)) or not isinstance(chi,(int,np.integer)) or chi<1:
        raise ValueError('chi must be a positive integer')
    try:
        cc=[np.asarray(c).copy() for c in cores]
    except (TypeError,ValueError) as exc:
        raise ValueError('cores must be a nonempty sequence of finite rank three arrays') from exc
    if not cc:
        raise ValueError('cores must be nonempty')
    physical=None
    previous=1
    for c in cc:
        if c.ndim!=3 or min(c.shape)<1 or not np.isrealobj(c):
            raise ValueError('each core must be a nonempty real rank three array')
        try:
            finite=bool(np.all(np.isfinite(c)))
        except TypeError as exc:
            raise ValueError('core entries must be finite numbers') from exc
        if not finite or c.shape[0]!=previous or (physical is not None and c.shape[1]!=physical):
            raise ValueError('core entries, adjacent bonds and physical dimensions must be consistent')
        previous=c.shape[2]
        physical=c.shape[1]
    if previous!=1:
        raise ValueError('last core must close the right bond')
    for j in range(len(cc)-1,0,-1):
        r,d,t=cc[j].shape
        q,rr=np.linalg.qr(cc[j].reshape(r,d*t).T,mode='reduced')
        cc[j]=q.T.reshape(q.shape[1],d,t)
        cc[j-1]=np.tensordot(cc[j-1],rr.T,axes=([2],[0]))
    discarded=0.
    for j in range(len(cc)-1):
        r,d,t=cc[j].shape
        u,s,vh=np.linalg.svd(cc[j].reshape(r*d,t),full_matrices=False)
        rank=min(chi,len(s)); discarded+=float(np.dot(s[rank:],s[rank:]))
        cc[j]=u[:,:rank].reshape(r,d,rank)
        cc[j+1]=np.tensordot(s[:rank,None]*vh[:rank],cc[j+1],axes=([1],[0]))
    return cc,float(np.sqrt(discarded))


def run_sampling(repeats=32):
    results={}
    for scenario in s1.PUBLISHED:
        annual,calibration=s1.annual_model(scenario)
        tilt=TerminalTilt(annual,scenario)
        truth=tilt.term@tilt.features
        curves={}
        for budget in BUDGETS:
            methods={}
            for method in BASELINES+['MPS-IS','MPS-QMC-IS']:
                start=time.perf_counter_ns()
                values=[]
                for repeat in range(repeats):
                    seed=1729+repeat
                    if method in BASELINES:estimate=s1.sample_estimate(method,budget,seed,annual,scenario)
                    else:estimate=tilt.sample(budget,seed,qmc=method=='MPS-QMC-IS')
                    values.append(estimate)
                elapsed=(time.perf_counter_ns()-start)/1e6/repeats
                arr=np.array(values)
                rmse=np.sqrt(np.mean((arr-truth)**2,axis=0))
                methods[method]={'mean_estimate':dict_targets(arr.mean(axis=0)),'relative_rmse':dict_targets(rmse/truth),
                                 'rmse':dict_targets(rmse),'replicate_estimates':arr.tolist(),
                                 'replicate_error_quantiles_relative':np.quantile((arr-truth)/truth,[.025,.5,.975],axis=0).tolist(),
                                 'zero_exhaustion_estimate_replicates':int(np.count_nonzero(arr[:,4]==0)),
                                 'paths_per_replicate':budget,'mean_runtime_ms':elapsed,
                                 'annual_state_values_generated_per_replicate':3*budget,
                                 'conditional_bridge_pair_draws_per_replicate':0 if method in BASELINES else 2*budget,
                                 'terminal_state_draws_per_replicate':0 if method in BASELINES else budget}
            curves[str(budget)]=methods
        primary=curves['4096']
        wins={target:{b:primary['MPS-QMC-IS']['relative_rmse'][target]<primary[b]['relative_rmse'][target] for b in BASELINES} for target in TARGETS}
        results[scenario]={'exact_targets':dict_targets(truth),'calibration':calibration,
                           'term_state_probabilities':tilt.term.tolist(),'sampling_at_4096':primary,'budget_curves':curves,
                           'mps_qmc_wins_by_target':wins,'all_20_comparisons_won':all(all(v.values()) for v in wins.values()),
                           'tilting':{'theta_dimensionless_portfolio_loss':tilt.tilts,'proposal':tilt.proposal.tolist(),
                                      'proposal_mixture_weights':[.2]*5,'setup_ms':tilt.setup_ms,
                                      'bridge_probability_table_entries':tilt.setup_entries,
                                      'exact_mps_bond_dimensions':[int(c.shape[2]) for c in tilt.cores[:-1]],
                                      'mps_vs_official_term_max_error':tilt.mps_reference_error,
                                      'tilt_operator':'Product diagonal MPO exp(theta times local payout divided by USD160m)',
                                      'proposal_exhaustion_mass':float(tilt.proposal[tilt.features[:,4]>0].sum()),
                                      'target_moment_fit_uses_full_pmf':True,
                                      'exact_target_used_as_estimate':False}}
        print(json.dumps({'scenario':scenario,'all_20_comparisons_won':results[scenario]['all_20_comparisons_won'],
                          'qmc_relative_rmse':primary['MPS-QMC-IS']['relative_rmse']}),flush=True)
    return results


def run_scaling():
    rows=[]; rank_rows=[]; checked=[]
    memory_limit=4*1024**3
    for graph in GRAPHS:
        for n in COUNTRY_COUNTS:
            start=time.perf_counter_ns(); built=model(n,graph); setup=(time.perf_counter_ns()-start)/1e6
            cores=built['cores']
            start=time.perf_counter_ns(); pmf=contract(cores); elapsed=(time.perf_counter_ns()-start)/1e6
            marginals=np.array([marginal(cores,j) for j in range(n)])
            expected=[]
            for j in range(n):
                rates,total=s1.compound_rates([.03883,.02331,.01790,.01184] if j%2==0 else [.03925,.01743,.01332,.01043])
                eq=compound_capped(total,rates/total)
                rr=np.array([.98**3,1-.98**3])
                expected.append(np.outer(eq,rr).ravel())
            row={'n':n,'graph':graph,'chi':max(max(c.shape[0],c.shape[2]) for c in cores),
                 'full_state_count':10**n,'dense_probability_float64_bytes':8*10**n,
                 'naive_pair_capped_convolution_float64_bytes':8*10**(2*n),
                 'setup_ms':setup,'runtime_ms':elapsed,'total_runtime_ms':setup+elapsed,
                 'core_entries':sum(c.size for c in cores),'probability_sum':float(pmf.sum()),
                 'normalization_error':float(abs(pmf.sum()-1)),'min_probability':float(pmf.min()),
                 'truncation_error':0.,'count_folding_error':0.,'poisson_states':'0,1,2,3,at least 4',
                 'marginal_max_error':float(np.max(abs(marginals-np.array(expected)))),
                 'cut_crossing_group_counts':built['cut_crossing_group_counts'],
                 'shared_group_count':len(built['shared_groups']),
                 'shared_country_degree_max':int(max(built['country_shared_degree'])),
                 'portfolio_pmf':pmf.tolist(),'portfolio_grid_usd':2.5e6,
                 'expected_portfolio_loss_usd':float(pmf@np.arange(len(pmf))*2.5e6),
                 'stop_loss_at_half_notional_usd':float(pmf@np.maximum(np.arange(len(pmf))*2.5e6-40e6*n,0)),
                 'dense_feasible_under_4gb':8*10**n<=memory_limit,
                 'naive_pair_convolution_feasible_under_4gb':8*10**(2*n)<=memory_limit}
            # Enumeration remains feasible for 10^6 states. Dense joins need
            # temporary memory too, so this probe explicitly measures n at most 6.
            if n<=6:
                start=time.perf_counter_ns(); array=dense(cores); tensor_ms=(time.perf_counter_ns()-start)/1e6
                aggregate_start=time.perf_counter_ns()
                shifts=np.zeros(array.shape,dtype=np.int16)
                for j in range(n):
                    shape=[1]*n; shape[j]=10
                    shifts+=LOSS_UNITS.reshape(shape)
                enumerated=np.bincount(shifts.ravel(),weights=array.ravel(),minlength=len(pmf))
                aggregate_ms=(time.perf_counter_ns()-aggregate_start)/1e6
                dense_ms=(time.perf_counter_ns()-start)/1e6
                row.update({'dense_measured':True,'dense_runtime_ms':dense_ms,
                            'dense_probability_tensor_ms':tensor_ms,
                            'dense_portfolio_aggregation_ms':aggregate_ms,
                            'dense_actual_probability_array_bytes':array.nbytes,
                            'dense_portfolio_l1_error':float(np.abs(enumerated-pmf).sum())})
                del shifts,array
            else:
                row.update({'dense_measured':False,'dense_runtime_ms':None,
                            'dense_skip_reason':'Explicit enumeration measurement scope ends at n=6; larger memory is a calculated requirement, not an observed allocation failure'})
            if n==2 and graph in ['global','chain','blocks']:
                ordered=dense(cores).reshape(5,2,5,2).transpose(0,2,1,3).ravel()
                official=s1.exact_term(s1.annual_model('adb-base')[0])
                row['official_n2_max_probability_error']=float(max(abs(ordered-official)))
                checked.append(row['official_n2_max_probability_error'])
            rows.append(row)
            if n in [2,5,10,15]:
                for chi in [1,2,4,8,10]:
                    start=time.perf_counter_ns(); rounded,bound=round_mps(cores,chi); rounded_pmf=contract(rounded)
                    rank_rows.append({'n':n,'graph':graph,'requested_chi':chi,'chi':max(max(c.shape[0],c.shape[2]) for c in rounded),
                                      'runtime_ms':(time.perf_counter_ns()-start)/1e6,'frobenius_error_bound':bound,
                                      'tv_bound_from_frobenius':min(1.,.5*np.sqrt(10.**n)*bound),
                                      'portfolio_l1_error':float(abs(rounded_pmf-pmf).sum()),
                                      'normalization_error':float(abs(rounded_pmf.sum()-1)),
                                      'min_portfolio_probability':float(rounded_pmf.min()),
                                      'strictly_nonnegative_cores':bool(all(c.min()>=0 for c in rounded)),
                                      'used_for_exact_results':False})
    reference=model(15,'global')
    portable={'physical_states':LOCAL_STATES.tolist(),'country_order':COUNTRY_LABELS,
              'marginal_template_order':reference['marginal_template_order'],
              'country_marginals_are_repeated_contract_templates':True,'global_latent_weights':reference['latent_weights'].tolist(),
              'conditional_local_pmf':reference['local_conditionals'].tolist(),
              'loss_units':LOSS_UNITS.tolist(),'grid_usd':2.5e6,
              'common_eq_rate':reference['common_eq_rate'],'rain_rate':reference['country_rain_rate'],
              'shared_rain_rate':reference['common_rain_rate']}
    np.savez_compressed(ROOT/'data/mps_s2_scaling_operator.npz',**{f'core_{j}':c for j,c in enumerate(reference['cores'])})
    return {'rows':rows,'rank_truncation_rows':rank_rows,'portable_global_model':portable,
            'official_n2_max_probability_error':max(checked),
            'construction':'Independent compound Poisson marks plus independent shared group Poisson counts; within each group marks are conditionally independent across countries. Aggregate three year independent increments before the cap. Count at least four folds exactly because every EQ mark is positive. Shared rain count folds exactly to a bit.',
            'graph_scope':'Independent, disjoint pair blocks, nearest neighbor chain and a single global hyperedge. Each selected graph has at most one 10 state latent group crossing a country order cut. General dense pair graphs can require exponentially increasing chi.',
            'complexity':'Core expectation O(n chi^2 d). Exact portfolio polynomial DP adds the loss grid length L=32n+1: O(n chi^2 d L), with global diagonal cores admitting O(n chi d L). Core construction and TT rounding are timed separately.',
            'year_semigroup':'Three year cap equals cap of the three annual sums by independent increments and associativity of capped addition. It is not an uncontrolled cubed latent approximation.',
            'probability_mps_not_amplitude_mps':True,'loss_pmf_truncation':'Exact chosen graph cores use no SVD truncation; rounded cores are separate diagnostic approximations and may lose positivity.'}


def run(repeats=32):
    start=time.perf_counter_ns()
    result={'schema':'adb.official.s2.v1','targets':TARGETS,'budgets':BUDGETS,'repeats':repeats,'primary_budget':4096,
            'source':'benchmark/pyproject.toml',
            'source_sha256':hashlib.sha256((ROOT/'benchmark/pyproject.toml').read_bytes()).hexdigest(),
            'quantum_hardware_used':False,
            'sampling_cost_scope':'Same number of complete contract paths, different information access: MPS methods use all 100 terminal probabilities, optimized exponential proposals and exact conditional bridge tables. Baselines retain S1 annual probability queries. Finite 2 country exact contraction is already cheaper than this proposal setup. This is classical structural variance reduction, not a black box query or quantum advantage.',
            'proposal_objective':'For each of five fixed payoffs minimize log E exp(theta L/160m) plus log E[f^2 exp(-theta L/160m)] for theta in [-20,80], mix equally, use unbiased p/q terminal likelihood weights. All calibration uses the known official model only; replicate errors are not used for proposal selection.',
            'independent_replicates':'32 independent randomizations with seeds 1729 through 1760, nested powers of two within each seed.',
            'environment':{'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()}}
    result['scenarios']=run_sampling(repeats)
    result['scaling']=run_scaling()
    result['all_100_sampling_comparisons_won']=all(v['all_20_comparisons_won'] for v in result['scenarios'].values())
    result['runtime_seconds']=(time.perf_counter_ns()-start)/1e9
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeats',type=int,default=32)
    args=parser.parse_args()
    if args.repeats<2:parser.error('At least two independent repeats required')
    output=run(args.repeats)
    path=ROOT/'data/probe_official_s2_results.json'
    path.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(path),'all_100_sampling_comparisons_won':output['all_100_sampling_comparisons_won'],
                      'n2_error':output['scaling']['official_n2_max_probability_error'],'runtime_seconds':output['runtime_seconds']}))
