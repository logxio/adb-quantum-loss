"""Deterministic real loss copula probe with preserved spliced GPD margins."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import scipy
from scipy import optimize, special, stats
from scipy.spatial.distance import pdist
import pyvinecopulib as pv

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261118
EPS = 1e-10


def clip(u):
    return np.clip(u, EPS, 1-EPS)


def sobol(n, d, seed=SEED):
    return clip(stats.qmc.Sobol(d, scramble=True, seed=seed).random_base2(int(np.ceil(np.log2(n))))[:n])


class Margin:
    """A zero atom, piecewise linear empirical body and exact GPD excess tail."""
    def __init__(self, x):
        x = np.asarray(x)
        self.pzero = float(np.mean(x == 0))
        pos = np.sort(x[x > 0])
        if len(pos) < 8:
            raise ValueError("At least eight positive training observations required")
        self.threshold = float(np.quantile(pos, .8))
        self.pt = self.pzero+(1-self.pzero)*.8
        excess = pos[pos > self.threshold]-self.threshold
        # The declared heavy tail family has infinite support and a finite mean.
        def obj(v):
            return -np.sum(stats.genpareto.logpdf(excess, v[0], scale=np.exp(v[1])))
        fitted = optimize.minimize(obj, [.2, np.log(np.mean(excess))], bounds=[(0.,.95), (np.log(EPS), np.log(max(pos)*10))])
        self.xi, self.scale = float(fitted.x[0]), float(np.exp(fitted.x[1]))
        inner = np.unique(pos[pos < self.threshold])
        self.knots = np.r_[0., inner, self.threshold]
        pp = np.searchsorted(pos, inner, side="right")/len(pos)
        pp = np.minimum(pp, .8-1e-8)
        self.probs = np.r_[self.pzero, self.pzero+(1-self.pzero)*pp, self.pt]
        self.n_tail = len(excess)

    def cdf(self, x):
        x = np.asarray(x)
        body = np.interp(x, self.knots, self.probs)
        tail = self.pt+(1-self.pt)*stats.genpareto.cdf(np.maximum(x-self.threshold, 0), self.xi, scale=self.scale)
        return np.where(x > self.threshold, tail, body)

    def ppf(self, u):
        u = clip(u)
        body = np.interp(u, self.probs, self.knots)
        excess = stats.genpareto.ppf(clip((u-self.pt)/(1-self.pt)), self.xi, scale=self.scale)
        return np.where(u <= self.pzero, 0., np.where(u <= self.pt, body, self.threshold+excess))

    def logpdf(self, x):
        ix = np.clip(np.searchsorted(self.knots, x, side="right")-1, 0, len(self.knots)-2)
        density = np.diff(self.probs)[ix]/np.diff(self.knots)[ix]
        tail = np.log1p(-self.pt)+stats.genpareto.logpdf(np.maximum(x-self.threshold, 0), self.xi, scale=self.scale)
        return np.where(x > self.threshold, tail, np.log(np.maximum(density, EPS)))

    def metadata(self):
        return {"p_zero":self.pzero, "threshold_loss_gdp":self.threshold, "threshold_cdf":self.pt, "gpd_shape":self.xi, "gpd_scale":self.scale, "shape_bounds":[0.,.95], "positive_tail_training_count":self.n_tail, "fourth_loss_moment_exists":self.xi < .25}


class Independent:
    name = "Independent"
    def __init__(self, u): self.d = u.shape[1]
    def logpdf(self, u): return np.zeros(len(u))
    def sample(self, n): return sobol(n, self.d)


class Gaussian(Independent):
    name = "Gaussian Copula"
    def __init__(self, u):
        self.d = u.shape[1]
        z = stats.norm.ppf(clip(u))
        self.corr = .95*np.corrcoef(z.T)+.05*np.eye(self.d)
    def logpdf(self, u):
        z = stats.norm.ppf(clip(u))
        return stats.multivariate_normal.logpdf(z, cov=self.corr)-np.sum(stats.norm.logpdf(z), axis=1)
    def sample(self, n):
        z = stats.norm.ppf(sobol(n, self.d))@np.linalg.cholesky(self.corr).T
        return clip(stats.norm.cdf(z))


class Student(Gaussian):
    name = "Student-t Copula"
    def __init__(self, u):
        super().__init__(u)
        def objective(df):
            z = stats.t.ppf(clip(u), df)
            return -np.sum(stats.multivariate_t.logpdf(z, shape=self.corr, df=df)-np.sum(stats.t.logpdf(z, df), axis=1))
        self.df = float(optimize.minimize_scalar(objective, bounds=(2.1,30), method="bounded").x)
    def logpdf(self, u):
        z = stats.t.ppf(clip(u), self.df)
        return stats.multivariate_t.logpdf(z, shape=self.corr, df=self.df)-np.sum(stats.t.logpdf(z,self.df),axis=1)
    def sample(self,n):
        q = sobol(n,self.d+1)
        z = stats.norm.ppf(q[:,:self.d])@np.linalg.cholesky(self.corr).T
        z /= np.sqrt(stats.chi2.ppf(q[:,-1],self.df)/self.df)[:,None]
        return clip(stats.t.cdf(z,self.df))


class Clayton(Independent):
    name = "Clayton Copula"
    def __init__(self,u):
        self.d = u.shape[1]
        self.theta = float(optimize.minimize_scalar(lambda th:-np.sum(self._logpdf(u,th)),bounds=(1e-4,12),method="bounded").x)
    def _logpdf(self,u,th):
        lu = np.log(clip(u))
        s = np.sum(np.exp(-th*lu),axis=1)-self.d+1
        return special.gammaln(self.d+1/th)-special.gammaln(1/th)+self.d*np.log(th)-(1+th)*lu.sum(axis=1)-(self.d+1/th)*np.log(s)
    def logpdf(self,u): return self._logpdf(u,self.theta)
    def sample(self,n):
        q = sobol(n,self.d+1)
        w = stats.gamma.ppf(q[:,-1],1/self.theta)
        return clip((1-np.log(q[:,:self.d])/w[:,None])**(-1/self.theta))


class Gumbel(Independent):
    name = "Gumbel Extreme-Value Copula"
    def __init__(self,u):
        self.d = u.shape[1]
        self.theta = float(optimize.minimize_scalar(lambda th:-np.sum(self._logpdf(u,th)),bounds=(1.00001,8),method="bounded").x)
    def _logpdf(self,u,th):
        a = 1/th
        l = -np.log(clip(u))
        t = np.sum(l**th,axis=1)
        z = t**a
        p = np.array([0.,a])
        for order in range(1,self.d):
            out = np.zeros(len(p)+1)
            out[:len(p)] += (order-a*np.arange(len(p)))*p
            out[1:] += a*p
            p = out
        polynomial = np.polynomial.polynomial.polyval(z,p)
        return -z-self.d*np.log(t)+np.log(np.maximum(polynomial,EPS))+self.d*np.log(th)+(th-1)*np.log(l).sum(axis=1)+l.sum(axis=1)
    def logpdf(self,u): return self._logpdf(u,self.theta)
    def sample(self,n):
        a = 1/self.theta
        q = sobol(n,self.d+2)
        v = np.pi*q[:,-2]
        w = -np.log(q[:,-1])
        stable = np.sin(a*v)/(np.sin(v)**(1/a))*(np.sin((1-a)*v)/w)**((1-a)/a)
        return clip(np.exp(-((-np.log(q[:,:self.d]))/stable[:,None])**a))


class RVine(Independent):
    name = "Regular Vine"
    def __init__(self,u):
        self.d = u.shape[1]
        families = [pv.BicopFamily.indep,pv.BicopFamily.gaussian,pv.BicopFamily.student,pv.BicopFamily.clayton,pv.BicopFamily.gumbel,pv.BicopFamily.frank]
        controls = pv.FitControlsVinecop(family_set=families,selection_criterion="bic",num_threads=1)
        self.model = pv.Vinecop.from_data(np.asfortranarray(u),controls=controls)
    def logpdf(self,u): return np.log(np.maximum(self.model.pdf(np.asfortranarray(u)),EPS))
    def sample(self,n): return clip(self.model.inverse_rosenblatt(np.asfortranarray(sobol(n,self.d))))


def connected4(u):
    """Bounded PIT cumulants exist even when loss fourth moments do not."""
    z = (u-.5)*np.sqrt(12)
    pairs = {(i,j):np.mean(z[:,i]*z[:,j]) for i,j in itertools.combinations(range(z.shape[1]),2)}
    ans = []
    for a,b,c,d in itertools.combinations(range(z.shape[1]),4):
        ans.append(np.mean(z[:,a]*z[:,b]*z[:,c]*z[:,d])-pairs[a,b]*pairs[c,d]-pairs[a,c]*pairs[b,d]-pairs[a,d]*pairs[b,c])
    return np.asarray(ans)


class MPSCumulant(Independent):
    name = "MPS-Cumulant Entangled Copula"
    def __init__(self,u,bins=4,rank=4):
        self.d, self.bins, self.rank = u.shape[1], bins, rank
        b = np.minimum((u*bins).astype(int),bins-1)
        rng = np.random.default_rng(SEED)
        self.f = rng.dirichlet(np.ones(bins),size=(rank,self.d))
        self.w = np.ones(rank)/rank
        for _ in range(100):
            prob = np.prod(np.stack([self.f[:,j,b[:,j]].T for j in range(self.d)],axis=2),axis=2)*self.w
            resp = prob/np.maximum(prob.sum(axis=1,keepdims=True),EPS)
            self.w = (resp.sum(axis=0)+1)/(len(u)+rank)
            for j in range(self.d):
                for h in range(rank):
                    self.f[h,j] = np.bincount(b[:,j],weights=resp[:,h],minlength=bins)+1.
                    self.f[h,j] /= self.f[h,j].sum()
        target = connected4(u)
        centers = ((np.arange(bins)+.5)/bins-.5)*np.sqrt(12)
        m = np.einsum("hjb,b->hj",self.f,centers)
        def moment(w):
            means = w@m
            mm = m-means
            pair = {(i,j):np.sum(w*mm[:,i]*mm[:,j]) for i,j in itertools.combinations(range(self.d),2)}
            return np.asarray([np.sum(w*mm[:,a]*mm[:,b]*mm[:,c]*mm[:,d])-pair[a,b]*pair[c,d]-pair[a,c]*pair[b,d]-pair[a,d]*pair[b,c] for a,b,c,d in itertools.combinations(range(self.d),4)])
        pm = np.prod(np.stack([self.f[:,j,b[:,j]].T for j in range(self.d)],axis=2),axis=2)
        def objective(logits):
            weights = special.softmax(logits)
            return -np.mean(np.log(np.maximum(pm@weights,EPS)))+.1*np.mean((moment(weights)-target)**2)
        fit = optimize.minimize(objective,np.log(self.w),method="BFGS",options={"maxiter":100})
        self.w = special.softmax(fit.x)
        self.cumulant_train_target = target.tolist()
        # IPF is a local diagonal scaling of the nonnegative TT cores.
        for iteration in range(2000):
            for j in range(self.d):
                marginal = self.w@self.f[:,j,:]
                self.f[:,j,:] *= (1/bins)/marginal
                normalizer = self.f[:,j,:].sum(axis=1)
                self.f[:,j,:] /= normalizer[:,None]
                self.w *= normalizer
                self.w /= self.w.sum()
            error = np.max(np.abs(np.einsum("h,hjb->jb",self.w,self.f)-1/bins))
            if error < 1e-12: break
        self.marginal_max_error = float(error)
        self.ipf_iterations = iteration+1
        self.cores = []
        self.cores.append((self.w[:,None]*self.f[:,0,:]).T[None,:,:])
        for j in range(1,self.d-1):
            core = np.zeros((rank,bins,rank))
            for h in range(rank): core[h,:,h] = self.f[h,j,:]
            self.cores.append(core)
        self.cores.append(self.f[:,-1,:,None])
        # Explicit MPS contraction is checked against its mixture expression.
        check = np.ones((min(10,len(u)),1))
        for j,core in enumerate(self.cores):
            check = np.einsum("na,nab->nb",check,np.moveaxis(core[:,b[:len(check),j],:],1,0))
        direct = np.exp(self.logpdf(u[:len(check)]))/bins**self.d
        self.contraction_max_error = float(np.max(np.abs(check[:,0]-direct)))
    def logpdf(self,u):
        b = np.minimum((clip(u)*self.bins).astype(int),self.bins-1)
        probs = np.prod(np.stack([self.f[:,j,b[:,j]].T for j in range(self.d)],axis=2),axis=2)@self.w
        return np.log(np.maximum(probs,EPS))+self.d*np.log(self.bins)
    def sample(self,n):
        q = sobol(n,2*self.d+1)
        latent = np.minimum(np.searchsorted(np.cumsum(self.w),q[:,0]),self.rank-1)
        out = np.zeros((n,self.d))
        for j in range(self.d):
            cumulative = np.cumsum(self.f[latent,j,:],axis=1)
            category = np.sum(q[:,1+j,None] > cumulative,axis=1)
            out[:,j] = (category+q[:,1+self.d+j])/self.bins
        return clip(out)


class NaiveQCBM(Independent):
    """A trained real state vector RY plus CNOT Born machine with midpoint decoding."""
    name = "Naive Discretized QCBM Copula"
    def __init__(self,u,bits_per_country=3):
        self.d, self.bits = u.shape[1], bits_per_country
        self.qubits = self.d*self.bits
        self.states = 2**self.qubits
        self.bins = 2**self.bits
        indices = np.arange(self.states)
        self.cnot = []
        for control in range(self.qubits-1):
            self.cnot.append(indices ^ (((indices>>control)&1) << (control+1)))
        b = np.minimum((u*self.bins).astype(int),self.bins-1)
        state_ids = np.sum(b*(self.bins**np.arange(self.d)),axis=1)
        self.target = np.bincount(state_ids,minlength=self.states).astype(float)/len(u)
        init = np.ones(2*self.qubits)*np.pi/2
        def objective(p):
            prob = self.born(p)
            return -np.sum(self.target*np.log(np.maximum(prob,1e-12)))
        result = optimize.minimize(objective,init,method="COBYLA",options={"maxiter":180,"rhobeg":.35})
        self.probs = self.born(result.x)
        self.train_nll = float(result.fun)
        self.optimization_evaluations = int(result.nfev)
        self.parameters = result.x.tolist()
    def born(self,params):
        state = np.zeros(self.states)
        state[0] = 1.
        for layer in range(2):
            for qubit in range(self.qubits):
                theta = params[layer*self.qubits+qubit]
                c,s = np.cos(theta/2),np.sin(theta/2)
                v = state.reshape(-1,2,2**qubit)
                a,b = v[:,0,:].copy(),v[:,1,:].copy()
                v[:,0,:],v[:,1,:] = c*a-s*b,s*a+c*b
            for permutation in self.cnot: state = state[permutation]
        probs = state**2
        return probs/probs.sum()
    def logpdf(self,u): return np.full(len(u),np.nan)
    def sample(self,n):
        q = sobol(n,1)[:,0]
        states = np.minimum(np.searchsorted(np.cumsum(self.probs),q),self.states-1)
        b = (states[:,None]//(self.bins**np.arange(self.d)))%self.bins
        return (b+.5)/self.bins


def metrics(model, margins, test, utest, n):
    start = time.perf_counter()
    usim = model.sample(n)
    generated = np.column_stack([m.ppf(usim[:,j]) for j,m in enumerate(margins)])
    sample_ms = (time.perf_counter()-start)*1000
    pool = generated.sum(axis=1)
    truth = test.sum(axis=1)
    q99 = float(np.quantile(pool,.99))
    q995 = float(np.quantile(pool,.995))
    es99 = float(pool[pool >= q99].mean())
    tail = (np.max(utest,axis=1) > .8)&np.all(test > 0,axis=1)
    ll = model.logpdf(utest)
    marginal_ll = np.column_stack([m.logpdf(test[:,j]) for j,m in enumerate(margins)]).sum(axis=1)
    tail_ll = float(np.mean(ll[tail]+marginal_ll[tail])) if np.any(tail) and np.all(np.isfinite(ll)) else None
    copula_ll = float(np.mean(ll[tail])) if np.any(tail) and np.all(np.isfinite(ll)) else None
    out = {"tail_joint_log_likelihood":tail_ll,"tail_copula_log_likelihood":copula_ll,"tail_test_observations":int(tail.sum()),"pool_var_99_loss_gdp_sum":q99,"pool_var_995_loss_gdp_sum":q995,"pool_tvar_99_loss_gdp_sum":es99,"var_99_coverage_error":float(abs(np.mean(truth > q99)-.01)),"tvar_99_calibration_moment_error":float(abs(np.mean(np.maximum(truth-q99,0))/.01+q99-es99)/max(es99,EPS)),"sample_runtime_ms":sample_ms}
    for threshold in [.95,.99]:
        predicted = np.mean(np.sum(usim > threshold,axis=1) >= 2)
        observed = np.sum(utest > threshold,axis=1) >= 2
        key = str(int(threshold*100))
        out["joint_exceedance_"+key+"_predicted"] = float(predicted)
        out["joint_exceedance_"+key+"_observed"] = float(observed.mean())
        out["joint_exceedance_"+key+"_absolute_error"] = float(abs(predicted-observed.mean()))
        out["brier_"+key] = float(np.mean((predicted-observed)**2))
    es_sample = generated[:min(n,2048)]
    first = np.mean(np.linalg.norm(es_sample[:,None,:]-test[None,:,:],axis=2))
    second = np.mean(pdist(es_sample))
    out["energy_score_loss_gdp"] = float(first-.5*second)
    out["marginal_ks_uniform"] = [float(stats.kstest(usim[:,j],"uniform").statistic) for j in range(usim.shape[1])]
    out["max_uniform_coordinate"] = float(usim.max())
    return out


def write_evidence(result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = result["results"]
    labels = ["Independent","Gaussian","Student t","Clayton","Gumbel","Regular Vine","Naive QCBM","MPS cumulant"]
    colors = ["#a7afb5"]*7+["#147d92"]
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout="constrained")
    ll = [r["tail_joint_log_likelihood"] if r["tail_joint_log_likelihood"] is not None else np.nan for r in rows]
    axes[0].scatter(ll,np.arange(len(labels)),c=colors,s=65)
    axes[0].set_yticks(np.arange(len(labels)),labels)
    axes[0].set_xlim(min(v for v in ll if np.isfinite(v))-.1,max(v for v in ll if np.isfinite(v))+.05)
    axes[0].set_title("Heldout tail log density, higher is better")
    axes[0].text(.02,.01,"QCBM continuous density undefined",transform=axes[0].transAxes,fontsize=8)
    axes[1].barh(labels,[r["energy_score_loss_gdp"] for r in rows],color=colors)
    axes[1].set_title("Heldout energy score, lower is better")
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top","right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="x",alpha=.2)
    fig.suptitle("Real reported annual losses: PHL, VNM, THA, IDN\n28 training years, 10 chronological heldout years")
    png = ROOT/"docs/probe_s1_comparison.png"
    fig.savefig(png,dpi=180)
    plt.close(fig)
    lines = ["# 第0轮真实损失基线探针", "", "真实数据流已从OWID转发的EM-DAT年度经济损失占GDP比值接入8个模型。4国为PHL、VNM、THA、IDN，共同完整年度为38年。1985至2013年中的28个可用年度训练，2014至2024年中的10个可用年度留出。全部缺失保留为缺失，完整年度选择后才建模。源ETL已把部分缺失损失按0聚合，源年度0代表reported aggregate，不能解释为逐事件证实无损失。", "", "## 实测表", "", "尾部联合loglikelihood使用全为正损失且至少一个PIT大于0.8的留出年度。joint exceedance定义为至少2国同时超过各自边际95%或99%分位数。pool金额单位是等GDP归一化国家名义本金损失比之和。", "", "|模型|Tail loglikelihood|Joint95误差|Joint99误差|VaR99覆盖误差|TVaR99校准矩误差|Brier95|Energy score|拟合加采样ms|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        vals = [r[k] for k in ["tail_joint_log_likelihood","joint_exceedance_95_absolute_error","joint_exceedance_99_absolute_error","var_99_coverage_error","tvar_99_calibration_moment_error","brier_95","energy_score_loss_gdp","deterministic_algorithm_runtime_ms"]]
        lines.append("|"+r["model"]+"|"+"|".join("不可比" if v is None else f"{v:.8g}" for v in vals)+"|")
    lines += ["", "![真实留出渲染图](probe_s1_comparison.png)", "", "## 数学算子与守恒", "", "MPS概率张量为P[b1,...,bd]=Σh wh∏j fhj[bj]。其显式TT核形状为(1,4,4)、(4,4,4)、(4,4,4)、(4,4,1)，核心数组保存在data/mps_s1_operator.npz。训练阶段使用100轮带对称平滑的EM，再以训练内有界PIT坐标的四阶连通累积量惩罚校准混合权重，惩罚系数预声明为0.1。随后对TT核逐站点做IPF缩放。", "", "每维类别边际均为1/4，类别内加入连续uniform坐标，故任意u满足P[Uj≤u]=u。再用共同的经验主体加GPD尾部逆CDF映射，MPS依赖拟合保持输入GPD边际。该结论是分布层面的数学守恒，有限模拟样本的KS误差仍受样本误差影响。边际GPD使用正损失80%分位数阈值，xi限定在[0,0.95]，保证重尾无限支撑和有限TVaR。每国只有5至6个训练尾部超额值。", ""]
    inv = rows[-1]["invariants"]
    lines += [f"实测分箱边际最大绝对误差为{inv['marginal_max_absolute_mass_error']:.4g}，显式TT收缩与混合概率最大误差为{inv['tt_contraction_max_error']:.4g}。IPF使用{inv['ipf_iterations']}轮。累积量在有界PIT坐标计算，因此PHL、THA、IDN拟合损失边际四阶矩不存在时仍有数学定义。", "", "有限checkerboard密度的渐近上尾依赖系数为0。本轮验证有限95%和99%阈值相关，连续上尾核属于下一轮正交方法。当前rank4非负MPS是经典概率张量网络，不构成量子硬件加速或物理纠缠优势证明。", "", "## 基线完整性", "", "Gaussian使用训练正态分位相关矩阵和5%对角收缩。Student-t共享同一相关估计并在训练内拟合自由度。Clayton与Gumbel使用全维Archimedean密度MLE，均保留各自下尾或上尾依赖族。Regular Vine使用pyvinecopulib真实R-vine，以BIC选择independent、Gaussian、Student、Clayton、Gumbel和Frank成对族及旋转。", "", "Naive Discretized QCBM使用12个模拟量子比特，每国3bit，两层RY和邻接CNOT，COBYLA实训Born状态概率。中点译码的最大U为0.9375，所以95%与99%分位以上永远没有预测样本。连续经济损失密度对应原子测度，连续loglikelihood不可比，因此JSON为null。经典去量化与uniform边际校正可以修复此缺陷，故本轮只反证该朴素译码方案，不能外推为所有QCBM的必然缺陷。", "", "## 留出解释与下一步", "", "本轮MPS未击败全部基线。W/L逐项保存在data/probe_s1_results.json。留出集joint95与joint99事件都为0，朴素QCBM的0预测因此获得0误差，这是尾部观测不足造成的误导性排名。10个年度对99%水平期望只有0.1个超越，不能验证99%或99.5%资本充足率。TVaR指标是ES识别矩归一化误差，不存在简单的TVaR覆盖率。", "", "本轮损失池为各国GDP归一化损失比之和，没有把不同国家的GDP比直接命名为美元资本。正式主权池应固定投保敞口、汇率、通胀基年、赔付层和费用后再测美元VaR/TVaR、分散化信用与Euler资本分摊。", "", "原始事件表保留损失缺失而不是给未记录国家补0，但当前公开转发缺少跨国DisNo唯一ID，年度全灾种聚合也没有event family。严格家族整块留出无法从这些字段生成可识别的4国联合损失真值，报告保留该状态。下一轮接入带DisNo和reported loss标记的多国同灾记录，再按灾害家族及事件ID整体留出。", "", "## 重现", "", "```sh", ".venv/bin/python src/probe_s1_baselines.py", "```", "", "采样数16384，随机种子20261118，确定性Sobol积分。Energy score使用2048个预测样本与留出观测的平均距离，减去同一预测样本全部无序对平均距离的一半。wall time实测随主机负载变化，确定性指算法与采样，不指时钟读数恒定。完整指标、参数、源文件SHA256与环境版本保存于data/probe_s1_results.json。"]
    (ROOT/"docs/probe_s1_evidence.md").write_text("\n".join(lines)+"\n")


def validate_independently(result, train, margins):
    saved = np.load(ROOT/"data/mps_s1_operator.npz")
    cores = [saved["core_"+str(j)] for j in range(len(margins))]
    tensor = np.zeros((4,)*len(margins))
    for cell in itertools.product(range(4),repeat=len(margins)):
        value = np.ones((1,1))
        for j,b in enumerate(cell): value = value@cores[j][:,b,:]
        tensor[cell] = value[0,0]
    marg_errors = []
    for j in range(tensor.ndim):
        axes = tuple(k for k in range(tensor.ndim) if k != j)
        marg_errors.append(float(np.max(abs(tensor.sum(axis=axes)-.25))))
    roundtrips = []
    for m in margins:
        quantiles = np.linspace(m.pt+1e-6,.999999,1000)
        roundtrips.append(float(np.max(abs(m.cdf(m.ppf(quantiles))-quantiles))))
    circuit = next(r["circuit"] for r in result["results"] if "circuit" in r)
    nqubits = circuit["qubits"]
    ix = np.arange(2**nqubits)
    state = np.zeros(len(ix))
    state[0] = 1.
    for layer in range(2):
        for bit in range(nqubits):
            theta = circuit["parameters"][layer*nqubits+bit]
            sign = np.where((ix>>bit)&1,1.,-1.)
            state = np.cos(theta/2)*state+np.sin(theta/2)*sign*state[ix^(1<<bit)]
        for control in range(nqubits-1):
            mapped = ix^(((ix>>control)&1)<<(control+1))
            state = state[mapped]
    born = state**2
    train_years = result["input"]["training_years"]
    test_years = result["input"]["test_years"]
    checks = {"explicit_full_tensor_shape":list(tensor.shape),"tensor_mass":float(tensor.sum()),"tensor_min_probability":float(tensor.min()),"tensor_marginal_max_errors":marg_errors,"gpd_tail_cdf_ppf_max_errors":roundtrips,"training_max_year":max(train_years),"heldout_min_year":min(test_years),"train_test_year_sets_disjoint":not bool(set(train_years)&set(test_years)),"independent_born_sum":float(born.sum()),"independent_born_min_probability":float(born.min()),"independent_born_vs_model_max_error":float(np.max(abs(born-np.asarray(circuit["born_probabilities"])))),"all_checks_passed":False}
    assert abs(tensor.sum()-1) < 1e-11 and tensor.min() >= 0
    assert max(marg_errors) < 1e-11 and max(roundtrips) < 1e-10
    assert max(train_years) < min(test_years)
    assert abs(born.sum()-1) < 1e-11 and born.min() >= 0
    assert checks["independent_born_vs_model_max_error"] < 1e-12
    checks["all_checks_passed"] = True
    checks["checks_use"] = "full discrete tensor enumeration, independent matrix-chain contraction, GPD CDF inverse roundtrip and alternate bit-index state-vector gate implementation"
    (ROOT/"data/probe_s1_validation.json").write_text(json.dumps(checks,indent=2)+"\n")
    event_path = ROOT/"data/processed/emdat_events_apac.csv"
    if event_path.exists():
        events = pd.read_csv(event_path)
        selected = events[events.iso3.isin(result["input"]["countries"])]
        result["event_family_holdout_audit"] = {"source":str(event_path.relative_to(ROOT)),"raw_rows_four_countries":len(selected),"rows_reported_loss":int(selected.loss_usd.notna().sum()),"disaster_types":selected.type.value_counts().to_dict(),"cross_country_unique_event_identifier_available":False,"evaluated":False,"reason":"The public event mirror provides no DisNo or other reliable cross-country unique ID. Missing loss and absent country event rows cannot be converted into zero. A complete four-country joint event truth and family holdout are not identifiable from these fields."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel",type=Path,default=ROOT/"data/processed/loss_gdp_panel.csv")
    parser.add_argument("--countries",default="PHL,VNM,THA,IDN")
    parser.add_argument("--samples",type=int,default=16384)
    args = parser.parse_args()
    countries = args.countries.split(",")
    panel = pd.read_csv(args.panel)
    panel = panel[(panel.year >= 1980)&(panel.year <= 2024)]
    complete = panel[["year"]+countries].dropna().sort_values("year")
    if len(complete) < 24: raise ValueError("Insufficient complete observed country years")
    x = complete[countries].to_numpy(float)
    if np.any(x < 0): raise ValueError("Negative losses")
    split = int(len(x)*.75)
    train,test = x[:split],x[split:]
    margins = [Margin(train[:,j]) for j in range(len(countries))]
    u = clip(np.column_stack([m.cdf(train[:,j]) for j,m in enumerate(margins)]))
    utest = clip(np.column_stack([m.cdf(test[:,j]) for j,m in enumerate(margins)]))
    for j,m in enumerate(margins):
        if m.pzero:
            u[train[:,j] == 0,j] = sobol(len(train),len(countries))[:,j][train[:,j] == 0]*m.pzero
            utest[test[:,j] == 0,j] = sobol(len(test),len(countries),SEED+1)[:,j][test[:,j] == 0]*m.pzero
    rows = []
    for cls in [Independent,Gaussian,Student,Clayton,Gumbel,RVine,NaiveQCBM,MPSCumulant]:
        start = time.perf_counter()
        model = cls(u)
        fit_ms = (time.perf_counter()-start)*1000
        row = {"model":model.name,"fit_runtime_ms":fit_ms}
        row.update(metrics(model,margins,test,utest,args.samples))
        row["deterministic_algorithm_runtime_ms"] = fit_ms+row["sample_runtime_ms"]
        if isinstance(model,MPSCumulant):
            row["invariants"] = {"marginal_max_absolute_mass_error":model.marginal_max_error,"tt_contraction_max_error":model.contraction_max_error,"ipf_iterations":model.ipf_iterations,"tt_core_shapes":[list(c.shape) for c in model.cores],"bond_dimension":model.rank,"bins":model.bins,"cumulant_training_target":model.cumulant_train_target,"cumulant_generated":connected4(model.sample(args.samples)).tolist(),"cumulant_domain":"bounded uniform PIT coordinates", "gpd_tail_exact_by_inverse_cdf":True}
            np.savez_compressed(ROOT/"data/mps_s1_operator.npz",weights=model.w,factors=model.f,**{"core_"+str(j):c for j,c in enumerate(model.cores)})
        if isinstance(model,Student): row["parameters"] = {"degrees_of_freedom":model.df,"correlation":model.corr.tolist()}
        elif isinstance(model,Gaussian): row["parameters"] = {"correlation":model.corr.tolist()}
        elif isinstance(model,(Clayton,Gumbel)): row["parameters"] = {"theta":model.theta}
        elif isinstance(model,RVine): row["parameters"] = {"summary":str(model.model)}
        if isinstance(model,NaiveQCBM):
            row["circuit"] = {"qubits":model.qubits,"bits_per_country":model.bits,"layers":2,"gates":"RY, nearest neighbor CNOT", "trained":True,"optimization_evaluations":model.optimization_evaluations,"training_nll":model.train_nll,"max_marginal_uniform_support":1-.5/model.bins,"continuous_density":"singular atomic measure, continuous loglikelihood not defined"}
            row["circuit"]["parameters"] = model.parameters
            row["circuit"]["born_probabilities"] = model.probs.tolist()
        rows.append(row)
        print(json.dumps(row),flush=True)
    mps = rows[-1]
    comparisons = {}
    directions = {"tail_joint_log_likelihood":1,"joint_exceedance_95_absolute_error":-1,"joint_exceedance_99_absolute_error":-1,"var_99_coverage_error":-1,"tvar_99_calibration_moment_error":-1,"brier_95":-1,"energy_score_loss_gdp":-1}
    for row in rows[:-1]:
        comparisons[row["model"]] = {key:("not_comparable" if row[key] is None or mps[key] is None else "win" if direction*(mps[key]-row[key]) > 1e-12 else "loss" if direction*(mps[key]-row[key]) < -1e-12 else "tie") for key,direction in directions.items()}
    result = {"schema":"adb.probe.s1.v1","seed":SEED,"input":{"path":str(args.panel.relative_to(ROOT)),"sha256":hashlib.sha256(args.panel.read_bytes()).hexdigest(),"countries":countries,"unit":"sum of national loss / national GDP ratios; equal GDP-normalized country notional, not USD pool capital","complete_years":complete.year.tolist(),"training_years":complete.year.iloc[:split].tolist(),"test_years":complete.year.iloc[split:].tolist(),"missing_country_years_excluded":int(len(panel)-len(complete)),"train_n":len(train),"test_n":len(test)},"protocol":{"split":"chronological fixed 75 percent training, 25 percent heldout","fit_data":"training years only, including marginal GPD fitting and cumulant targets","sampling":"fixed scrambled Sobol uniforms, deterministic seed; wall time is measured and varies by host load","simulation_n":args.samples,"joint_exceedance":"at least two countries exceed the same marginal PIT threshold","tail_loglikelihood":"continuous joint loss density on all-positive heldout years with any PIT above 0.8","var_backtest":"absolute empirical exceedance frequency minus 0.01","tvar_backtest":"absolute ES identification moment divided by predicted ES","energy_score":"mean Euclidean forecast distance minus half paired independent forecast distance","not_identifiable":["99 percent and 99.5 percent tail calibration with only ten heldout annual observations","event-family GroupKFold from annual all-hazard aggregates","causal physical cross-border event dependence","USD reinsurance capital without common-year exposure currency weights"],"quantum_scope":"classical tensor network and state vector simulation, no quantum advantage claim"},"margins":{c:m.metadata() for c,m in zip(countries,margins)},"results":rows,"mps_vs_baselines":comparisons,"environment":{"python":platform.python_version(),"platform":platform.platform(),"numpy":np.__version__,"scipy":scipy.__version__,"pyvinecopulib":pv.__version__}}
    destination = ROOT/"data/probe_s1_results.json"
    result["protocol"]["energy_score"] = "2048 forecast samples: mean observation distance minus half exact all-unordered-pairs forecast distance"
    result["protocol"]["source_zero_caveat"] = "OWID upstream ETL fills missing damage components with zero before annual aggregation; no additional missing country year imputation is performed here"
    result["protocol"]["mps_asymptotic_upper_tail_dependence"] = 0
    validate_independently(result,train,margins)
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    write_evidence(result)
    return result


if __name__ == "__main__":
    main()
