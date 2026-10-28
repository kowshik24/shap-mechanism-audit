"""Finite-state reference-transfer tests and non-Shapley response baselines."""
from functools import lru_cache
import numpy as np
from .exact_oracle import STATES,ALIGNED,conformal_threshold
from .fixed_reference import shap_kernel
from .conditional_audit import observed_overlap,table_f1

METHODS=['global','fixed3','source_q3','source_q5']
LAWS=[.01,.10,.25,.50,1.]

@lru_cache(maxsize=32)
def kernel(values):return shap_kernel(values)


def scores(values,counts,kind='shap'):
    values=np.asarray(values);counts=np.asarray(counts,dtype=float)
    if kind=='shap':
        local=np.einsum('sfb,kb->ksf',kernel(tuple(values)),counts/counts.sum(axis=1,keepdims=True))
        weighted=abs(local)*counts[:,:,None]
    else:
        response=np.stack([abs(values-values[np.arange(16)^(1<<(3-j))]) for j in range(4)],axis=1)
        weighted=counts[:,:,None]*response[None,:,:]
    valid=np.ones(len(counts),bool);all_scores=[];context=[]
    for m in range(3):
        own=(STATES[:,m]==1)&ALIGNED.astype(bool);rest=(~own)&ALIGNED.astype(bool)
        a,b=weighted[:,own].sum(axis=1),weighted[:,rest].sum(axis=1)
        da=a.sum(axis=1) if kind=='shap' else counts[:,own].sum(axis=1)
        db=b.sum(axis=1) if kind=='shap' else counts[:,rest].sum(axis=1)
        valid&=(da>0)&(db>0)
        a=np.divide(a,da[:,None],out=np.zeros_like(a),where=da[:,None]>0)
        b=np.divide(b,db[:,None],out=np.zeros_like(b),where=db[:,None]>0)
        excess=a-b;excess[:,m]=-np.inf
        s=excess.max(axis=1);j=excess.argmax(axis=1)
        all_scores.append(s);context.append(np.where(j==3,s,-np.inf))
    a=np.max(all_scores,axis=0);l=np.max(context,axis=0)
    a[~valid]=np.inf;l[~valid]=-np.inf
    return a,l,valid


def bins(counts,edges):
    o=observed_overlap(counts);b=np.searchsorted(edges[1:-1],o,side='right');b[~np.isfinite(o)]=-1
    return b


def cuts(base_scores,base_counts,tune_scores,tune_counts,source_counts,tie_cal,randomized=False):
    o=observed_overlap(source_counts);o=o[np.isfinite(o)]
    edges=[np.array([0.,1.]),np.array([0.,.8,.9,1.]),
           np.r_[0.,np.quantile(o,[1/3,2/3]),1.],np.r_[0.,np.quantile(o,[.2,.4,.6,.8]),1.]]
    result=[]
    for edge in edges:
        bb=bins(base_counts,edge);q=np.array([conformal_threshold(base_scores[bb==k]) for k in range(len(edge)-1)])
        tb=bins(tune_counts,edge);usable=(tb>=0)&np.isfinite(q[np.maximum(tb,0)])&np.isfinite(tune_scores)
        residual=np.full(len(tune_scores),-np.inf);residual[usable]=tune_scores[usable]-q[tb[usable]]
        if len(edge)==2:
            # Canonical raw cut avoids floating cancellation for a global offset.
            residual=tune_scores.copy();q=np.zeros(1)
        rank=int(np.ceil((len(residual)+1)*.95))
        order=np.lexsort((tie_cal,residual)) if randomized else np.argsort(residual)
        index=order[rank-1]
        result.append({'edges':edge.tolist(),'q':q.tolist(),'delta':float(residual[index]),
                       'tie_cut':float(tie_cal[index]) if randomized else 1.,'randomized':randomized})
    return result


def apply(score,counts,valid,cut,ties):
    b=bins(counts,cut['edges']);q=np.asarray(cut['q']);usable=valid&(b>=0)&np.isfinite(q[np.maximum(b,0)])
    residual=np.full(len(score),-np.inf);residual[usable]=score[usable]-q[b[usable]]
    return usable&((residual>cut['delta'])|((residual==cut['delta'])&(ties>cut['tie_cut'])))


def shifted_counts(test,rng):
    zcounts=test.reshape(-1,8,2).sum(axis=2)
    categories=rng.multinomial(zcounts,[.10,.15,.25,.50]);cumulative=categories.cumsum(axis=2)
    out=[test]
    for k in range(4):
        flips=cumulative[:,:,k];ones=np.where(STATES[::2,0][None,:]==0,flips,zcounts-flips)
        out.append(np.stack([zcounts-ones,ones],axis=2).reshape(-1,16))
    return np.asarray(out)


def eta_squared(score,counts,edges):
    b=bins(counts,edges);total=np.sum((score-score.mean())**2)
    between=sum(np.sum(b==k)*(score[b==k].mean()-score.mean())**2 for k in np.unique(b))
    return float(between/total) if total else 0.
