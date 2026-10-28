"""Sharp identification ranges for audits with unresolved expected features."""
import math
import numpy as np


def _partition(d,required,excluded):
    p=list(required);n=list(excluded)
    if any(not isinstance(i,(int,np.integer)) or i<0 or i>=d for i in p+n):
        raise ValueError('Feature indices must belong to the declared universe')
    pset,nset=set(p),set(n)
    if len(pset)!=len(p) or len(nset)!=len(n) or pset&nset:
        raise ValueError('Required and excluded features must be disjoint without duplicates')
    occupied=pset|nset;u=[i for i in range(d) if i not in occupied]
    return np.array(p,dtype=int),np.array(n,dtype=int),np.array(u,dtype=int)


def identification_ranges(own,comparison,required,excluded=()):
    a=np.asarray(own,dtype=float)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all() or (a<0).any() or not np.isclose(a.sum(),1,rtol=0,atol=1e-10):
        raise ValueError('Own shares must be a finite probability vector')
    p,n,u=_partition(len(a),required,excluded)
    result={'required':p.tolist(),'excluded':n.tolist(),'unknown':u.tolist(),
            'R':[float(a[p].sum()),float(a[np.r_[p,u]].sum())],
            'D':None,'S':None,'comparison_available':comparison is not None}
    if comparison is None:return result
    b=np.asarray(comparison,dtype=float)
    if b.shape!=a.shape or not np.isfinite(b).all() or (b<0).any() or not np.isclose(b.sum(),1,rtol=0,atol=1e-10):
        raise ValueError('Comparison shares must be a finite probability vector of the same size')
    delta=a-b;base=delta[p].sum()
    result['D']=[float(base+np.minimum(delta[u],0).sum()),float(base+np.maximum(delta[u],0).sum())]
    result['S']=[float(delta[n].max()) if len(n) else -np.inf,
                 float(delta[np.r_[n,u]].max()) if len(n)+len(u) else -np.inf]
    return result


def conservative_rank_cut(calibration_upper,alpha=.05):
    upper=np.asarray(calibration_upper,dtype=float)
    if upper.ndim!=1 or np.isnan(upper).any() or not 0<alpha<1:
        raise ValueError('Expected one-dimensional extended-real bounds and alpha in (0,1)')
    rank=math.ceil((len(upper)+1)*(1-alpha))
    return (float(np.sort(upper)[rank-1]) if rank<=len(upper) else np.inf),rank
