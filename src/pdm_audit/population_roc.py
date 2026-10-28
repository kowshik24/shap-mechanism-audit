"""Population-calibrated ranks and retrospective matched-warning comparisons."""
import numpy as np


def upper_ranks(calibration, cal_ties, values, ties, available, randomized):
    """Conservative upper-tail p-values, or lexicographic randomized ranks."""
    c=np.asarray(calibration); x=np.asarray(values)
    if randomized:
        dtype=[('score','f8'),('tie','f8')]
        reference=np.empty(c.shape,dtype=dtype)
        reference['score']=c; reference['tie']=cal_ties
        query=np.empty(x.shape,dtype=dtype)
        query['score']=x; query['tie']=ties
        ordered=np.sort(reference)
        below=np.searchsorted(ordered,query,side='left')
    else:
        below=np.searchsorted(np.sort(c),x,side='left')
    p=(1+len(c)-below)/(len(c)+1)
    return np.where(available & np.isfinite(x),p,2.)


def population_pvalues(calibration, cal_bins, cal_ties, values, value_bins,
                       ties, available, randomized, groups):
    out=np.full(np.shape(values),2.)
    for b in range(groups):
        selected=np.asarray(cal_bins)==b; use=np.asarray(value_bins)==b
        if selected.sum()<19: continue
        out[use]=upper_ranks(np.asarray(calibration)[selected],np.asarray(cal_ties)[selected],
                            np.asarray(values)[use],np.asarray(ties)[use],
                            np.asarray(available)[use],randomized)
    return out


def matched_boundary(null_p, rate):
    """Empirical expected size exactly rate, including null-score atoms."""
    p=np.asarray(null_p).ravel(); desired=rate*len(p)
    q=float(np.partition(p,max(0,int(np.ceil(desired))-1))[max(0,int(np.ceil(desired))-1)])
    if q>1: raise ValueError('Requested rate exceeds supported null fraction')
    below=int(np.count_nonzero(p<q)); equal=int(np.count_nonzero(p==q))
    weight=float((desired-below)/equal)
    assert -1e-12<=weight<=1+1e-12
    return q,float(np.clip(weight,0,1))


def boundary_rate(p,boundary):
    q,w=boundary
    return float(np.mean(np.asarray(p)<q)+w*np.mean(np.asarray(p)==q))
