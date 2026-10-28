"""Evidence states for physical feature maps; unknown is never a negative edge."""
import numpy as np
from scipy.stats import fisher_exact


def holm_adjust(values):
    p=np.asarray(values,dtype=float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():
        raise ValueError('Expected finite p-values in [0,1]')
    order=np.argsort(p,kind='stable');out=np.empty_like(p)
    out[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return out


def randomized_response(values,flags):
    y=np.asarray(values,dtype=float);z=np.asarray(flags)
    if y.ndim!=1 or z.shape!=y.shape or not np.isfinite(y).all() or not set(z)<= {0,1} or len(set(z))!=2:
        raise ValueError('Expected finite outcomes and both binary treatment groups')
    center=float(np.median(y));deviation=abs(y-center)
    tests=[]
    for name,indicator,threshold in [('location',y>center,center),
                                    ('dispersion',deviation>np.median(deviation),float(np.median(deviation)))]:
        counts=[[int(np.count_nonzero((z==t)&(indicator==v))) for v in [0,1]] for t in [0,1]]
        p=float(fisher_exact(counts,alternative='two-sided').pvalue)
        risk=counts[1][1]/sum(counts[1])-counts[0][1]/sum(counts[0])
        tests.append({'endpoint':name,'threshold':threshold,'counts':counts,'p':p,'risk_difference':risk})
    return {'median_A':float(np.median(y[z==0])),'median_B':float(np.median(y[z==1])),
            'mean_difference':float(y[z==1].mean()-y[z==0].mean()),'tests':tests}


def path_status(adjacency,names,source,target):
    if source not in names or target not in names:return 'unmodeled'
    a=np.asarray(adjacency);i=names.index(source);j=names.index(target)
    if a[i,j]:return 'direct'
    reached={i};frontier=[i]
    while frontier:
        current=frontier.pop()
        for child in np.flatnonzero(a[current]):
            if child==j:return 'path'
            if child not in reached:reached.add(int(child));frontier.append(int(child))
    return 'no_listed_path'


def map_coverage(universe,supported,excluded):
    universe=set(universe);supported=set(supported);excluded=set(excluded)
    if not supported<=universe or not excluded<=universe or supported&excluded:
        raise ValueError('Evidence states must be disjoint subsets of the feature universe')
    unknown=universe-supported-excluded
    return {'supported':sorted(supported),'excluded':sorted(excluded),'unknown':sorted(unknown),
            'complete_relative_to_universe':not unknown}
