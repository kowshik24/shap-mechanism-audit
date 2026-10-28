"""Conformal calibration on one independent batch per aligned training draw."""
import pdm_audit
import hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.stats import beta
from pdm_audit.reference_transfer import scores,bins,METHODS,LAWS
from pdm_audit.conditional_audit import observed_overlap
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def cut_for(values,counts,edges,uniforms,randomized):
    b=bins(counts,edges);cuts=[]
    for j in range(len(edges)-1):
        ids=np.flatnonzero(b==j);rank=int(np.ceil((len(ids)+1)*.95))
        if rank>len(ids):cuts.append({'q':'Infinity','tie':1.,'n':len(ids)});continue
        order=np.lexsort((uniforms[ids],values[ids])) if randomized else np.argsort(values[ids]);i=ids[order[rank-1]]
        cuts.append({'q':float(values[i]),'tie':float(uniforms[i]) if randomized else 1.,'n':len(ids)})
    return cuts


def apply(a,counts,valid,edges,cuts,ties):
    b=bins(counts,edges);q=np.full(len(a),np.inf);u=np.ones(len(a))
    for j,c in enumerate(cuts):q[b==j]=float(c['q']);u[b==j]=c['tie']
    usable=valid&np.isfinite(q)&(b>=0)
    return usable&((a>q)|((a==q)&(ties>u))),~usable


def main():
    start=time.perf_counter();r=json.loads((OUT/'reference_transfer.json').read_text());d=np.load(OUT/'reference_transfer_draws.npz')
    old=np.load(OUT/'observed_overlap_draws.npz');base=old['target_counts'];source=old['source_counts'];test=old['test_counts']
    shifted=d['shift_counts'];ties=d['ties'];f1=d['f1'];N=len(r['records'])
    null=np.zeros((N,2,4,200),bool);local=np.zeros((N,2,4,5,200),bool);abstain=np.zeros_like(null);banks=[];summary=[]
    rng=np.random.default_rng(2026102547)
    for n in [1000,10000]:
        for family in ['tree','forest','logistic']:
            ids=[i for i,x in enumerate(r['records']) if x['n']==n and x['family']==family]
            calibration=np.array([base[i,0] for i in ids]);source_o=observed_overlap(np.array([source[i,0] for i in ids]))
            edges=[[0.,1.],[0.,.8,.9,1.],np.r_[0.,np.quantile(source_o,[1/3,2/3]),1.].tolist(),np.r_[0.,np.quantile(source_o,[.2,.4,.6,.8]),1.].tolist()]
            uniform=rng.random(200);bank_scores=[];cut_records=[]
            for k,kind in enumerate(['shap','flip']):
                a=np.array([scores(r['records'][i]['control_tables'][0],calibration[j:j+1],kind)[0][0] for j,i in enumerate(ids)])
                bank_scores.append(a.tolist());current=[cut_for(a,calibration,edge,uniform,kind=='flip') for edge in edges];cut_records.append(current)
                for i in ids:
                    row=r['records'][i];x,_,v=scores(row['reference_table'],test[i],kind)
                    for m,edge in enumerate(edges):null[i,k,m],abstain[i,k,m]=apply(x,test[i],v,edge,current[m],ties[i])
                    for law in range(5):
                        _,l,v=scores(row['candidate_table'],shifted[i,law],kind)
                        for m,edge in enumerate(edges):local[i,k,m,law]=apply(l,shifted[i,law],v,edge,current[m],ties[i])[0]
            banks.append({'n':n,'family':family,'case_indices':ids,'edges':edges,'scores':bank_scores,'cuts':cut_records,'tie_uniforms':uniform.tolist()})
            boot=rng.integers(0,200,(1000,200));loss=f1[ids,0].mean(axis=1)-f1[ids,3].mean(axis=1);harm=loss>=.05
            def estimate(a):return {'mean':float(a.mean()),'ci95':np.quantile(a[boot].mean(axis=1),[.025,.975]).tolist()}
            for k,kind in enumerate(['shap','flip']):
                for m,method in enumerate(METHODS):
                    item={'n':n,'family':family,'score':kind,'method':method,'harm_cases':int(harm.sum()),
                      'null_warning':estimate(null[ids,k,m].mean(axis=1)),'abstention':estimate(abstain[ids,k,m].mean(axis=1)),'laws':[]}
                    for law,e in enumerate(LAWS):
                        rates=local[ids,k,m,law].mean(axis=1)
                        item['laws'].append({'error':e,'all_localization':estimate(rates),'harm_localization':float(rates[harm].mean()) if harm.any() else None,
                                              'no_harm_localization':float(rates[~harm].mean()) if (~harm).any() else None})
                    summary.append(item)
            print('composite',n,family,flush=True)
    np.savez_compressed(OUT/'composite_reference_draws.npz',null_warning=null,localization=local,abstention=abstain)
    paths=[Path(__file__),ROOT/'experiments/REVIEW3_COMPOSITE_PROTOCOL.md',ROOT/'src/pdm_audit/reference_transfer.py',OUT/'reference_transfer.json',OUT/'reference_transfer_draws.npz',OUT/'observed_overlap_draws.npz']
    out={'complete':True,'banks':banks,'summary':summary,'calibration_models_per_stratum':200,
         'PAC_95_warning_upper_M200':float(beta.ppf(.95,10,191)),'seconds':time.perf_counter()-start,
         'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'composite_reference.json').write_text(json.dumps(out,indent=1,allow_nan=False))
    print('COMPLETE composite',round(out['seconds'],1),flush=True)


if __name__=='__main__':main()
