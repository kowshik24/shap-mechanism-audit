"""Matched empirical acceptable-model size with paired bank/case bootstrap."""
import pdm_audit
import hashlib,json,time
from pathlib import Path
import numpy as np
from pdm_audit.reference_transfer import scores,bins
from pdm_audit.population_roc import population_pvalues,matched_boundary,boundary_rate

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'
RATES=np.array([.01,.025,.05,.10,.20]);REPLICATES=400


def evaluate(cal,cb,ct,na,nv,nb,lo,lv,ties,case_ids,bank_ids,harm,groups):
    """Evaluate paired scores/partitions for one resampling of complete cases."""
    selected=harm[case_ids]; result=np.empty((2,2,2,len(RATES)))
    nominal=np.empty((2,2,3));boundaries=np.empty((2,2,len(RATES),2))
    for k in range(2):
        for m in range(2):
            n=population_pvalues(cal[k,bank_ids],cb[m,bank_ids],ct[bank_ids],
                na[k,case_ids],nb[m,case_ids],ties[case_ids],nv[k,case_ids],k==1,groups[m])
            powers=[]
            for law in range(2):
                p=population_pvalues(cal[k,bank_ids],cb[m,bank_ids],ct[bank_ids],
                    lo[k,law,case_ids],nb[m,case_ids],ties[case_ids],lv[k,law,case_ids],k==1,groups[m])
                powers.append(p[selected])
            nominal[k,m]=[np.mean(n<=.05),*[np.mean(p<=.05) for p in powers]]
            for j,rate in enumerate(RATES):
                cut=matched_boundary(n,rate);boundaries[k,m,j]=cut
                assert np.isclose(boundary_rate(n,cut),rate,atol=1e-13)
                for law,p in enumerate(powers):result[k,m,law,j]=boundary_rate(p,cut)
    return result,nominal,boundaries


def main():
    start=time.perf_counter();r=json.loads((OUT/'reference_transfer.json').read_text())
    c=json.loads((OUT/'composite_reference.json').read_text())
    old=np.load(OUT/'observed_overlap_draws.npz');raw=np.load(OUT/'reference_transfer_draws.npz')
    base=old['target_counts'];test=old['test_counts'];shift=raw['shift_counts'];all_ties=raw['ties'];f1=raw['f1']
    arrays={};summary=[];rng=np.random.default_rng(2026102548)
    def store(prefix,**kwargs):
        for key,value in kwargs.items():arrays[prefix+'_'+key]=value
    for n in [1000,10000]:
        for family in ['tree','forest','logistic']:
            prefix=f'{family}_{n}';bank=next(b for b in c['banks'] if b['n']==n and b['family']==family)
            ids=np.asarray(bank['case_indices']);cal=np.asarray(bank['scores']);ct=np.asarray(bank['tie_uniforms'])
            calibration=np.asarray([base[i,0] for i in ids]);eval_counts=test[ids]
            cb=np.asarray([bins(calibration,e) for e in bank['edges'][:2]])
            nb=np.asarray([bins(eval_counts.reshape(-1,16),e).reshape(200,200) for e in bank['edges'][:2]])
            na=np.empty((2,200,200));nv=np.zeros_like(na,dtype=bool)
            lo=np.empty((2,2,200,200));lv=np.zeros_like(lo,dtype=bool)
            for k,kind in enumerate(['shap','flip']):
                for j,i in enumerate(ids):
                    row=r['records'][i];na[k,j],_,nv[k,j]=scores(row['reference_table'],eval_counts[j],kind)
                    for law,index in enumerate([0,3]):
                        _,lo[k,law,j],lv[k,law,j]=scores(row['candidate_table'],shift[i,index],kind)
            harm=f1[ids,0].mean(axis=1)-f1[ids,3].mean(axis=1)>=.05
            ties=all_ties[ids];identity=np.arange(200)
            point,nominal,boundary=evaluate(cal,cb,ct,na,nv,nb,lo,lv,ties,identity,identity,harm,[1,3])
            bank_draws=rng.integers(0,200,(REPLICATES,200));case_draws=rng.integers(0,200,(REPLICATES,200))
            boot=np.empty((REPLICATES,*point.shape));nominal_boot=np.empty((REPLICATES,*nominal.shape))
            boot_boundaries=np.empty((REPLICATES,*boundary.shape))
            for b in range(REPLICATES):
                boot[b],nominal_boot[b],boot_boundaries[b]=evaluate(cal,cb,ct,na,nv,nb,lo,lv,ties,case_draws[b],bank_draws[b],harm,[1,3])
            store(prefix,cal=cal,cal_bins=cb,cal_ties=ct,null_score=na,null_valid=nv,null_bins=nb,
                  localization_score=lo,localization_valid=lv,ties=ties,harm=harm,case_indices=ids,
                  bank_draws=bank_draws,case_draws=case_draws,point=point,nominal=nominal,boundary=boundary,
                  bootstrap=boot,nominal_bootstrap=nominal_boot,bootstrap_boundaries=boot_boundaries)
            for m,method in enumerate(['global','fixed3']):
                item={'n':n,'family':family,'method':method,'harm_cases':int(harm.sum()),'scores':[]}
                for k,kind in enumerate(['shap','flip']):
                    item['scores'].append({'score':kind,'nominal':nominal[k,m].tolist(),
                        'nominal_ci95':np.quantile(nominal_boot[:,k,m],[.025,.975],axis=0).T.tolist(),
                        'matched_localization':point[k,m].tolist(),
                        'matched_ci95':np.moveaxis(np.quantile(boot[:,k,m],[.025,.975],axis=0),0,-1).tolist()})
                diff=point[1,m]-point[0,m];diff_boot=boot[:,1,m]-boot[:,0,m]
                item['flip_minus_shap']=diff.tolist()
                item['difference_ci95']=np.moveaxis(np.quantile(diff_boot,[.025,.975],axis=0),0,-1).tolist()
                summary.append(item)
            print('matched population',n,family,round(time.perf_counter()-start,1),flush=True)
    path=OUT/'matched_population_draws.npz';np.savez_compressed(path,**arrays)
    paths=[Path(__file__),ROOT/'src/pdm_audit/population_roc.py',ROOT/'experiments/MATCHED_POPULATION_PROTOCOL.md',
           OUT/'reference_transfer.json',OUT/'reference_transfer_draws.npz',OUT/'composite_reference.json',
           OUT/'observed_overlap_draws.npz',path]
    result={'complete':True,'rates':RATES.tolist(),'replicates':REPLICATES,'summary':summary,
        'scope':'Retrospective matched empirical acceptable-model size; paired fit-bank/case bootstrap, not a deployment guarantee',
        'seconds':time.perf_counter()-start,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'matched_population.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('COMPLETE matched population',round(result['seconds'],1),flush=True)


if __name__=='__main__':main()
