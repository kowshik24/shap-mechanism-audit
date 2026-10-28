"""Independent aligned fits, transfer harm, non-SHAP controls, and bin sensitivity."""
import pdm_audit
import hashlib,json,time
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed
from scipy.stats import binomtest,beta
from pdm_audit.reference_transfer import METHODS,LAWS,scores,cuts,apply,shifted_counts,eta_squared
from pdm_audit.conditional_audit import table_f1,observed_overlap
from pdm_audit.learned_reference import FAMILIES,functional_gate
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def case(row,source,base,tune,test,f,g,controls):
    rng=np.random.default_rng([20261025,43,row['n'],row['case_seed'],FAMILIES.index(row['family'])])
    shifts=shifted_counts(test,rng);tie_cal=rng.random(400);ties=rng.random(200)
    targeted_cut=float(np.sort(tie_cal)[380]);direct_null=ties>targeted_cut
    warnings=np.zeros((2,4,5,200),bool);local=np.zeros_like(warnings)
    control_warnings=np.zeros((2,4,4,200),bool);control_local=np.zeros_like(control_warnings)
    null_warning=np.zeros((2,4,200),bool);cut_records=[];variance=[]
    for k,kind in enumerate(['shap','flip']):
        ba,_,_=scores(f,base,kind);ta,_,_=scores(f,tune,kind);xa,_,valid=scores(f,test,kind)
        references=cuts(ba,base,ta,tune,source,tie_cal,randomized=(kind=='flip'));cut_records.append(references)
        variance.append([eta_squared(ba,base,c['edges']) for c in references])
        for m,c in enumerate(references):null_warning[k,m]=apply(xa,test,valid,c,ties)
        for law,counts in enumerate(shifts):
            a,l,v=scores(g,counts,kind)
            for m,c in enumerate(references):
                warnings[k,m,law]=apply(a,counts,v,c,ties);local[k,m,law]=apply(l,counts,v,c,ties)
        for j,values in enumerate(controls):
            a,l,v=scores(values,test,kind)
            for m,c in enumerate(references):
                control_warnings[k,m,j]=apply(a,test,v,c,ties);control_local[k,m,j]=apply(l,test,v,c,ties)
    direct=[];f1=[];proxy_f1=[]
    for counts in shifts:
        response=abs(np.asarray(g)-np.asarray(g)[np.arange(16)^1]);rho=counts@response/counts.sum(axis=1)
        direct.append((rho>0)|((rho==0)&direct_null))
        f1.append(table_f1(g,counts));proxy_f1.append(table_f1(controls[-1],counts))
    o=observed_overlap(base);ba=scores(f,base)[0]
    corr=float(np.corrcoef(o,ba)[0,1]**2) if np.std(ba)>0 else 0.
    return {'record':{'n':row['n'],'case_seed':row['case_seed'],'family':row['family'],'array_index':row['array_index'],
                     'cuts':cut_records,'variance_explained':variance,'overlap_score_r2':corr,
                     'control_gates':[functional_gate(x) for x in controls],
                     'reference_table':f,'candidate_table':g,'control_tables':controls},
            'shift_counts':shifts,'warning':warnings,'localization':local,'control_warning':control_warnings,
            'control_localization':control_local,'null_warning':null_warning,'targeted':np.array(direct),
            'targeted_null':direct_null,'f1':np.array(f1),'proxy_f1':np.array(proxy_f1),
            'tie_cal':tie_cal,'ties':ties}


def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,float) and not np.isfinite(x):return 'Infinity' if x>0 else '-Infinity'
    return x


def ci(values,indices):
    a=np.asarray(values,float)
    if not len(a):return None
    return {'mean':float(a.mean()),'cases':len(a),'ci95':np.quantile(a[indices].mean(axis=1),[.025,.975]).tolist()}


def main():
    started=time.perf_counter();old=json.loads((OUT/'observed_overlap.json').read_text())
    learned=json.loads((OUT/'learned_calibration.json').read_text());lookup={(r['n'],r['case_seed'],r['family'],r['role']):r for r in learned['records']}
    archive=np.load(OUT/'observed_overlap_draws.npz');banks=[archive[k] for k in ['source_counts','target_counts','tune_counts','test_counts']]
    arguments=[]
    for r in old['records']:
        a=lookup[r['n'],r['case_seed'],r['family'],'evaluation'];seed=r['case_seed']-100000
        families=[r['family']]+[f for f in FAMILIES if f!=r['family']]
        controls=[lookup[r['n'],seed,f,'calibration']['views']['mechanism_only']['probability_table'] for f in families]
        assert all(functional_gate(v)['eligible'] for v in controls)
        controls.append(a['views']['proxy_only']['probability_table']);i=r['array_index']
        arguments.append((r,*[x[i] for x in banks],a['views']['mechanism_only']['probability_table'],a['views']['unrestricted']['probability_table'],controls))
    records=[];arrays={k:[] for k in ['shift_counts','warning','localization','control_warning','control_localization','null_warning','targeted','targeted_null','f1','proxy_f1','tie_cal','ties']}
    stream=Parallel(n_jobs=4,return_as='generator',batch_size=4)(delayed(case)(*a) for a in arguments)
    for i,result in enumerate(stream):
        records.append(result.pop('record'))
        for key,value in result.items():arrays[key].append(value)
        if (i+1)%100==0:print('reference transfer',i+1,'/1200',round(time.perf_counter()-started,1),flush=True)
    arrays={k:np.asarray(v,dtype=np.int16 if k=='shift_counts' else None) for k,v in arrays.items()}
    summaries=[];paired=[];rng=np.random.default_rng(2026102543)
    for n in [1000,10000]:
        for family in FAMILIES:
            ids=[i for i,r in enumerate(records) if r['n']==n and r['family']==family];boot=rng.integers(0,200,(1000,200))
            f1=arrays['f1'][ids];harm=f1[:,0].mean(axis=1)-f1[:,3].mean(axis=1)>=.05
            harmful=np.flatnonzero(harm);harm_boot=rng.integers(0,len(harmful),(1000,len(harmful))) if len(harmful) else None
            for k,kind in enumerate(['shap','flip']):
                for m,method in enumerate(METHODS):
                    item={'n':n,'family':family,'score':kind,'method':method,'harm_cases':len(harmful),
                          'self_warning':ci(arrays['null_warning'][ids,k,m].mean(axis=1),boot),
                          'aligned_warning':[ci(arrays['control_warning'][ids,k,m,j].mean(axis=1),boot) for j in range(3)],
                          'proxy_warning':ci(arrays['control_warning'][ids,k,m,3].mean(axis=1),boot),
                          'eta2_mean':float(np.mean([records[i]['variance_explained'][k][m] for i in ids])),'laws':[]}
                    for law,e in enumerate(LAWS):
                        rate=arrays['localization'][ids,k,m,law].mean(axis=1)
                        item['laws'].append({'error':e,'all_localization':ci(rate,boot),
                          'harm_localization':ci(rate[harm],harm_boot) if len(harmful) else None,
                          'mean_F1':float(f1[:,law].mean()),'mean_F1_loss':float((f1[:,0]-f1[:,law]).mean())})
                    summaries.append(item)
                delta=arrays['localization'][ids,k,1,0].mean(axis=1)-arrays['localization'][ids,k,0,0].mean(axis=1)
                plus,minus=int((delta>0).sum()),int((delta<0).sum())
                paired.append({'n':n,'family':family,'score':kind,'positive':plus,'negative':minus,
                  'p':float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.,'mean_difference':ci(delta,boot)})
            for law,e in enumerate(LAWS):
                rate=arrays['targeted'][ids,law].mean(axis=1)
                summaries.append({'n':n,'family':family,'score':'targeted_C','error':e,'harm_cases':len(harmful),
                    'self_warning':ci(arrays['targeted_null'][ids].mean(axis=1),boot),'all_localization':ci(rate,boot),
                    'harm_localization':ci(rate[harm],harm_boot) if len(harmful) else None,
                    'mean_F1':float(f1[:,law].mean()),'mean_F1_loss':float((f1[:,0]-f1[:,law]).mean())})
    running=0.
    for rank,index in enumerate(np.argsort([r['p'] for r in paired])):
        running=max(running,min(1.,(12-rank)*paired[index]['p']));paired[index]['holm_p']=running
    path=OUT/'reference_transfer_draws.npz';np.savez_compressed(path,**arrays)
    paths=[Path(__file__),ROOT/'src/pdm_audit/reference_transfer.py',ROOT/'experiments/REVIEW3_PROTOCOL.md',OUT/'observed_overlap.json',OUT/'observed_overlap_draws.npz',OUT/'learned_calibration.json']
    result={'complete':True,'records':records,'summary':summaries,'paired_case_tests':paired,'methods':METHODS,'laws':LAWS,
            'harm_definition':'mean paired F1 loss >= .05 under independent context, error=.50',
            'PAC_95_warning_upper_M400':float(beta.ppf(.95,20,381)),'seconds':time.perf_counter()-started,
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'reference_transfer.json').write_text(json.dumps(safe(result),indent=1,allow_nan=False))
    print('COMPLETE reference transfer',round(result['seconds'],1),flush=True)


if __name__=='__main__':main()
