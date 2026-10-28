"""Reconstruct population ROC comparisons without the production rank helpers."""
import pdm_audit
import hashlib,json
from pathlib import Path
import numpy as np
from pdm_audit.reference_transfer import scores,bins

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def independent_ranks(cal,cb,u,x,xb,ties,valid,randomized,groups):
    p=np.full(x.shape,2.)
    for b in range(groups):
        selected=cb==b;use=(xb==b)&valid&np.isfinite(x)
        if selected.sum()<19:continue
        reference=cal[selected];query=x[use]
        if randomized:
            z=np.empty(reference.shape,complex);z.real=reference;z.imag=u[selected]
            w=np.empty(query.shape,complex);w.real=query;w.imag=ties[use]
            count=np.searchsorted(np.sort(z),w,side='left')
        else:count=np.searchsorted(np.sort(reference),query,side='left')
        p[use]=(1+len(reference)-count)/(len(reference)+1)
    return p


def reconstruct(a,prefix,case_ids,bank_ids,rates):
    get=lambda name:a[prefix+'_'+name]
    out=np.empty((2,2,2,len(rates)));nominal=np.empty((2,2,3));boundaries=np.empty((2,2,len(rates),2))
    harm=get('harm')[case_ids]
    for k in range(2):
        for m in range(2):
            args=(get('cal')[k,bank_ids],get('cal_bins')[m,bank_ids],get('cal_ties')[bank_ids])
            n=independent_ranks(*args,get('null_score')[k,case_ids],get('null_bins')[m,case_ids],
                                get('ties')[case_ids],get('null_valid')[k,case_ids],k==1,[1,3][m])
            p=[]
            for law in range(2):
                l=independent_ranks(*args,get('localization_score')[k,law,case_ids],get('null_bins')[m,case_ids],
                                    get('ties')[case_ids],get('localization_valid')[k,law,case_ids],k==1,[1,3][m])
                p.append(l[harm])
            nominal[k,m]=[np.mean(n<=.05),*[np.mean(x<=.05) for x in p]]
            values,counts=np.unique(n,return_counts=True);cumulative=counts.cumsum()
            for j,rate in enumerate(rates):
                index=np.flatnonzero(cumulative>=rate*n.size)[0];q=values[index]
                lower=0 if index==0 else cumulative[index-1]
                weight=(rate*n.size-lower)/counts[index];assert q<=1
                assert np.isclose((lower+weight*counts[index])/n.size,rate)
                boundaries[k,m,j]=q,weight
                for law,x in enumerate(p):out[k,m,law,j]=np.mean(x<q)+weight*np.mean(x==q)
    return out,nominal,boundaries


def main():
    r=json.loads((OUT/'matched_population.json').read_text());assert r['complete']
    for path,expected in r['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    a=dict(np.load(OUT/'matched_population_draws.npz'));saved=np.load(OUT/'observed_overlap_draws.npz')
    old={key:saved[key] for key in ['target_counts','test_counts']}
    transfer=json.loads((OUT/'reference_transfer.json').read_text());raw=dict(np.load(OUT/'reference_transfer_draws.npz'))
    composite=json.loads((OUT/'composite_reference.json').read_text());rates=np.asarray(r['rates'])
    rng=np.random.default_rng(2026102548);identity=np.arange(200)
    for n in [1000,10000]:
        for family in ['tree','forest','logistic']:
            prefix=f'{family}_{n}';get=lambda name:a[prefix+'_'+name]
            bank=next(b for b in composite['banks'] if b['n']==n and b['family']==family);ids=bank['case_indices']
            assert np.array_equal(ids,get('case_indices'))
            assert np.array_equal(get('cal'),bank['scores']) and np.array_equal(get('cal_ties'),bank['tie_uniforms'])
            loss=raw['f1'][ids,0].mean(axis=1)-raw['f1'][ids,3].mean(axis=1)
            assert np.array_equal(get('harm'),loss>=.05)
            calibration=np.asarray([old['target_counts'][i,0] for i in ids]);test=old['test_counts'][ids]
            for m,e in enumerate(bank['edges'][:2]):
                assert np.array_equal(get('cal_bins')[m],bins(calibration,e))
                assert np.array_equal(get('null_bins')[m],bins(test.reshape(-1,16),e).reshape(200,200))
            assert np.array_equal(get('ties'),raw['ties'][ids])
            for j,i in enumerate(ids):
                row=transfer['records'][i]
                for k,kind in enumerate(['shap','flip']):
                    cal=scores(row['control_tables'][0],calibration[j:j+1],kind)[0][0]
                    assert np.isclose(cal,get('cal')[k,j])
                    score,_,valid=scores(row['reference_table'],test[j],kind)
                    assert np.allclose(score,get('null_score')[k,j]) and np.array_equal(valid,get('null_valid')[k,j])
                    for law,index in enumerate([0,3]):
                        _,local,valid=scores(row['candidate_table'],raw['shift_counts'][i,index],kind)
                        assert np.allclose(local,get('localization_score')[k,law,j])
                        assert np.array_equal(valid,get('localization_valid')[k,law,j])
            for name,value in zip(['point','nominal','boundary'],reconstruct(a,prefix,identity,identity,rates)):
                assert np.allclose(get(name),value),name
            bank_draws=rng.integers(0,200,(r['replicates'],200));case_draws=rng.integers(0,200,(r['replicates'],200))
            assert np.array_equal(bank_draws,get('bank_draws')) and np.array_equal(case_draws,get('case_draws'))
            for b in range(r['replicates']):
                values=reconstruct(a,prefix,case_draws[b],bank_draws[b],rates)
                for name,value in zip(['bootstrap','nominal_bootstrap','bootstrap_boundaries'],values):
                    assert np.allclose(get(name)[b],value),(prefix,b,name)
            for m,method in enumerate(['global','fixed3']):
                item=next(x for x in r['summary'] if x['n']==n and x['family']==family and x['method']==method)
                assert item['harm_cases']==get('harm').sum()
                for k,record in enumerate(item['scores']):
                    assert record['score']==['shap','flip'][k]
                    assert np.allclose(record['nominal'],get('nominal')[k,m])
                    assert np.allclose(record['nominal_ci95'],np.quantile(get('nominal_bootstrap')[:,k,m],[.025,.975],axis=0).T)
                    assert np.allclose(record['matched_localization'],get('point')[k,m])
                    assert np.allclose(record['matched_ci95'],np.moveaxis(np.quantile(get('bootstrap')[:,k,m],[.025,.975],axis=0),0,-1))
                assert np.allclose(item['flip_minus_shap'],get('point')[1,m]-get('point')[0,m])
                assert np.allclose(item['difference_ci95'],np.moveaxis(np.quantile(get('bootstrap')[:,1,m]-get('bootstrap')[:,0,m],[.025,.975],axis=0),0,-1))
            print('verified matched population',n,family,flush=True)
    paths=[OUT/'matched_population.json',OUT/'matched_population_draws.npz',Path(__file__)]
    manifest={'verified':True,'replicates':r['replicates'],'strata':6,
        'scope':'All raw scores, fit roles, paired bank/case resamples, matched boundaries, curves and intervals',
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'matched_population_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS matched acceptable-population curves, prospective nominal rates, and joint bootstrap reconstruction')


if __name__=='__main__':main()
