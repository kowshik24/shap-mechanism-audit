"""Verify matched-size, observed-overlap, proxy-strength, and spectral results."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from pdm_audit.exact_oracle import conformal_threshold, exact_shap, statistics
from pdm_audit.learned_reference import functional_gate
from pdm_audit.conditional_audit import (ALPHAS,METHODS,batch_scores,reference_thresholds,
    thresholds_on,residual,warning,table_f1)

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def load(name):
    r=json.loads((OUT/name).read_text());assert r['complete']
    for p,digest in r['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    return r


def f1(rows):
    scores=[]
    for c in ['B','IR','OR']:
        tp=sum(x['truth']==c and x['prediction']==c for x in rows)
        scores.append(2*tp/(sum(x['truth']==c for x in rows)+sum(x['prediction']==c for x in rows)))
    return float(np.mean(scores))


def main():
    conditional=load('observed_overlap.json');power=load('proxy_power.json');spectral=load('cwru_spectral_transfer.json')
    learned=json.loads((OUT/'learned_calibration.json').read_text())
    lookup={(r['n'],r['case_seed'],r['family']):r for r in learned['records'] if r['role']=='evaluation'}
    d=np.load(OUT/'observed_overlap_draws.npz');p=np.load(OUT/'proxy_power_draws.npz')
    assert len(conditional['records'])==len(power['records'])==1200
    assert d['source_counts'].shape==d['target_counts'].shape==(1200,1000,16)
    assert d['tune_counts'].shape==(1200,400,16) and d['test_counts'].shape==(1200,200,16)
    assert d['strength_counts'].shape==(1200,7,200,16)
    assert not conditional['algorithm_uses_latent_overlap']
    flags={k:d[k] for k in ['raw_warning','raw_localization','matched_warning','matched_localization','matched_abstain']}
    assert np.array_equal(flags['matched_warning'][:,0],flags['matched_warning'][:,1])
    assert np.array_equal(flags['matched_localization'][:,0],flags['matched_localization'][:,1])
    source_counts,target_counts,tune_counts,test_counts=(d[k] for k in ['source_counts','target_counts','tune_counts','test_counts'])
    thresholds=d['matched_thresholds'];strength=d['strength_counts'];permuted=p['permuted_counts'];pf=p['localization_flags']
    for i,row in enumerate(conditional['records']):
        old=lookup[row['n'],row['case_seed'],row['family']];f=old['views']['mechanism_only']['probability_table']
        assert all(np.all(c[i].sum(axis=1)==row['n']) for c in [source_counts,target_counts,tune_counts,test_counts])
        sa,_,_=batch_scores(f,source_counts[i]);ta,_,_=batch_scores(f,target_counts[i])
        qa,_,qv=batch_scores(f,tune_counts[i]);xa,_,xv=batch_scores(f,test_counts[i])
        _,local,gv=batch_scores(old['views']['unrestricted']['probability_table'],test_counts[i])
        refs=[reference_thresholds(sa,source_counts[i]),reference_thresholds(ta,target_counts[i])]
        for m,method in enumerate(METHODS):
            bank=0 if method.endswith('source') else 1;observed=method.startswith('observed')
            qx=thresholds_on(refs[bank],test_counts[i],observed);qt=thresholds_on(refs[bank],tune_counts[i],observed)
            for a,alpha in enumerate(ALPHAS):
                ref=reference_thresholds(sa if bank==0 else ta,source_counts[i] if bank==0 else target_counts[i],alpha)
                qraw=thresholds_on(ref,test_counts[i],observed)
                assert np.array_equal(warning(xa,qraw,xv),flags['raw_warning'][i,m,a])
                assert np.array_equal(warning(local,qraw,gv),flags['raw_localization'][i,m,a])
                if not observed:
                    q=conformal_threshold(qa,alpha);mw=warning(xa,np.full(200,q),xv);ml=warning(local,np.full(200,q),gv)
                    assert np.all(thresholds[i,m,a]==q)
                else:
                    delta=conformal_threshold(residual(qa,qt,qv),alpha)
                    assert float(row['matched_offsets'][m][a])==delta
                    mw=np.isfinite(qx)&xv&(residual(xa,qx,xv)>delta)
                    ml=np.isfinite(qx)&gv&(residual(local,qx,gv)>delta)
                assert np.array_equal(mw,flags['matched_warning'][i,m,a])
                assert np.array_equal(ml,flags['matched_localization'][i,m,a])
        if i%200==0:
            counts=test_counts[i,0];scalar=statistics(exact_shap(f,counts/counts.sum()),counts)
            assert np.isclose(xa[0],max(v['S'] for v in scalar.values()))
        pr=power['records'][i];assert (pr['n'],pr['case_seed'],pr['family'])==(row['n'],row['case_seed'],row['family'])
        assert np.array_equal(pr['tables'][1],old['views']['unrestricted']['probability_table'])
        for e,values in enumerate(pr['tables']):
            assert functional_gate(values)==pr['gates'][e]
            assert np.array_equal(strength[i,e].reshape(200,8,2).sum(axis=2),test_counts[i].reshape(200,8,2).sum(axis=2))
            assert np.array_equal(permuted[i,e].reshape(200,8,2).sum(axis=2),strength[i,e].reshape(200,8,2).sum(axis=2))
            assert np.array_equal(permuted[i,e,:,1::2].sum(axis=1),strength[i,e,:,1::2].sum(axis=1))
            _,ls,valid=batch_scores(values,strength[i,e])
            assert np.isclose(pr['mean_F1'][e],table_f1(values,strength[i,e]).mean())
            assert np.isclose(pr['mean_context_permuted_F1'][e],table_f1(values,permuted[i,e]).mean())
            for m,method in enumerate(METHODS):
                if m<2:found=warning(ls,thresholds[i,m,2],valid)
                else:
                    q=thresholds_on(refs[0 if method.endswith('source') else 1],strength[i,e],True)
                    found=np.isfinite(q)&valid&(residual(ls,q,valid)>float(row['matched_offsets'][m][2]))
                assert np.array_equal(found,pf[i,e,m])
        if i%200==0:print('verified functions',i+1,'/1200',flush=True)
    assert np.array_equal(pf[:,1],flags['matched_localization'][:,:,2])
    bootstrap = np.random.default_rng(2026102436)
    for s in conditional['summary']:
        ids=[r['array_index'] for r in conditional['records'] if r['n']==s['n'] and r['family']==s['family']]
        material=[i for i in ids if conditional['records'][i]['material_context']]
        m=METHODS.index(s['method']);a=ALPHAS.index(s['alpha'])
        for key,flag in [('raw_warning_rate','raw_warning'),('raw_localization_rate','raw_localization'),
                         ('matched_warning_rate','matched_warning'),('matched_localization_rate','matched_localization'),
                         ('matched_abstention_rate','matched_abstain')]:
            cases=material if 'localization' in key else ids
            values=flags[flag][cases,m,a].mean(axis=1)
            assert np.isclose(s[key]['mean'],values.mean()) and s[key]['case_n']==len(cases)
            indices=bootstrap.integers(0,len(values),size=(1000,len(values)))
            assert np.allclose(s[key]['case_bootstrap_95'],np.quantile(values[indices].mean(axis=1),[.025,.975]))
    bootstrap=np.random.default_rng(2026102437)
    errors=[0.,.01,.05,.10,.20,.30,.50]
    for row in power['summary']:
        ids=[i for i,r in enumerate(power['records']) if r['n']==row['n'] and r['family']==row['family']]
        e=errors.index(row['error_probability']);m=METHODS.index(row['method'])
        material=[i for i in ids if power['records'][i]['gates'][e]['material_context_dependence']]
        for key,cases in [('all_cases',ids),('material_cases',material)]:
            if not cases:
                assert row[key] is None
                continue
            values=pf[cases,e,m].mean(axis=1)
            indices=bootstrap.integers(0,len(values),size=(1000,len(values)))
            assert np.isclose(row[key]['mean'],values.mean()) and row[key]['case_n']==len(cases)
            assert np.allclose(row[key]['case_bootstrap_95'],np.quantile(values[indices].mean(axis=1),[.025,.975]))
        for key,record_key in [('mean_F1','mean_F1'),('mean_context_permuted_F1','mean_context_permuted_F1')]:
            assert np.isclose(row[key],np.mean([power['records'][i][record_key][e] for i in ids]))
    for t in conditional['paired_primary_tests']:
        ids=[r['array_index'] for r in conditional['records'] if r['n']==t['n'] and r['family']==t['family'] and r['material_context']]
        m=METHODS.index(t['comparison']);a=flags['matched_localization'][ids,0,2,0];b=flags['matched_localization'][ids,m,2,0]
        n01,n10=int((~a&b).sum()),int((a&~b).sum())
        assert (n01,n10)==(t['conditional_only'],t['global_only'])
        assert t['p']==(float(binomtest(n01,n01+n10,.5).pvalue) if n01+n10 else 1.)
    order=np.argsort([t['p'] for t in conditional['paired_primary_tests']]);running=0.
    for position,index in enumerate(order):
        t=conditional['paired_primary_tests'][index];running=max(running,min(1.,(12-position)*t['p']))
        assert t['holm_p']==running
    for r in spectral['results']:
        for key in ['matched','filter_removed']:
            assert len(r[key]['recordings'])==36 and np.isclose(f1(r[key]['recordings']),r[key]['recording_macro_f1'])
        assert all(not set(f['training_recordings'])&set(f['test_recordings']) for f in r['folds'])
    inventory=json.loads((ROOT/'data/cwru/manifest.json').read_text())
    # Manifest structure is intentionally read independently of the producing script.
    from pdm_audit.cwru import load_features
    _, inventory=load_features(ROOT/'data/cwru')
    groups={r['recording']:(r['fault'],r['diameter_in']) for r in inventory}
    for result in spectral['results']:
        a=result['matched']['recordings'];b=result['filter_removed']['recordings']
        labels=['B','IR','OR']; matrices=[]
        for location in labels:
            group_matrices=[]
            for group in sorted({groups[r['id']] for r in a if r['truth']==location}):
                pair=np.zeros((2,3,3),int)
                for i,r in enumerate(a):
                    if groups[r['id']]==group:
                        pair[0,labels.index(r['truth']),labels.index(r['prediction'])]+=1
                        pair[1,labels.index(b[i]['truth']),labels.index(b[i]['prediction'])]+=1
                group_matrices.append(pair)
            matrices.append(np.array(group_matrices))
        rng=np.random.default_rng(20261024381);drops=[]
        for _ in range(10000):
            matrix=sum(group[rng.integers(0,3,size=3)].sum(axis=0) for group in matrices)
            scores=[]
            for confusion in matrix:
                scores.append(np.mean(2*np.diag(confusion)/(confusion.sum(axis=0)+confusion.sum(axis=1))))
            drops.append(scores[0]-scores[1])
        assert np.allclose(np.quantile(drops,[.025,.975]),result['F1_drop_cluster_bootstrap_95'])
    assert spectral['primary_axis']=='diameter_in' and spectral['primary_bootstrap_clusters']==9
    assert all(x['max_log_RMS_roundoff']<1e-12 and x['minimum_feature_change_L2']>0 for x in spectral['input_checks'])
    assert len(spectral['single_anchor_masks'])==84
    print('PASS: all observed-bin thresholds, independent target tuning cuts, untouched tests, and raw score checks')
    print('PASS: 8,400 proxy alternatives, coupled strengths, exact base reproduction, F1 and context localization')
    print('PASS: paired case-level tests and 12-comparison Holm family; global matched rules identical')
    print('PASS: spectral transfer predictions, nine bootstrap clusters, RMS design checks, and 84 partial masks')


if __name__=='__main__':main()
