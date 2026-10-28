"""Independent arithmetic, pairing, role, and provenance checks for follow-up studies."""
import pdm_audit
import hashlib,itertools,json,time
from fractions import Fraction as F
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import binomtest,beta
from pdm_audit.exact_oracle import STATES,ALIGNED,exact_shap,conformal_threshold
from pdm_audit.conditional_audit import batch_scores,observed_overlap
from pdm_audit.learned_reference import functional_gate
from pdm_audit.reference_transfer import scores
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def load(name):
    r=json.loads((OUT/name).read_text());assert r['complete']
    for path,value in r['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    return r


def decision(a,c,v,cut,u):
    edges=np.array(cut['edges']);o=observed_overlap(c);b=np.searchsorted(edges[1:-1],o,side='right')
    q=np.array([float(x) for x in cut['q']]);threshold=q[b]+float(cut['delta'])
    usable=v&np.isfinite(q[b])&np.isfinite(o)
    # Compare the implemented residual, rather than adding/subtracting a rounded cut.
    residual=a-q[b]
    return usable&((residual>float(cut['delta']))|((residual==float(cut['delta']))&(u>cut['tie_cut'])))


def f1(values,counts):
    p=np.asarray(values)>=.5;y=ALIGNED.astype(bool)
    tp=counts[:,p&y].sum(axis=1);denominator=counts[:,p].sum(axis=1)+counts[:,y].sum(axis=1)
    return 2*tp/denominator


def algebra():
    states=list(itertools.product([0,1],repeat=3))
    for p,q in [(F(1,20),F(1,20)),(F(1,5),F(2,5)),(F(2,5),F(1,5))]:
        mass=[(p if x[0] else 1-p)*(q if x[2] else 1-q) if x[0]==x[1] else F(0) for x in states]
        phi=[]
        for x in states:
            row=[]
            for j in range(3):
                total=F(0)
                for order in itertools.permutations(range(3)):
                    chosen=set(order[:order.index(j)])
                    before=sum(w*max([x[k] if k in chosen else z[k] for k in range(3)]) for w,z in zip(mass,states))
                    chosen.add(j)
                    after=sum(w*max([x[k] if k in chosen else z[k] for k in range(3)]) for w,z in zip(mass,states))
                    total+=(after-before)/6
                row.append(total)
            phi.append(row)
        own=[i for i,x in enumerate(states) if x[0]==1];rest=[i for i,x in enumerate(states) if x[0]==0 and x[2]==1]
        means=[]
        for ids in [own,rest]:means.append([sum(mass[i]*abs(phi[i][j]) for i in ids)/sum(mass[i] for i in ids) for j in range(3)])
        assert means[0]==[(1-p)*(1-q)/2,(1-p)*(1-q)/2,2*q*(1-p)*(1-q)/3]
        assert means[1]==[p*(1-q)/6,p*(1-q)/6,(1-q)*(3-2*p)/3]
        assert means[0][1]/sum(means[0])==3/(6+4*q)
        assert means[1][1]/sum(means[1])==p/(6-2*p)


def main():
    start=time.perf_counter();algebra();r=load('reference_transfer.json');composite=load('composite_reference.json');cwru=load('cwru_attribution_audit.json')
    raw=np.load(OUT/'reference_transfer_draws.npz');d={k:raw[k] for k in raw.files};old=np.load(OUT/'observed_overlap_draws.npz')
    source,base,tune,test=[old[k] for k in ['source_counts','target_counts','tune_counts','test_counts']]
    learned=json.loads((OUT/'learned_calibration.json').read_text());lookup={(x['n'],x['case_seed'],x['family'],x['role']):x for x in learned['records']}
    assert len(r['records'])==1200
    for i,row in enumerate(r['records']):
        assert row['array_index']==i
        original=lookup[row['n'],row['case_seed'],row['family'],'evaluation']
        assert original['views']['mechanism_only']['probability_table']==row['reference_table']
        assert original['views']['unrestricted']['probability_table']==row['candidate_table']
        families=[row['family']]+[x for x in ['tree','forest','logistic'] if x!=row['family']]
        for j,family in enumerate(families):
            candidate=lookup[row['n'],row['case_seed']-100000,family,'calibration']['views']['mechanism_only']['probability_table']
            assert candidate==row['control_tables'][j] and functional_gate(candidate)['eligible']
        assert row['control_tables'][3]==original['views']['proxy_only']['probability_table']
        assert np.array_equal(d['shift_counts'][i,0],test[i])
        z=test[i].reshape(200,8,2).sum(axis=2)
        for law,counts in enumerate(d['shift_counts'][i]):
            assert np.array_equal(counts.reshape(200,8,2).sum(axis=2),z)
            assert np.allclose(d['f1'][i,law],f1(row['candidate_table'],counts))
            assert np.allclose(d['proxy_f1'][i,law],f1(row['control_tables'][3],counts))
            if law==4:
                expected=np.where(STATES[::2,0][None,:]==0,z,0)
                assert np.array_equal(counts[:,1::2],expected)
            response=abs(np.array(row['candidate_table'])-np.array(row['candidate_table'])[np.arange(16)^1])
            rho=counts@response/row['n'];cut=np.sort(d['tie_cal'][i])[380]
            targeted=(rho>0)|((rho==0)&(d['ties'][i]>cut))
            assert np.array_equal(targeted,d['targeted'][i,law])
        for k,kind in enumerate(['shap','flip']):
            ba,_,_=scores(row['reference_table'],base[i],kind);ta,_,_=scores(row['reference_table'],tune[i],kind)
            for m,cut in enumerate(row['cuts'][k]):
                edges=np.array(cut['edges']);o=observed_overlap(base[i]);b=np.searchsorted(edges[1:-1],o,side='right')
                expected_q=[conformal_threshold(ba[b==j]) for j in range(len(edges)-1)]
                if m==0:expected_q=[0.]
                assert np.array_equal(expected_q,np.array(cut['q'],float))
                if m>=2:
                    source_o=observed_overlap(source[i]);quantiles=[1/3,2/3] if m==2 else [.2,.4,.6,.8]
                    assert np.allclose(edges[1:-1],np.quantile(source_o,quantiles))
                tb=np.searchsorted(edges[1:-1],observed_overlap(tune[i]),side='right');q=np.array(expected_q)
                usable=np.isfinite(q[tb])&np.isfinite(ta);res=np.full(400,-np.inf);res[usable]=ta[usable]-q[tb[usable]]
                order=np.lexsort((d['tie_cal'][i],res)) if k==1 else np.argsort(res);index=order[380]
                assert float(cut['delta'])==res[index]
                if k==1:assert cut['tie_cut']==d['tie_cal'][i,index]
                a,_,v=scores(row['reference_table'],test[i],kind)
                assert np.array_equal(decision(a,test[i],v,cut,d['ties'][i]),d['null_warning'][i,k,m])
            for j,values in enumerate(row['control_tables']):
                a,l,v=scores(values,test[i],kind)
                for m,cut in enumerate(row['cuts'][k]):
                    assert np.array_equal(decision(a,test[i],v,cut,d['ties'][i]),d['control_warning'][i,k,m,j])
                    assert np.array_equal(decision(l,test[i],v,cut,d['ties'][i]),d['control_localization'][i,k,m,j])
            for law,counts in enumerate(d['shift_counts'][i]):
                a,l,v=scores(row['candidate_table'],counts,kind)
                for m,cut in enumerate(row['cuts'][k]):
                    assert np.array_equal(decision(a,counts,v,cut,d['ties'][i]),d['warning'][i,k,m,law])
                    assert np.array_equal(decision(l,counts,v,cut,d['ties'][i]),d['localization'][i,k,m,law])
        if i%100==0:
            a,l,v=scores(row['candidate_table'],test[i]);aa,ll,vv=batch_scores(row['candidate_table'],test[i])
            assert np.allclose(a,aa) and np.allclose(l,ll) and np.array_equal(v,vv)
        if (i+1)%200==0:print('verified follow-up cases',i+1,flush=True)
    for item in r['summary']:
        ids=[i for i,x in enumerate(r['records']) if x['n']==item['n'] and x['family']==item['family']]
        harm=d['f1'][ids,0].mean(axis=1)-d['f1'][ids,3].mean(axis=1)>=.05
        assert item['harm_cases']==harm.sum()
        if item['score']=='targeted_C':
            law=r['laws'].index(item['error']);assert np.isclose(item['all_localization']['mean'],d['targeted'][ids,law].mean());continue
        k=['shap','flip'].index(item['score']);m=r['methods'].index(item['method'])
        assert np.isclose(item['self_warning']['mean'],d['null_warning'][ids,k,m].mean())
        for j in range(3):assert np.isclose(item['aligned_warning'][j]['mean'],d['control_warning'][ids,k,m,j].mean())
        for law,summary in enumerate(item['laws']):
            assert np.isclose(summary['all_localization']['mean'],d['localization'][ids,k,m,law].mean())
            assert np.isclose(summary['harm_localization']['mean'],d['localization'][ids,k,m,law].mean(axis=1)[harm].mean())
    bootstrap_rng=np.random.default_rng(2026102543)
    def check_interval(saved,values,indices):
        assert saved['cases']==len(values) and np.isclose(saved['mean'],values.mean())
        assert np.allclose(saved['ci95'],np.quantile(values[indices].mean(axis=1),[.025,.975]))
    for n in [1000,10000]:
        for family in ['tree','forest','logistic']:
            ids=[i for i,x in enumerate(r['records']) if x['n']==n and x['family']==family]
            boot=bootstrap_rng.integers(0,200,(1000,200))
            harm=d['f1'][ids,0].mean(axis=1)-d['f1'][ids,3].mean(axis=1)>=.05
            hb=bootstrap_rng.integers(0,int(harm.sum()),(1000,int(harm.sum())))
            for item in r['summary']:
                if item['n']!=n or item['family']!=family:continue
                if item['score']=='targeted_C':
                    law=r['laws'].index(item['error']);values=d['targeted'][ids,law].mean(axis=1)
                    check_interval(item['all_localization'],values,boot);check_interval(item['harm_localization'],values[harm],hb);continue
                k=['shap','flip'].index(item['score']);m=r['methods'].index(item['method'])
                check_interval(item['self_warning'],d['null_warning'][ids,k,m].mean(axis=1),boot)
                for j in range(3):check_interval(item['aligned_warning'][j],d['control_warning'][ids,k,m,j].mean(axis=1),boot)
                for law,item_law in enumerate(item['laws']):
                    values=d['localization'][ids,k,m,law].mean(axis=1)
                    check_interval(item_law['all_localization'],values,boot);check_interval(item_law['harm_localization'],values[harm],hb)
    for item in r['paired_case_tests']:
        ids=[i for i,x in enumerate(r['records']) if x['n']==item['n'] and x['family']==item['family']];k=['shap','flip'].index(item['score'])
        delta=d['localization'][ids,k,1,0].mean(axis=1)-d['localization'][ids,k,0,0].mean(axis=1)
        plus,minus=int((delta>0).sum()),int((delta<0).sum());assert (plus,minus)==(item['positive'],item['negative'])
        assert item['p']==(float(binomtest(plus,plus+minus,.5).pvalue) if plus+minus else 1.)
    running=0.
    for position,index in enumerate(np.argsort([x['p'] for x in r['paired_case_tests']])):
        item=r['paired_case_tests'][index];running=max(running,min(1.,(12-position)*item['p']));assert item['holm_p']==running
    composite_archive=np.load(OUT/'composite_reference_draws.npz')
    composite_draws={key:composite_archive[key] for key in composite_archive.files}
    composite_rng=np.random.default_rng(2026102547)
    for bank in composite['banks']:
        ids=bank['case_indices'];counts=np.array([base[i,0] for i in ids])
        for k,kind in enumerate(['shap','flip']):
            cal=np.array([scores(r['records'][i]['control_tables'][0],counts[j:j+1],kind)[0][0] for j,i in enumerate(ids)])
            assert np.array_equal(cal,bank['scores'][k])
            cached={i:[scores(r['records'][i]['candidate_table'],x,kind) for x in d['shift_counts'][i]] for i in ids}
            null_cached={i:scores(r['records'][i]['reference_table'],test[i],kind) for i in ids}
            for m,edges in enumerate(bank['edges']):
                b=np.searchsorted(np.array(edges)[1:-1],observed_overlap(counts),side='right')
                for j,cut in enumerate(bank['cuts'][k][m]):
                    idx=np.flatnonzero(b==j);rank=int(np.ceil((len(idx)+1)*.95));assert cut['n']==len(idx)
                    if rank>len(idx):assert cut['q']=='Infinity';continue
                    order=np.lexsort((np.array(bank['tie_uniforms'])[idx],cal[idx])) if k==1 else np.argsort(cal[idx]);selected=idx[order[rank-1]]
                    assert cut['q']==cal[selected]
                for i in ids:
                    null_score,_,null_valid=null_cached[i]
                    bb=np.searchsorted(np.array(edges)[1:-1],observed_overlap(test[i]),side='right')
                    q=np.array([float(x['q']) for x in bank['cuts'][k][m]])[bb];u=np.array([x['tie'] for x in bank['cuts'][k][m]])[bb]
                    usable=null_valid&np.isfinite(q)
                    found=usable&((null_score>q)|((null_score==q)&(d['ties'][i]>u)))
                    assert np.array_equal(found,composite_draws['null_warning'][i,k,m])
                    assert np.array_equal(~usable,composite_draws['abstention'][i,k,m])
                    for law,counts_eval in enumerate(d['shift_counts'][i]):
                        _,l,v=cached[i][law]
                        bb=np.searchsorted(np.array(edges)[1:-1],observed_overlap(counts_eval),side='right')
                        q=np.array([float(x['q']) for x in bank['cuts'][k][m]])[bb];u=np.array([x['tie'] for x in bank['cuts'][k][m]])[bb]
                        found=v&np.isfinite(q)&((l>q)|((l==q)&(d['ties'][i]>u)))
                        assert np.array_equal(found,composite_draws['localization'][i,k,m,law])
        assert np.array_equal(composite_rng.random(200),bank['tie_uniforms'])
        boot=composite_rng.integers(0,200,(1000,200))
        harm=d['f1'][ids,0].mean(axis=1)-d['f1'][ids,3].mean(axis=1)>=.05
        for item in composite['summary']:
            if item['n']!=bank['n'] or item['family']!=bank['family']:continue
            k=['shap','flip'].index(item['score']);m=r['methods'].index(item['method'])
            for key,array in [('null_warning','null_warning'),('abstention','abstention')]:
                values=composite_draws[array][ids,k,m].mean(axis=1)
                assert np.isclose(item[key]['mean'],values.mean())
                assert np.allclose(item[key]['ci95'],np.quantile(values[boot].mean(axis=1),[.025,.975]))
            for law,summary in enumerate(item['laws']):
                values=composite_draws['localization'][ids,k,m,law].mean(axis=1)
                assert np.isclose(summary['all_localization']['mean'],values.mean())
                assert np.isclose(summary['harm_localization'],values[harm].mean())
                assert np.isclose(summary['no_harm_localization'],values[~harm].mean())
    # All measured attribution arrays must add back to their actual model probabilities.
    ca=np.load(OUT/'cwru_attribution_draws.npz');metadata=pd.DataFrame(cwru['recording_metadata']);columns=cwru['features']
    for proof in cwru['proof']:
        if 'folds' not in proof:continue
        condition=['original','spectral'].index(proof['condition'])
        for fold in proof['folds']:
            assert hashlib.sha256((ROOT/fold['model_path']).read_bytes()).hexdigest()==fold['model_sha256']
            model=joblib.load(ROOT/fold['model_path']);assert list(model.classes_)==['B','IR','OR'];ids=fold['test_indices'];bg=fold['background_indices']
            assert not set(ids)&set(bg)
            for view in range(2):
                features=ca['original_features'] if condition==0 or view==1 else ca['spectral_features']
                assert np.allclose(model.predict_proba(pd.DataFrame(features[ids],columns=columns)),ca['probabilities'][condition,view,ids])
                assert np.max(abs(ca['signed_shap'][condition,view,ids].sum(axis=1)+fold['expected_values']-ca['probabilities'][condition,view,ids]))<1e-6
    for item in cwru['records']:
        condition=['original','spectral'].index(item['condition']);view=['matched','filter_removed'].index(item['view']);head=['B','IR','OR'].index(item['fault'])
        table=pd.DataFrame(abs(ca['signed_shap'][condition,view,:,:,head]),columns=columns).groupby(metadata.recording.to_numpy()).mean()
        label=metadata.groupby('recording').fault.first();a=table.loc[label[label==item['fault']].index].mean();b=table.loc[label[label!=item['fault']].index].mean();a=a/a.sum();b=b/b.sum()
        expected=[f'{domain}_{item["fault"]}_{h}' for domain in ['raw','envelope'] for h in [1,2]]
        excess=(a-b).drop(expected)
        assert np.isclose(item['R'],a[expected].sum()) and np.isclose(item['D'],a[expected].sum()-b[expected].sum())
        assert np.isclose(item['S'],excess.max()) and item['S_feature']==excess.idxmax()
    # Reconstruct finite-inventory swaps and bootstrap outcomes from recording votes.
    locations=['B','IR','OR'];group_keys=sorted(set(zip(metadata.fault,metadata.diameter_in)))
    confusion=np.zeros((9,2,3,3),int)
    for recording,frame in metadata.groupby('recording'):
        indices=frame.index;truth=locations.index(frame.fault.iloc[0]);group=group_keys.index((frame.fault.iloc[0],frame.diameter_in.iloc[0]))
        for view in range(2):
            prediction=int(ca['probabilities'][1,view,indices].mean(axis=0).argmax())
            confusion[group,view,truth,prediction]+=1
    def macro(matrix):return float(np.mean(2*np.diag(matrix)/(matrix.sum(axis=0)+matrix.sum(axis=1))))
    swaps=[]
    for mask in itertools.product([0,1],repeat=9):
        a=sum(confusion[j,bit] for j,bit in enumerate(mask));b=sum(confusion[j,1-bit] for j,bit in enumerate(mask))
        swaps.append(macro(a)-macro(b))
    assert np.allclose(swaps,ca['swaps'])
    bootstrap=[]
    for choice in itertools.product(range(3),repeat=9):
        ids=[3*(j//3)+value for j,value in enumerate(choice)];pair=confusion[ids].sum(axis=0)
        bootstrap.append(macro(pair[0])-macro(pair[1]))
    assert np.allclose(bootstrap,ca['bootstrap'])
    exact=cwru['finite_inventory'];assert ca['swaps'].shape==(512,) and ca['bootstrap'].shape==(19683,)
    assert np.isclose(np.mean(abs(ca['swaps'])>=abs(exact['observed_F1_drop'])-1e-12),exact['swap_two_sided_p'])
    assert np.allclose(np.quantile(ca['bootstrap'],[.025,.975]),exact['exact_bootstrap_percentiles'])
    assert np.isclose(r['PAC_95_warning_upper_M400'],beta.ppf(.95,20,381))
    paths=[Path(__file__),OUT/'reference_transfer.json',OUT/'reference_transfer_draws.npz',OUT/'composite_reference.json',OUT/'composite_reference_draws.npz',OUT/'cwru_attribution_audit.json',OUT/'cwru_attribution_draws.npz']
    manifest={'verified':True,'algebra_exact_rational':True,'case_pairs':1200,'cwru_audit_heads':9,'seconds':time.perf_counter()-start,
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'reference_transfer_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS corrected algebra, independent fit roles, all calibration cuts and paired outcomes, composite model calibration, CWRU attribution/additivity, finite-inventory summaries',flush=True)


if __name__=='__main__':main()
