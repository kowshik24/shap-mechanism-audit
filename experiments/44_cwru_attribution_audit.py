"""Replay the measured spectral pressure test with descriptive TreeSHAP audits."""
import pdm_audit
import hashlib,importlib.util,itertools,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,shap,sklearn,platform,subprocess
from scipy.io import loadmat
from pdm_audit.cwru import EXPECTED,load_features,features,recording_shares
from pdm_audit.acquisition import resonant_transfer
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'
spec=importlib.util.spec_from_file_location('cwru_base',ROOT/'experiments/12_cwru.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def macro(matrix):
    denominator=matrix.sum(axis=0)+matrix.sum(axis=1)
    return float(np.mean(np.divide(2*np.diag(matrix),denominator,out=np.zeros(3),where=denominator>0)))


def finite_inventory(matched,removed,inventory):
    labels=['B','IR','OR'];mapping={r['recording']:(r['fault'],r['diameter_in']) for r in inventory}
    keys=sorted(set(mapping.values()));groups=np.zeros((9,2,3,3),int)
    a=matched['recordings'];b=removed['recordings']
    for i,row in enumerate(a):
        assert row['id']==b[i]['id']
        group=keys.index(mapping[row['id']]);truth=labels.index(row['truth'])
        groups[group,0,truth,labels.index(row['prediction'])]+=1
        groups[group,1,truth,labels.index(b[i]['prediction'])]+=1
    observed=macro(groups[:,0].sum(axis=0))-macro(groups[:,1].sum(axis=0));swaps=[]
    for mask in itertools.product([0,1],repeat=9):
        one=sum(groups[k,bit] for k,bit in enumerate(mask));two=sum(groups[k,1-bit] for k,bit in enumerate(mask))
        swaps.append(macro(one)-macro(two))
    bootstrap=[]
    for selection in itertools.product(range(3),repeat=9):
        ids=[3*(j//3)+index for j,index in enumerate(selection)]
        matrix=groups[ids].sum(axis=0);bootstrap.append(macro(matrix[0])-macro(matrix[1]))
    return {'groups':keys,'observed_F1_drop':observed,'swap_count':512,
            'swap_two_sided_p':float(np.mean(np.abs(swaps)>=abs(observed)-1e-12)),
            'bootstrap_configurations':19683,'exact_bootstrap_percentiles':np.quantile(bootstrap,[.025,.975]).tolist()},np.array(swaps),np.array(bootstrap)


def main():
    start=time.perf_counter();df,inventory=load_features(ROOT/'data/cwru');df['diameter_in']=df.recording.map({r['recording']:r['diameter_in'] for r in inventory})
    columns=[c for c in df if c not in ['recording','fault','load_hp','window','diameter_in']];changed=df[columns].copy()
    for row in inventory:
        x=loadmat(ROOT/'data/cwru'/(row['recording']+'.mat'))[row['channel']].ravel()
        for i in df.index[df.recording==row['recording']]:
            w=int(df.loc[i,'window']);value=features(resonant_transfer(x[w*4096:(w+1)*4096],row['fault']),row['rpm'])
            value['log_RMS']=df.loc[i,'log_RMS'];changed.loc[i]=[value[c] for c in columns]
    old=json.loads((OUT/'cwru_spectral_transfer.json').read_text());records=[];all_attrs=[];all_probabilities=[];proof=[]
    modeldir=OUT/'cwru_attribution_models';modeldir.mkdir(exist_ok=True)
    for condition,X in [('original',df[columns]),('spectral',changed)]:
        previous=next(x for x in old['results'] if x['condition']==condition)
        attrs=np.zeros((2,len(df),len(columns),3));probabilities=np.zeros((2,len(df),3));folds=[]
        for fold in previous['folds']:
            diameter=fold['diameter'];tr=df.index[df.diameter_in!=diameter];te=df.index[df.diameter_in==diameter]
            model=base.model(fold['depth']).fit(X.loc[tr],df.loc[tr,'fault'])
            bg=tr[np.linspace(0,len(tr)-1,96,dtype=int)];tick=time.perf_counter()
            explainer=shap.TreeExplainer(model,data=X.loc[bg],feature_perturbation='interventional',model_output='probability')
            shap_seconds=0.
            for view,V in enumerate([X,df[columns]]):
                probabilities[view,te]=model.predict_proba(V.loc[te])
                shap_started=time.perf_counter()
                attrs[view,te]=explainer.shap_values(V.loc[te],check_additivity=True)
                shap_seconds+=time.perf_counter()-shap_started
                error=np.max(abs(attrs[view,te].sum(axis=1)+explainer.expected_value-probabilities[view,te]))
                assert error<1e-6
                proof.append({'condition':condition,'diameter':diameter,'view':view,'max_additivity_error':float(error)})
            filename=modeldir/f'{condition}_{diameter:.3f}.joblib';joblib.dump(model,filename)
            folds.append({'diameter':diameter,'depth':fold['depth'],'background_indices':bg.tolist(),'test_indices':te.tolist(),
                          'model_path':str(filename.relative_to(ROOT)),'model_sha256':hashlib.sha256(filename.read_bytes()).hexdigest(),
                          'expected_values':explainer.expected_value.tolist(),'tree_shap_seconds':shap_seconds,'explanation_seconds':time.perf_counter()-tick})
            print(condition,diameter,'SHAP seconds',round(folds[-1]['explanation_seconds'],2),flush=True)
        for view,name in enumerate(['matched','filter_removed']):
            measured=base.metrics(df.fault,probabilities[view],df);assert measured==previous[name]
            if condition=='original' and view==1:continue
            for index,location in enumerate(base.CLASSES):
                attribution=pd.DataFrame(abs(attrs[view,:,:,index]),columns=columns)
                result=recording_shares(attribution,df,location)
                result['band_share']=sum(result['shares'][c] for c in columns if c.startswith('band_'))
                result['maximizer_is_band']=result['S_feature'].startswith('band_')
                records.append({'condition':condition,'view':name,'fault':location,**result})
        all_attrs.append(attrs);all_probabilities.append(probabilities)
        if condition=='spectral':exact,swaps,bootstrap=finite_inventory(previous['matched'],previous['filter_removed'],inventory)
        proof.append({'condition':condition,'folds':folds})
    np.savez_compressed(OUT/'cwru_attribution_draws.npz',signed_shap=np.array(all_attrs),probabilities=np.array(all_probabilities),
        original_features=df[columns].to_numpy(),spectral_features=changed.to_numpy(),swaps=swaps,bootstrap=bootstrap)
    paths=[Path(__file__),ROOT/'experiments/REVIEW3_PROTOCOL.md',OUT/'cwru_spectral_transfer.json',ROOT/'src/pdm_audit/acquisition.py',ROOT/'src/pdm_audit/cwru.py',ROOT/'data/cwru/manifest.json']
    result={'complete':True,'records':records,'proof':proof,'features':columns,'recording_metadata':df[['recording','fault','diameter_in','window']].to_dict('records'),
            'finite_inventory':exact,'seconds':time.perf_counter()-start,'background_n':96,
            'environment':{'platform':platform.platform(),'machine':platform.machine(),'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'numpy':np.__version__,'shap':shap.__version__,'sklearn':sklearn.__version__},
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'cwru_attribution_audit.json').write_text(json.dumps(result,indent=1,allow_nan=False))
    print('COMPLETE CWRU attribution',round(result['seconds'],1),'exact paired swaps',exact,flush=True)


if __name__=='__main__':main()
