"""Audit earlier continuous attributions using an equivalent threshold encoding.

Original predictions, models, and study records are preserved. Corrections, if
needed, are saved separately with parent hashes and recomputed correspondence.
"""
import pdm_audit
import hashlib,json,time,copy
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from pdm_audit.tree_precision import probability_explainer
from pdm_audit.cwru import recording_shares
from pdm_audit.sca import correspondence

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def main():
    began=time.perf_counter();cw=json.loads((OUT/'cwru_attribution_audit.json').read_text());raw=dict(np.load(OUT/'cwru_attribution_draws.npz'))
    attrs=np.zeros_like(raw['signed_shap']);expected=np.zeros_like(raw['probabilities']);checks=[];corrected=copy.deepcopy(cw)
    metadata=pd.DataFrame(cw['recording_metadata']);columns=cw['features']
    for c,condition in enumerate(['original','spectral']):
        x=raw['original_features'] if c==0 else raw['spectral_features']
        folds=next(p['folds'] for p in corrected['proof'] if p.get('condition')==condition and 'folds' in p)
        for fold in folds:
            model=joblib.load(ROOT/fold['model_path']);bg=fold['background_indices'];test=fold['test_indices']
            explainer=probability_explainer(model,x[bg]);elapsed=0.
            for view,z in enumerate([x,raw['original_features']]):
                tick=time.perf_counter();a=explainer.shap_values(z[test],check_additivity=False);elapsed+=time.perf_counter()-tick
                p=model.predict_proba(z[test]);error=float(abs(np.asarray(a).sum(axis=1)+explainer.expected_value-p).max());assert error<1e-6
                assert np.allclose(p,raw['probabilities'][c,view,test],atol=1e-12)
                attrs[c,view,test]=a;expected[c,view,test]=explainer.expected_value
                checks.append({'study':'cwru','condition':condition,'diameter':fold['diameter'],'view':view,
                    'max_attribution_change':float(abs(np.asarray(a)-raw['signed_shap'][c,view,test]).max()),'additivity_error':error})
            fold['expected_values']=explainer.expected_value.tolist();fold['tree_shap_seconds']=elapsed
        print('precision audit CWRU',condition,flush=True)
    corrected['records']=[]
    for c,condition in enumerate(['original','spectral']):
        for v,view in enumerate(['matched','filter_removed']):
            if c==0 and v==1:continue
            for j,fault in enumerate(['B','IR','OR']):
                a=pd.DataFrame(abs(attrs[c,v,:,:,j]),columns=columns);result=recording_shares(a,metadata,fault)
                result['band_share']=sum(result['shares'][x] for x in columns if x.startswith('band_'))
                result['maximizer_is_band']=result['S_feature'].startswith('band_')
                corrected['records'].append({'condition':condition,'view':view,'fault':fault,**result})
    arrays={'cwru_shap':attrs,'cwru_expected':expected}
    field=json.loads((OUT/'sca_field.json').read_text());fraw=dict(np.load(OUT/'sca_field_arrays.npz'));df=pd.read_csv(OUT/'sca_features.csv');fcorrected=copy.deepcopy(field)
    for item in fcorrected['summary']:
        view=item['view'];columns=item['columns'];x=df[columns].to_numpy();a=np.full_like(fraw[view+'_shap'],np.nan);offset=np.full(len(df),np.nan)
        folds=next(p['folds'] for p in fcorrected['proof'] if p['view']==view)
        for fold in folds:
            model=joblib.load(ROOT/fold['model']);bg=fold['background_rows'];test=fold['test_rows'];explainer=probability_explainer(model,x[bg])
            a[test]=np.asarray(explainer.shap_values(x[test],check_additivity=False))[:,:,1];offset[test]=explainer.expected_value[1]
            p=model.predict_proba(x[test])[:,1];error=float(abs(a[test].sum(axis=1)+offset[test]-p).max());assert error<1e-6
            assert np.allclose(p,fraw[view+'_probabilities'][test],atol=1e-12)
            checks.append({'study':'sca','view':view,'held_case':fold['held_case'],
                'max_attribution_change':float(abs(a[test]-fraw[view+'_shap'][test]).max()),'additivity_error':error})
            fold['expected_value']=float(explainer.expected_value[1]);fold['max_additivity_error']=error
        use=np.flatnonzero(df.case.to_numpy()<=10)
        item['audit']=correspondence(a[use],df.iloc[use],columns)
        item['within_case_audit']=[{'case':fold['held_case'],'audit':correspondence(a[fold['test_rows']],df.iloc[fold['test_rows']],columns)} for fold in folds]
        arrays['sca_'+view+'_shap']=a;arrays['sca_'+view+'_expected']=offset
        print('precision audit SCA',view,flush=True)
    path=OUT/'tree_precision_draws.npz';np.savez_compressed(path,**arrays)
    for result,parent,name in [(corrected,'cwru_attribution_audit.json','cwru_attribution_precision.json'),(fcorrected,'sca_field.json','sca_field_precision.json')]:
        result['parent_study']=parent;result['parent_sha256']=hashlib.sha256((OUT/parent).read_bytes()).hexdigest()
        result['precision_representation']='prediction-equivalent float32 threshold floor; original fitted model unchanged'
        result['precision_draws']='experiments/results/tree_precision_draws.npz'
        (OUT/name).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    paths=[Path(__file__),ROOT/'src/pdm_audit/tree_precision.py',OUT/'cwru_attribution_audit.json',OUT/'cwru_attribution_draws.npz',OUT/'sca_field.json',OUT/'sca_field_arrays.npz',OUT/'cwru_attribution_precision.json',OUT/'sca_field_precision.json',path]
    out={'complete':True,'checks':checks,'max_attribution_change':max(x['max_attribution_change'] for x in checks),
        'original_models_and_predictions_preserved':True,'seconds':time.perf_counter()-began,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'tree_precision_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print('COMPLETE numerical representation audit; maximum attribution change',out['max_attribution_change'],flush=True)


if __name__=='__main__':main()
