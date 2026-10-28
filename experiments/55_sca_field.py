"""Frozen case-held-out field diagnosis and descriptive TreeSHAP audit."""
import pdm_audit
import hashlib,json,time
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.sca import extract,correspondence,ANCHORS,FEATURES

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';DATA=ROOT/'data/sca'


def metrics(y,p,weights=None):
    prediction=p>=.5;w=np.ones(len(y)) if weights is None else np.asarray(weights)
    tp=float(w[(y==1)&prediction].sum());fp=float(w[(y==0)&prediction].sum());fn=float(w[(y==1)&~prediction].sum());tn=float(w[(y==0)&~prediction].sum())
    return {'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'healthy_FPR':fp/(fp+tn) if fp+tn else None,
            'recall':tp/(tp+fn) if tp+fn else None,'TP':tp,'FP':fp,'FN':fn,'TN':tn}


def main():
    start=time.perf_counter();source=json.loads((OUT/'sca_download.json').read_text());assert source['complete']
    for x in source['files']:
        with (DATA/x['name']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==x['sha256']
    df,excluded,duplicates=extract(DATA)
    gate={'complete':True,'retained':len(df),'excluded':excluded,'cross_case_duplicate_pairs':duplicates,'raw_files':len(source['files'])}
    (OUT/'sca_feature_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    assert not duplicates,'Cross-case duplicate signals invalidate the frozen split; do not fit'
    df.to_csv(OUT/'sca_features.csv',index=False)
    print('SCA feature gate',len(df),'retained;',len(excluded),'excluded',flush=True)
    y=(df.label.to_numpy()>0).astype(int);cases=df.case.to_numpy();external=np.flatnonzero(cases==11)
    arrays={};proof=[];summary=[];model_dir=OUT/'sca_models';model_dir.mkdir(exist_ok=True)
    for view,columns in [('anchors',ANCHORS),('all',FEATURES)]:
        x=df[columns];probability=np.full(len(df),np.nan);attr=np.full((len(df),len(columns)),np.nan)
        expected=np.full(len(df),np.nan);external_probs=[];folds=[];fold_metrics=[]
        for held in range(1,11):
            train=np.flatnonzero((cases!=held)&(cases<=10));test=np.flatnonzero(cases==held)
            train_cases=sorted(set(cases[train]));background=[]
            for j,c in enumerate(train_cases):
                ids=train[cases[train]==c];number=96//len(train_cases)+(j<96%len(train_cases))
                background.extend(ids[np.linspace(0,len(ids)-1,number,dtype=int)].tolist())
            weights=1/df.iloc[train].groupby('case').case.transform('size').to_numpy()
            model=RandomForestClassifier(n_estimators=64,max_depth=6,min_samples_leaf=5,max_features='sqrt',
                class_weight='balanced',random_state=2026,n_jobs=4).fit(x.iloc[train],y[train],sample_weight=weights)
            assert np.array_equal(model.classes_,[0,1])
            probability[test]=model.predict_proba(x.iloc[test])[:,1]
            external_probs.append(model.predict_proba(x.iloc[external])[:,1])
            began=time.perf_counter();explainer=shap.TreeExplainer(model,data=x.iloc[background],feature_perturbation='interventional',model_output='probability')
            values=np.asarray(explainer.shap_values(x.iloc[test],check_additivity=False))[:,:,1]
            seconds=time.perf_counter()-began;offset=float(explainer.expected_value[1])
            error=float(np.max(abs(values.sum(axis=1)+offset-probability[test])))
            assert error<1e-6
            attr[test]=values;expected[test]=offset
            path=model_dir/f'{view}_{held}.joblib';joblib.dump(model,path)
            folds.append({'held_case':held,'train_rows':train.tolist(),'test_rows':test.tolist(),'background_rows':background,
                'expected_value':offset,'max_additivity_error':error,'tree_shap_seconds':seconds,
                'model':str(path.relative_to(ROOT)),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            fold_metrics.append({'case':held,'records':len(test),**metrics(y[test],probability[test])})
            print('SCA',view,'held',held,'F1',round(fold_metrics[-1]['F1'],3),flush=True)
        use=np.flatnonzero(cases<=10);external_probs=np.asarray(external_probs)
        arrays[view+'_probabilities']=probability;arrays[view+'_shap']=attr;arrays[view+'_expected']=expected
        arrays[view+'_external_probabilities']=external_probs
        weights=1/df.iloc[use].groupby('case').case.transform('size').to_numpy()
        item={'view':view,'columns':columns,'case_weighted_metrics':metrics(y[use],probability[use],weights),
            'fold_metrics':fold_metrics,'audit':correspondence(attr[use],df.iloc[use],columns),'external':[],'within_case_audit':[]}
        for fold in folds:
            indices=fold['test_rows'];item['within_case_audit'].append({'case':fold['held_case'],'audit':correspondence(attr[indices],df.iloc[indices],columns)})
        for split in ['train','test']:
            selected=df.iloc[external].native_split.to_numpy()==split
            assert not y[external[selected]].any()
            item['external'].append({'native_split':split,'records':int(selected.sum()),
                **metrics(y[external[selected]],external_probs.mean(axis=0)[selected])})
        summary.append(item);proof.append({'view':view,'folds':folds})
    path=OUT/'sca_field_arrays.npz';np.savez_compressed(path,**arrays)
    paths=[Path(__file__),ROOT/'src/pdm_audit/sca.py',ROOT/'experiments/SCA_FIELD_PROTOCOL.md',OUT/'sca_download.json',OUT/'sca_features.csv',OUT/'sca_feature_gate.json',path]
    result={'complete':True,'summary':summary,'proof':proof,'records':len(df),'cases':sorted(set(cases.tolist())),
        'labels':dict(Counter(df.label.to_list())),'excluded':dict(Counter(x['reason'] for x in excluded)),
        'scope':'Descriptive case-held-out field diagnosis; pooled attribution contrasts include fold-model variation',
        'independent_sessions_verified':False,'exhaustive_physical_alignment_verified':False,'calibrated_industrial_warning':False,
        'seconds':time.perf_counter()-start,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'sca_field.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('COMPLETE descriptive SCA field audit',round(result['seconds'],1),flush=True)


if __name__=='__main__':main()
