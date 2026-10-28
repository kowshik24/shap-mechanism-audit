"""Diameter-primary spectral acquisition confound and nonconstant partial masks."""
import pdm_audit
import hashlib
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
from scipy.io import loadmat
from pdm_audit.cwru import EXPECTED, load_features, features
from pdm_audit.acquisition import resonant_transfer, RESONANCE_HZ
from pdm_audit.industrial import recording_binary

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'experiments/results'
spec = importlib.util.spec_from_file_location('original_cwru',ROOT/'experiments/12_cwru.py')
base = importlib.util.module_from_spec(spec);spec.loader.exec_module(base)


def main():
    df, inventory = load_features(ROOT/'data/cwru')
    df['diameter_in'] = df.recording.map({r['recording']:r['diameter_in'] for r in inventory})
    cols = [c for c in df if c not in ['recording','fault','load_hp','window','diameter_in']]
    changed = df[cols].copy(); proof = []
    for r in inventory:
        x = loadmat(ROOT/'data/cwru'/(r['recording']+'.mat'))[r['channel']].ravel()
        indices = df.index[df.recording == r['recording']]
        max_rms_error = 0.; differences = []
        for index in indices:
            w = int(df.loc[index,'window']); signal = x[w*4096:(w+1)*4096]
            transformed = resonant_transfer(signal,r['fault'])
            row = features(transformed,r['rpm']); old = df.loc[index,cols]
            max_rms_error = max(max_rms_error, abs(row['log_RMS']-old['log_RMS']))
            assert max_rms_error < 1e-12
            # Enforce the exact intended design equality, avoiding roundoff leakage.
            row['log_RMS'] = old['log_RMS']; changed.loc[index] = [row[c] for c in cols]
            differences.append(float(np.linalg.norm(changed.loc[index].to_numpy()-old.to_numpy())))
        proof.append({'recording':r['recording'],'max_log_RMS_roundoff':max_rms_error,
                      'minimum_feature_change_L2':min(differences)})
    results = []
    for condition, X in [('original',df[cols]),('spectral',changed)]:
        probabilities,removed = np.zeros((len(df),3)),np.zeros((len(df),3));folds=[]
        for diameter in sorted(df.diameter_in.unique()):
            tr = df.index[df.diameter_in != diameter];te = df.index[df.diameter_in == diameter]
            scores=[]
            for depth in [3,6,None]:
                values=[]
                for inner in sorted(df.loc[tr,'diameter_in'].unique()):
                    a=tr[df.loc[tr,'diameter_in'] != inner];b=tr[df.loc[tr,'diameter_in'] == inner]
                    model=base.model(depth).fit(X.loc[a],df.loc[a,'fault'])
                    values.append(base.metrics(df.loc[b,'fault'],model.predict_proba(X.loc[b]),df.loc[b])['recording_macro_f1'])
                scores.append(float(np.mean(values)))
            depth=[3,6,None][int(np.argmax(scores))]
            model=base.model(depth).fit(X.loc[tr],df.loc[tr,'fault'])
            probabilities[te]=model.predict_proba(X.loc[te]);removed[te]=model.predict_proba(df.loc[te,cols])
            folds.append({'diameter':float(diameter),'depth':depth,'inner_F1':scores,
                          'training_recordings':sorted(set(df.loc[tr,'recording'])),
                          'test_recordings':sorted(set(df.loc[te,'recording']))})
        item={'condition':condition,'matched':base.metrics(df.fault,probabilities,df),
              'filter_removed':base.metrics(df.fault,removed,df),'folds':folds}
        rng=np.random.default_rng(2026102438);a=item['matched']['recordings'];b=item['filter_removed']['recordings'];drops=[]
        from sklearn.metrics import f1_score
        for _ in range(10000):
            ids=np.concatenate([rng.choice([i for i,r in enumerate(a) if r['truth']==m],12) for m in ['B','IR','OR']])
            y=[a[i]['truth'] for i in ids]
            drops.append(f1_score(y,[a[i]['prediction'] for i in ids],average='macro')-
                         f1_score(y,[b[i]['prediction'] for i in ids],average='macro'))
        item['F1_drop_recording_bootstrap_95']=np.quantile(drops,[.025,.975]).tolist()
        clusters={rec['recording']:(rec['fault'],rec['diameter_in']) for rec in inventory}
        cluster_rng=np.random.default_rng(20261024381);cluster_drops=[]
        for _ in range(10000):
            selected=[]
            for location in ['B','IR','OR']:
                choices=sorted({clusters[x['id']] for x in a if x['truth']==location})
                for k in cluster_rng.integers(0,len(choices),size=3):
                    selected.extend(i for i,x in enumerate(a) if clusters[x['id']]==choices[k])
            y=[a[i]['truth'] for i in selected]
            cluster_drops.append(f1_score(y,[a[i]['prediction'] for i in selected],average='macro')-
                                 f1_score(y,[b[i]['prediction'] for i in selected],average='macro'))
        item['F1_drop_cluster_bootstrap_95']=np.quantile(cluster_drops,[.025,.975]).tolist()
        results.append(item)
        print(condition,item['matched']['recording_macro_f1'],item['filter_removed']['recording_macro_f1'],flush=True)
    previous=json.loads((OUT/'cwru_followup.json').read_text())
    expected=next(r for r in previous['records'] if r['axis']=='diameter_in' and r['view']=='all' and not r['gain_correlated'])
    assert results[0]['matched']==expected['matched']
    controls=json.loads((OUT/'industrial_alignment.json').read_text()); partial=[]
    for r in controls['records']:
        if r['view']!='own_anchors':continue
        model=joblib.load(ROOT/r['model_path']);test=df.loc[df[r['axis']]==r['held_out']]
        for col in r['features']:
            x=test[r['features']].copy();x[col]=r['training_anchor_medians'][col]
            p=model.predict_proba(x)[:,1]
            metrics=recording_binary(test,p,r['fault'])
            partial.append({'axis':r['axis'],'held_out':r['held_out'],'fault':r['fault'],'masked_feature':col,
                            'metrics':metrics,'constant_window_probability':bool(np.ptp(p)==0)})
    paths=[Path(__file__),ROOT/'experiments/REVIEW2_PROTOCOL.md',ROOT/'experiments/REVIEW2_CLUSTER_AMENDMENT.md',ROOT/'src/pdm_audit/acquisition.py',
           OUT/'industrial_alignment.json',OUT/'cwru_followup.json',ROOT/'data/cwru/manifest.json']
    output={'complete':True,'primary_axis':'diameter_in','normal_recordings_used':False,
            'resonance_centers_Hz':RESONANCE_HZ,'width_Hz':250,'response_amplitude':3,
            'RMS_preserved_by_design':True,'input_checks':proof,'results':results,'single_anchor_masks':partial,
            'primary_bootstrap_clusters':9,'cluster_bootstrap_added_after_point_outcomes':True,
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'cwru_spectral_transfer.json').write_text(json.dumps(output,indent=1,allow_nan=False))


if __name__=='__main__':main()
