"""Explain frozen measured-load controls without claiming calibrated shortcut detection."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from pdm_audit.kaist import fit_predict, metrics, weights

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def main():
    frame = pd.read_csv(OUT/'kaist_features.csv')
    names = [x for x in frame if x.startswith('ch')]
    saved = pd.read_csv(OUT/'kaist_predictions.csv')
    rows, profiles, arrays, seed_rows, seed_predictions = [], [], {}, [], []
    max_error = 0.
    for source_loads in [[0],[0,2]]:
        tag = 'source_'+'_'.join(map(str,source_loads))
        train = frame[frame.load_Nm.isin(source_loads)]
        predicted, model, scaler = fit_predict(train,frame,names,'forest',4)
        for load in [0,2,4]:
            if load in source_loads:
                continue
            existing = saved[(saved.family == 'forest')&(saved.load_Nm==load)]
            existing = existing[(existing.role=='source_validation')&(existing.training_load==0)&(existing.parameter==4)] if len(source_loads)==1 else existing[existing.role=='target']
            existing = existing.drop_duplicates(['recording','window'])
            expected = frame[frame.load_Nm==load][['recording','window']].merge(existing[['recording','window','prediction']],on=['recording','window'],validate='one_to_one')
            assert np.array_equal(expected.prediction,predicted[frame.load_Nm==load])
        background = pd.concat([g.iloc[np.linspace(0,len(g)-1,10).astype(int)] for _,g in train.groupby('label')])
        transformed = scaler.transform(frame[names])
        explainer = shap.TreeExplainer(model,data=scaler.transform(background[names]),
            feature_perturbation='interventional',model_output='probability')
        phi = explainer.shap_values(transformed,check_additivity=True)
        assert phi.shape == (len(frame),40,3)
        fault_phi = phi[:,:,1:].sum(axis=2)
        probability = model.predict_proba(transformed)[:,1:].sum(axis=1)
        base = float(np.asarray(explainer.expected_value)[1:].sum())
        error = float(np.max(abs(base+fault_phi.sum(axis=1)-probability)))
        assert error < 1e-6
        max_error = max(max_error,error)
        arrays[tag+'_fault_phi'] = fault_phi
        arrays[tag+'_fault_probability'] = probability
        arrays[tag+'_multiclass_prediction'] = predicted
        for load in [0,2,4]:
            selected = frame.load_Nm==load
            test = frame[selected]
            score = metrics(test,predicted[selected])
            healthy = selected & (frame.label==0)
            rows.append(dict(source_loads_Nm=source_loads,load_Nm=load,
                role='in_sample' if load in source_loads else 'held_out',metrics=score,
                fault_probability_threshold_healthy_FPR=float(np.mean(probability[healthy]>=.5)),
                healthy_mean_fault_probability=float(np.mean(probability[healthy])),
                background_fault_probability=base))
            for label in [0,1,2]:
                mask = selected & (frame.label==label)
                signed = fault_phi[mask].mean(axis=0)
                absolute = abs(fault_phi[mask]).mean(axis=0)
                profiles.append(dict(source_loads_Nm=source_loads,load_Nm=load,label=label,
                    mean_signed=signed.tolist(),mean_absolute=absolute.tolist()))
        fp = (frame.load_Nm==4)&(frame.label==0)&(predicted!=0)
        top = np.argsort(-fault_phi[fp].mean(axis=0))[:5] if fp.any() else np.array([],dtype=int)
        print(tag,'max additivity error',error,'top positive FP features',[names[i] for i in top],flush=True)
        for seed in range(20261026,20261046):
            repeated = clone(model).set_params(random_state=seed)
            repeated.fit(scaler.transform(train[names]),train.label,sample_weight=weights(train))
            pred = repeated.predict(transformed)
            prob = repeated.predict_proba(transformed)[:,1:].sum(axis=1)
            if seed == 20261026:
                assert np.array_equal(pred,predicted) and np.allclose(prob,probability)
            for load in [0,2,4]:
                mask = frame.load_Nm==load
                healthy = mask&(frame.label==0)
                seed_rows.append(dict(source_loads_Nm=source_loads,seed=seed,load_Nm=load,
                    role='in_sample' if load in source_loads else 'held_out',
                    metrics=metrics(frame[mask],pred[mask]),
                    fault_probability_threshold_healthy_FPR=float(np.mean(prob[healthy]>=.5))))
            for i,r in frame.iterrows():
                seed_predictions.append(dict(source_tag=tag,seed=seed,recording=r.recording,load_Nm=int(r.load_Nm),
                    window=int(r.window),label=int(r.label),prediction=int(pred[i]),fault_probability=float(prob[i])))
    pd.DataFrame(seed_predictions).to_csv(OUT/'kaist_seed_predictions.csv',index=False)
    np.savez_compressed(OUT/'kaist_diagnostic_arrays.npz',**arrays)
    inputs = ['experiments/KAIST_DIAGNOSTIC_PROTOCOL.md','experiments/96_kaist_diagnostics.py',
        'src/pdm_audit/kaist.py','experiments/results/kaist_features.csv',
        'experiments/results/kaist_predictive.json','experiments/results/kaist_predictions.csv']
    result = dict(scope='Exploratory explanation of frozen one-rig load controls; not calibrated or causal shortcut detection',
        output='P(BPFI)+P(BPFO); multiclass argmax collapse and probability >=0.5 reported separately',
        features=names,results=rows,profiles=profiles,seed_results=seed_rows,
        seed_scope='Twenty algorithm seeds on fixed recordings per coverage set; no independent-asset inference',
        max_additivity_error=max_error,
        input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs})
    (OUT/'kaist_diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
