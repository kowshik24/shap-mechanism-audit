"""Source-selected three-load pilot, using fixed features and no target tuning."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pdm_audit.kaist import CLASSES, fit_predict, metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def main():
    frame = pd.read_csv(OUT/'kaist_features.csv')
    names = [x for x in frame if x.startswith('ch')]
    assert len(names) == 40 and len(frame) == 630
    assert frame.groupby('recording').size().eq(30).all()
    families = {'logistic':[.1,1.,10.], 'forest':[4,8,None]}
    results, predictions = [], []
    for target in [0,2,4]:
        source = [x for x in [0,2,4] if x != target]
        for family, parameters in families.items():
            validation = []
            for parameter in parameters:
                directed = []
                for training_load, validation_load in [source,source[::-1]]:
                    train = frame[frame.load_Nm == training_load]
                    test = frame[frame.load_Nm == validation_load]
                    assert set(train.recording).isdisjoint(test.recording)
                    predicted, _, _ = fit_predict(train,test,names,family,parameter)
                    score = metrics(test,predicted)
                    directed.append(dict(training_load=training_load,validation_load=validation_load,metrics=score))
                    for (_, r), yhat in zip(test.iterrows(),predicted):
                        predictions.append(dict(outer_target_load=target,family=family,parameter=parameter,
                            role='source_validation',training_load=training_load,recording=r.recording,
                            load_Nm=r.load_Nm,window=int(r.window),label=int(r.label),prediction=int(yhat)))
                validation.append(dict(parameter=parameter,mean_macro_F1=float(np.mean([x['metrics']['macro_F1'] for x in directed])),directed=directed))
            chosen = max(range(len(validation)),key=lambda i:validation[i]['mean_macro_F1'])
            parameter = parameters[chosen]
            train = frame[frame.load_Nm.isin(source)]
            test = frame[frame.load_Nm == target].copy()
            assert set(train.recording).isdisjoint(test.recording)
            predicted, model, scaler = fit_predict(train,test,names,family,parameter)
            test['prediction'] = predicted
            vote = test.groupby('recording').agg(label=('label','first'),
                prediction=('prediction',lambda x:int(np.bincount(x,minlength=3).argmax()))).reset_index()
            results.append(dict(target_load_Nm=target,source_loads_Nm=source,family=family,
                selected_parameter=parameter,source_validation=validation,
                metrics=metrics(test,predicted),record_vote_metrics=metrics(vote,vote.prediction),
                training_records=int(train.recording.nunique()),target_records=int(test.recording.nunique()),
                training_windows=len(train),target_windows=len(test)))
            for _, r in test.iterrows():
                predictions.append(dict(outer_target_load=target,family=family,parameter=parameter,
                    role='target',training_load=None,recording=r.recording,load_Nm=r.load_Nm,
                    window=int(r.window),label=int(r.label),prediction=int(r.prediction)))
            print(target,family,'parameter',parameter,'macro F1',results[-1]['metrics']['macro_F1'],flush=True)
    pd.DataFrame(predictions).to_csv(OUT/'kaist_predictions.csv',index=False)
    paths = ['experiments/KAIST_PREDICTIVE_PROTOCOL.md','experiments/KAIST_LOAD_PILOT_CONTRACT.json',
             'experiments/94_kaist_predictive.py','experiments/93_kaist_features.py',
             'src/pdm_audit/kaist.py','experiments/results/kaist_features.csv',
             'experiments/results/kaist_signal_inventory.json','experiments/results/kaist_predictions.csv']
    record = dict(scope='Within-rig measured-load transfer; no independent-bearing confidence or conformal claim',
        classes=CLASSES,seed=20261026,features=names,results=results,
        input_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (OUT/'kaist_predictive.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
