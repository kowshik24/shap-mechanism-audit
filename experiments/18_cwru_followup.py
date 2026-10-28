"""Prespecified feature-view and held-out-diameter CWRU follow-up."""
import pdm_audit
import importlib.util
import json
from pathlib import Path
import numpy as np
from pdm_audit.cwru import EXPECTED, load_features

ROOT = Path(__file__).resolve().parents[1]
# Reuse the initial study's estimator and recording metric definitions exactly.
spec = importlib.util.spec_from_file_location('cwru_initial', Path(__file__).with_name('12_cwru.py'))
initial = importlib.util.module_from_spec(spec); spec.loader.exec_module(initial)


def evaluate(df, cols, axis, gain):
    X = df[cols].copy()
    if gain and 'log_RMS' in cols:
        X['log_RMS'] += df.fault.map(lambda m: np.log(initial.GAINS[m]))
    groups = sorted(df[axis].unique())
    probabilities = np.zeros((len(df),3)); shifted = np.zeros_like(probabilities)
    folds = []
    for group in groups:
        tr = df.index[df[axis] != group].to_numpy(); te = df.index[df[axis] == group].to_numpy()
        assert set(df.loc[tr,'recording']).isdisjoint(df.loc[te,'recording'])
        scores = []
        for depth in [3,6,None]:
            values = []
            for inner in sorted(df.loc[tr,axis].unique()):
                a = tr[df.loc[tr,axis].to_numpy() != inner]
                b = tr[df.loc[tr,axis].to_numpy() == inner]
                est = initial.model(depth).fit(X.loc[a], df.loc[a,'fault'])
                values.append(initial.metrics(df.loc[b,'fault'], est.predict_proba(X.loc[b]), df.loc[b])['recording_macro_f1'])
            scores.append(float(np.mean(values)))
        depth = [3,6,None][int(np.argmax(scores))]
        est = initial.model(depth).fit(X.loc[tr],df.loc[tr,'fault'])
        probabilities[te] = est.predict_proba(X.loc[te])
        shifted[te] = est.predict_proba(df.loc[te,cols])
        folds.append({'held_out_group': float(group), 'depth': depth, 'inner_scores': scores,
                      'training_recordings': sorted(set(df.loc[tr,'recording'])),
                      'test_recordings': sorted(set(df.loc[te,'recording']))})
    return {'matched': initial.metrics(df.fault, probabilities, df),
            'gain_removed': initial.metrics(df.fault, shifted, df), 'folds': folds}


def main():
    df, inventory = load_features(ROOT/'data/cwru')
    diameter = {r['recording']:r['diameter_in'] for r in inventory}
    df['diameter_in'] = df.recording.map(diameter)
    all_cols = [c for c in df if c not in ['recording','fault','load_hp','window','diameter_in']]
    anchors = [f for values in EXPECTED.values() for f in values]
    views = {'all':all_cols,'anchors':anchors,'broad':[f for f in all_cols if f not in anchors], 'RMS':['log_RMS']}
    records = []
    for axis in ['load_hp','diameter_in']:
        for view, cols in views.items():
            for gain in [False,True]:
                result = evaluate(df,cols,axis,gain)
                result.update(axis=axis,view=view,gain_correlated=gain,features=cols)
                records.append(result)
                print(axis,view,gain,result['matched']['recording_macro_f1'],result['gain_removed']['recording_macro_f1'],flush=True)
    old = json.loads((ROOT/'experiments/results/cwru.json').read_text())
    for row in records:
        if row['axis'] == 'load_hp' and row['view'] == 'all':
            expected = old['conditions']['gain_correlated' if row['gain_correlated'] else 'original']
            assert row['matched'] == expected['matched']
            assert row['gain_removed'] == expected['gain_removed']
        if row['view'] == 'anchors':
            assert row['matched'] == row['gain_removed']
    out = {'complete':True,'protocol':'experiments/CWRU_FOLLOWUP_PROTOCOL.md',
           'followup_after_initial_results':True,'records':records}
    (ROOT/'experiments/results/cwru_followup.json').write_text(json.dumps(out,indent=1,allow_nan=False))


if __name__ == '__main__':
    main()
