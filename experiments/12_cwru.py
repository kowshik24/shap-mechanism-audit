"""Independent recorded-signal study; see SCIENTIFIC_GAPS_PROTOCOL.md."""
import pdm_audit  # thread settings before numerical imports
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pdm_audit.cwru import EXPECTED, load_features, recording_shares
from pdm_audit.validation import disjoint_feature_decoys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/cwru.json'
CLASSES = ['B', 'IR', 'OR']
GAINS = {'IR': .25, 'B': 1., 'OR': 4.}


def model(depth):
    return RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=3,
                                  class_weight='balanced', random_state=2026, n_jobs=1)


def metrics(y, p, meta):
    votes = pd.DataFrame(p, columns=CLASSES).assign(recording=meta['recording'].to_numpy()).groupby('recording').mean()
    truth = meta.groupby('recording')['fault'].first().loc[votes.index]
    pred = votes.idxmax(axis=1)
    return {'window_macro_f1': float(f1_score(y, np.array(CLASSES)[p.argmax(axis=1)], average='macro')),
            'recording_accuracy': float(accuracy_score(truth, pred)),
            'recording_macro_f1': float(f1_score(truth, pred, average='macro')),
            'confusion': confusion_matrix(truth, pred, labels=CLASSES).tolist(),
            'recordings': [{'id': k, 'truth': str(truth[k]), 'prediction': str(pred[k]),
                            'probabilities': votes.loc[k].to_dict()} for k in votes.index]}


def main():
    started = time.time()
    df, inventory = load_features(ROOT/'data/cwru')
    cols = [c for c in df if c not in ['recording', 'load_hp', 'fault', 'window']]
    results = {}
    for condition in ['original', 'gain_correlated']:
        X = df[cols].copy()
        if condition == 'gain_correlated':
            X['log_RMS'] += df['fault'].map(lambda m: np.log(GAINS[m]))
        folds, probabilities, shifted_probabilities = [], np.zeros((len(df), 3)), np.zeros((len(df), 3))
        attrs = {m: np.zeros((len(df), len(cols))) for m in CLASSES}
        shifted_attrs = {m: np.zeros((len(df), len(cols))) for m in CLASSES}
        for test_load in range(4):
            tr = df.index[df.load_hp != test_load].to_numpy(); te = df.index[df.load_hp == test_load].to_numpy()
            assert set(df.loc[tr, 'recording']).isdisjoint(df.loc[te, 'recording'])
            scores = []
            for depth in [3, 6, None]:
                fold_scores = []
                for val_load in sorted(set(df.loc[tr, 'load_hp'])):
                    it = tr[df.loc[tr, 'load_hp'].to_numpy() != val_load]
                    iv = tr[df.loc[tr, 'load_hp'].to_numpy() == val_load]
                    est = model(depth).fit(X.loc[it], df.loc[it, 'fault'])
                    fold_scores.append(metrics(df.loc[iv, 'fault'], est.predict_proba(X.loc[iv]), df.loc[iv])['recording_macro_f1'])
                scores.append(float(np.mean(fold_scores)))
            depth = [3, 6, None][int(np.argmax(scores))]
            est = model(depth).fit(X.loc[tr], df.loc[tr, 'fault'])
            assert list(est.classes_) == CLASSES
            probabilities[te] = est.predict_proba(X.loc[te])
            clean = df.loc[te, cols]
            shifted_probabilities[te] = est.predict_proba(clean)
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(X.loc[te], check_additivity=True)
            shifted = explainer.shap_values(clean, check_additivity=True)
            assert sv.shape == (len(te), len(cols), 3)
            for c, m in enumerate(CLASSES):
                attrs[m][te] = np.abs(sv[:, :, c]); shifted_attrs[m][te] = np.abs(shifted[:, :, c])
            folds.append({'test_load_hp': test_load, 'selected_depth': depth,
                          'inner_recording_macro_f1': dict(zip(['3', '6', 'None'], scores)),
                          'train_recordings': sorted(set(df.loc[tr, 'recording'])),
                          'test_recordings': sorted(set(df.loc[te, 'recording']))})
            print(condition, test_load, depth, flush=True)
        item = {'folds': folds, 'matched': metrics(df.fault, probabilities, df),
                'gain_removed': metrics(df.fault, shifted_probabilities, df), 'audit': {}, 'gain_removed_audit': {}}
        for m in CLASSES:
            A = pd.DataFrame(attrs[m], columns=cols)
            item['audit'][m] = recording_shares(A, df, m)
            # Every recording has the same number of windows, so row/recording means agree here.
            item['audit'][m]['decoys'] = disjoint_feature_decoys(A, df.index[df.fault == m], EXPECTED[m])
            item['gain_removed_audit'][m] = recording_shares(pd.DataFrame(shifted_attrs[m], columns=cols), df, m)
        results[condition] = item
    # Recording bootstrap describes this fixed inventory, not independent physical bearings.
    rng = np.random.default_rng(20261022)
    for condition, item in results.items():
        matched, removed = item['matched']['recordings'], item['gain_removed']['recordings']
        drops = []
        for _ in range(10000):
            sample = np.concatenate([rng.choice([i for i, r in enumerate(matched) if r['truth'] == m], 12)
                                     for m in CLASSES])
            y = [matched[i]['truth'] for i in sample]
            drops.append(f1_score(y, [matched[i]['prediction'] for i in sample], average='macro') -
                         f1_score(y, [removed[i]['prediction'] for i in sample], average='macro'))
        item['recording_macro_f1_drop_bootstrap_95'] = np.quantile(drops, [.025, .975]).tolist()
    out = {'complete': True, 'task': 'fault location diagnosis among seeded faults',
           'protocol': 'experiments/SCIENTIFIC_GAPS_PROTOCOL.md', 'inventory': inventory,
           'feature_names': cols, 'expected_features': EXPECTED, 'conditions': results,
           'seconds': time.time()-started, 'limitations': ['Physical bearing identity unknown',
           'Spectral anchor maps are not exhaustive causal truth', 'Four loads on one test rig',
           'Gain confound is constructed, not discovered', 'Bootstrap conditions on fixed recordings']}
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False))
    print('saved', OUT, 'seconds', out['seconds'], flush=True)


if __name__ == '__main__':
    main()
