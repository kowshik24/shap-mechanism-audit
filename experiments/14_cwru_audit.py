"""
Stage 14: the mechanism audit on measured CWRU bearing vibration.

A random forest classifies the fault location (IR, B, OR) of 4,096-sample windows from 36 drive-end
recordings (3 locations x 3 defect sizes x 4 motor loads). Each window is explained with
path-dependent TreeSHAP through the output of its own fault class, the analogue of explaining the
failure output on a failure row, and the audit computes R_m, D_m and C_m with S_m = windows of
recordings with fault m and E_m = the defect-frequency features of m; D_m then asks whether windows
of fault m rely on m's defect frequencies more than other windows do for their own diagnosis.
Cross-fitting holds out whole recordings (stratified group 4-fold), so no window is explained by a
model that saw its recording. The null permutes fault labels across recordings (windows keep their
recording's label), refits and re-explains, and keeps the true fault groups, as in the AI4I audit.

  original   the measured features
  gain       an acquisition shortcut: the recorded amplitude is scaled by a fault-dependent gain
             (x0.25 IR, x1 B, x4 OR), which shifts only log_RMS, a feature outside every E_m

Usage: python experiments/14_cwru_audit.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

from pdm_audit.audit import holm, permutation_p
from pdm_audit.config import RESULTS_DIR, ROOT, SEED
from pdm_audit.cwru import EXPECTED, FAULTS, FEATURES, load

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 1000
GAINS = {'IR': 0.25, 'B': 1.0, 'OR': 4.0}

cache = RESULTS_DIR / 'cwru_features.csv'
if cache.exists():
    df = pd.read_csv(cache)
else:
    df = load(ROOT / 'data' / 'cwru')
    df.to_csv(cache, index=False)
recordings = df['recording'].to_numpy()
truth = df['fault'].to_numpy()


def forest(seed):
    return RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=3, class_weight='balanced',
                                  random_state=seed, n_jobs=1)


def cross_fit(X, labels, seed=SEED):
    """Held-out |SHAP| of each class head (rows x features) and held-out class probabilities."""
    heads = {m: np.zeros(X.shape) for m in FAULTS}
    proba = np.zeros((len(X), len(FAULTS)))
    for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=seed).split(X, labels, recordings):
        rf = forest(seed).fit(X[tr], labels[tr])
        sv = np.asarray(shap.TreeExplainer(rf).shap_values(X[te]))  # (rows, features, classes)
        for c, m in enumerate(rf.classes_):
            heads[m][te] = np.abs(sv[:, :, c])
            proba[te, FAULTS.index(m)] = rf.predict_proba(X[te])[:, c]
    return heads, proba


def statistics(heads):
    own_head = pd.DataFrame(np.select([truth[:, None] == m for m in FAULTS], [heads[m] for m in FAULTS]),
                            columns=FEATURES)  # each window explained through its own fault's output
    out = {}
    for m in FAULTS:
        own, rest = own_head[truth == m].mean(), own_head[truth != m].mean()
        a, b = own / own.sum(), rest / rest.sum()
        excess = (a - b).drop(EXPECTED[m])
        out[m] = {'R': float(a[EXPECTED[m]].sum()), 'D': float(a[EXPECTED[m]].sum() - b[EXPECTED[m]].sum()),
                  'S': float(excess.max()), 'S_feature': str(excess.idxmax()),
                  'top_feature': str(a.idxmax()), 'share_log_RMS': float(a['log_RMS'])}
    return out


def null_draw(k, X):
    rng = np.random.default_rng(SEED * 100_003 + k)
    rec = pd.Series(truth, index=recordings).groupby(level=0).first()
    perm = dict(zip(rec.index, rng.permutation(rec.to_numpy())))
    heads, _ = cross_fit(X, np.array([perm[r] for r in recordings]))
    return statistics(heads)


results = {'K': K, 'n_windows': len(df), 'n_recordings': int(len(set(recordings)))}
for condition in ('original', 'gain'):
    t0 = time.time()
    Xdf = df[FEATURES].copy()
    if condition == 'gain':
        Xdf['log_RMS'] += np.log(df['fault'].map(GAINS))
    X = Xdf.to_numpy()
    heads, proba = cross_fit(X, truth)
    obs = statistics(heads)
    rec_pred = pd.DataFrame(proba, columns=FAULTS).groupby(recordings).mean().idxmax(axis=1)
    rec_true = pd.Series(truth, index=recordings).groupby(level=0).first()
    window_acc = float((np.array(FAULTS)[proba.argmax(axis=1)] == truth).mean())
    draws = Parallel(n_jobs=-1)(delayed(null_draw)(k, X) for k in range(K))
    p = {f'{m}.{s}': permutation_p(obs[m][s], [d[m][s] for d in draws]) for m in FAULTS for s in 'RDS'}
    results[condition] = {
        'observed': obs, 'window_accuracy': window_acc,
        'recording_accuracy': float((rec_pred.loc[rec_true.index] == rec_true).mean()),
        **{f'p_holm_{s}': holm({m: p[f'{m}.{s}'] for m in FAULTS}) for s in 'RDS'},
        'null_mean': {f'{m}.{s}': float(np.mean([d[m][s] for d in draws])) for m in FAULTS for s in 'RDS'},
        'null_q95': {f'{m}.{s}': float(np.quantile([d[m][s] for d in draws], 0.95)) for m in FAULTS for s in 'RDS'},
        'seconds': time.time() - t0}
    r = results[condition]
    print(condition, f'{r["seconds"]:.0f}s acc(window/recording)={window_acc:.3f}/{r["recording_accuracy"]:.3f}')
    for m in FAULTS:
        print(f'  {m}: R={obs[m]["R"]:.3f} (null {r["null_mean"][m + ".R"]:.3f}, p={r["p_holm_R"][m]:.3f})  '
              f'D={obs[m]["D"]:.3f} (p={r["p_holm_D"][m]:.3f})  C={obs[m]["S"]:.3f} {obs[m]["S_feature"]} '
              f'(p={r["p_holm_S"][m]:.3f})  logRMS share={obs[m]["share_log_RMS"]:.3f}', flush=True)

name = 'cwru_audit_quick.json' if args.quick else 'cwru_audit.json'
(RESULTS_DIR / name).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / name)
