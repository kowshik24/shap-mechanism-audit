"""
Stage 15: the mechanism audit on Tennessee Eastman process simulations (DTU TEP, mode 1).

400 complete 100-hour trajectories, 200 per fault: IDV4 (step in the reactor cooling-water inlet
temperature) and IDV5 (step in the condenser cooling-water inlet temperature), injected at 30 h. From
each run we take 16 samples at 10-25 h (normal operation) and 16 at 40-95 h (fault active), so the
overall label y is "fault active" and the mechanism flag is the injected fault. Expected features
E_m are the measurements and manipulated variables of the disturbed cooling loop (the partial maps of
the paper/new-experiments branch).

XGBoost (300 trees, depth 6) is cross-fitted with whole runs held out (5 stratified group folds) and
explained with path-dependent TreeSHAP; R, D and C follow the AI4I audit with F = all fault-active
samples. The null permutes y within each run (each run keeps its 16/16 split, so run structure and
class balance are preserved), refits and re-explains, and keeps the true mechanism groups. With
--null shift, each run's time-ordered label sequence is instead rotated by a random offset, which keeps
the labels in one contiguous fault-active block and so preserves their temporal dependence.

Temporal context: with --window W, each sample also carries every variable's mean over the last W
samples and its change over them (3 x 53 inputs); a variable's attribution is the absolute sum of its
three signed SHAP values, so the statistics stay on the 53 variables.

Usage: python experiments/15_tep_audit.py [--quick] [--window 10] [--null shift]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

from pdm_audit.audit import holm, permutation_p
from pdm_audit.config import RESULTS_DIR, ROOT, SEED

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--window', type=int, default=0, help='samples of history (3 min each); 0 = none')
ap.add_argument('--null', choices=['within', 'shift'], default='within',
                help='within-run label permutation or within-run circular shift')
args = ap.parse_args()
W = args.window
K = 4 if args.quick else 200

LABELS = ['A Feed', 'D Feed', 'E Feed', 'A and C Feed', 'Recycle Flow', 'Reactor Feed Rate', 'Reactor Pressure',
          'Reactor Level', 'Reactor Temperature', 'Purge Rate', 'Product Sep Temp', 'Product Sep Level',
          'Product Sep Pressure', 'Product Sep Underflow', 'Stripper Level', 'Stripper Pressure',
          'Stripper Underflow', 'Stripper Temp', 'Stripper Steam Flow', 'Compressor Work', 'Reactor Coolant Temp',
          'Separator Coolant Temp'] + \
         [f'Component {c} to Reactor' for c in 'ABCDEF'] + [f'Component {c} in Purge' for c in 'ABCDEFGH'] + \
         [f'Component {c} in Product' for c in 'DEFGH'] + \
         ['D Feed (MV)', 'E Feed (MV)', 'A Feed (MV)', 'A and C Feed (MV)', 'Recycle (MV)', 'Purge (MV)',
          'Separator (MV)', 'Stripper (MV)', 'Steam (MV)', 'Reactor Coolant (MV)', 'Condenser Coolant (MV)',
          'Agitator (MV)']
assert len(LABELS) == 53
EXPECTED = {'IDV4': ['Reactor Temperature', 'Reactor Coolant Temp', 'Reactor Coolant (MV)'],
            'IDV5': ['Product Sep Temp', 'Separator Coolant Temp', 'Condenser Coolant (MV)']}
MECH = list(EXPECTED)

# ---- data: 32 samples per run
rows = []
for fault in (4, 5):
    for run in range(1, 201):
        full = np.fromfile(ROOT / 'data' / 'dtu_tep' / 'prospective_runs' / f'{fault}_{run:03d}.bin', dtype='<f8')
        full = full.reshape(2001, 54)
        idx = np.abs(full[:, 0, None] - np.r_[np.linspace(10, 25, 16), np.linspace(40, 95, 16)]).argmin(axis=0)
        for i in idx:
            history = [] if not W else [*full[i - W + 1:i + 1, 1:].mean(axis=0), *(full[i, 1:] - full[i - W, 1:])]
            rows.append([fault, run, full[i, 0], *full[i, 1:], *history])
INPUTS = LABELS + ([] if not W else [f'{v} (mean)' for v in LABELS] + [f'{v} (change)' for v in LABELS])
df = pd.DataFrame(rows, columns=['fault', 'run', 'hour', *INPUTS])
df['group'] = df['fault'] * 1000 + df['run']
y = (df['hour'] > 30).astype(int).to_numpy()
mech = {f'IDV{f}': ((df['fault'] == f) & (y == 1)).to_numpy() for f in (4, 5)}
X = df[INPUTS].to_numpy()
groups = df['group'].to_numpy()
fault_of_row = df['fault'].to_numpy()


def cross_fit(labels, seed=SEED):
    """Held-out |SHAP| of the fault-active samples (others zero) and held-out probabilities."""
    A, proba = np.zeros((len(X), len(LABELS))), np.zeros(len(X))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, fault_of_row, groups):
        model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                                  random_state=seed, n_jobs=1).fit(X[tr], labels[tr])
        proba[te] = model.predict_proba(X[te])[:, 1]
        te_f = te[y[te] == 1]
        sv = shap.TreeExplainer(model).shap_values(X[te_f])
        A[te_f] = np.abs(sv.reshape(len(te_f), -1, len(LABELS)).sum(axis=1))  # per variable
    return A, proba


def statistics(A):
    out = {}
    for m in MECH:
        own, rest = A[mech[m]].mean(axis=0), A[(y == 1) & ~mech[m]].mean(axis=0)
        a, b = pd.Series(own / own.sum(), LABELS), pd.Series(rest / rest.sum(), LABELS)
        excess = (a - b).drop(EXPECTED[m])
        out[m] = {'R': float(a[EXPECTED[m]].sum()), 'D': float(a[EXPECTED[m]].sum() - b[EXPECTED[m]].sum()),
                  'S': float(excess.max()), 'S_feature': str(excess.idxmax()), 'top_feature': str(a.idxmax())}
    return out


def null_draw(k):
    rng = np.random.default_rng(SEED * 100_003 + k)
    perm = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)  # rows of a run are in time order
        perm[idx] = np.roll(y[idx], rng.integers(len(idx))) if args.null == 'shift' else rng.permutation(y[idx])
    return statistics(cross_fit(perm)[0])


t0 = time.time()
A, proba = cross_fit(y)
obs = statistics(A)
pred = (proba >= 0.5).astype(int)
tp, fp, fn = ((pred == 1) & (y == 1)).sum(), ((pred == 1) & (y == 0)).sum(), ((pred == 0) & (y == 1)).sum()
draws = Parallel(n_jobs=-1)(delayed(null_draw)(k) for k in range(K))
p = {f'{m}.{s}': permutation_p(obs[m][s], [d[m][s] for d in draws]) for m in MECH for s in 'RDS'}
results = {'K': K, 'window': W, 'null': args.null, 'n_samples': len(df), 'n_runs': 400, 'observed': obs,
           'f1': float(2 * tp / (2 * tp + fp + fn)), 'recall_by_mechanism': {m: float(pred[mech[m]].mean()) for m in MECH},
           'false_positive_rate': float(fp / (y == 0).sum()),
           **{f'p_holm_{s}': holm({m: p[f'{m}.{s}'] for m in MECH}) for s in 'RDS'},
           'null_mean': {f'{m}.{s}': float(np.mean([d[m][s] for d in draws])) for m in MECH for s in 'RDS'},
           'seconds': time.time() - t0}
print(f'F1={results["f1"]:.3f} recall={results["recall_by_mechanism"]} FPR={results["false_positive_rate"]:.3f} '
      f'({results["seconds"]:.0f}s)')
for m in MECH:
    print(f'  {m}: R={obs[m]["R"]:.3f} (null {results["null_mean"][m + ".R"]:.3f}, p={results["p_holm_R"][m]:.3f})  '
          f'D={obs[m]["D"]:.3f} (p={results["p_holm_D"][m]:.3f})  C={obs[m]["S"]:.3f} {obs[m]["S_feature"]} '
          f'(p={results["p_holm_S"][m]:.3f})  top={obs[m]["top_feature"]}')
suffix = (f'_window{W}' if W else '') + ('_shift' if args.null == 'shift' else '')
name = 'tep_audit_quick.json' if args.quick else f'tep_audit{suffix}.json'
(RESULTS_DIR / name).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / name)
