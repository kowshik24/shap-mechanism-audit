"""
Stage 9: the primary audit against a null whose models are re-tuned on each permutation.

The primary null carries over the hyperparameters selected on the real labels. Here every
permuted-label pipeline repeats the full selection step of stage 1 (randomized search on the
training split, Borderline-SMOTE inside the CV folds) before it is cross-fitted and explained, so the
null matches the procedure that produced the audited model. Draw k uses the same permutation as draw k
of the primary null, so the first K primary draws are a paired carry-over reference.

Usage: python experiments/09_retuned_null.py [--quick] [--K 200]
"""
import argparse
import json
import time
import warnings

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pdm_audit.audit import holm, permutation_p, shap_cross_fitted, statistics
from pdm_audit.config import ENGINEERED, PHYSICAL, RESULTS_DIR, SEED
from pdm_audit.data import load, mechanism_groups, split
from pdm_audit.models import make_preprocessor, tune_xgb

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true', help='tiny K for a smoke test')
ap.add_argument('--K', type=int, default=200)
args = ap.parse_args()
K = 2 if args.quick else args.K
STATS = ('R', 'D', 'S')  # 'S' is the shortcut statistic C_m in the paper

df = load()
y = df['Machine failure']
idx_train, _, _ = split(df)
groups = mechanism_groups(df)
primary = json.loads((RESULTS_DIR / 'audit.json').read_text())['primary']


def draw(k):
    warnings.filterwarnings('ignore')  # F1 is often undefined in CV folds of permuted labels
    t0 = time.time()
    rng = np.random.default_rng(SEED * 100_003 + k)  # same permutation as primary draw k
    perm = pd.Series(rng.permutation(y.values), index=y.index)
    pre = make_preprocessor(ENGINEERED)
    X_train = pre.fit_transform(df.loc[idx_train, ENGINEERED + ['Type']])
    params, cv_f1 = tune_xgb(X_train, perm.loc[idx_train].values)
    A = shap_cross_fitted(df, ENGINEERED, params, SEED, labels=perm)
    return {'stats': statistics(A, df, groups), 'params': params, 'cv_f1': cv_f1, 'seconds': time.time() - t0}


t0 = time.time()
draws = Parallel(n_jobs=-1, verbose=5)(delayed(draw)(k) for k in range(K))
observed = primary['observed']['all']

out = {'K': K, 'seconds': time.time() - t0, 'params': [d['params'] for d in draws],
       'cv_f1': [d['cv_f1'] for d in draws]}
for name, null_of in [('retuned', lambda m, s: [d['stats'][m][s] for d in draws]),
                      ('carry_over_paired', lambda m, s: primary['all']['null_draws'][f'{m}.{s}'][:K])]:
    p = {f'{m}.{s}': permutation_p(observed[m][s], null_of(m, s)) for m in PHYSICAL for s in STATS}
    out[name] = {
        'p': p,
        **{f'p_holm_{s}': holm({m: p[f'{m}.{s}'] for m in PHYSICAL}) for s in STATS},
        'null_mean': {f'{m}.{s}': float(np.mean(null_of(m, s))) for m in PHYSICAL for s in STATS},
        'null_q95': {f'{m}.{s}': float(np.quantile(null_of(m, s), 0.95)) for m in PHYSICAL for s in STATS},
        'null_draws': {f'{m}.{s}': list(map(float, null_of(m, s))) for m in PHYSICAL for s in STATS},
    }
    print(name, {s: {m: round(v, 3) for m, v in out[name][f'p_holm_{s}'].items()} for s in STATS})

name = 'retuned_null_quick.json' if args.quick else 'retuned_null.json'
(RESULTS_DIR / name).write_text(json.dumps(out, indent=1))
print('saved', RESULTS_DIR / name, f'{out["seconds"]:.0f}s')
