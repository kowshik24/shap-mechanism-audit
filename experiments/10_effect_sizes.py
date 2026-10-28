"""
Stage 10: effect sizes and confidence intervals for the primary audit statistics.

  effect size   z = (T - null mean) / null s.d., from the K = 1000 permuted-label draws of stage 2
  CI            95% percentile bootstrap of R_m and D_m: the cross-fitted |SHAP| matrix of the primary
                audit is recomputed (same seed, same folds) and rows are resampled with replacement
                within S_m and within F minus S_m. The models are held fixed, so the interval covers
                row sampling only, not model refitting.

Usage: python experiments/10_effect_sizes.py [--B 5000]
"""
import argparse
import json

import pdm_audit  # noqa: F401
import numpy as np

from pdm_audit.audit import shap_cross_fitted, statistics
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED, TRIGGER_FEATURES
from pdm_audit.data import load, mechanism_groups

ap = argparse.ArgumentParser()
ap.add_argument('--B', type=int, default=5000)
args = ap.parse_args()

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
primary = json.loads((RESULTS_DIR / 'audit.json').read_text())['primary']
groups = mechanism_groups(df)
failures = df.index[df['Machine failure'] == 1]

A = shap_cross_fitted(df, ENGINEERED, params, SEED)
observed = statistics(A, df, groups)
for m in PHYSICAL:  # the recomputed matrix must reproduce the stage-2 statistics exactly
    for s in ('R', 'D', 'S'):
        assert np.isclose(observed[m][s], primary['observed']['all'][m][s]), (m, s)

rng = np.random.default_rng(SEED)
out = {'B': args.B, 'K': primary['K'], 'mechanisms': {}}
for m in PHYSICAL:
    own, rest = A.loc[groups[m]].to_numpy(), A.loc[np.setdiff1d(failures, groups[m])].to_numpy()
    trig = np.isin(A.columns, TRIGGER_FEATURES[m])
    share = lambda X: X[:, trig].sum() / X.sum()  # ratio of column means = ratio of sums, same n
    R, D = np.empty(args.B), np.empty(args.B)
    for b in range(args.B):
        o = own[rng.integers(0, len(own), len(own))]
        r = rest[rng.integers(0, len(rest), len(rest))]
        R[b], D[b] = share(o), share(o) - share(r)
    entry = {}
    for s, boot in (('R', R), ('D', D)):
        null = np.asarray(primary['all']['null_draws'][f'{m}.{s}'])
        entry[s] = {'observed': observed[m][s], 'ci95': [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                    'null_mean': float(null.mean()), 'null_sd': float(null.std(ddof=1)),
                    'null_q95': float(np.quantile(null, 0.95)),
                    'z': float((observed[m][s] - null.mean()) / null.std(ddof=1)),
                    'excess_over_null_mean': float(observed[m][s] - null.mean())}
    out['mechanisms'][m] = entry
    print(m, {s: (round(e['observed'], 3), [round(c, 3) for c in e['ci95']], round(e['z'], 1)) for s, e in entry.items()})

(RESULTS_DIR / 'effect_sizes.json').write_text(json.dumps(out, indent=1))
print('saved', RESULTS_DIR / 'effect_sizes.json')
