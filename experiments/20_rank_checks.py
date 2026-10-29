"""
Stage 20: rank-based alignment statistics and a null-referenced shortcut score.

  Rk_m, Dk_m   rank analogues of R_m and D_m: for each observation, the probability that a trigger
               feature outranks a non-trigger feature in |SHAP|, averaged over S_m (Rk) and compared
               with the other failures (Dk); a missing trigger costs one comparison, not its whole share
  N_m          each non-trigger feature's share on S_m, standardized by its own null mean and s.d.,
               maximum over features; unlike C_m it is not compared with other failures

Run on the primary audit, the two misspecified trigger sets of stage 16, and the planted shortcut of
stage 3 at sigma 0.4, 0.35, 0.3, 0.25, each against a K-model null.

With --transform positive every audit uses only the failure-pushing part of SHAP, max(SHAP, 0): a
shortcut that pushes other failures away from the failure class then no longer inflates their share,
so C_m compares like with like (writes rank_checks_positive.json).

Usage: python experiments/20_rank_checks.py [--quick] [--transform positive]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np

from pdm_audit.audit import run_audit
from pdm_audit.config import DERIVED, ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED, TRIGGER_FEATURES
from pdm_audit.data import load

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--transform', default='abs', choices=('abs', 'positive'))
args = ap.parse_args()
K = 4 if args.quick else 200
STATS = ('R', 'D', 'S', 'Rk', 'Dk', 'N')

base = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
mean_abs = json.loads((RESULTS_DIR / 'audit.json').read_text())['primary']['mean_abs_shap']
no_derived = {m: [f for f in TRIGGER_FEATURES[m] if f not in DERIVED] for m in PHYSICAL}
drop_top = {m: ([f for f in TRIGGER_FEATURES[m] if f != max(TRIGGER_FEATURES[m], key=lambda f: mean_abs[m][f])]
                if len(TRIGGER_FEATURES[m]) > 1 else TRIGGER_FEATURES[m]) for m in PHYSICAL}

variants = [('primary', base, ENGINEERED, None), ('triggers_no_derived', base, ENGINEERED, no_derived),
            ('triggers_drop_top', base, ENGINEERED, drop_top)]
for sigma in ((0.5, 0.4, 0.35, 0.3, 0.25) if args.transform == 'positive' else (0.4, 0.35, 0.3, 0.25)):
    df = base.copy()
    df['Shortcut sensor'] = df['HDF'] + np.random.default_rng(SEED).normal(0, sigma, len(df))  # as in stage 3
    variants.append((f'shortcut_sigma{sigma:g}', df, ENGINEERED + ['Shortcut sensor'], None))

results = {'K': K, 'transform': args.transform}
for name, df, cols, triggers in variants:
    t0 = time.time()
    res = run_audit(df, cols, params, K=K, triggers=triggers, transform=args.transform)
    a, o = res['all'], res['observed']['all']
    results[name] = {'observed': {m: {s: o[m][s] for s in ('R', 'D', 'S', 'S_feature', 'Rk', 'Dk', 'N', 'N_feature')}
                                  for m in PHYSICAL},
                     **{f'p_holm_{s}': a[f'p_holm_{s}'] for s in STATS},
                     'null_mean': {k: v for k, v in a['null_mean'].items() if k.split('.')[1] in STATS},
                     'seconds': time.time() - t0}
    print(name, f'{time.time() - t0:.0f}s', {s: {m: round(a[f'p_holm_{s}'][m], 3) for m in PHYSICAL} for s in STATS},
          {m: (round(o[m]['Rk'], 3), o[m]['N_feature']) for m in PHYSICAL}, flush=True)

suffix = '' if args.transform == 'abs' else f'_{args.transform}'
name = f'rank_checks{suffix}_quick.json' if args.quick else f'rank_checks{suffix}.json'
(RESULTS_DIR / name).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / name)
