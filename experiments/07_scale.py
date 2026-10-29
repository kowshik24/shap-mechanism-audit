"""
Stage 7: audit power and cost as the data grow.

Generates AI4I 2020 afresh at several sizes and seeds (pdm_audit.sim; no released row is copied),
plants the shortcut sensor s = HDF + N(0, sigma) (sigma = None: no sensor), and runs the
cross-fitted audit. Records R/D/S statistics with their null quantiles and Holm-adjusted p-values,
and times one uncontended cross-fitted run per size. Answers: how do the relevant-share test and the
shortcut test behave as n grows, and what does an audit cost?

Usage: python experiments/07_scale.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np

from pdm_audit.audit import run_audit, shap_cross_fitted
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR
from pdm_audit.sim import simulate

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
if args.quick:
    PLAN, SIGMAS, K, TIMING = [(2_000, [0])], [None, 0.25], 4, [2_000]
else:
    PLAN = [(10_000, [0, 1, 2]), (30_000, [0, 1, 2]), (100_000, [0, 1, 2]), (300_000, [0])]
    SIGMAS, K, TIMING = [None, 0.35, 0.25], 200, [10_000, 30_000, 100_000, 300_000, 1_000_000]

params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
rows = []
for n, seeds in PLAN:
    for seed in seeds:
        base = simulate(n, seed=seed)
        for sigma in SIGMAS:
            df, cols, extra = base, ENGINEERED, ()
            if sigma is not None:
                df = base.copy()
                df['Shortcut sensor'] = df['HDF'] + np.random.default_rng(seed).normal(0, sigma, len(df))
                cols, extra = ENGINEERED + ['Shortcut sensor'], ('Shortcut sensor',)
            t0 = time.time()
            res = run_audit(df, numeric_cols=cols, params=params, seed=seed, K=K, extra_features=extra)
            a, obs = res['all'], res['observed']['all']
            q = {f'{m}.{s}': np.quantile(a['null_draws'][f'{m}.{s}'], [0.5, 0.95]).tolist()
                 for m in PHYSICAL for s in ('R', 'D', 'S')}
            row = {'n': n, 'seed': seed, 'sigma': sigma, 'failures': int(df['Machine failure'].sum()),
                   'mode_counts': {m: int(df[m].sum()) for m in PHYSICAL},
                   'sensor_share_HDF': obs['HDF'].get('share[Shortcut sensor]'),
                   **{s: {m: obs[m][s] for m in PHYSICAL} for s in ('R', 'D', 'S')},
                   'S_feature': {m: obs[m]['S_feature'] for m in PHYSICAL},
                   'p_holm_R': a['p_holm_R'], 'p_holm_D': a['p_holm_D'], 'p_holm_S': a['p_holm_S'],
                   'null_quantiles_50_95': q, 'seconds': time.time() - t0}
            rows.append(row)
            print(f"n={n:<7} seed={seed} sigma={sigma!s:<5} share={row['sensor_share_HDF'] or 0:.3f} "
                  f"pR={ {m: round(v, 3) for m, v in a['p_holm_R'].items()} } "
                  f"pD={ {m: round(v, 3) for m, v in a['p_holm_D'].items()} } "
                  f"pS={ {m: round(v, 3) for m, v in a['p_holm_S'].items()} } ({row['seconds']:.0f}s)", flush=True)

# cost: one cross-fitted run (5 fits + held-out SHAP) on one core, nothing else running; an audit needs K+1
cost = []
for n in TIMING:
    df = simulate(n, seed=0)
    t0 = time.time()
    shap_cross_fitted(df, ENGINEERED, params, 0)
    cost.append({'n': n, 'failures': int(df['Machine failure'].sum()), 'seconds_per_run': time.time() - t0})
    print(f'n={n}: one cross-fitted run {cost[-1]["seconds_per_run"]:.0f}s', flush=True)

out = 'scale_quick.json' if args.quick else 'scale.json'
(RESULTS_DIR / out).write_text(json.dumps({'K': K, 'rows': rows, 'cost': cost}, indent=1))
print('saved', RESULTS_DIR / out)
