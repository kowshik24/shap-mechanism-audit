"""
Stage 7: audit power and cost as the data grow.

Re-simulates AI4I 2020 at several sizes (pdm_audit.sim), plants the shortcut sensor
s = HDF + N(0, sigma) at a few strengths (sigma = None: no sensor), and runs the cross-fitted audit.
Records the sensor's attribution share, the Holm-adjusted HDF verdicts, the verdicts of the other
mechanisms, and wall-clock cost. Answers: does a larger dataset let the audit detect lighter
reliance on a shortcut, and what does the audit cost at that size?

Usage: python experiments/07_scale.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np

from pdm_audit.audit import run_audit, shap_cross_fitted
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED
from pdm_audit.sim import simulate

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
SIZES = [10_000, 2_000] if args.quick else [10_000, 30_000, 100_000, 300_000]
COST_ONLY = [20_000] if args.quick else [1_000_000]  # time one cross-fitted run, no null
SIGMAS = [None, 0.25] if args.quick else [None, 0.5, 0.35, 0.25]
K = 4 if args.quick else 100

params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
rows = []
for n in SIZES:
    base = simulate(n, seed=SEED)
    for sigma in SIGMAS:
        df, cols, extra = base, ENGINEERED, ()
        if sigma is not None:
            df = base.copy()
            df['Shortcut sensor'] = df['HDF'] + np.random.default_rng(SEED).normal(0, sigma, len(df))
            cols, extra = ENGINEERED + ['Shortcut sensor'], ('Shortcut sensor',)
        t0 = time.time()
        res = run_audit(df, numeric_cols=cols, params=params, seed=SEED, K=K, extra_features=extra)
        a, obs = res['all'], res['observed']['all']
        row = {'n': n, 'sigma': sigma, 'failures': int(df['Machine failure'].sum()),
               'mode_counts': {m: int(df[m].sum()) for m in PHYSICAL},
               'sensor_share_HDF': obs['HDF'].get('share[Shortcut sensor]'),
               'R': {m: obs[m]['R'] for m in PHYSICAL}, 'D': {m: obs[m]['D'] for m in PHYSICAL},
               'p_holm_R': a['p_holm_R'], 'p_holm_D': a['p_holm_D'],
               'null_mean_R_HDF': a['null_mean']['HDF.R'], 'seconds': time.time() - t0}
        rows.append(row)
        print(f"n={n:<7} sigma={sigma!s:<5} share={row['sensor_share_HDF'] or 0:.3f} "
              f"pR={ {m: round(v, 3) for m, v in a['p_holm_R'].items()} } "
              f"pD={ {m: round(v, 3) for m, v in a['p_holm_D'].items()} } ({row['seconds']:.0f}s)", flush=True)

# cost of one cross-fitted SHAP run (5 fits + held-out SHAP); an audit needs K+1 of them
cost = []
for n in COST_ONLY:
    df = simulate(n, seed=SEED)
    t0 = time.time()
    shap_cross_fitted(df, ENGINEERED, params, SEED)
    cost.append({'n': n, 'failures': int(df['Machine failure'].sum()), 'seconds_per_run': time.time() - t0})
    print(f'n={n}: one cross-fitted run {cost[-1]["seconds_per_run"]:.0f}s', flush=True)
cost += [{'n': r['n'], 'failures': r['failures'], 'seconds_per_run': r['seconds'] / (K + 1) * 8}
         for r in rows if r['sigma'] is None]  # 8 parallel workers

out = 'scale_quick.json' if args.quick else 'scale.json'
(RESULTS_DIR / out).write_text(json.dumps({'K': K, 'rows': rows, 'cost': cost}, indent=1))
print('saved', RESULTS_DIR / out)
