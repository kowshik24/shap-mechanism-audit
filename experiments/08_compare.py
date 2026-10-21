"""
Stage 8: the audit-design combinations missing from the comparison table.

  naive_shortcut     naive null (random failure rows, same model) on generated data with the shortcut
                     sensor (n = 300,000, sigma = 0.25, seed 0): does it flag the shortcut?
  in_sample_permuted in-sample SHAP with the permuted-label null, audited model trained on permuted
                     labels (the same permutation as stage 2): does it reject a model that learned nothing?
  in_sample_shortcut in-sample SHAP with the permuted-label null on the same shortcut data: does it
                     flag the shortcut?

The shortcut is flagged when the HDF relevant-share test fails (p_R >= 0.05) or the shortcut score
fires (p_S < 0.05); the naive null has only the relevant-share test.

Usage: python experiments/08_compare.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd

from pdm_audit.audit import holm, naive_null_p, run_audit, shap_cross_fitted
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED
from pdm_audit.data import load, mechanism_groups
from pdm_audit.sim import simulate

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
N, SIGMA, K = (2_000, 0.25, 4) if args.quick else (300_000, 0.25, 200)

params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
results = {'n': N, 'sigma': SIGMA, 'K': K}

# shortcut data exactly as in stage 7 (seed 0)
sim = simulate(N, seed=0)
sim['Shortcut sensor'] = sim['HDF'] + np.random.default_rng(0).normal(0, SIGMA, len(sim))
cols = ENGINEERED + ['Shortcut sensor']

t0 = time.time()
A = shap_cross_fitted(sim, cols, params, 0)
p = naive_null_p(A, sim, mechanism_groups(sim))
results['naive_shortcut'] = {'p': p, 'p_holm_R': holm(p), 'seconds': time.time() - t0}
print('naive null, shortcut data:', {m: round(v, 4) for m, v in holm(p).items()}, flush=True)


def summary(res):
    a = res['all']
    return {'observed': res['observed']['all'], 'p_holm_R': a['p_holm_R'], 'p_holm_D': a['p_holm_D'],
            'p_holm_S': a['p_holm_S'], 'null_mean': a['null_mean']}


df = load()
y = df['Machine failure']
perm = pd.Series(np.random.default_rng(7).permutation(y.values), index=y.index)
t0 = time.time()
res = run_audit(df, numeric_cols=ENGINEERED, params=params, seed=SEED, K=K, cross_fit=False,
                train_idx=df.index, observed_labels=perm)
results['in_sample_permuted'] = {**summary(res), 'seconds': time.time() - t0}
print('in-sample, permuted labels:', {s: {m: round(results['in_sample_permuted'][f'p_holm_{s}'][m], 3)
                                          for m in PHYSICAL} for s in 'RDS'}, flush=True)

t0 = time.time()
res = run_audit(sim, numeric_cols=cols, params=params, seed=0, K=K, cross_fit=False, train_idx=sim.index,
                extra_features=('Shortcut sensor',))
results['in_sample_shortcut'] = {**summary(res), 'seconds': time.time() - t0}
print('in-sample, shortcut data:', {s: {m: round(results['in_sample_shortcut'][f'p_holm_{s}'][m], 3)
                                        for m in PHYSICAL} for s in 'RDS'}, flush=True)

out = 'compare_quick.json' if args.quick else 'compare.json'
(RESULTS_DIR / out).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / out)
