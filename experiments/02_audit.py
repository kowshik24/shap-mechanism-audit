"""
Stage 2: the SHAP failure-mechanism audit and its robustness variants.

  primary       cross-fitted (out-of-sample SHAP), seed 42, K permuted-label models
  seeds         the primary analysis repeated with other seeds (smaller K)
  raw           raw sensor features only (circularity check)
  in_sample     one model fitted on all rows explains those same rows (fully in-sample SHAP)
  permuted      a model trained on permuted labels (sanity check only: exchangeable with the null)

Usage: python experiments/02_audit.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd

from pdm_audit.audit import run_audit
from pdm_audit.config import ENGINEERED, MODELS_DIR, RAW_NUMERIC, RESULTS_DIR, SEED
from pdm_audit.data import load

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true', help='tiny K for a smoke test')
args = ap.parse_args()
K_MAIN, K_ROBUST = (4, 4) if args.quick else (1000, 200)
SEEDS = [1, 2, 3, 4]

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
results = {'K_main': K_MAIN, 'K_robust': K_ROBUST}


def timed(name, **kw):
    t0 = time.time()
    res = run_audit(df, **kw)
    a = res['all']
    print(f'{name:<24} {time.time() - t0:6.0f}s  p(R)={ {m: round(v, 4) for m, v in a["p_holm_R"].items()} }'
          f'  p(D)={ {m: round(v, 4) for m, v in a["p_holm_D"].items()} }', flush=True)
    res['seconds'] = time.time() - t0
    return res


results['primary'] = timed('primary', numeric_cols=ENGINEERED, params=params, seed=SEED, K=K_MAIN)
results['seeds'] = {str(s): timed(f'seed {s}', numeric_cols=ENGINEERED, params=params, seed=s, K=K_ROBUST)
                    for s in SEEDS}
results['raw'] = timed('raw features', numeric_cols=RAW_NUMERIC, params=params, seed=SEED, K=K_MAIN)
results['in_sample'] = timed('in-sample', numeric_cols=ENGINEERED, params=params, seed=SEED, K=K_MAIN,
                             cross_fit=False, train_idx=df.index)
y = df['Machine failure']
perm = pd.Series(np.random.default_rng(7).permutation(y.values), index=y.index)
results['permuted'] = timed('permuted labels', numeric_cols=ENGINEERED, params=params, seed=SEED,
                            K=K_ROBUST, observed_labels=perm)

name = 'audit_quick.json' if args.quick else 'audit.json'
(RESULTS_DIR / name).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / name)
