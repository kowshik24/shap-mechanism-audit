"""
Stage 6: explainer and model variants of the primary audit.

  interventional   XGBoost, TreeSHAP with interventional feature perturbation (background of 100
                   training rows per fold) instead of the path-dependent default
  random_forest    random forest (300 trees, depth 12) in place of XGBoost, path-dependent TreeSHAP

Usage: python experiments/06_explainers.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401

from pdm_audit.audit import run_audit
from pdm_audit.config import ENGINEERED, MODELS_DIR, RESULTS_DIR, SEED
from pdm_audit.data import load

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 200

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
results = {'K': K}
for name, kw in [('interventional', {'explainer': 'interventional'}),
                 ('random_forest', {'model': 'rf'})]:
    t0 = time.time()
    res = run_audit(df, numeric_cols=ENGINEERED, params=params, seed=SEED, K=K, **kw)
    res.pop('single')
    for part in ('all',):
        res[part].pop('null_draws')
    results[name] = res
    a = res['all']
    print(f'{name:<16} {time.time() - t0:6.0f}s  p(R)={ {m: round(v, 4) for m, v in a["p_holm_R"].items()} }'
          f'  p(D)={ {m: round(v, 4) for m, v in a["p_holm_D"].items()} }', flush=True)

out = 'explainers_quick.json' if args.quick else 'explainers.json'
(RESULTS_DIR / out).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / out)
