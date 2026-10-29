"""
Stage 12: the audit beyond tree ensembles and TreeSHAP.

The MLP of the model comparison (64-32, Borderline-SMOTE, fixed hyperparameters) is audited with
  kernel   KernelSHAP (model-agnostic; 25 training rows as background, 1000 coalitions per row)
  ig       Integrated Gradients (gradient-based; baseline = mean of 100 training rows, 1024 steps)
Both attribute the MLP's logit. Cross-fitting and the shuffled-label null are unchanged: every null
draw retrains the MLP on permuted labels and is explained the same way.

Usage: python experiments/12_other_explainers.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401

from pdm_audit.audit import run_audit
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED
from pdm_audit.data import load

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 200

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())  # unused by the MLP
results = {'K': K}
for name in ('ig', 'kernel'):
    t0 = time.time()
    res = run_audit(df, numeric_cols=ENGINEERED, params=params, seed=SEED, K=K, model='mlp', explainer=name)
    res.pop('single')
    res['all'].pop('null_draws')
    res['seconds'] = time.time() - t0
    results[f'mlp_{name}'] = res
    a, o = res['all'], res['observed']['all']
    print(f'mlp_{name:<7} {res["seconds"]:6.0f}s  R={ {m: round(o[m]["R"], 3) for m in PHYSICAL} }'
          f'  p(R)={ {m: round(a["p_holm_R"][m], 3) for m in PHYSICAL} }'
          f'  p(D)={ {m: round(a["p_holm_D"][m], 3) for m in PHYSICAL} }'
          f'  p(C)={ {m: round(a["p_holm_S"][m], 3) for m in PHYSICAL} }', flush=True)

out = 'other_explainers_quick.json' if args.quick else 'other_explainers.json'
(RESULTS_DIR / out).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / out)
