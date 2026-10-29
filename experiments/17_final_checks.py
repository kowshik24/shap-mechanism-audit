"""
Stage 17: remaining audit variants from the second review, each against its own K-model null.

  bsmote_k3, bsmote_k10   Borderline-SMOTE with 3 or 10 neighbours instead of 5
  focal                   XGBoost with a focal-loss objective (gamma 2), no oversampling
  positive                statistics on max(SHAP, 0), the attribution that pushes towards failure
  pairwise                each row explained against its nearest normal-operation training row
  mlp_deep, mlp_smooth    the MLP with DeepSHAP and with SmoothGrad x (input - baseline)
  kernel_bg500            the MLP with exact KernelSHAP and a 500-row background (K=100; slow)

Usage: python experiments/17_final_checks.py [--quick] [--only name,name]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401

from pdm_audit.audit import run_audit
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR
from pdm_audit.data import load

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--only', default='')
args = ap.parse_args()

VARIANTS = {
    'bsmote_k3': dict(model='xgb:Borderline-SMOTE:3'),
    'bsmote_k10': dict(model='xgb:Borderline-SMOTE:10'),
    'focal': dict(model='xgb_focal'),
    'positive': dict(transform='positive'),
    'pairwise': dict(explainer='pairwise'),
    'mlp_deep': dict(model='mlp', explainer='deep'),
    'mlp_smooth': dict(model='mlp', explainer='smoothgrad'),
    'kernel_bg500': dict(model='mlp', explainer='kernel', kernel_background=500, kernel_samples=4096),
}
K_OF = {'kernel_bg500': 100}
names = [n for n in args.only.split(',') if n] or list(VARIANTS)

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
path = RESULTS_DIR / ('final_checks_quick.json' if args.quick else 'final_checks.json')
results = json.loads(path.read_text()) if path.exists() else {}
for name in names:
    K = 4 if args.quick else K_OF.get(name, 200)
    t0 = time.time()
    res = run_audit(df, ENGINEERED, params, K=K, **VARIANTS[name])
    res.pop('single')
    res['all'].pop('null_draws')
    res['seconds'] = time.time() - t0
    results[name] = res
    a, o = res['all'], res['observed']['all']
    print(f'{name:<13} K={K} {res["seconds"]:5.0f}s R={ {m: round(o[m]["R"], 3) for m in PHYSICAL} } '
          + ' '.join(f'p{s}={ {m: round(a[f"p_holm_{s}"][m], 3) for m in PHYSICAL} }' for s in 'RDS'), flush=True)
    path.write_text(json.dumps(results, indent=1))  # save after each variant
print('saved', path)
