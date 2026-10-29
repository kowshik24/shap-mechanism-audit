"""
Stage 6: explainer and model variants of the primary audit.

  interventional   XGBoost with interventional TreeSHAP, for background sizes 100 / 500 / 1000 and
                   several background seeds (the background is drawn from each fold's training rows)
  random_forest    random forest (300 trees, depth 12) with path-dependent TreeSHAP
  checks           additivity of the interventional values against the booster's margin output, and
                   the correlation of tool wear with the features built from it

Usage: python experiments/06_explainers.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np
import shap

from pdm_audit.audit import audit_rows, run_audit
from pdm_audit.config import ENGINEERED, MODELS_DIR, RESULTS_DIR, SEED
from pdm_audit.data import load, split
from pdm_audit.models import fit_pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 200
BACKGROUNDS = [(100, 0)] if args.quick else [(100, 0), (100, 1), (100, 2), (500, 0), (500, 1), (500, 2), (1000, 0)]

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
results = {'K': K, 'interventional': {}}


def audit(name, **kw):
    t0 = time.time()
    res = run_audit(df, numeric_cols=ENGINEERED, params=params, seed=SEED, K=K, **kw)
    res.pop('single')
    res['all'].pop('null_draws')
    a = res['all']
    print(f'{name:<22} {time.time() - t0:6.0f}s  p(R)={ {m: round(v, 4) for m, v in a["p_holm_R"].items()} }'
          f'  p(D)={ {m: round(v, 4) for m, v in a["p_holm_D"].items()} }', flush=True)
    return res


for size, bseed in BACKGROUNDS:
    results['interventional'][f'{size}/{bseed}'] = audit(f'interventional {size}/{bseed}', explainer='interventional',
                                                         n_background=size, background_seed=bseed)
results['random_forest'] = audit('random_forest', model='rf')

# ---- checks
idx_train, _, _ = split(df)
pre, model = fit_pipeline(df, idx_train, ENGINEERED, params)
cols = ENGINEERED + ['Type']
X = pre.transform(df.loc[audit_rows(df), cols])
bg = pre.transform(df.loc[np.random.default_rng(0).choice(idx_train, 100, replace=False), cols])
booster = model.get_booster()
sv = shap.TreeExplainer(booster, data=bg, feature_perturbation='interventional').shap_values(X)
expl = shap.TreeExplainer(booster, data=bg, feature_perturbation='interventional')
margin = model.predict(X, output_margin=True)
wear = df['Tool wear [min]']
results['checks'] = {
    'additivity_max_abs_error': float(np.max(np.abs(sv.sum(axis=1) + expl.expected_value - margin))),
    'spearman_tool_wear': {c: float(wear.corr(df[c], method='spearman'))
                           for c in ['Overstrain [minNm]', 'Overstrain_ratio', 'Torque [Nm]']},
    'spearman_tool_wear_twf_rows': {c: float(wear[df['TWF'] == 1].corr(df.loc[df['TWF'] == 1, c], method='spearman'))
                                    for c in ['Overstrain [minNm]', 'Overstrain_ratio']},
}
print('checks:', results['checks'])

out = 'explainers_quick.json' if args.quick else 'explainers.json'
(RESULTS_DIR / out).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / out)
