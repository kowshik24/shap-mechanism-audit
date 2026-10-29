"""
Stage 11: why does the TWF verdict flip between path-dependent and interventional TreeSHAP?

Interventional TreeSHAP (background 100, seed 0) audits, each against a matching K-model null:

  failures        background drawn only from training rows labelled failure
  non_failures    background drawn only from training rows labelled non-failure
  oversampled     background drawn from the Borderline-SMOTE output the model was trained on, which is
                  the distribution path-dependent TreeSHAP's node covers reflect
  raw             the stage-6 default background, raw sensor features only (no overstrain features)

plus, without new fits, the share of each feature in TWF-row attribution under path-dependent
(stage 2) and interventional (stage 6) SHAP.

Usage: python experiments/11_twf_diagnostics.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401

from pdm_audit.audit import run_audit
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RAW_NUMERIC, RESULTS_DIR, SEED
from pdm_audit.data import load

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 200

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
results = {'K': K}

VARIANTS = {
    'failures': dict(numeric_cols=ENGINEERED, background='failures'),
    'non_failures': dict(numeric_cols=ENGINEERED, background='non_failures'),
    'oversampled': dict(numeric_cols=ENGINEERED, background='oversampled'),
    'raw': dict(numeric_cols=RAW_NUMERIC),
}
for name, kw in VARIANTS.items():
    t0 = time.time()
    res = run_audit(df, params=params, seed=SEED, K=K, explainer='interventional', n_background=100,
                    background_seed=0, **kw)
    res.pop('single')
    res['all'].pop('null_draws')
    res['seconds'] = time.time() - t0
    results[name] = res
    a = res['all']
    print(f'{name:<14} {res["seconds"]:6.0f}s  TWF R={res["observed"]["all"]["TWF"]["R"]:.3f}'
          f'  p(R)={ {m: round(a["p_holm_R"][m], 3) for m in PHYSICAL} }'
          f'  p(D)={ {m: round(a["p_holm_D"][m], 3) for m in PHYSICAL} }', flush=True)


def shares(mean_abs):
    total = sum(mean_abs.values())
    return {f: v / total for f, v in sorted(mean_abs.items(), key=lambda kv: -kv[1])}


path = json.loads((RESULTS_DIR / 'audit.json').read_text())['primary']['mean_abs_shap']['TWF']
interventional = json.loads((RESULTS_DIR / 'explainers.json').read_text())['interventional']['100/0']
results['twf_shares'] = {'path': shares(path), 'interventional': shares(interventional['mean_abs_shap']['TWF'])}
for k, v in results['twf_shares'].items():
    print(k, {f: round(x, 3) for f, x in list(v.items())[:5]})

out = 'twf_diagnostics_quick.json' if args.quick else 'twf_diagnostics.json'
(RESULTS_DIR / out).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / out)
