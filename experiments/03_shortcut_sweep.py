"""
Stage 3: graded shortcut sweep.

Adds a synthetic sensor s = HDF + N(0, sigma) with no physical role, for a grid of noise levels, and
records how much attribution the sensor receives on HDF rows, the HDF audit verdict, and test F1.
Answers: at what shortcut strength does the model start to use it, and does the audit then flag HDF?

Usage: python experiments/03_shortcut_sweep.py [--quick]
"""
import argparse
import json
import time

import pdm_audit  # noqa: F401
import numpy as np

from pdm_audit.audit import run_audit
from pdm_audit.config import ENGINEERED, MODELS_DIR, RESULTS_DIR, SEED
from pdm_audit.data import load, split
from pdm_audit.metrics import point_metrics
from pdm_audit.models import f1_threshold, fit_pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 500
SIGMAS = [2.0, 1.0, 0.67, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]

base = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
idx_train, idx_val, idx_test = split(base)
y = base['Machine failure']
rows = []
for sigma in SIGMAS:
    t0 = time.time()
    df = base.copy()
    df['Shortcut sensor'] = df['HDF'] + np.random.default_rng(SEED).normal(0, sigma, len(df))
    cols = ENGINEERED + ['Shortcut sensor']

    pre, model = fit_pipeline(df, idx_train, cols, params)
    Xv, Xt = (pre.transform(df.loc[i, cols + ['Type']]) for i in (idx_val, idx_test))
    thr = f1_threshold(y[idx_val], model.predict_proba(Xv)[:, 1])
    pt = model.predict_proba(Xt)[:, 1]
    perf = point_metrics(y[idx_test], (pt >= thr).astype(int), pt)

    res = run_audit(df, numeric_cols=cols, params=params, seed=SEED, K=K, extra_features=('Shortcut sensor',))
    a, obs = res['all'], res['observed']['all']
    row = {'sigma': sigma, 'test_f1': perf['f1'], 'sensor_share_HDF': obs['HDF']['share[Shortcut sensor]'],
           'R_HDF': obs['HDF']['R'], 'D_HDF': obs['HDF']['D'], 'top_HDF': obs['HDF']['top_feature'],
           'p_holm_R': a['p_holm_R'], 'p_holm_D': a['p_holm_D'], 'p_holm_S': a['p_holm_S'],
           'S_HDF': obs['HDF']['S'], 'S_feature_HDF': obs['HDF']['S_feature'],
           'null_mean_S_HDF': a['null_mean']['HDF.S'],
           'null_mean_R_HDF': a['null_mean']['HDF.R']}
    rows.append(row)
    print(f'sigma={sigma:<5} F1={perf["f1"]:.3f} sensor_share={row["sensor_share_HDF"]:.3f} '
          f'R_HDF={row["R_HDF"]:.3f} pR_HDF={a["p_holm_R"]["HDF"]:.3f} pD_HDF={a["p_holm_D"]["HDF"]:.3f} '
          f'pS_HDF={a["p_holm_S"]["HDF"]:.3f} others_R={ {m: round(v, 3) for m, v in a["p_holm_R"].items() if m != "HDF"} } ({time.time()-t0:.0f}s)',
          flush=True)

name = 'shortcut_sweep_quick.json' if args.quick else 'shortcut_sweep.json'
(RESULTS_DIR / name).write_text(json.dumps({'K': K, 'rows': rows}, indent=1))
print('saved', RESULTS_DIR / name)
