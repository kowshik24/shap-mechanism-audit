"""
Stage 24: several weak shortcut proxies instead of one.

Adds n_proxies synthetic sensors s_k = HDF + N(0, sigma), each with its own noise draw and none with a
physical role, and runs the cross-fitted audit for a grid of noise levels. Compared with the single
sensor of stage 3 at the same sigma, the proxies together carry more information about HDF, but the
model can spread attribution over them, and C_m takes the maximum over single features. Also reports
the summed excess of all proxies, which C_m does not test.

Usage: python experiments/24_multi_proxy.py [--quick] [--proxies 3]
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
ap.add_argument('--proxies', type=int, default=3)
args = ap.parse_args()
K = 4 if args.quick else 500
SIGMAS = [1.0, 0.67, 0.5, 0.4, 0.35, 0.25]

base = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
idx_train, idx_val, idx_test = split(base)
y = base['Machine failure']
proxies = [f'Shortcut sensor {k + 1}' for k in range(args.proxies)]
rows = []
for sigma in SIGMAS:
    t0 = time.time()
    df = base.copy()
    rng = np.random.default_rng(SEED)
    for name in proxies:
        df[name] = df['HDF'] + rng.normal(0, sigma, len(df))
    cols = ENGINEERED + proxies

    pre, model = fit_pipeline(df, idx_train, cols, params)
    Xv, Xt = (pre.transform(df.loc[i, cols + ['Type']]) for i in (idx_val, idx_test))
    thr = f1_threshold(y[idx_val], model.predict_proba(Xv)[:, 1])
    pt = model.predict_proba(Xt)[:, 1]
    perf = point_metrics(y[idx_test], (pt >= thr).astype(int), pt)

    res = run_audit(df, numeric_cols=cols, params=params, seed=SEED, K=K, extra_features=tuple(proxies))
    a, obs = res['all'], res['observed']['all']['HDF']
    row = {'sigma': sigma, 'n_proxies': len(proxies), 'test_f1': perf['f1'],
           'proxy_shares_HDF': [obs[f'share[{p}]'] for p in proxies],
           'proxy_excess_HDF': [obs['excess'][p] for p in proxies],
           'R_HDF': obs['R'], 'D_HDF': obs['D'], 'S_HDF': obs['S'], 'S_feature_HDF': obs['S_feature'],
           'p_holm_R': a['p_holm_R'], 'p_holm_D': a['p_holm_D'], 'p_holm_S': a['p_holm_S'],
           'null_mean_R_HDF': a['null_mean']['HDF.R'], 'null_mean_S_HDF': a['null_mean']['HDF.S']}
    rows.append(row)
    print(f'sigma={sigma:<5} F1={perf["f1"]:.3f} proxy_share={sum(row["proxy_shares_HDF"]):.3f} '
          f'max_proxy={max(row["proxy_shares_HDF"]):.3f} R_HDF={row["R_HDF"]:.3f} '
          f'pR={a["p_holm_R"]["HDF"]:.3f} pD={a["p_holm_D"]["HDF"]:.3f} '
          f'C={row["S_HDF"]:.3f} {row["S_feature_HDF"]} pC={a["p_holm_S"]["HDF"]:.3f} ({time.time()-t0:.0f}s)',
          flush=True)

name = 'multi_proxy_quick.json' if args.quick else f'multi_proxy_{len(proxies)}.json'
(RESULTS_DIR / name).write_text(json.dumps({'K': K, 'rows': rows}, indent=1))
print('saved', RESULTS_DIR / name)
