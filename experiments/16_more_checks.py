"""
Stage 16: further checks from the second review.

  samplers      primary audit with plain SMOTE or ADASYN instead of Borderline-SMOTE (K null models)
  triggers      primary audit with misspecified expected sets: no_derived (E_m without the four
                engineered features) and drop_top (E_m without its most attributed feature, for the
                mechanisms with more than one trigger feature)
  hsic          dependence-based corroboration: for each mechanism, the HSIC between each input and
                the cross-fitted failure probability on the mechanism's rows plus 1,000 normal rows
                (Gaussian kernels, median bandwidth); reports the share of HSIC on E_m and its rank
                agreement with the SHAP shares

Usage: python experiments/16_more_checks.py [--quick]
"""
import argparse
import json

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold

from pdm_audit.audit import run_audit
from pdm_audit.config import DERIVED, ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED, TRIGGER_FEATURES
from pdm_audit.data import load, mechanism_groups
from pdm_audit.models import fit_pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 200

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
primary = json.loads((RESULTS_DIR / 'audit.json').read_text())['primary']
results = {'K': K}


def summary(name, res):
    res.pop('single')
    res['all'].pop('null_draws')
    a = res['all']
    print(name, {s: {m: round(a[f'p_holm_{s}'][m], 3) for m in PHYSICAL} for s in 'RDS'}, flush=True)
    return res


for sampler in ('SMOTE', 'ADASYN'):
    results[f'sampler_{sampler}'] = summary(sampler, run_audit(df, ENGINEERED, params, K=K, model=f'xgb:{sampler}'))

mean_abs = primary['mean_abs_shap']
no_derived = {m: [f for f in TRIGGER_FEATURES[m] if f not in DERIVED] for m in PHYSICAL}
drop_top = {m: ([f for f in TRIGGER_FEATURES[m] if f != max(TRIGGER_FEATURES[m], key=lambda f: mean_abs[m][f])]
                if len(TRIGGER_FEATURES[m]) > 1 else TRIGGER_FEATURES[m]) for m in PHYSICAL}
results['trigger_sets'] = {'no_derived': no_derived, 'drop_top': drop_top}
for name, triggers in (('no_derived', no_derived), ('drop_top', drop_top)):
    results[f'triggers_{name}'] = summary(name, run_audit(df, ENGINEERED, params, K=K, triggers=triggers))

# ---- HSIC corroboration
y = df['Machine failure']
proba = pd.Series(np.nan, index=df.index)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y):
    pre, est = fit_pipeline(df, df.index[tr], ENGINEERED, params, SEED)
    proba.iloc[te] = est.predict_proba(pre.transform(df.iloc[te][ENGINEERED + ['Type']]))[:, 1]


def gram(v):
    d = (v[:, None] - v[None, :]) ** 2
    bw = np.median(d[d > 0]) if (d > 0).any() else 1.0
    return np.exp(-d / bw)


def hsic(a, b):
    n = len(a)
    H = np.eye(n) - 1 / n
    return float(np.trace(gram(a) @ H @ gram(b) @ H) / (n - 1) ** 2)


groups = mechanism_groups(df)
normal = df.index[(y == 0) & (df[['TWF', 'HDF', 'PWF', 'OSF', 'RNF']].sum(axis=1) == 0)]
rng = np.random.default_rng(SEED)
results['hsic'] = {}
for m in PHYSICAL:
    idx = np.r_[groups[m], rng.choice(normal, 200 if args.quick else 1000, replace=False)]
    out = np.log(proba.loc[idx].clip(1e-6, 1 - 1e-6) / (1 - proba.loc[idx].clip(1e-6, 1 - 1e-6))).to_numpy()
    h = pd.Series({f: hsic(df.loc[idx, f].to_numpy(dtype=float), out) for f in ENGINEERED})
    share = h / h.sum()
    shap_share = pd.Series({f: mean_abs[m][f] for f in ENGINEERED})
    results['hsic'][m] = {'E_share': float(share[TRIGGER_FEATURES[m]].sum()), 'shares': share.to_dict(),
                          'spearman_with_shap': float(spearmanr(share, shap_share).correlation),
                          'top_feature': str(share.idxmax())}
    print('hsic', m, round(results['hsic'][m]['E_share'], 3), round(results['hsic'][m]['spearman_with_shap'], 2),
          results['hsic'][m]['top_feature'], flush=True)

name = 'more_checks_quick.json' if args.quick else 'more_checks.json'
(RESULTS_DIR / name).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / name)
