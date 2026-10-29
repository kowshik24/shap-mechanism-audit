"""
Stage 13: checks requested in the second review of the TUAI paper.

  variants        primary audit (K permuted-label models each) with
                    no_unflagged    the nine failures without any mechanism flag removed
                    or_labels       overall label rebuilt as the OR of the five flags
                    class_weights   no oversampling; XGBoost weights failures by the class ratio
                    kernel_exact    the MLP with exact KernelSHAP (100-row background, all 4,096
                                    coalitions); run with --kernel, it is the slow one
  k_stability     p-values and effect sizes of the primary audit from its first K null draws
  holm_all        Holm correction over all twelve R/D/C tests of the primary audit
  signed          share of trigger-feature attribution that pushes towards failure
  removal         recall on each mechanism's rows after retraining without its raw trigger features
  attributions    per-row signed SHAP of the primary audit, saved for third-party statistics

Usage: python experiments/13_review_checks.py [--quick] [--kernel]
"""
import argparse
import json

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold

from pdm_audit.audit import audit_rows, holm, permutation_p, run_audit
from pdm_audit.config import (ENGINEERED, MODES, MODELS_DIR, PHYSICAL, RAW_NUMERIC, RESULTS_DIR, SEED,
                              TRIGGER_FEATURES)
from pdm_audit.data import load, mechanism_groups
from pdm_audit.models import feature_names, fit_pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
ap.add_argument('--kernel', action='store_true', help='only the exact-KernelSHAP MLP audit')
args = ap.parse_args()
K = 4 if args.quick else 200

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
primary = json.loads((RESULTS_DIR / 'audit.json').read_text())['primary']
out_name = ('review_checks_kernel' if args.kernel else 'review_checks') + ('_quick' if args.quick else '')


def summary(res):
    res.pop('single')
    res['all'].pop('null_draws')
    a = res['all']
    print({s: {m: round(a[f'p_holm_{s}'][m], 3) for m in PHYSICAL} for s in 'RDS'}, flush=True)
    return res


results = {'K': K}
if args.kernel:
    results['kernel_exact'] = summary(run_audit(df, ENGINEERED, params, K=K, model='mlp', explainer='kernel',
                                                kernel_background=100, kernel_samples=4096))
    (RESULTS_DIR / f'{out_name}.json').write_text(json.dumps(results, indent=1))
    raise SystemExit

# ---- label anomalies and imbalance handling
flags = df[MODES].sum(axis=1)
unflagged = df.index[(df['Machine failure'] == 1) & (flags == 0)]
results['n_unflagged'] = len(unflagged)
results['no_unflagged'] = summary(run_audit(df.drop(index=unflagged), ENGINEERED, params, K=K))
df_or = df.copy()
df_or['Machine failure'] = (flags > 0).astype(int)
results['or_labels'] = summary(run_audit(df_or, ENGINEERED, params, K=K))
results['class_weights'] = summary(run_audit(df, ENGINEERED, params, K=K, model='xgb_cw'))

# ---- K stability and Holm over all twelve tests (primary audit, no new fits)
draws, obs = primary['all']['null_draws'], primary['observed']['all']
results['k_stability'] = {}
for k in (50, 100, 200, 500, 1000):
    p = {f'{m}.{s}': permutation_p(obs[m][s], draws[f'{m}.{s}'][:k]) for m in PHYSICAL for s in 'RDS'}
    results['k_stability'][k] = {
        **{f'p_holm_{s}': holm({m: p[f'{m}.{s}'] for m in PHYSICAL}) for s in 'RDS'},
        'z': {f'{m}.{s}': float((obs[m][s] - np.mean(draws[f'{m}.{s}'][:k])) / np.std(draws[f'{m}.{s}'][:k], ddof=1))
              for m in PHYSICAL for s in 'RD'}}
results['holm_all'] = holm(primary['all']['p'])

# ---- signed attributions of the primary audit (same folds and models as stage 2)
groups = mechanism_groups(df)
rows = audit_rows(df)
y = df['Machine failure']
parts = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y):
    pre, est = fit_pipeline(df, df.index[tr], ENGINEERED, params, SEED)
    held = np.intersect1d(df.index[te], rows)
    sv = shap.TreeExplainer(est).shap_values(pre.transform(df.loc[held, ENGINEERED + ['Type']]))
    parts.append(pd.DataFrame(sv, index=held, columns=feature_names(pre, ENGINEERED)))
phi = pd.concat(parts).loc[rows]
failures = df.index[y == 1]
results['signed'] = {}
for m in PHYSICAL:
    trig = [f for f in TRIGGER_FEATURES[m] if f in phi.columns]
    def pos_share(idx):
        t = phi.loc[idx, trig].to_numpy()
        return float(np.clip(t, 0, None).sum() / np.abs(t).sum())
    results['signed'][m] = {'own': pos_share(groups[m]), 'other_failures': pos_share(np.setdiff1d(failures, groups[m]))}
print('signed', results['signed'])
phi.join(df.loc[rows, MODES + ['Machine failure']]).to_csv(RESULTS_DIR / 'attributions_primary.csv')

# ---- mechanism-level removal: retrain without the raw trigger features (cross-fitted predictions)
def cross_fitted_proba(cols):
    p = pd.Series(np.nan, index=df.index)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y):
        pre, est = fit_pipeline(df, df.index[tr], cols, params, SEED)
        p.iloc[te] = est.predict_proba(pre.transform(df.iloc[te][cols + ['Type']]))[:, 1]
    return p


full = cross_fitted_proba(RAW_NUMERIC)
results['removal'] = {}
for m in PHYSICAL:
    keep = [c for c in RAW_NUMERIC if c not in TRIGGER_FEATURES[m]]
    reduced = cross_fitted_proba(keep)
    own, rest = groups[m], np.setdiff1d(failures, groups[m])
    recall = lambda p, idx: float((p.loc[idx] >= 0.5).mean())
    results['removal'][m] = {'removed': [c for c in RAW_NUMERIC if c in TRIGGER_FEATURES[m]],
                             'recall_own': [recall(full, own), recall(reduced, own)],
                             'recall_other_failures': [recall(full, rest), recall(reduced, rest)]}
print('removal', {m: (np.round(v['recall_own'], 2).tolist(), np.round(v['recall_other_failures'], 2).tolist())
                  for m, v in results['removal'].items()})

(RESULTS_DIR / f'{out_name}.json').write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / f'{out_name}.json')
