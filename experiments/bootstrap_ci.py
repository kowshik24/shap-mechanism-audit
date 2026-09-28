"""
Test-set uncertainty and calibration for the tuned model.

  - 95% bootstrap CIs (percentile, 5000 resamples, stratified by class) for precision, recall,
    F1 at the validation-tuned threshold, plus ROC-AUC and PR-AUC
  - Brier score and expected calibration error (ECE, 10 quantile bins), with a prevalence-only
    baseline for context

Run from the repo root:  python experiments/bootstrap_ci.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
N_BOOT = 5000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'experiments', 'results')
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, 'dataset', 'ai4i2020.csv'))
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Temp_diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Overstrain [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['Overstrain_ratio'] = df['Overstrain [minNm]'] / df['Type'].map({'L': 11000, 'M': 12000, 'H': 13000})

meta = joblib.load(os.path.join(ROOT, 'models', 'metadata.joblib'))
pre = joblib.load(os.path.join(ROOT, 'models', 'preprocessor.joblib'))
model = joblib.load(os.path.join(ROOT, 'models', 'xgboost_tuned.joblib'))

X, y = df[meta['feature_cols']], df['Machine failure']
_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
_, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp, random_state=RANDOM_STATE)

y_true = y_test.values
proba = model.predict_proba(pre.transform(X_test))[:, 1]
pred = (proba >= meta['threshold']).astype(int)


def metrics(t, p, s):
    return {'precision': precision_score(t, p, zero_division=0), 'recall': recall_score(t, p, zero_division=0),
            'f1': f1_score(t, p, zero_division=0), 'roc_auc': roc_auc_score(t, s),
            'pr_auc': average_precision_score(t, s)}


point = metrics(y_true, pred, proba)
rng = np.random.default_rng(RANDOM_STATE)
pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
boot = {k: [] for k in point}
for _ in range(N_BOOT):
    idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
    for k, v in metrics(y_true[idx], pred[idx], proba[idx]).items():
        boot[k].append(v)
ci = {k: {'point': float(point[k]), 'lo': float(np.percentile(v, 2.5)), 'hi': float(np.percentile(v, 97.5))}
      for k, v in boot.items()}


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


# The percentile bootstrap is degenerate for precision here: with zero false positives no resample
# can contain one, so it returns [1, 1]. Exact binomial intervals for precision and recall instead.
tp = int(((pred == 1) & (y_true == 1)).sum())
exact = {'precision': dict(zip(('lo', 'hi'), clopper_pearson(tp, int(pred.sum())))),
         'recall': dict(zip(('lo', 'hi'), clopper_pearson(tp, int(y_true.sum())))),
         'tp': tp, 'fp': int(((pred == 1) & (y_true == 0)).sum()), 'fn': int(((pred == 0) & (y_true == 1)).sum())}


def ece(t, s, bins=10):
    edges = np.quantile(s, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, s, side='right') - 1, 0, bins - 1)
    return float(sum(abs(t[which == b].mean() - s[which == b].mean()) * np.mean(which == b)
                     for b in range(bins) if np.any(which == b)))


calibration = {'brier': float(brier_score_loss(y_true, proba)),
               'brier_prevalence_baseline': float(brier_score_loss(y_true, np.full_like(proba, y_true.mean()))),
               'ece_10_quantile_bins': ece(y_true, proba)}

out = {'n_test': int(len(y_true)), 'n_test_positive': int(y_true.sum()), 'threshold': float(meta['threshold']),
       'n_boot': N_BOOT, 'ci95_bootstrap': ci, 'ci95_clopper_pearson': exact, 'calibration': calibration}
print(json.dumps(out, indent=2))
with open(os.path.join(OUT, 'bootstrap_ci.json'), 'w') as f:
    json.dump(out, f, indent=2)
