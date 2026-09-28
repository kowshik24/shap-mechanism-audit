"""
Stage 1: predictive benchmark and model artifacts.

Oversampler comparison, seven model families, validation-set selection, XGBoost tuning, threshold
selection, test metrics with confidence intervals for every model, calibration, and 5-fold CV of the
full pipeline. Writes models/ and experiments/results/benchmark.json (+ test_predictions.npz).
"""
import json

import pdm_audit  # noqa: F401  (threading env vars first)
import joblib
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

from pdm_audit.config import ENGINEERED, MODELS_DIR, RESULTS_DIR, SEED, STRAIN_LIMIT
from pdm_audit.data import label_facts, load, split
from pdm_audit.metrics import bootstrap_ci, calibration, exact_ci, point_metrics
from pdm_audit.models import (baseline_models, default_xgb, f1_threshold, fit_pipeline, make_preprocessor,
                              resampler, train_mlp, tune_xgb, tuned_xgb)

COLS = ENGINEERED + ['Type']
df = load()
y = df['Machine failure']
idx_train, idx_val, idx_test = split(df)

pre = make_preprocessor(ENGINEERED)
X_train = pre.fit_transform(df.loc[idx_train, COLS])
X_val, X_test = pre.transform(df.loc[idx_val, COLS]), pre.transform(df.loc[idx_test, COLS])
y_train, y_val, y_test = y[idx_train].values, y[idx_val].values, y[idx_test].values

# ---- oversampler comparison (validation F1 of a default XGBoost)
samplers = {}
for name in ['SMOTE', 'Borderline-SMOTE', 'ADASYN']:
    Xr, yr = resampler(name).fit_resample(X_train, y_train)
    p = default_xgb().fit(Xr, yr).predict_proba(X_val)[:, 1]
    samplers[name] = point_metrics(y_val, (p >= 0.5).astype(int), p)
best_sampler = max(samplers, key=lambda k: samplers[k]['f1'])
X_res, y_res = resampler(best_sampler).fit_resample(X_train, y_train)
print('best oversampler:', best_sampler)

# ---- model families (validation for selection, test for reporting)
predictors = {name: m.fit(X_res, y_res).predict_proba for name, m in baseline_models().items()}
predictors['MLP'] = train_mlp(X_res, y_res)
val_metrics = {n: point_metrics(y_val, (f(X_val)[:, 1] >= 0.5).astype(int), f(X_val)[:, 1])
               for n, f in predictors.items()}
selected = max(val_metrics, key=lambda k: val_metrics[k]['f1'])
print('selected on validation:', selected, round(val_metrics[selected]['f1'], 3))
assert selected == 'XGBoost', 'tuning below assumes XGBoost is selected'

# ---- tuning + threshold
params, cv_f1 = tune_xgb(X_res, y_res)
tuned = tuned_xgb(params).fit(X_res, y_res)
predictors['XGBoost (tuned)'] = tuned.predict_proba
threshold = f1_threshold(y_val, tuned.predict_proba(X_val)[:, 1])
print('tuned params:', params, 'threshold:', round(threshold, 4))

# ---- test metrics with CIs for every model
test, probas = {}, {}
for name, f in predictors.items():
    p = f(X_test)[:, 1]
    thr = threshold if name == 'XGBoost (tuned)' else 0.5
    pred = (p >= thr).astype(int)
    test[name] = {**point_metrics(y_test, pred, p), 'threshold': thr,
                  'ci_bootstrap': bootstrap_ci(y_test, pred, p), 'ci_exact': exact_ci(y_test, pred)}
    probas[name] = p
p_tuned = probas['XGBoost (tuned)']
cm = confusion_matrix(y_test, (p_tuned >= threshold).astype(int)).tolist()

# ---- 5-fold CV of the full pipeline at the default threshold (every step refit per fold)
cv = []
for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y):
    fpre, fmodel = fit_pipeline(df, df.index[tr], ENGINEERED, params)
    p = fmodel.predict_proba(fpre.transform(df.iloc[te][COLS]))[:, 1]
    cv.append(point_metrics(y.iloc[te].values, (p >= 0.5).astype(int), p))
cv_summary = {k: [float(np.mean([c[k] for c in cv])), float(np.std([c[k] for c in cv]))] for k in cv[0]}

# ---- artifacts
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pre, MODELS_DIR / 'preprocessor.joblib')
joblib.dump(tuned, MODELS_DIR / 'xgboost_tuned.joblib')
joblib.dump({'threshold': threshold, 'feature_cols': COLS, 'numeric_cols': ENGINEERED,
             'strain_limit': STRAIN_LIMIT}, MODELS_DIR / 'metadata.joblib')
(MODELS_DIR / 'xgboost_params.json').write_text(json.dumps(params, indent=2))
np.savez(RESULTS_DIR / 'test_predictions.npz', y_test=y_test, **{k: v for k, v in probas.items()})

out = {'label_facts': label_facts(df), 'splits': {'train': len(idx_train), 'val': len(idx_val),
                                                   'test': len(idx_test), 'test_failures': int(y_test.sum())},
       'oversamplers': samplers, 'best_oversampler': best_sampler, 'validation': val_metrics,
       'selected_model': selected, 'xgb_params': params, 'xgb_search_cv_f1': cv_f1,
       'threshold': threshold, 'test': test, 'confusion_matrix_tuned': cm,
       'calibration_tuned': calibration(y_test, p_tuned), 'cv5_default_threshold': cv_summary, 'cv5_folds': cv}
(RESULTS_DIR / 'benchmark.json').write_text(json.dumps(out, indent=2))
print('tuned test:', {k: round(v, 3) for k, v in test['XGBoost (tuned)'].items() if isinstance(v, float)})
print('CV5 F1 mean/std:', [round(v, 3) for v in cv_summary['f1']])
