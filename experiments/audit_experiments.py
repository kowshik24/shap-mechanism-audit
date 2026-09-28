"""
Robustness experiments for the SHAP failure-mode audit.

  1. RNF negative control tested against a null (replaces the fixed 0.35 cutoff)
  2. Ablation: model trained on raw sensor features only (no mechanism-derived features)
  3. Positive controls: (a) injected spurious shortcut at two strengths, (b) label permutation
  4. Audit restricted to rows where exactly one mechanism fired

Statistic: for mechanism m, the share of the model's mean |SHAP| on m's rows that falls on the
features m's published trigger depends on ("relevant share").

Two nulls are computed. The "naive" null draws random failure rows under the same model; it
turned out to be confounded (mechanism rows are atypical in their own features, and |SHAP| is
larger at atypical values even for a model that learned nothing), so a label-permuted model
passes it. The verdict therefore uses a model-randomization null: the same statistic from K
models retrained on permuted labels with the same features and pipeline.

Run from the repo root:  python experiments/audit_experiments.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import BorderlineSMOTE

RANDOM_STATE = 42
N_NAIVE = 2000
K_PERM = 100
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'experiments', 'results')
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)

MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
REAL = ['TWF', 'HDF', 'PWF', 'OSF']
RAW_NUMERIC = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
               'Torque [Nm]', 'Tool wear [min]']
ENGINEERED = ['Power [W]', 'Temp_diff [K]', 'Overstrain [minNm]', 'Overstrain_ratio']

# Features each mechanism's published trigger depends on (raw inputs + derived versions).
EXPECTED = {
    'TWF': ['Tool wear [min]'],
    'HDF': ['Temp_diff [K]', 'Rotational speed [rpm]', 'Air temperature [K]', 'Process temperature [K]'],
    'PWF': ['Power [W]', 'Torque [Nm]', 'Rotational speed [rpm]'],
    'OSF': ['Overstrain [minNm]', 'Overstrain_ratio', 'Tool wear [min]', 'Torque [Nm]'],
}

# ---------------------------------------------------------------- data (mirrors notebook)
df = pd.read_csv(os.path.join(ROOT, 'dataset', 'ai4i2020.csv'))
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Temp_diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Overstrain [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['Overstrain_ratio'] = df['Overstrain [minNm]'] / df['Type'].map({'L': 11000, 'M': 12000, 'H': 13000})
# spurious "sensor" channels that track HDF with no physical role (shortcut controls)
df['Sensor_weak'] = rng.normal(0, 1, len(df)) + 1.5 * df['HDF']
df['Sensor_strong'] = rng.normal(0, 0.1, len(df)) + df['HDF']

y = df['Machine failure']
n_modes = df[MODES].sum(axis=1)
idx_train, idx_temp = train_test_split(df.index, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
idx_val, idx_test = train_test_split(idx_temp, test_size=1/3, stratify=y[idx_temp], random_state=RANDOM_STATE)

tuned_params = joblib.load(os.path.join(ROOT, 'models', 'xgboost_tuned.joblib')).get_params()
tuned_params.update(n_jobs=1, random_state=RANDOM_STATE)

# rows SHAP is needed on: every failure, every RNF row, plus a fixed pool of non-failures
neg_pool = rng.choice(df.index[y == 0], 1500, replace=False)
AUDIT_ROWS = np.unique(np.concatenate([df.index[y == 1], df.index[df['RNF'] == 1], neg_pool]))

label_facts = {
    'machine_failure_total': int(y.sum()),
    'failures_with_no_mode': int(((y == 1) & (n_modes == 0)).sum()),
    'rows_with_multiple_modes': int((n_modes > 1).sum()),
    'RNF_total': int(df['RNF'].sum()),
    'RNF_with_machine_failure_0': int(((df['RNF'] == 1) & (y == 0)).sum()),
}
print('label facts:', label_facts)


def fit_variant(numeric_cols, y_train=None, evaluate=True):
    cols = numeric_cols + ['Type']
    pre = ColumnTransformer([('num', StandardScaler(), numeric_cols),
                             ('cat', OneHotEncoder(), ['Type'])])
    X_tr = pre.fit_transform(df.loc[idx_train, cols])
    y_tr = y[idx_train] if y_train is None else y_train
    X_res, y_res = BorderlineSMOTE(random_state=RANDOM_STATE).fit_resample(X_tr, y_tr)
    model = xgb.XGBClassifier(**tuned_params).fit(X_res, y_res)

    perf = None
    if evaluate:
        val_p = model.predict_proba(pre.transform(df.loc[idx_val, cols]))[:, 1]
        prec, rec, thr = precision_recall_curve(y[idx_val], val_p)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        threshold = float(thr[np.nanargmax(f1[:-1])])
        pred = (model.predict_proba(pre.transform(df.loc[idx_test, cols]))[:, 1] >= threshold).astype(int)
        perf = {'threshold': threshold,
                'test_precision': float(precision_score(y[idx_test], pred, zero_division=0)),
                'test_recall': float(recall_score(y[idx_test], pred, zero_division=0)),
                'test_f1': float(f1_score(y[idx_test], pred, zero_division=0))}

    names = numeric_cols + list(pre.named_transformers_['cat'].get_feature_names_out(['Type']))
    sv = shap.TreeExplainer(model).shap_values(pre.transform(df.loc[AUDIT_ROWS, cols]))
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    return pd.DataFrame(np.abs(sv), columns=names, index=AUDIT_ROWS), perf


def subgroups(exclusive):
    groups = {}
    for m in MODES:
        mask = df['RNF'] == 1 if m == 'RNF' else df[m] == 1
        if exclusive:
            mask &= n_modes == 1
        groups[m] = df.index[mask]
    return groups


def relevant_share(abs_shap, rows, feats):
    imp = abs_shap.loc[rows].mean(axis=0)
    return float(imp[[f for f in feats if f in imp.index]].sum() / imp.sum())


def top_share(abs_shap, rows):
    imp = abs_shap.loc[rows].mean(axis=0)
    return float(imp.max() / imp.sum())


def statistics(abs_shap, groups):
    s = {m: relevant_share(abs_shap, groups[m], EXPECTED[m]) for m in REAL}
    s['RNF'] = top_share(abs_shap, groups['RNF'])
    return s


def naive_null(abs_shap, groups):
    """Random same-size subsets under the SAME model (kept to show why it is insufficient)."""
    fail = df.index[y == 1]
    out = {}
    for m in REAL:
        obs = relevant_share(abs_shap, groups[m], EXPECTED[m])
        null = np.array([relevant_share(abs_shap, rng.choice(fail, len(groups[m]), replace=False), EXPECTED[m])
                         for _ in range(N_NAIVE)])
        out[m] = float((np.sum(null >= obs) + 1) / (N_NAIVE + 1))
    return out


def randomization_null(numeric_cols, groups):
    """Same statistics from K models trained on permuted labels (model-randomization null)."""
    draws = {m: [] for m in MODES}
    for k in range(K_PERM):
        y_perm = pd.Series(rng.permutation(y[idx_train].values), index=idx_train)
        abs_shap, _ = fit_variant(numeric_cols, y_train=y_perm, evaluate=False)
        for m, v in statistics(abs_shap, groups).items():
            draws[m].append(v)
    return {m: np.array(v) for m, v in draws.items()}


def verdict(obs, null):
    out = {}
    for m in REAL:
        p = float((np.sum(null[m] >= obs[m]) + 1) / (K_PERM + 1))
        out[m] = {'relevant_share': obs[m], 'null_mean': float(null[m].mean()),
                  'null_max': float(null[m].max()), 'p_value': p, 'pass': p < 0.05}
    # RNF: passes if the model is NOT more concentrated on RNF rows than a know-nothing model
    p = float((np.sum(null['RNF'] >= obs['RNF']) + 1) / (K_PERM + 1))
    out['RNF'] = {'top_share': obs['RNF'], 'null_mean': float(null['RNF'].mean()),
                  'null_max': float(null['RNF'].max()),
                  'p_value_more_concentrated': p, 'pass': p >= 0.05}
    return out


def run(name, numeric_cols, exclusive=False, y_train=None):
    t0 = time.time()
    groups = subgroups(exclusive)
    abs_shap, perf = fit_variant(numeric_cols, y_train=y_train)
    obs = statistics(abs_shap, groups)
    null = randomization_null(numeric_cols, groups)
    res = {'perf': perf, 'n': {m: int(len(groups[m])) for m in MODES},
           'n_RNF_machine_failure': int(y[groups['RNF']].sum()),
           'top_feature': {m: abs_shap.loc[groups[m]].mean(axis=0).idxmax() for m in MODES},
           'naive_null_p': naive_null(abs_shap, groups),
           'audit': verdict(obs, null),
           'observed': obs,
           'null_draws': {m: [float(v) for v in null[m]] for m in MODES}}
    print(f'\n=== {name} === ({time.time()-t0:.0f}s)  test F1={perf["test_f1"]:.3f} '
          f'P={perf["test_precision"]:.3f} R={perf["test_recall"]:.3f}')
    for m in MODES:
        a = res['audit'][m]
        stat = a.get('relevant_share', a.get('top_share'))
        p = a.get('p_value', a.get('p_value_more_concentrated'))
        naive = res['naive_null_p'].get(m, float('nan'))
        print(f'  {m}: n={res["n"][m]:3d} top={res["top_feature"][m]:<24} stat={stat:.3f} '
              f'perm_null_mean={a["null_mean"]:.3f} max={a["null_max"]:.3f} p={p:.3f} '
              f'naive_p={naive:.4f} {"PASS" if a["pass"] else "FAIL"}')
    return res


results = {'label_facts': label_facts, 'K_PERM': K_PERM}
eng = RAW_NUMERIC + ENGINEERED

results['main'] = run('main (engineered features)', eng)
results['main_exclusive'] = run('main, exclusive-mode rows only', eng, exclusive=True)
results['raw_ablation'] = run('ablation: raw features only', RAW_NUMERIC)
results['shortcut_weak'] = run('positive control: weak HDF shortcut', eng + ['Sensor_weak'])
results['shortcut_strong'] = run('positive control: strong HDF shortcut', eng + ['Sensor_strong'])
y_perm = pd.Series(rng.permutation(y[idx_train].values), index=idx_train)
results['label_permutation'] = run('positive control: permuted training labels', eng, y_train=y_perm)

with open(os.path.join(OUT, 'audit_experiments.json'), 'w') as f:
    json.dump(results, f, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
print('\nsaved', os.path.join(OUT, 'audit_experiments.json'))
