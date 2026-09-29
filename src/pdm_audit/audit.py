"""
Per-mechanism audit of SHAP attributions.

For mechanism m with rows S_m and trigger features E_m, with I_{m,j} the mean |SHAP| of feature j
over S_m:

  relevant share   R_m = sum_{j in E_m} I_{m,j} / sum_j I_{m,j}
  specificity      D_m = R_m(S_m) - R_m(F minus S_m), F = all failure rows

  shortcut score   S_m = max_{j not in E_m} [ s_j(S_m) - s_j(F minus S_m) ], s_j = share of feature j

R_m asks whether attribution on the mechanism's rows sits on its trigger features; D_m asks whether
that concentration is specific to the mechanism's rows rather than common to every failure. S_m asks
the converse: does any feature outside E_m take attribution specifically on the mechanism's rows?
R_m and D_m pass when large; S_m flags a shortcut when large. All three are compared with a
model-randomization null: the same pipeline retrained on permuted labels.
"""
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from .config import PHYSICAL, SEED, TRIGGER_FEATURES
from .data import mechanism_groups
from .models import feature_names, fit_pipeline


def abs_shap(model, pre, df, rows, numeric_cols, explainer='path', background_rows=None, seed=SEED,
             n_background=100, background_seed=0):
    """
    |SHAP| per row and feature. explainer='path' is path-dependent TreeSHAP; 'interventional'
    averages over a background of `n_background` rows drawn from `background_rows` (training rows).
    """
    cols = list(numeric_cols) + ['Type']
    X = pre.transform(df.loc[rows, cols])
    if explainer == 'path':
        tree = shap.TreeExplainer(model)
    else:
        bg = np.random.default_rng([seed, background_seed]).choice(background_rows, n_background, replace=False)
        # XGBoost >= 3 marks sklearn models as categorical-enabled, which shap refuses for the
        # interventional algorithm; the booster carries the same trees (all splits here are numeric).
        # check_additivity below verifies that the SHAP values sum to the booster's margin output.
        est = model.get_booster() if hasattr(model, 'get_booster') else model
        masker = shap.maskers.Independent(pre.transform(df.loc[bg, cols]), max_samples=n_background)
        tree = shap.TreeExplainer(est, data=masker.data, feature_perturbation='interventional')
    sv = tree.shap_values(X, check_additivity=True)
    if isinstance(sv, list):
        sv = sv[1]
    elif np.ndim(sv) == 3:
        sv = sv[:, :, 1]
    return pd.DataFrame(np.abs(sv), index=rows, columns=feature_names(pre, numeric_cols))


def audit_rows(df):
    """Rows the statistics need: every failure and every RNF row."""
    return np.union1d(df.index[df['Machine failure'] == 1], df.index[df['RNF'] == 1])


def shap_in_sample(df, train_idx, numeric_cols, params, seed, labels=None, model='xgb', explainer='path',
                   **shap_kw):
    pre, est = fit_pipeline(df, train_idx, numeric_cols, params, seed, y=labels, model=model)
    return abs_shap(est, pre, df, audit_rows(df), numeric_cols, explainer, train_idx, seed, **shap_kw)


def shap_cross_fitted(df, numeric_cols, params, seed, labels=None, n_folds=5, model='xgb', explainer='path',
                      **shap_kw):
    """Out-of-sample SHAP: each row is explained by a model that never saw it in training."""
    y = df['Machine failure']
    rows = audit_rows(df)
    parts = []
    for tr, te in StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(df, y):
        pre, est = fit_pipeline(df, df.index[tr], numeric_cols, params, seed, y=labels, model=model)
        held_out = np.intersect1d(df.index[te], rows)
        parts.append(abs_shap(est, pre, df, held_out, numeric_cols, explainer, df.index[tr], seed, **shap_kw))
    return pd.concat(parts).loc[rows]


def _share(imp, feats):
    feats = [f for f in feats if f in imp.index]
    return float(imp[feats].sum() / imp.sum())


def statistics(A: pd.DataFrame, df: pd.DataFrame, groups: dict, extra_features=()):
    """R_m, D_m for physical mechanisms; top-feature share for RNF; shares of any extra features."""
    failures = df.index[df['Machine failure'] == 1]
    out = {}
    for m in PHYSICAL:
        own = A.loc[groups[m]].mean(axis=0)
        rest = A.loc[np.setdiff1d(failures, groups[m])].mean(axis=0)
        excess = (own / own.sum() - rest / rest.sum()).drop([f for f in TRIGGER_FEATURES[m] if f in own.index])
        out[m] = {'R': _share(own, TRIGGER_FEATURES[m]),
                  'D': _share(own, TRIGGER_FEATURES[m]) - _share(rest, TRIGGER_FEATURES[m]),
                  'S': float(excess.max()), 'S_feature': str(excess.idxmax()),
                  'top_feature': str(own.idxmax()),
                  **{f'share[{f}]': _share(own, [f]) for f in extra_features}}
    rnf = A.loc[groups['RNF']].mean(axis=0)
    out['RNF'] = {'top_share': float(rnf.max() / rnf.sum()), 'top_feature': str(rnf.idxmax())}
    return out


def permutation_p(observed, null, greater=True):
    null = np.asarray(null)
    hits = np.sum(null >= observed) if greater else np.sum(null <= observed)
    return float((1 + hits) / (len(null) + 1))


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values over a family of tests."""
    keys = sorted(pvals, key=pvals.get)
    n, running, adj = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (n - i) * pvals[k]))
        adj[k] = running
    return adj


def naive_null_p(A, df, groups, n_draws=2000, seed=0):
    """
    The confounded null: R_m on S_m against R_m on random, equally sized subsets of failure rows
    under the same model. Kept to show that it passes a model trained on permuted labels.
    """
    rng = np.random.default_rng(seed)
    failures = df.index[df['Machine failure'] == 1].to_numpy()
    out = {}
    for m in PHYSICAL:
        obs = _share(A.loc[groups[m]].mean(axis=0), TRIGGER_FEATURES[m])
        null = [_share(A.loc[rng.choice(failures, len(groups[m]), replace=False)].mean(axis=0),
                       TRIGGER_FEATURES[m]) for _ in range(n_draws)]
        out[m] = permutation_p(obs, null)
    return out


def _null_draw(k, df, numeric_cols, params, seed, train_idx, cross_fit, groups_by_name, extra, fit_kw):
    rng = np.random.default_rng(seed * 100_003 + k)
    y = df['Machine failure']
    perm = pd.Series(y.values, index=y.index)
    if cross_fit:
        perm[:] = rng.permutation(y.values)
        A = shap_cross_fitted(df, numeric_cols, params, seed, labels=perm, **fit_kw)
    else:
        perm.loc[train_idx] = rng.permutation(y.loc[train_idx].values)
        A = shap_in_sample(df, train_idx, numeric_cols, params, seed, labels=perm, **fit_kw)
    return {name: statistics(A, df, g, extra) for name, g in groups_by_name.items()}


def run_audit(df, numeric_cols, params, *, seed=SEED, K=100, cross_fit=True, train_idx=None,
              extra_features=(), n_jobs=-1, observed_labels=None, model='xgb', explainer='path',
              **shap_kw):
    """
    Observed statistics, a K-model randomization null, p-values and Holm-adjusted p-values, for
    all flagged rows ('all') and single-mechanism rows ('single'). `model` ('xgb', 'rf') and
    `explainer` ('path', 'interventional') apply to the audited model and every null run alike.
    """
    fit_kw = {'model': model, 'explainer': explainer, **shap_kw}
    groups_by_name = {'all': mechanism_groups(df), 'single': mechanism_groups(df, exclusive=True)}
    if cross_fit:
        A = shap_cross_fitted(df, numeric_cols, params, seed, labels=observed_labels, **fit_kw)
    else:
        A = shap_in_sample(df, train_idx, numeric_cols, params, seed, labels=observed_labels, **fit_kw)
    observed = {name: statistics(A, df, g, extra_features) for name, g in groups_by_name.items()}

    draws = Parallel(n_jobs=n_jobs)(
        delayed(_null_draw)(k, df, numeric_cols, params, seed, train_idx, cross_fit, groups_by_name,
                            extra_features, fit_kw) for k in range(K))

    result = {'seed': seed, 'K': K, 'cross_fit': cross_fit, 'observed': observed,
              'naive_null_p': naive_null_p(A, df, groups_by_name['all']),
              'mean_abs_shap': {m: A.loc[g].mean(axis=0).to_dict()
                                for m, g in groups_by_name['all'].items()}}
    for name in groups_by_name:
        tests, null_store = {}, {}
        for m in PHYSICAL:
            for stat in ('R', 'D', 'S'):
                null = [d[name][m][stat] for d in draws]
                null_store[f'{m}.{stat}'] = null
                tests[f'{m}.{stat}'] = permutation_p(observed[name][m][stat], null)
        rnf_null = [d[name]['RNF']['top_share'] for d in draws]
        null_store['RNF.top_share'] = rnf_null
        result[name] = {
            'p': tests,
            'p_holm_R': holm({m: tests[f'{m}.R'] for m in PHYSICAL}),
            'p_holm_D': holm({m: tests[f'{m}.D'] for m in PHYSICAL}),
            'p_holm_S': holm({m: tests[f'{m}.S'] for m in PHYSICAL}),  # small = shortcut flagged
            'p_RNF_more_concentrated': permutation_p(observed[name]['RNF']['top_share'], rnf_null),
            'null_mean': {k: float(np.mean(v)) for k, v in null_store.items()},
            'null_draws': null_store,
        }
    return result
