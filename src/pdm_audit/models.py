import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import SEED, XGB_SEARCH_ITER, XGB_SEARCH_SPACE


def make_preprocessor(numeric_cols):
    return ColumnTransformer([('num', StandardScaler(), list(numeric_cols)),
                              ('cat', OneHotEncoder(), ['Type'])])


def feature_names(pre, numeric_cols):
    return list(numeric_cols) + list(pre.named_transformers_['cat'].get_feature_names_out(['Type']))


def resampler(name, seed=SEED):
    """'SMOTE', 'Borderline-SMOTE' or 'ADASYN'; a ':k' suffix sets k_neighbors (e.g. 'Borderline-SMOTE:3')."""
    name, _, k = name.partition(':')
    kw = {'k_neighbors': int(k)} if k else {}
    return {'SMOTE': SMOTE, 'Borderline-SMOTE': BorderlineSMOTE, 'ADASYN': ADASYN}[name](random_state=seed, **kw)


def focal_objective(alpha, gamma=2.0, eps=1e-3):
    """Binary focal loss on the margin for XGBoost; gradient and hessian by central differences."""
    def loss(y, z):
        p = np.clip(1 / (1 + np.exp(-z)), 1e-12, 1 - 1e-12)
        return -(y * alpha * (1 - p) ** gamma * np.log(p) + (1 - y) * (1 - alpha) * p ** gamma * np.log(1 - p))

    def objective(y, z):
        lo, mid, hi = loss(y, z - eps), loss(y, z), loss(y, z + eps)
        return (hi - lo) / (2 * eps), np.maximum((hi - 2 * mid + lo) / eps ** 2, 1e-6)
    return objective


def default_xgb(seed=SEED):
    # n_jobs=1: xgboost's own OpenMP threading segfaulted on the development machine
    return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             eval_metric='logloss', random_state=seed, n_jobs=1)


def tuned_xgb(params, seed=SEED):
    return xgb.XGBClassifier(**params, eval_metric='logloss', random_state=seed, n_jobs=1)


def baseline_models(seed=SEED):
    return {
        'Logistic regression': LogisticRegression(max_iter=1000, random_state=seed),
        'SVM': SVC(kernel='rbf', probability=True, random_state=seed),
        'k-NN': KNeighborsClassifier(n_neighbors=7),
        'Decision tree': DecisionTreeClassifier(max_depth=10, random_state=seed),
        'Random forest': RandomForestClassifier(n_estimators=300, max_depth=12, random_state=seed, n_jobs=1),
        'XGBoost': default_xgb(seed),
    }


def tune_xgb(X, y, sampler='Borderline-SMOTE', seed=SEED):
    """
    Randomized search on the original (not resampled) training rows. The oversampler sits inside
    the pipeline, so each CV fold resamples only its own training part and is scored on real rows.
    """
    pipe = Pipeline([('sampler', resampler(sampler, seed)),
                     ('xgb', xgb.XGBClassifier(eval_metric='logloss', random_state=seed, n_jobs=1))])
    search = RandomizedSearchCV(
        pipe, {f'xgb__{k}': v for k, v in XGB_SEARCH_SPACE.items()}, n_iter=XGB_SEARCH_ITER, scoring='f1',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        random_state=seed, n_jobs=1)
    search.fit(X, y)
    params = {k.removeprefix('xgb__'): v for k, v in search.best_params_.items()}
    return params, float(search.best_score_)


def f1_threshold(y_true, proba):
    """Threshold maximizing F1 on a precision-recall curve."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return float(thr[np.nanargmax(f1[:-1])])


def train_mlp(X, y, seed=SEED, epochs=40):
    """Feed-forward network (64-32) trained with BCE; returns a predict_proba-style callable."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = nn.Sequential(
        nn.Linear(X.shape[1], 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
        nn.Linear(32, 1))
    loader = DataLoader(TensorDataset(torch.tensor(np.asarray(X), dtype=torch.float32),
                                      torch.tensor(np.asarray(y), dtype=torch.float32).unsqueeze(1)),
                        batch_size=64, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()
    net.eval()

    def predict_proba(Xn):
        with torch.no_grad():
            p = torch.sigmoid(net(torch.tensor(np.asarray(Xn), dtype=torch.float32))).numpy().ravel()
        return np.column_stack([1 - p, p])

    def logit(Xn):
        with torch.no_grad():
            return net(torch.tensor(np.asarray(Xn), dtype=torch.float32)).numpy().ravel()

    predict_proba.logit = logit  # the explainers attribute the logit, as TreeSHAP does the margin
    predict_proba.net = net
    return predict_proba


def fit_pipeline(df: pd.DataFrame, train_idx, numeric_cols, params, seed=SEED, y=None, model='xgb'):
    """
    Preprocess -> Borderline-SMOTE -> tuned XGBoost (or, with model='rf', the random forest of
    baseline_models; with model='mlp', the MLP of train_mlp, returned as its predict_proba callable;
    with model='xgb_cw', the tuned XGBoost without oversampling, weighting failures by the class ratio)
    on the given rows. `y` overrides the labels.
    """
    labels = df['Machine failure'] if y is None else y
    cols = list(numeric_cols) + ['Type']
    pre = make_preprocessor(numeric_cols)
    X_tr = pre.fit_transform(df.loc[train_idx, cols])
    if model.startswith('xgb:'):  # tuned XGBoost after another oversampler, e.g. 'xgb:SMOTE', 'xgb:ADASYN'
        X_res, y_res = resampler(model.split(':', 1)[1], seed).fit_resample(X_tr, labels.loc[train_idx])
        return pre, tuned_xgb(params, seed).fit(X_res, y_res)
    if model == 'xgb_focal':  # focal loss (gamma 2, alpha = share of non-failures), no oversampling
        y_tr = np.asarray(labels.loc[train_idx])
        est = xgb.XGBClassifier(**params, objective=focal_objective(alpha=float((y_tr == 0).mean())),
                                random_state=seed, n_jobs=1)
        return pre, est.fit(X_tr, y_tr)
    if model == 'xgb_cw':
        y_tr = np.asarray(labels.loc[train_idx])
        est = tuned_xgb({**params, 'scale_pos_weight': (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)}, seed)
        return pre, est.fit(X_tr, y_tr)
    X_res, y_res = resampler('Borderline-SMOTE', seed).fit_resample(X_tr, labels.loc[train_idx])
    if model == 'mlp':
        return pre, train_mlp(X_res, np.asarray(y_res), seed)
    est = tuned_xgb(params, seed) if model == 'xgb' else baseline_models(seed)['Random forest']
    return pre, est.fit(X_res, y_res)
