import numpy as np
from scipy.stats import beta
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)


def point_metrics(y_true, pred, proba):
    return {'precision': float(precision_score(y_true, pred, zero_division=0)),
            'recall': float(recall_score(y_true, pred, zero_division=0)),
            'f1': float(f1_score(y_true, pred, zero_division=0)),
            'macro_f1': float(f1_score(y_true, pred, average='macro', zero_division=0)),
            'roc_auc': float(roc_auc_score(y_true, proba)),
            'pr_auc': float(average_precision_score(y_true, proba))}


def clopper_pearson(k, n, alpha=0.05):
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def bootstrap_ci(y_true, pred, proba, n_boot=5000, seed=0):
    """Class-stratified percentile bootstrap for F1, ROC-AUC, PR-AUC."""
    y_true, pred, proba = map(np.asarray, (y_true, pred, proba))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    draws = {'f1': [], 'roc_auc': [], 'pr_auc': []}
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        draws['f1'].append(f1_score(y_true[i], pred[i], zero_division=0))
        draws['roc_auc'].append(roc_auc_score(y_true[i], proba[i]))
        draws['pr_auc'].append(average_precision_score(y_true[i], proba[i]))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


def exact_ci(y_true, pred):
    """Clopper-Pearson intervals for precision and recall (the bootstrap degenerates at 0 FP)."""
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    return {'precision': list(clopper_pearson(tp, int(pred.sum()))),
            'recall': list(clopper_pearson(tp, int(y_true.sum()))),
            'tp': tp, 'fp': int(((pred == 1) & (y_true == 0)).sum()),
            'fn': int(((pred == 0) & (y_true == 1)).sum())}


def calibration(y_true, proba, bins=10):
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    edges = np.quantile(proba, np.linspace(0, 1, bins + 1))
    which = np.clip(np.searchsorted(edges, proba, side='right') - 1, 0, bins - 1)
    ece = sum(abs(y_true[which == b].mean() - proba[which == b].mean()) * np.mean(which == b)
              for b in range(bins) if np.any(which == b))
    return {'brier': float(brier_score_loss(y_true, proba)),
            'brier_prevalence_baseline': float(brier_score_loss(y_true, np.full_like(proba, y_true.mean()))),
            'ece': float(ece)}
