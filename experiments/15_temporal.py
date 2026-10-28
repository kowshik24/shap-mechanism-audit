"""Nested chronological wear-cycle study with paired cluster uncertainty."""
import pdm_audit
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from pdm_audit.config import ENGINEERED, PHYSICAL
from pdm_audit.data import load
from pdm_audit.models import fit_pipeline, f1_threshold
from pdm_audit.sim import simulate
from pdm_audit.temporal import add_context, chronological_split, inner_folds, CONTEXT

ROOT = Path(__file__).resolve().parents[1]


def fit(df, cols, params, tr, va, te):
    scores = []
    for depth in [3, 6, 8]:
        values = []
        for a, b in inner_folds(df, tr):
            pre, est = fit_pipeline(df, a, cols, {**params, 'max_depth': depth}, 42)
            p = est.predict_proba(pre.transform(df.loc[b, cols+['Type']]))[:, 1]
            values.append(float(f1_score(df.loc[b, 'Machine failure'], p >= .5, zero_division=0)))
        scores.append(float(np.mean(values)))
    depth = [3, 6, 8][int(np.argmax(scores))]
    pre, est = fit_pipeline(df, tr, cols, {**params, 'max_depth': depth}, 42)
    threshold = f1_threshold(df.loc[va, 'Machine failure'], est.predict_proba(pre.transform(df.loc[va, cols+['Type']]))[:, 1])
    proba = est.predict_proba(pre.transform(df.loc[te, cols+['Type']]))[:, 1]
    pred = (proba >= threshold).astype(int)
    return {'selected_depth': depth, 'inner_F1': scores, 'threshold': threshold,
            'test_F1': float(f1_score(df.loc[te, 'Machine failure'], pred, zero_division=0)),
            'mechanism_recall': {m: {'detected': int(pred[df.loc[te, m].to_numpy() == 1].sum()),
                                     'total': int(df.loc[te, m].sum())} for m in PHYSICAL},
            'predictions': pred.tolist(), 'probabilities': proba.tolist()}


def evaluate(df, name, params):
    df = add_context(df); tr, va, te = chronological_split(df)
    models = {name: fit(df, cols, params, tr, va, te)
              for name, cols in [('baseline', ENGINEERED), ('past_context', ENGINEERED+CONTEXT)]}
    # Summarize each independent resampling unit once; avoid repeatedly copying row arrays.
    y = df.loc[te, 'Machine failure'].to_numpy(); twf = df.loc[te, 'TWF'].to_numpy()
    cycle = df.loc[te, 'Wear cycle'].to_numpy(); ids = np.unique(cycle)
    parts = {}
    for kind, r in models.items():
        p = np.array(r['predictions'])
        parts[kind] = np.array([[np.sum(y[cycle == c]*p[cycle == c]), np.sum(y[cycle == c]),
                                np.sum(p[cycle == c]), np.sum(twf[cycle == c]*p[cycle == c]),
                                np.sum(twf[cycle == c])] for c in ids])
    rng = np.random.default_rng(20261023); differences, recall_differences = [], []
    for _ in range(3000):
        i = rng.choice(len(ids), len(ids))
        a, b = parts['baseline'][i].sum(axis=0), parts['past_context'][i].sum(axis=0)
        differences.append(2*b[0]/max(1,b[1]+b[2])-2*a[0]/max(1,a[1]+a[2]))
        if a[4]:
            recall_differences.append(b[3]/b[4]-a[3]/a[4])
    item = {'dataset': name, 'n': len(df), 'cycle_counts': [len(df.loc[x, 'Wear cycle'].unique()) for x in [tr,va,te]],
            'row_counts': [len(x) for x in [tr,va,te]], 'test_rows': te.tolist(),
            'test_cycles': cycle.tolist(), 'test_labels': y.tolist(), 'test_TWF': twf.tolist(), 'models': models,
            'paired_F1_difference_cluster_95': np.quantile(differences, [.025,.975]).tolist(),
            'paired_TWF_recall_difference_cluster_95': np.quantile(recall_differences, [.025,.975]).tolist() if recall_differences else None,
            'undefined_TWF_bootstrap_draws': 3000-len(recall_differences)}
    print(name, {k: (v['test_F1'], v['mechanism_recall']['TWF']) for k,v in models.items()}, flush=True)
    return item


def main():
    params = json.loads((ROOT/'models/xgboost_params.json').read_text())
    records = [evaluate(load(), 'released_order', params)]
    for seed in range(20, 30):
        records.append(evaluate(simulate(30000, seed), f'generated_seed_{seed}', params))
    out = {'complete': True, 'protocol': 'experiments/TEMPORAL_PROTOCOL.md', 'records': records}
    (ROOT/'experiments/results/temporal.json').write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == '__main__':
    main()
