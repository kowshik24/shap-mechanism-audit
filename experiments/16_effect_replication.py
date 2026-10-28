"""Five-seed effect and held-out reliance replication; no inferred p-values."""
import pdm_audit
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from pdm_audit.audit import abs_shap, audit_rows, statistics
from pdm_audit.config import ENGINEERED, PHYSICAL
from pdm_audit.data import mechanism_groups
from pdm_audit.models import fit_pipeline
from pdm_audit.sim import simulate

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT/'experiments/results/effect_replication_checkpoints'


def condition(n, seed, injected, params, resume=False):
    CHECK.mkdir(exist_ok=True)
    path = CHECK/f'{n}_{seed}_{int(injected)}.json'
    inputs = [Path(__file__), *[ROOT/'src/pdm_audit'/f for f in ['audit.py','models.py','sim.py','data.py','config.py']],
              ROOT/'dataset/ai4i2020.csv']
    provenance = {'parameters': params, 'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    if resume and path.exists():
        saved = json.loads(path.read_text())
        if saved.get('provenance') != provenance:
            raise ValueError('Checkpoint provenance differs or is missing; rerun without --resume.')
        return saved
    started = time.time(); df = simulate(n, seed); cols = list(ENGINEERED)
    if injected:
        df['Shortcut sensor'] = df.HDF+np.random.default_rng(seed).normal(0, .25, n)
        cols += ['Shortcut sensor']
    probabilities, shifted_probabilities = np.zeros(n), np.zeros(n)
    parts = []
    for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(df, df['Machine failure'])):
        pre, est = fit_pipeline(df, tr, cols, params, 42)
        probabilities[te] = est.predict_proba(pre.transform(df.loc[te, cols+['Type']]))[:, 1]
        shifted = df.loc[te, cols+['Type']].copy()
        if injected:
            shifted['Shortcut sensor'] = np.random.default_rng([20261023, seed, n, fold]).permutation(shifted['Shortcut sensor'].to_numpy())
        shifted_probabilities[te] = est.predict_proba(pre.transform(shifted))[:, 1]
        rows = np.intersect1d(te, audit_rows(df))
        parts.append(abs_shap(est, pre, df, rows, cols))
    observed = statistics(pd.concat(parts), df, mechanism_groups(df), ['Shortcut sensor'] if injected else [])
    metrics = {}
    for name, p in [('matched', probabilities), ('shortcut_permuted', shifted_probabilities)]:
        pred = p >= .5
        metrics[name] = {'F1': float(f1_score(df['Machine failure'], pred)),
                         'mechanism_recall': {m: {'detected': int(pred[df[m].to_numpy() == 1].sum()),
                                                  'total': int(df[m].sum())} for m in PHYSICAL}}
    out = {'n': n, 'generation_seed': seed, 'model_fold_seed': 42, 'sigma': .25 if injected else None,
           'observed': observed, 'metrics': metrics, 'seconds': time.time()-started, 'provenance': provenance}
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(out, allow_nan=False)); temp.replace(path)
    print(n, seed, injected, 'seconds', round(out['seconds']), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--resume',action='store_true'); args = ap.parse_args()
    params = json.loads((ROOT/'models/xgboost_params.json').read_text())
    records = [condition(n, seed, injected, params, args.resume) for n in [10000,30000,100000,300000]
               for seed in range(5) for injected in [False, True]]
    out = {'complete': True, 'protocol': 'experiments/SCIENTIFIC_GAPS_PROTOCOL.md', 'records': records,
           'reference_draws': 0, 'interpretation': 'Descriptive effects and held-out model reliance only'}
    (ROOT/'experiments/results/effect_replication.json').write_text(json.dumps(out, indent=1, allow_nan=False))


if __name__ == '__main__':
    main()
