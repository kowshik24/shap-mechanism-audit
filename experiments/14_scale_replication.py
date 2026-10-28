"""Checkpointed 300k replication with independently generated reference draws.

The 200-draw block is preliminary; the full planned analysis uses 1000 draws.
Every reference draw is saved, so interruption does not discard completed work.
"""
import pdm_audit
import argparse
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pdm_audit.audit import shap_cross_fitted, statistics
from pdm_audit.config import ENGINEERED, PHYSICAL
from pdm_audit.data import mechanism_groups
from pdm_audit.sim import simulate
from pdm_audit.validation import monte_carlo_family

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'


def write(path, obj):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(obj, allow_nan=False)); temp.replace(path)


def draw(k, df, cols, params, seed, checkpoint):
    path = checkpoint/f'draw_{k:04d}.json'
    if path.exists():
        return json.loads(path.read_text())
    labels = pd.Series(np.random.default_rng([20261023, seed, int(len(cols) > len(ENGINEERED)), k])
                       .permutation(df['Machine failure']), index=df.index)
    a = shap_cross_fitted(df, cols, params, 42, labels=labels)
    st = statistics(a, df, mechanism_groups(df))
    obj = {f'{m}.{s}': st[m][s] for m in PHYSICAL for s in 'RDS'}
    write(path, obj)
    return obj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=1000)
    ap.add_argument('--jobs', type=int, default=8); args = ap.parse_args()
    params = json.loads((ROOT/'models/xgboost_params.json').read_text())
    records = []
    for seed in [1, 2]:
        for injected in [False, True]:
            name = f'scale_300000_seed{seed}_injected{int(injected)}'
            checkpoint = RESULTS/'scale_replication_checkpoints'/name
            checkpoint.mkdir(parents=True, exist_ok=True)
            started = time.time(); df = simulate(300000, seed=seed); cols = list(ENGINEERED)
            if injected:
                # Same historical noise stream, distinct from the generator's RNG instance.
                df['Shortcut sensor'] = df.HDF+np.random.default_rng(seed).normal(0, .25, len(df))
                cols += ['Shortcut sensor']
            observed_path = checkpoint/'observed.json'
            if observed_path.exists():
                observed = json.loads(observed_path.read_text())
            else:
                a = shap_cross_fitted(df, cols, params, 42)
                observed = statistics(a, df, mechanism_groups(df), ['Shortcut sensor'] if injected else [])
                write(observed_path, observed)
            stream = Parallel(n_jobs=args.jobs, return_as='generator_unordered')(
                delayed(draw)(k, df, cols, params, seed, checkpoint) for k in range(args.K))
            for i, _ in enumerate(stream, 1):
                if i % 50 == 0:
                    print(name, i, '/', args.K, 'elapsed', round(time.time()-started), flush=True)
            draws = [json.loads((checkpoint/f'draw_{k:04d}.json').read_text()) for k in range(args.K)]
            families = {s: monte_carlo_family({m: observed[m][s] for m in PHYSICAL},
                                              {m: [d[f'{m}.{s}'] for d in draws] for m in PHYSICAL}) for s in 'RDS'}
            records.append({'n': len(df), 'generation_seed': seed, 'model_fold_seed': 42,
                            'sigma': .25 if injected else None, 'K': args.K,
                            'observed': observed, 'families': families,
                            'raw_draws': draws, 'seconds_this_invocation': time.time()-started})
            write(RESULTS/f'scale_replication_K{args.K}.json',
                  {'complete': len(records) == 4, 'preliminary': args.K < 1000,
                   'protocol': 'experiments/SCIENTIFIC_GAPS_PROTOCOL.md', 'records': records})
    print('COMPLETE', args.K, flush=True)


if __name__ == '__main__':
    main()
