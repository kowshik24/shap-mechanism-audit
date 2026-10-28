"""Independent calibration/evaluation seeds for explicit oracle functions."""
import pdm_audit
import argparse
import json
from pathlib import Path
import numpy as np
from pdm_audit.exact_oracle import (ALIGNED, MISALIGNED, STATE_WEIGHTS, exact_shap,
                                    statistics, conformal_threshold)
from pdm_audit.metrics import clopper_pearson

OUT = Path(__file__).resolve().parents[1]/'experiments/results/known_truth.json'


def make(n, seed, overlap=False):
    rng = np.random.default_rng([20261023, n, seed, int(overlap)])
    q = rng.uniform(.1, .4)
    low, high = rng.uniform(.01, .04, 3), rng.uniform(.1, .3, 3)
    regime = rng.random(n) < q
    z = (rng.random((n, 3)) < np.where(regime[:, None], high, low)).astype(int)
    overlap_probability = None
    if overlap:
        overlap_probability = float(rng.uniform(0, 1))
        z[:, 1] |= ((z[:, 0] == 1) & (rng.random(n) < overlap_probability)).astype(int)
    c = z[:, 0] ^ (rng.random(n) < .01)
    x = np.column_stack([z, c]); counts = np.bincount(x@STATE_WEIGHTS, minlength=16)
    background = counts/n
    result = {'seed': seed, 'n': n, 'regime_probability': q,
              'low_probabilities': low.tolist(), 'high_probabilities': high.tolist(),
              'counts': counts.tolist(), 'overlap_probability': overlap_probability}
    for name, output in [('aligned', ALIGNED), ('misaligned', MISALIGNED)]:
        stats = statistics(exact_shap(output, background), counts)
        result[name] = {'statistics': stats, 'max_S': max(v['S'] for v in stats.values())}
    y = z.max(axis=1)
    pred = MISALIGNED[x@STATE_WEIGHTS].astype(int)
    removed = x.copy(); removed[:, 3] = rng.permutation(removed[:, 3])
    shifted = MISALIGNED[removed@STATE_WEIGHTS].astype(int)
    def f1(p):
        return float(2*np.sum((y == 1) & (p == 1))/(y.sum()+p.sum()))
    result.update(misaligned_F1=f1(pred), context_permuted_F1=f1(shifted))
    return result


def count_interval(k, n=200):
    return {'count': int(k), 'n': n, 'rate': k/n, 'exact_95_interval': list(clopper_pearson(k, n))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--overlap', action='store_true'); args = ap.parse_args()
    output = {'complete': True, 'overlap_followup': args.overlap,
              'protocol': 'experiments/KNOWN_TRUTH_CALIBRATION_PROTOCOL.md', 'sizes': []}
    for n in [1000, 10000, 30000]:
        cal = [make(n, s, args.overlap) for s in range(100000, 100200)]
        evaluation = [make(n, s, args.overlap) for s in range(200000, 200200)]
        threshold = conformal_threshold([r['aligned']['max_S'] for r in cal])
        summary = {'n': n, 'threshold': threshold, 'calibration': cal, 'evaluation': evaluation}
        for name in ['aligned', 'misaligned']:
            summary[name+'_calibrated_warning'] = count_interval(sum(r[name]['max_S'] > threshold for r in evaluation))
            summary[name+'_fixed_floor_warning'] = count_interval(sum(r[name]['max_S'] > .05 for r in evaluation))
        summary['misaligned_mean_F1'] = float(np.mean([r['misaligned_F1'] for r in evaluation]))
        summary['permuted_mean_F1'] = float(np.mean([r['context_permuted_F1'] for r in evaluation]))
        output['sizes'].append(summary)
        print(n, threshold, summary['aligned_calibrated_warning'], summary['misaligned_calibrated_warning'], flush=True)
    path = OUT.with_name('known_truth_overlap.json') if args.overlap else OUT
    path.write_text(json.dumps(output, indent=1, allow_nan=False))


if __name__ == '__main__':
    main()
