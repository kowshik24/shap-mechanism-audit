"""Observed-stratum calibration and independent target-size tuning."""
import numpy as np
from .exact_oracle import STATES, ALIGNED, conformal_threshold
from .fixed_reference import shap_kernel

BINS = np.array([0., .8, .9, 1.000000001])
ALPHAS = [.01, .025, .05, .10, .20]
METHODS = ['global_source', 'global_target', 'observed_source', 'observed_target']


def sample_batches(n, rng, batches, target=False):
    q = rng.uniform(.1, .4, (batches, 1))
    low, high = rng.uniform(.01, .04, (batches, 3)), rng.uniform(.1, .3, (batches, 3))
    u = rng.uniform(.8 if target else 0., 1., (batches, 1))
    z = STATES[:, :3]
    pieces = []
    for p in [low, high]:
        probabilities = np.broadcast_to(p[:, None, :], (batches, 16, 3)).copy()
        probabilities[:, :, 1] += (1-p[:, 1:2])*u*z[None, :, 0]
        mass = np.prod(np.where(z[None, :, :] == 1, probabilities, 1-probabilities), axis=2)
        context_p = .01+.98*z[:, 0]
        mass *= np.where(STATES[:, 3] == 1, context_p, 1-context_p)[None, :]
        pieces.append(mass)
    mass = (1-q)*pieces[0]+q*pieces[1]
    assert np.allclose(mass.sum(axis=1), 1)
    return rng.multinomial(n, mass), u[:, 0]


def observed_overlap(counts):
    counts = np.asarray(counts)
    denominator = counts[:, STATES[:, 0] == 1].sum(axis=1)
    numerator = counts[:, (STATES[:, 0] == 1) & (STATES[:, 1] == 1)].sum(axis=1)
    return np.divide(numerator, denominator, out=np.full(len(counts), np.nan), where=denominator > 0)


def overlap_bins(counts):
    overlap = observed_overlap(counts)
    bins = np.searchsorted(BINS[1:-1], overlap, side='right')
    bins[~np.isfinite(overlap)] = -1
    return bins


def batch_scores(values, counts):
    counts = np.asarray(counts, dtype=float)
    signed = np.einsum('sfb,kb->ksf', shap_kernel(values), counts/counts.sum(axis=1, keepdims=True))
    weighted = np.abs(signed)*counts[:, :, None]
    valid = np.ones(len(counts), dtype=bool); all_s, context_s = [], []
    for m in range(3):
        own = (STATES[:, m] == 1) & ALIGNED.astype(bool)
        rest = ~own & ALIGNED.astype(bool)
        a, b = weighted[:, own].sum(axis=1), weighted[:, rest].sum(axis=1)
        da, db = a.sum(axis=1, keepdims=True), b.sum(axis=1, keepdims=True)
        valid &= (da[:, 0] > 0) & (db[:, 0] > 0)
        a = np.divide(a, da, out=np.zeros_like(a), where=da > 0)
        b = np.divide(b, db, out=np.zeros_like(b), where=db > 0)
        difference = a-b; difference[:, m] = -np.inf
        j = difference.argmax(axis=1); s = difference.max(axis=1)
        all_s.append(s); context_s.append(np.where(j == 3, s, -np.inf))
    score = np.max(all_s, axis=0); local = np.max(context_s, axis=0)
    score[~valid] = np.inf; local[~valid] = -np.inf
    return score, local, valid


def reference_thresholds(scores, counts, alpha=.05):
    bins = overlap_bins(counts)
    return {'global': conformal_threshold(scores, alpha),
            'observed': [conformal_threshold(scores[bins == k], alpha) for k in range(3)],
            'bin_counts': [int((bins == k).sum()) for k in range(3)]}


def thresholds_on(reference, counts, observed):
    if not observed:
        return np.full(len(counts), reference['global'])
    bins = overlap_bins(counts); q = np.full(len(counts), np.inf)
    for k in range(3):
        q[bins == k] = reference['observed'][k]
    return q


def residual(score, threshold, valid):
    result = np.full(len(score), -np.inf)
    usable = valid & np.isfinite(threshold)
    result[usable] = score[usable]-threshold[usable]
    return result


def warning(score, threshold, valid):
    return valid & np.isfinite(threshold) & (score > threshold)


def coupled_context_counts(counts, errors, rng):
    """Common Z counts and nested bit-flip uniforms for each proxy strength."""
    zcounts = np.asarray(counts).reshape(-1, 8, 2).sum(axis=2)
    probabilities = np.diff(np.r_[0., errors, 1.])
    intervals = rng.multinomial(zcounts, probabilities)
    cumulative = np.cumsum(intervals, axis=2)
    result = []
    z0 = STATES[::2, 0]
    for k in range(len(errors)):
        flips = cumulative[:, :, k]
        ones = np.where(z0[None, :] == 0, flips, zcounts-flips)
        out = np.empty((len(counts), 8, 2), dtype=np.int64)
        out[:, :, 1] = ones; out[:, :, 0] = zcounts-ones
        result.append(out.reshape(-1, 16))
    return np.array(result)


def permute_context_counts(counts, rng):
    output = []
    for row in counts:
        z = row.reshape(8, 2).sum(axis=1); total_ones = row[1::2].sum()
        ones = rng.multivariate_hypergeometric(z, total_ones)
        output.append(np.stack([z-ones, ones], axis=1).reshape(16))
    return np.asarray(output)


def table_f1(values, counts):
    prediction = np.asarray(values) >= .5; y = ALIGNED.astype(bool)
    return 2*counts[:, prediction & y].sum(axis=1)/(counts[:, prediction].sum(axis=1)+counts[:, y].sum(axis=1))
