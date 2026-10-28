"""Explicit background, normalization, and comparison-cohort ablations."""
import numpy as np
from .exact_oracle import STATES, ALIGNED
from .reference_transfer import kernel


def contributions(values, counts, background=None):
    values = np.asarray(values, dtype=float)
    counts = np.asarray(counts, dtype=float)
    probability = counts / counts.sum(axis=1, keepdims=True)
    if background is not None:
        background = np.asarray(background, dtype=float)
        if background.shape != (16,) or not np.isclose(background.sum(), 1):
            raise ValueError('Expected a probability distribution on 16 states')
        probability = np.broadcast_to(background, probability.shape)
    signed = np.einsum('sfb,kb->ksf', kernel(tuple(values)), probability)
    error = np.max(abs(signed.sum(axis=2) -
                       (np.asarray(values)[None, :] - probability @ values[:, None])))
    if error > 1e-12:
        raise AssertionError(f'SHAP additivity error {error}')
    return abs(signed)


def excess(local, counts, normalize=True, healthy=False, bounds=False):
    """Return maximum excess, C-localization, validity, and optional map bounds."""
    counts = np.asarray(counts, dtype=float)
    weighted = local * counts[:, :, None]
    valid = np.ones(len(counts), dtype=bool)
    maxima, context, lower = [], [], []
    for m in range(3):
        own = STATES[:, m] == 1
        rest = ~ALIGNED.astype(bool) if healthy else ALIGNED.astype(bool) & ~own
        left, right = weighted[:, own].sum(axis=1), weighted[:, rest].sum(axis=1)
        dl = left.sum(axis=1) if normalize else counts[:, own].sum(axis=1)
        dr = right.sum(axis=1) if normalize else counts[:, rest].sum(axis=1)
        valid &= (dl > 0) & (dr > 0)
        left = np.divide(left, dl[:, None], out=np.zeros_like(left), where=dl[:, None] > 0)
        right = np.divide(right, dr[:, None], out=np.zeros_like(right), where=dr[:, None] > 0)
        delta = left - right
        delta[:, m] = -np.inf
        maximum = delta.max(axis=1)
        maxima.append(maximum)
        context.append(np.where(delta.argmax(axis=1) == 3, maximum, -np.inf))
        lower.append(delta[:, 3])
    a, l = np.max(maxima, axis=0), np.max(context, axis=0)
    a[~valid], l[~valid] = np.inf, -np.inf
    if bounds:
        lo = np.max(lower, axis=0)
        lo[~valid] = -np.inf
        return a, lo, valid
    return a, l, valid


def targeted(local, counts):
    value = (local[:, :, 3] * counts).sum(axis=1) / counts.sum(axis=1)
    return value, value, np.isfinite(value)
