"""Exhaustive interventional SHAP for four Boolean inputs and known functions."""
from itertools import product
from math import factorial, ceil
import numpy as np

STATES = np.array(list(product([0, 1], repeat=4)), dtype=int)
ALIGNED = STATES[:, :3].max(axis=1).astype(float)
MISALIGNED = STATES[:, [1, 2, 3]].max(axis=1).astype(float)
STATE_WEIGHTS = np.array([8, 4, 2, 1])


def exact_shap(output, background):
    """All state-wise signed values, retaining joint dependence among missing inputs."""
    output, background = np.asarray(output), np.asarray(background)
    assert output.shape == background.shape == (16,)
    assert np.isclose(background.sum(), 1)
    values = np.empty((16, 16))
    for mask in range(16):
        present = np.array([(mask >> j) & 1 for j in range(4)], dtype=bool)
        hybrid = np.where(present[None, None, :], STATES[:, None, :], STATES[None, :, :])
        values[:, mask] = output[hybrid @ STATE_WEIGHTS] @ background
    phi = np.zeros((16, 4))
    for j in range(4):
        for mask in range(16):
            if (mask >> j) & 1:
                continue
            size = mask.bit_count()
            weight = factorial(size)*factorial(3-size)/factorial(4)
            phi[:, j] += weight*(values[:, mask | (1 << j)]-values[:, mask])
    assert np.allclose(phi.sum(axis=1), output-output@background, atol=1e-12)
    return phi


def statistics(phi, counts):
    result = {}
    failure = ALIGNED.astype(bool)
    for m in range(3):
        own = failure & (STATES[:, m] == 1)
        other = failure & ~own
        a = np.average(np.abs(phi[own]), weights=counts[own], axis=0)
        b = np.average(np.abs(phi[other]), weights=counts[other], axis=0)
        a, b = a/a.sum(), b/b.sum()
        excess = a-b; excess[m] = -np.inf
        result[str(m)] = {'R': float(a[m]), 'D': float(a[m]-b[m]),
                          'S': float(excess.max()), 'S_feature': int(excess.argmax())}
    return result


def conformal_threshold(scores, alpha=.05):
    """Strict exceedance of a split-conformal order statistic; infinity if too few scores."""
    rank = ceil((len(scores)+1)*(1-alpha))
    return float(np.sort(scores)[rank-1]) if rank <= len(scores) else float('inf')
