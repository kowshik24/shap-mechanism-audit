"""Exact batched audit scores for fixed finite-state fitted functions."""
import numpy as np
from .exact_oracle import STATES,ALIGNED,exact_shap


def population_probabilities(regime_probability,low,high,overlap):
    """Sixteen-state mass under the original overlap population."""
    parts = []
    z = STATES[:,:3]
    for p in [np.asarray(low),np.asarray(high)]:
        probabilities = np.broadcast_to(p,(16,3)).copy()
        probabilities[:,1] += (1-p[1])*overlap*z[:,0]
        weight = np.prod(np.where(z == 1,probabilities,1-probabilities),axis=1)
        context_p = .01+.98*z[:,0]
        weight *= np.where(STATES[:,3] == 1,context_p,1-context_p)
        parts.append(weight)
    mass = (1-regime_probability)*parts[0]+regime_probability*parts[1]
    assert np.isclose(mass.sum(),1) and np.all(mass >= 0)
    return mass


def calibration_counts(n,rng,count=200):
    rows = []
    for _ in range(count):
        q = rng.uniform(.1,.4);low,high = rng.uniform(.01,.04,3),rng.uniform(.1,.3,3)
        mass = population_probabilities(q,low,high,rng.uniform(0,1))
        rows.append(rng.multinomial(n,mass))
    return np.asarray(rows)


def shap_kernel(values):
    return np.stack([exact_shap(values,p) for p in np.eye(16)],axis=-1)


def batch_max_score(values,counts):
    """Exact ratios of mean absolute attribution; no absolute-background averaging."""
    counts = np.asarray(counts,dtype=float)
    background = counts/counts.sum(axis=1,keepdims=True)
    signed = np.einsum('sfb,kb->ksf',shap_kernel(values),background)
    weighted = np.abs(signed)*counts[:,:,None]
    scores = []
    for m in range(3):
        own = (STATES[:,m] == 1) & ALIGNED.astype(bool)
        rest = (~own) & ALIGNED.astype(bool)
        a,b = weighted[:,own].sum(axis=1),weighted[:,rest].sum(axis=1)
        assert np.all(a.sum(axis=1) > 0) and np.all(b.sum(axis=1) > 0)
        a,b = a/a.sum(axis=1,keepdims=True),b/b.sum(axis=1,keepdims=True)
        excess = a-b;excess[:,m] = -np.inf
        scores.append(excess.max(axis=1))
    return np.stack(scores,axis=1).max(axis=1)
