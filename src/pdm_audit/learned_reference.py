"""Finite-world learned controls with explicit functional eligibility checks."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from .exact_oracle import STATES, ALIGNED, STATE_WEIGHTS, exact_shap, statistics

VIEWS = {'mechanism_only':[0,1,2], 'proxy_only':[3,1,2], 'unrestricted':[0,1,2,3]}
FAMILIES = ['tree','forest','logistic']


def training_data(n, seed):
    """Independent noisy training labels; no audit rows enter a fit."""
    rng = np.random.default_rng([20261023,20,n,seed,1])
    z = rng.integers(0,2,size=(512,3))
    context = z[:,0] ^ (rng.random(len(z)) < .01)
    x = np.column_stack([z,context])
    y = z.max(axis=1) ^ (rng.random(len(z)) < .05)
    return x,y.astype(int)


def audit_data(n, seed):
    """Paired overlap increase leaves aggregate outcomes and context unchanged."""
    rng = np.random.default_rng([20261023,20,n,seed,2])
    q = float(rng.uniform(.1,.4))
    low, high = rng.uniform(.01,.04,3),rng.uniform(.1,.3,3)
    regime = rng.random(n) < q
    z = (rng.random((n,3)) < np.where(regime[:,None],high,low)).astype(int)
    overlap = float(rng.uniform(0,1)); nested = rng.random(n)
    original, shifted = z.copy(),z.copy()
    original[:,1] |= ((z[:,0] == 1) & (nested < overlap)).astype(int)
    shifted[:,1] |= ((z[:,0] == 1) & (nested < .8+.2*overlap)).astype(int)
    context = z[:,0] ^ (rng.random(n) < .01)
    x = np.column_stack([original,context]); shift_x = np.column_stack([shifted,context])
    assert np.array_equal(x[:,:3].max(axis=1),shift_x[:,:3].max(axis=1))
    permutation = np.random.default_rng([20261023,20,n,seed,3]).permutation(n)
    changed = x.copy(); changed[:,3] = x[permutation,3]
    return x,shift_x,changed,{'regime_probability':q,'low_probabilities':low.tolist(),
                             'high_probabilities':high.tolist(),'overlap':overlap,
                             'shifted_overlap':.8+.2*overlap}


def estimator(family, n_features, seed):
    if family == 'tree':
        return DecisionTreeClassifier(max_depth=n_features,min_samples_leaf=5,random_state=seed)
    if family == 'forest':
        return RandomForestClassifier(n_estimators=64,max_depth=n_features,min_samples_leaf=5,
                                      max_features='sqrt',bootstrap=True,n_jobs=1,random_state=seed)
    if family == 'logistic':
        return LogisticRegression(C=1,solver='lbfgs',max_iter=1000,random_state=seed)
    raise ValueError(f'Unknown family: {family}')


def functional_gate(values):
    """Contract on the complete 16-state domain, not only an observed sample."""
    values = np.asarray(values)
    assert values.shape == (16,) and np.isfinite(values).all()
    delta = float(np.max(np.abs(values[1::2]-values[::2])))
    decisions = values >= .5
    agreement = float(np.mean(decisions == ALIGNED))
    return {'eligible':bool(agreement == 1 and delta <= 1e-12),
            'full_state_decision_agreement':agreement,'max_context_probability_difference':delta,
            'context_decision_change_pairs':int((decisions[1::2] != decisions[::2]).sum()),
            'material_context_dependence':bool(delta >= .05)}


def describe(values, x, changed_context):
    """Exact explanation and fixed-function reliance on an untouched audit batch."""
    counts = np.bincount(x@STATE_WEIGHTS,minlength=16)
    phi = exact_shap(values,counts/len(x))
    stats = statistics(phi,counts)
    y = x[:,:3].max(axis=1)
    def f1(inputs):
        pred = np.asarray(values)[inputs@STATE_WEIGHTS] >= .5
        return float(2*np.sum(pred*y)/(pred.sum()+y.sum()))
    return {'statistics':stats,'max_S':max(r['S'] for r in stats.values()),
            'F1':f1(x),'context_permuted_F1':f1(changed_context)}
