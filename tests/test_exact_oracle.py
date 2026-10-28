import numpy as np
from pdm_audit.exact_oracle import STATES, ALIGNED, MISALIGNED, exact_shap, conformal_threshold
from pdm_audit.exact_oracle import statistics


def test_exact_oracle_additivity_and_unused_input():
    p = np.arange(1, 17, dtype=float); p /= p.sum()
    phi = exact_shap(ALIGNED, p)
    assert np.allclose(phi.sum(axis=1), ALIGNED-ALIGNED@p)
    assert np.all(phi[:, 3] == 0)
    wrong = exact_shap(MISALIGNED, p)
    assert np.all(wrong[:, 0] == 0)
    assert np.max(np.abs(wrong[:, 3])) > .1


def test_linear_oracle_matches_analytic_shap():
    p = np.arange(1, 17, dtype=float); p /= p.sum()
    beta = np.array([2., -.5, 0., 4.])
    assert np.allclose(exact_shap(STATES@beta, p), (STATES-p@STATES)*beta)
    assert conformal_threshold(list(range(200))) == 190


def test_nested_true_causes_produce_large_outside_excess():
    p = q = .05
    weights = np.zeros(16)
    for i, (a, b, c, dummy) in enumerate(STATES):
        if a == b and dummy == 0:
            weights[i] = (p if a else 1-p)*(q if c else 1-q)
    result = statistics(exact_shap(ALIGNED, weights), weights)['0']
    own_share = 3/(6+4*q)
    other_share = p/(6-2*p)
    assert np.isclose(result['R'], own_share)
    assert np.isclose(result['S'], own_share-other_share)
    assert result['S'] > .47 and result['S_feature'] == 1
