import numpy as np
import pandas as pd
import pytest
from pdm_audit.validation import disjoint_feature_decoys, group_statistics, monte_carlo_family


def test_decoys_preserve_cardinality_and_do_not_reuse_expected_features():
    a = pd.DataFrame([[8., 1., 1.]], columns=['signal', 'a', 'b'])
    out = disjoint_feature_decoys(a, [0], ['signal'])
    assert out['n_decoys'] == 2
    assert out['R'] == pytest.approx(.8)
    assert out['decoy_max'] == pytest.approx(.1)
    assert out['fraction_decoys_ge_observed'] == 0


def test_monte_carlo_near_threshold_is_not_a_stable_decision():
    # One exceedance in 200 gives adjusted 4 * (2 / 201), but is imprecise.
    draws = {m: np.r_[np.zeros(199), 2.] for m in 'abcd'}
    out = monte_carlo_family({m: 1. for m in 'abcd'}, draws)
    for r in out.values():
        assert r['hits'] == 1
        assert r['p_holm'] == pytest.approx(8/201)
        assert r['holm_interval'][0] < .05 < r['holm_interval'][1]
        assert not r['mc_decision_stable']


def test_explicit_comparison_cohort_changes_only_the_reference_rows():
    a = pd.DataFrame({'wear': [8., 8., 1.], 'load': [2., 2., 9.]})
    overlap = group_statistics(a, [0], [1], ['wear'])
    excluded = group_statistics(a, [0], [2], ['wear'])
    assert overlap['D'] == pytest.approx(0)
    assert excluded['D'] == pytest.approx(.7)
    assert excluded['R'] == overlap['R']
