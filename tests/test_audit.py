import numpy as np
import pandas as pd
import pytest

import pdm_audit  # noqa: F401  (sets threading env vars)
from pdm_audit.audit import holm, permutation_p, statistics
from pdm_audit.config import MODES, PHYSICAL, TRIGGER_FEATURES
from pdm_audit.data import label_facts, load, mechanism_groups
from pdm_audit.metrics import clopper_pearson


def test_permutation_p_bounds():
    null = np.linspace(0, 1, 99)
    assert permutation_p(2.0, null) == pytest.approx(1 / 100)
    assert permutation_p(-1.0, null) == pytest.approx(1.0)


def test_holm_matches_manual():
    adj = holm({'a': 0.01, 'b': 0.04, 'c': 0.03})
    assert adj['a'] == pytest.approx(0.03)
    assert adj['c'] == pytest.approx(0.06)
    assert adj['b'] == pytest.approx(0.06)


def test_clopper_pearson_all_successes():
    lo, hi = clopper_pearson(28, 28)
    assert hi == 1.0 and 0.87 < lo < 0.88


def test_label_facts_match_paper():
    facts = label_facts(load())
    assert facts['machine_failure_total'] == 339
    assert facts['failures_with_no_mode'] == 9
    assert facts['rows_with_multiple_modes'] == 24
    assert facts['RNF_with_machine_failure_0'] == 18


def test_statistics_on_synthetic_attributions():
    """If attribution sits only on a mechanism's trigger features for its own rows, R = 1 and D > 0."""
    df = load()
    groups = mechanism_groups(df)
    rows = np.union1d(df.index[df['Machine failure'] == 1], df.index[df['RNF'] == 1])
    features = sorted({f for fs in TRIGGER_FEATURES.values() for f in fs} | {'Type_H'})
    A = pd.DataFrame(0.0, index=rows, columns=features)
    A['Type_H'] = 1.0
    A.loc[groups['PWF'], TRIGGER_FEATURES['PWF']] = 5.0
    A.loc[groups['PWF'], 'Type_H'] = 0.0
    stats = statistics(A, df, groups)
    assert stats['PWF']['R'] == pytest.approx(1.0)
    assert stats['PWF']['D'] > 0.5
    assert set(stats) == set(PHYSICAL) | {'RNF'}
    assert all(m in stats for m in MODES)
