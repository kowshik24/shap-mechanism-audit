import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.industrial import table_metadata, rank_capacity, split_support


def test_table_metadata_uses_rows_not_names_and_keeps_diameter():
    html = '''<table><tr><td>0.007"</td><td>2</td><td>1750</td>
    <td><a href="/a.mat">IR007_0</a></td><td></td><td></td></tr>
    <tr><td>&nbsp;</td><td>3</td><td>1730</td><td></td><td></td>
    <td><a href="/b.mat">OR007@6_3</a></td></tr></table>'''
    result = table_metadata(html, 'https://example.org/table')
    assert result['IR007_0']['load_hp'] == 2  # Deliberate inconsistent name.
    assert result['OR007@6_3']['diameter_in'] == .007
    assert result['OR007@6_3']['outer_race_position'] == '6:00'
    assert result['IR007_0']['url'] == 'https://example.org/a.mat'


def test_calibration_resolution_abstains_without_enough_batches():
    for n in [0, 2, 3, 11, 18]:
        assert not rank_capacity(n)['finite_threshold_available']
    assert rank_capacity(2)['rank'] == 3
    assert rank_capacity(11)['rank'] == 12
    assert rank_capacity(19)['finite_threshold_available']
    assert rank_capacity(19)['rank'] == 19


def test_split_gate_detects_probability_reliance_outside_allowed_features():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]*20)
    model = RandomForestClassifier(n_estimators=8, max_features=None, random_state=2).fit(x, x[:, 1])
    assert not split_support(model, ['anchor', 'context'], ['anchor'])['coordinate_invariant']
    assert split_support(model, ['anchor', 'context'], ['context'])['coordinate_invariant']
