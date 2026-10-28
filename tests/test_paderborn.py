import pytest
from pdm_audit.paderborn import parse_capture_name, capacity, FUTURE_SPLIT


def test_repeated_captures_and_conditions_keep_one_bearing_cluster():
    conditions = ['N15_M07_F10', 'N09_M07_F10']
    a = parse_capture_name('N15_M07_F10_KA01_1.mat', ['KA01'], conditions)
    b = parse_capture_name('N09_M07_F10_KA01_20.mat', ['KA01'], conditions)
    assert a['independent_cluster'] == b['independent_cluster'] == 'KA01'
    for name in ['N15_M07_F10_KA01_0.mat', 'N15_M07_F10_KA01_21.mat',
                 'N15_M07_F10_KA02_1.mat', 'N10_M07_F10_KA01_1.mat']:
        with pytest.raises(ValueError):
            parse_capture_name(name, ['KA01'], conditions)


def test_maximum_capacity_does_not_count_repeated_measurements_as_bearings():
    rows = ([{'diagnosis_class': 'healthy'}]*6+[{'diagnosis_class': 'IR'}]*11+
            [{'diagnosis_class': 'OR'}]*12+[{'diagnosis_class': 'mixed_or_secondary'}]*3)
    c = capacity(rows)
    assert c['optimistic_all_damaged_batches_before_training_or_test'] == 13
    assert c['optimistic_after_one_test']['calibration_batches'] == 12
    assert c['optimistic_after_one_test']['rank'] == 13
    assert c['pure_IR_OR_after_one_test']['calibration_batches'] == 10
    assert not c['optimistic_after_one_test']['finite_threshold_available']


def test_future_bearing_split_is_disjoint_and_mixed_damage_is_not_forced_pure():
    ids = [b for values in FUTURE_SPLIT.values() for b in values]
    assert len(ids) == len(set(ids)) == 32
    assert set(FUTURE_SPLIT['mixed_challenge']) == {'KB23', 'KB24', 'KB27'}
