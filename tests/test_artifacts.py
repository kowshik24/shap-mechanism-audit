from copy import deepcopy
from pdm_audit.artifacts import scientific_records_match


def test_runtime_variation_does_not_hide_changed_scientific_results():
    original = {'folds': [{'seconds': 2., 'R': .45, 'p_holm': .04975, 'supported': True}],
                'seconds_per_run': .5, 'features': ['wear', 'power']}
    repeated = deepcopy(original)
    repeated['folds'][0]['seconds'] = 3.
    repeated['seconds_per_run'] = .75
    assert scientific_records_match(original, repeated)
    for key, value in [('R', .46), ('p_holm', .05001), ('supported', False)]:
        changed = deepcopy(repeated); changed['folds'][0][key] = value
        assert not scientific_records_match(original, changed)
    changed = deepcopy(repeated); changed['features'].reverse()
    assert not scientific_records_match(original, changed)
    changed = deepcopy(repeated); changed['folds'][0]['seconds'] = 'missing'
    assert not scientific_records_match(original, changed)
