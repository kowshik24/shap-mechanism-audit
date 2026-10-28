import numpy as np
import pandas as pd
from pdm_audit.temporal import add_context, chronological_split, inner_folds, CONTEXT


def fixture():
    return pd.DataFrame({'Tool wear [min]': np.tile([2, 4, 6, 8], 20),
                         'Torque [Nm]': np.arange(80)+20., 'Temp_diff [K]': np.arange(80)/10})


def test_context_never_reads_future():
    original = fixture(); changed = original.copy()
    changed.loc[41:, ['Tool wear [min]', 'Torque [Nm]', 'Temp_diff [K]']] = 999
    pd.testing.assert_frame_equal(add_context(original).loc[:40, CONTEXT], add_context(changed).loc[:40, CONTEXT])
    result = add_context(original)
    assert result.loc[4, 'Previous wear'] == 0
    assert result.loc[6, 'Past mean torque'] == 24.5


def test_chronological_splits_keep_cycles_intact():
    df = add_context(fixture()); tr, va, te = chronological_split(df)
    assert max(tr) < min(va) and max(va) < min(te)
    for a, b in [(tr, va), (va, te), *list(inner_folds(df, tr))]:
        assert set(df.loc[a, 'Wear cycle']).isdisjoint(df.loc[b, 'Wear cycle'])
        assert max(a) < min(b)
