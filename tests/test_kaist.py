"""Checks for the scientific weighting and feature contract."""
import numpy as np
import pandas as pd
from pdm_audit.kaist import features, metrics, weights


def test_weights_are_invariant_to_windows_and_severity_counts():
    frame = pd.DataFrame({'recording':['n']*2+['i1']*3+['i2']*7+['o']*4,
                          'label':[0]*2+[1]*10+[2]*4})
    w = weights(frame)
    totals = frame.assign(weight=w).groupby('label').weight.sum().to_numpy()
    assert np.allclose(totals, totals[0])
    totals = frame.assign(weight=w).query('label == 1').groupby('recording').weight.sum()
    assert np.isclose(totals.iloc[0], totals.iloc[1])


def test_all_fault_baseline_has_zero_healthy_specificity():
    frame = pd.DataFrame({'recording':['n','i','o'],'label':[0,1,2]})
    result = metrics(frame,np.ones(3,dtype=int))
    assert result['healthy_FPR'] == 1
    assert result['binary_balanced_accuracy'] == .5
    assert result['balanced_accuracy'] == 1/3


def test_feature_contract_has_four_channels_and_unit_power():
    fs = 25600
    t = np.arange(fs)/fs
    x = np.column_stack([np.sin(2*np.pi*(200+100*j)*t) for j in range(4)])
    values = features(x,fs)
    assert len(values) == 40
    for j in range(4):
        assert np.isclose(sum(v for k,v in values.items() if k.startswith(f'ch{j+1}_band_')),1)
        assert np.isclose(values[f'ch{j+1}_log_rms'],values[f'ch{j+1}_log_std'])
