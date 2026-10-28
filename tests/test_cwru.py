import numpy as np
from pdm_audit.cwru import features, EXPECTED


def test_gain_isolated_to_amplitude():
    rng = np.random.default_rng(17)
    x = rng.normal(size=4096)
    original, changed = features(x, 1772), features(4*x, 1772)
    assert np.isclose(changed['log_RMS']-original['log_RMS'], np.log(4))
    for key in original.keys()-{'log_RMS'}:
        assert np.isclose(original[key], changed[key], rtol=1e-10, atol=1e-12), key


def test_frequency_anchor_not_broadband_amplitude():
    fs, rpm = 12000, 1800
    t = np.arange(4096)/fs
    x = np.sin(2*np.pi*5.4152*rpm/60*t) + .01*np.random.default_rng(9).normal(size=len(t))
    result = features(x, rpm)
    assert result['raw_IR_1'] > .9
    assert result['raw_OR_1'] < .001 and result['raw_B_1'] < .001
    assert len(EXPECTED['IR']) == 4
