import numpy as np
from pdm_audit.acquisition import resonant_transfer
from pdm_audit.cwru import features


def test_resonance_changes_spectrum_without_an_amplitude_shortcut():
    rng = np.random.default_rng(4); x = rng.normal(size=4096); x -= x.mean()
    y = resonant_transfer(x, 'IR')
    assert np.isclose(np.mean(x*x), np.mean(y*y), atol=1e-14)
    assert not np.allclose(np.abs(np.fft.rfft(x)), np.abs(np.fft.rfft(y)))
    a, b = features(x,1772),features(y,1772)
    assert np.isclose(a['log_RMS'],b['log_RMS'],atol=1e-12)
    assert any(abs(a[k]-b[k]) > .01 for k in a if k.startswith('band_'))
