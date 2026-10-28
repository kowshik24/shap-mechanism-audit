import numpy as np
from pdm_audit.physical_acquisition import ball_pass_orders, spectral_anchors, duplicate_groups, resample_native_time


def test_bearing_specific_geometry_changes_the_physical_anchor():
    g = {'rolling_elements': 8, 'ball_diameter_mm': 6.75, 'pitch_diameter_mm': 28.55,
         'contact_angle_degrees': 0}
    a = ball_pass_orders(g); b = ball_pass_orders({**g, 'pitch_diameter_mm': 29.05})
    assert np.isclose(a['IR']+a['OR'], 8)
    assert b['IR'] < a['IR'] and b['OR'] > a['OR']
    assert np.isclose(b['IR'], 4*(1+6.75/29.05))


def test_predeclared_envelope_transform_finds_amplitude_modulation():
    fs = 32000; t = np.arange(fs*2)/fs
    modulation = 76.25
    x = (1+.3*np.cos(2*np.pi*modulation*t)+.2*np.cos(4*np.pi*modulation*t)+
         .1*np.cos(6*np.pi*modulation*t))*np.cos(2*np.pi*5000*t)
    rows = spectral_anchors(x, fs, 1500, {'OR': modulation/25})['OR']['harmonics']
    assert all(abs(r['peak_Hz']-modulation*r['harmonic']) < .6 for r in rows)
    assert all(r['screen_above_3dB'] for r in rows)


def test_duplicate_capture_names_do_not_create_independent_payloads():
    rows = [{'file': 'a', 'channel_sha256': {'vibration_1': 'x'}, 'all_Y_sha256': 'A'},
            {'file': 'b', 'channel_sha256': {'vibration_1': 'x'}, 'all_Y_sha256': 'A'},
            {'file': 'c', 'channel_sha256': {'vibration_1': 'z'}, 'all_Y_sha256': 'C'}]
    assert duplicate_groups(rows, 'vibration') == [['a', 'b']]
    assert duplicate_groups(rows, 'all') == [['a', 'b']]
    t = np.arange(9)/8
    assert np.array_equal(resample_native_time(t, t, fs=8), t)
