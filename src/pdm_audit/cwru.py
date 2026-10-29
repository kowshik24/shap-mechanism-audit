"""
CWRU bearing data (12 kHz drive-end recordings) as a measured-data test of the mechanism audit.

Mechanisms are the three fault locations (inner race IR, rolling element B, outer race OR). Each has a
characteristic defect frequency, a fixed multiple of the shaft speed set by the bearing geometry; the
expected features E_m are the spectral power shares at that frequency and its second harmonic, in the
raw and the envelope spectrum. Features follow the paper/new-experiments branch (19 per window).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt
from scipy.stats import kurtosis

ORDERS = {'IR': 5.4152, 'B': 4.7135, 'OR': 3.5848}  # defect frequency / shaft frequency (SKF 6205-2RS)
FAULTS = list(ORDERS)
EXPECTED = {m: [f'{domain}_{m}_{h}' for domain in ('raw', 'envelope') for h in (1, 2)] for m in ORDERS}


def features(signal, rpm, fs=12000):
    """Nineteen prespecified features of one window; only log_RMS changes with a gain."""
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    env = np.abs(hilbert(sosfiltfilt(butter(4, [2000, 5000], fs=fs, btype='bandpass', output='sos'), x)))
    out = {}
    raw_spectrum = None
    for domain, values in [('raw', x), ('envelope', env - env.mean())]:
        freq, power = periodogram(values, fs=fs, window='hann')
        if domain == 'raw':
            raw_spectrum = (freq, power)
        total = max(power.sum(), np.finfo(float).tiny)
        for m, order in ORDERS.items():
            for harmonic in (1, 2):
                centre, width = order * harmonic * rpm / 60, 0.15 * rpm / 60
                out[f'{domain}_{m}_{harmonic}'] = float(power[np.abs(freq - centre) <= width].sum() / total)
    rms = np.sqrt(np.mean(x * x))
    out.update(log_RMS=float(np.log(max(rms, np.finfo(float).tiny))),
               kurtosis=float(kurtosis(x, fisher=False)), crest=float(np.max(np.abs(x)) / rms))
    freq, power = raw_spectrum
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6001)]:
        out[f'band_{lo}_{hi}'] = float(power[(freq >= lo) & (freq < hi)].sum() / power.sum())
    return out


def load(directory, windows=24, width=4096):
    """One row per window; fault, defect size and motor load parsed from file names like IR014_2.mat."""
    rows = []
    for path in sorted(Path(directory).glob('*.mat')):
        stem = path.stem
        fault = next(f for f in ('IR', 'OR', 'B') if stem.startswith(f))
        mat = loadmat(path)
        channel = next(k for k in mat if k.endswith('_DE_time'))
        rpm = float(mat[next(k for k in mat if k.endswith('RPM'))].ravel()[0])
        x = mat[channel].ravel()
        assert len(x) >= windows * width, stem
        for w in range(windows):
            rows.append({**features(x[w * width:(w + 1) * width], rpm), 'recording': stem, 'fault': fault,
                         'size': stem[len(fault):len(fault) + 3], 'load_hp': int(stem.rsplit('_', 1)[1]),
                         'window': w})
    return pd.DataFrame(rows)


FEATURES = [f'{d}_{m}_{h}' for d in ('raw', 'envelope') for m in ORDERS for h in (1, 2)] + \
           ['log_RMS', 'kurtosis', 'crest'] + [f'band_{lo}_{hi}' for lo, hi in
                                              [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6001)]]
