"""Recording-level CWRU features and grouped fault-location evaluation helpers."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt
from scipy.stats import kurtosis

ORDERS = {'IR': 5.4152, 'B': 4.7135, 'OR': 3.5848}
EXPECTED = {m: [f'{domain}_{m}_{h}' for domain in ('raw', 'envelope') for h in (1, 2)]
            for m in ORDERS}


def features(signal, rpm, fs=12000):
    """Nineteen prespecified features; only log_RMS changes with positive gain."""
    x = np.asarray(signal, dtype=float)
    x = x - x.mean()
    env = np.abs(hilbert(sosfiltfilt(butter(4, [2000, 5000], fs=fs, btype='bandpass',
                                          output='sos'), x)))
    result = {}
    spectra = {}
    for domain, values in [('raw', x), ('envelope', env-env.mean())]:
        freq, power = periodogram(values, fs=fs, window='hann')
        spectra[domain] = (freq, power)
        total = max(power.sum(), np.finfo(float).tiny)
        for m, order in ORDERS.items():
            for harmonic in (1, 2):
                center = order * harmonic * rpm/60
                width = .15*rpm/60
                result[f'{domain}_{m}_{harmonic}'] = float(power[np.abs(freq-center) <= width].sum()/total)
    rms = np.sqrt(np.mean(x*x))
    result.update(log_RMS=float(np.log(max(rms, np.finfo(float).tiny))),
                  kurtosis=float(kurtosis(x, fisher=False)), crest=float(np.max(np.abs(x))/rms))
    freq, power = spectra['raw']
    for lo, hi in [(0, 1000), (1000, 2000), (2000, 4000), (4000, 6001)]:
        result[f'band_{lo}_{hi}'] = float(power[(freq >= lo) & (freq < hi)].sum()/power.sum())
    assert len(result) == 19 and np.isfinite(list(result.values())).all()
    return result


def load_features(directory, windows=24, width=4096):
    """Read every recording with checksum verification; never infer labels from signals."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    rows, inventory = [], []
    for r in manifest['records']:
        path = directory / (r['recording']+'.mat')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r['sha256']
        mat = loadmat(path)
        channels = [k for k in mat if k.endswith('_DE_time')]
        rpm_keys = [k for k in mat if k.endswith('RPM')]
        assert len(channels) == len(rpm_keys) == 1, (r['recording'], list(mat))
        x = mat[channels[0]].ravel(); rpm = float(mat[rpm_keys[0]].ravel()[0])
        assert 1600 < rpm < 1900 and len(x) >= windows*width
        inventory.append({**r, 'rpm': rpm, 'samples': len(x), 'mat_keys': sorted(mat),
                          'channel': channels[0], 'n_windows': windows})
        for w in range(windows):
            rows.append({**features(x[w*width:(w+1)*width], rpm), 'recording': r['recording'],
                         'load_hp': r['load_hp'], 'fault': r['fault'], 'window': w})
    assert len(inventory) == 36
    return pd.DataFrame(rows), inventory


def recording_shares(attribution, metadata, expected):
    """Equal recording weight, after averaging absolute attributions within recording."""
    weights = attribution.groupby(metadata['recording'].to_numpy()).mean()
    labels = metadata.groupby('recording')['fault'].first()
    own = weights.loc[labels[labels == expected].index].mean()
    rest = weights.loc[labels[labels != expected].index].mean()
    a, b = own/own.sum(), rest/rest.sum()
    cols = EXPECTED[expected]
    excess = (a-b).drop(cols)
    return {'R': float(a[cols].sum()), 'D': float(a[cols].sum()-b[cols].sum()),
            'S': float(excess.max()), 'S_feature': str(excess.idxmax()),
            'gain_share': float(a['log_RMS']), 'shares': a.to_dict()}
