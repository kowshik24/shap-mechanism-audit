"""Paderborn native-recording checks without an independence or causal certificate."""
import hashlib
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def profile_metadata(path):
    text = subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)
    def number(pattern):
        match = re.search(pattern, text)
        assert match, pattern
        return float(match.group(1))
    geometry = {'rolling_elements': int(number(r'Number of rolling elements\s+pc\.\s+(\d+)')),
                'ball_diameter_mm': number(r'Rolling element diameter\s+mm\s+([\d.]+)'),
                'pitch_diameter_mm': number(r'Pitch circle diameter\s+mm\s+([\d.]+)'),
                'contact_angle_degrees': number(r'Nominal pressure angle\s+°\s+([\d.]+)')}
    components = re.search(r'(?m)^\s*Component\s+(.+)$', text).group(1).split()
    result = {'geometry': geometry, 'components': components,
              'damage_count': int(number(r'Number of damages\s+(\d+)')),
              'manufacturer': re.search(r'Manufacturer\s+-\s+(\S+)', text).group(1),
              'sample_number': re.search(r'Sample number\s+-\s+(\S+)', text).group(1),
              'lifetime': re.search(r'Lifetime\s+h:min\s+(\S+)', text).group(1),
              'load_cycles': int(number(r'Number of load cycles\s+cycles\s+(\d+)')),
              'source_sha256': file_sha256(path)}
    return result


def ball_pass_orders(geometry):
    z = geometry['rolling_elements']; d = geometry['ball_diameter_mm']
    D = geometry['pitch_diameter_mm']; theta = np.deg2rad(geometry['contact_angle_degrees'])
    ratio = d/D*np.cos(theta)
    assert 0 < ratio < 1
    return {'OR': float(z/2*(1-ratio)), 'IR': float(z/2*(1+ratio))}


def native_capture(path):
    raw = loadmat(path, simplify_cells=True)
    roots = [k for k in raw if not k.startswith('__')]
    assert len(roots) == 1
    root = raw[roots[0]]; channels = {v['Name']: v for v in root['Y']}
    info = root['Description']; times = root['X']
    digests, summaries = {}, {}
    for name, item in sorted(channels.items()):
        values = np.asarray(item['Data'], dtype='<f8')
        assert values.ndim == 1 and np.isfinite(values).all()
        axis = times[int(item['XIndex'])-1]; t = np.asarray(axis['Data'])
        assert len(values) == len(t) and np.all(np.diff(t) > 0)
        unit = item['Unit'] if isinstance(item['Unit'], str) else None
        digests[name] = hashlib.sha256(values.tobytes()).hexdigest()
        dt = np.diff(t)
        summaries[name] = {'n': len(values), 'unit': unit, 'raster': item['Raster'],
                           'mean_native_value': float(values.mean()),
                           'std_native_value': float(values.std()),
                           'start_relative_time': float(t[0]), 'end_relative_time': float(t[-1]),
                           'span_rate_Hz': float((len(t)-1)/(t[-1]-t[0])),
                           'median_time_step': float(np.median(dt)),
                           'min_time_step': float(dt.min()), 'max_time_step': float(dt.max()),
                           'strictly_increasing_time': True}
    metadata = {'file': Path(path).name, 'root_variable': roots[0],
                'root_matches_filename': roots[0] == Path(path).stem,
                'file_sha256': file_sha256(path),
                'measurement_id': int(root['Info']['MeasurementID']),
                'native_DateTime': info['General']['DateTime'],
                'mat_export_header': raw['__header__'].decode(errors='replace'),
                'measurement_length': float(info['Measurement']['Length']),
                'start_timestamp': float(info['Measurement']['StartTimestamp']),
                'stop_timestamp': float(info['Measurement']['StopTimestamp']),
                'metadata_field_names': {k: sorted(v) for k, v in info.items() if isinstance(v, dict)},
                'channel_summaries': summaries, 'channel_sha256': digests,
                'all_Y_sha256': hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()}
    vibration = channels['vibration_1']; axis = times[int(vibration['XIndex'])-1]
    return metadata, np.asarray(vibration['Data']), np.asarray(axis['Data'])


def spectral_anchors(signal, fs, rpm, orders):
    centered = np.asarray(signal)-np.mean(signal)
    filtered = sosfiltfilt(butter(4, [2000, 10000], fs=fs, btype='bandpass', output='sos'), centered)
    envelope = np.abs(hilbert(filtered)); envelope -= envelope.mean()
    frequency, psd = periodogram(envelope, fs=fs, window='hann')
    total = max(float(psd[(frequency >= 5) & (frequency <= 1000)].sum()), np.finfo(float).tiny)
    shaft = rpm/60
    result = {}
    for location, order in orders.items():
        rows = []
        for h in [1, 2, 3]:
            center = h*order*shaft; distance = np.abs(frequency-center)/(shaft*h)
            band = distance <= .15; flank = (distance >= .30) & (distance <= .60)
            assert band.any() and flank.any()
            candidates = np.flatnonzero(band); i = candidates[np.argmax(psd[candidates])]
            peak, noise = max(float(psd[i]), np.finfo(float).tiny), max(float(np.median(psd[flank])), np.finfo(float).tiny)
            db = float(10*np.log10(peak/noise))
            rows.append({'harmonic': h, 'center_Hz': float(center), 'peak_Hz': float(frequency[i]),
                         'order_error_per_harmonic': float(frequency[i]/shaft/h-order),
                         'band_fraction_5_1000_Hz': float(psd[band].sum()/total),
                         'peak_to_flank_dB': db, 'screen_above_3dB': db > 3})
        result[location] = {'harmonics': rows,
                            'two_of_three_screen': sum(r['screen_above_3dB'] for r in rows) >= 2}
    return result


def resample_native_time(signal, time, fs=64000):
    start, stop = int(np.ceil(time[0]*fs)), int(np.floor(time[-1]*fs))
    return np.interp(np.arange(start, stop+1)/fs, time, signal)


def duplicate_groups(rows, key):
    groups = {}
    for r in rows:
        digest = r['channel_sha256']['vibration_1'] if key == 'vibration' else r['all_Y_sha256']
        groups.setdefault(digest, []).append(r['file'])
    return sorted([sorted(names) for names in groups.values() if len(names) > 1])
