"""Replay all inspected payloads and independently check spectra and source claims."""
import pdm_audit
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt
from pdm_audit.physical_acquisition import file_sha256, native_capture, profile_metadata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT/'data/paderborn/raw'


def main():
    r = json.loads((ROOT/'experiments/results/paderborn_physical_acquisition.json').read_text())
    assert r['complete'] and len(r['records']) == 160 and r['fitted_models'] == 0
    assert r['scope'] == ['KA04', 'KI04'] and r['genuine_new_bearings_observed'] == 2
    assert not r['protected_calibration_or_test_signals_inspected']
    assert not r['official_byte_identity_verified'] and not r['acquisition_independence_verified']
    assert not r['bearing_exchangeability_verified'] and not r['exhaustive_physical_alignment_verified']
    assert not r['calibration_gate_reopened']
    contract = json.loads((ROOT/'experiments/PADERBORN_SPLIT_CONTRACT.json').read_text())
    assert set(r['scope']) <= set(contract['partitions']['training'])
    for p, digest in r['source_sha256'].items():
        assert file_sha256(ROOT/p) == digest, p
    api = json.loads((ROOT/'data/paderborn/zenodo_15845309.json').read_text())
    assert len(api['files']) == r['mirror_archive_count'] == 31
    assert r['mirror_missing_documented_bearings'] == ['KA08']
    expected_components = {'KA04': ['OR'], 'KI04': ['IR', 'OR']}
    expected_pitch = {'KA04': 28.55, 'KI04': 29.05}
    for a in r['archives']:
        b = a['bearing_id']; path = RAW/(b+'.rar')
        md5 = hashlib.md5()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024*1024), b''):
                md5.update(chunk)
        source = next(x for x in api['files'] if x['key'] == path.name)
        assert 'md5:'+md5.hexdigest() == a['mirror_md5_verified'] == source['checksum']
        assert path.stat().st_size == a['archive_bytes'] == source['size']
        assert file_sha256(path) == a['archive_sha256']
        assert subprocess.check_output(['bsdtar', '-tf', str(path)], text=True).splitlines() == a['members']
        profile = profile_metadata(RAW/b/(b+'.pdf'))
        assert profile == r['profiles'][b]
        assert profile['components'] == expected_components[b]
        assert profile['geometry']['pitch_diameter_mm'] == expected_pitch[b]
        for doc, description in r['documents'][b].items():
            assert file_sha256(RAW/b/doc) == description['sha256']
    sensor_groups, waveform_groups = defaultdict(list), defaultdict(list)
    for row in r['records']:
        native, signal, time = native_capture(RAW/row['bearing_id']/row['file'])
        for key, value in native.items():
            assert row[key] == value, (row['file'], key)
        sensor_groups[native['all_Y_sha256']].append(row['file'])
        waveform_groups[native['channel_sha256']['vibration_1']].append(row['file'])
        assert row['physical_components_from_profile'] == expected_components[row['bearing_id']]
        assert all(x['unit'] is None for x in row['channel_summaries'].values())
        assert row['primary_sampling_Hz'] == 64000 and np.isclose(row['measurement_length'], 4, atol=1e-12)
        # Independent direct spectrum calculation on all first captures (eight groups).
        if row['capture_index'] == 1:
            g = r['profiles'][row['bearing_id']]['geometry']
            ratio = g['ball_diameter_mm']/g['pitch_diameter_mm']
            assert g['contact_angle_degrees'] == 0 and g['rolling_elements'] == 8
            orders = {'OR': 4*(1-ratio), 'IR': 4*(1+ratio)}
            for variant in ['primary_anchors', 'timestamp_anchors']:
                if variant == 'primary_anchors':
                    x = signal
                else:
                    ticks = np.arange(np.ceil(time[0]*64000), np.floor(time[-1]*64000)+1)/64000
                    x = np.interp(ticks, time, signal)
                env = np.abs(hilbert(sosfiltfilt(butter(4, [2000, 10000], fs=64000,
                                                      btype='bandpass', output='sos'), x-x.mean())))
                f, power = periodogram(env-env.mean(), fs=64000, window='hann')
                total = power[(f >= 5) & (f <= 1000)].sum(); shaft = row['nominal_setting']['speed_rpm']/60
                for location, order in orders.items():
                    for line in row[variant][location]['harmonics']:
                        h = line['harmonic']; center = order*h*shaft
                        distance = np.abs(f-center)/(h*shaft)
                        inside = np.flatnonzero(distance <= .15)
                        i = inside[np.argmax(power[inside])]
                        flank = power[(distance >= .30) & (distance <= .60)]
                        assert line['center_Hz'] == center and line['peak_Hz'] == f[i]
                        assert np.isclose(line['band_fraction_5_1000_Hz'], power[inside].sum()/total)
                        assert np.isclose(line['peak_to_flank_dB'], 10*np.log10(power[i]/np.median(flank)))
        for variant in ['primary_anchors', 'timestamp_anchors']:
            for loc in ['IR', 'OR']:
                item = row[variant][loc]
                assert item['two_of_three_screen'] == (sum(x['peak_to_flank_dB'] > 3 for x in item['harmonics']) >= 2)
    groups = lambda d: sorted(sorted(v) for v in d.values() if len(v) > 1)
    assert groups(sensor_groups) == r['duplicate_all_Y_groups']
    assert groups(waveform_groups) == r['duplicate_vibration_groups']
    assert len(waveform_groups) == r['unique_vibration_payloads'] == 156
    expected_duplicates = sorted([sorted(['N09_M07_F10_KA04_17.mat', 'N09_M07_F10_KA04_18.mat']),
        sorted(['N15_M07_F04_KA04_6.mat', 'N15_M07_F04_KA04_15.mat', 'N15_M07_F04_KA04_20.mat']),
        sorted(['N15_M07_F04_KA04_5.mat', 'N15_M07_F04_KA04_19.mat'])])
    assert r['duplicate_all_Y_groups'] == expected_duplicates
    for names in expected_duplicates:
        # Compare source arrays directly, not just fingerprints or filenames.
        first = loadmat(RAW/'KA04'/names[0], simplify_cells=True)
        first = next(v for k, v in first.items() if not k.startswith('__'))
        for name in names[1:]:
            other = loadmat(RAW/'KA04'/name, simplify_cells=True)
            other = next(v for k, v in other.items() if not k.startswith('__'))
            assert all(np.array_equal(a['Data'], b['Data']) for a, b in zip(first['Y'], other['Y']))
            assert all(np.array_equal(a['Data'], b['Data']) for a, b in zip(first['X'], other['X']))
    for s in r['summary']:
        rows = [x for x in r['records'] if (x['bearing_id'], x['condition']) == (s['bearing_id'], s['condition'])]
        assert len(rows) == s['capture_n'] == 20
        assert s['unique_vibration_payloads'] == len({x['channel_sha256']['vibration_1'] for x in rows})
        for variant in ['primary_anchors', 'timestamp_anchors']:
            for loc in ['IR', 'OR']:
                assert s['screen_counts'][variant][loc] == sum(x[variant][loc]['two_of_three_screen'] for x in rows)
    log = r['documents']['KA04']['measuring_log_KA04.pdf']
    assert log['sample_KA04_17_replacement_documented']
    assert r['documents']['KI04']['measuring_log_KI04.pdf']['disabled_sensor_remarks_present']
    overlay = json.loads((ROOT/'experiments/PADERBORN_PHYSICAL_LABEL_OVERLAY.json').read_text())
    assert overlay['pure_IR_interpretation_invalidated'] == ['KI04']
    assert not overlay['new_training_with_old_pure_labels_permitted'] and not overlay['frozen_role_ids_changed']
    assert not overlay['protected_calibration_or_test_signals_inspected']
    for s in r['summary']:
        assert all(count == 20 for counts in s['screen_counts'].values() for count in counts.values())
    import csv
    with (ROOT/'experiments/results/paderborn_capture_metadata.csv').open() as f:
        captures = list(csv.DictReader(f))
    with (ROOT/'experiments/results/paderborn_physical_spectra.csv').open() as f:
        spectra = list(csv.DictReader(f))
    assert len(captures) == 160 and len(spectra) == 1920
    lookup = {x['file']: x for x in r['records']}
    for x in captures:
        assert x['vibration_sha256'] == lookup[x['file']]['channel_sha256']['vibration_1']
        assert x['all_Y_sha256'] == lookup[x['file']]['all_Y_sha256']
    for x in spectra:
        source = lookup[x['file']][x['variant']][x['candidate_location']]['harmonics'][int(x['harmonic'])-1]
        for key in ['center_Hz', 'peak_Hz', 'order_error_per_harmonic', 'band_fraction_5_1000_Hz', 'peak_to_flank_dB']:
            assert float(x[key]) == source[key]
    outputs = ['experiments/results/paderborn_physical_acquisition.json',
               'experiments/results/paderborn_capture_metadata.csv', 'experiments/results/paderborn_physical_spectra.csv',
               'experiments/PADERBORN_PHYSICAL_LABEL_OVERLAY.json',
               'IEEE-conference-template-062824/PADERBORN_PHYSICAL_ACQUISITION.md']
    manifest = {'verified': True, 'raw_files': 160, 'physical_bearings': 2,
                'unique_numerical_payloads': 156, 'duplicate_groups': 3,
                'protected_signals_inspected': False, 'official_byte_identity_verified': False,
                'acquisition_independence_verified': False, 'exhaustive_physical_alignment_verified': False,
                'output_sha256': {p: file_sha256(ROOT/p) for p in outputs}}
    (ROOT/'experiments/results/paderborn_physical_verification_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('PASS: two mirror MD5/SHA-256 identities, 160 native files, seven sensor channels, protected roles')
    print('PASS: bearing-specific geometry, detailed IR+OR KI04 profile, and measurement-log evidence')
    print('PASS: all source payloads replayed; three duplicate groups checked directly across sensors and time axes')
    print('PASS: 96 independently recomputed harmonic measurements; every screen count and summary checked')
    print('PASS: no model, independence, exhaustive physical-alignment, or reopened-calibration claim')


if __name__ == '__main__':
    main()
