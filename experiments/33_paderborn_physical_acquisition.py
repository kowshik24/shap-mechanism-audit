"""Inspect two training-bearing archives; do not fit models or infer independence."""
import pdm_audit
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import urllib.request
import numpy as np
from pdm_audit.paderborn import parse_capture_name
from pdm_audit.physical_acquisition import (file_sha256, profile_metadata, ball_pass_orders,
    native_capture, spectral_anchors, resample_native_time, duplicate_groups)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/paderborn'
RAW = DATA/'raw'
SELECTED = ['KA04', 'KI04']


def prepare(metadata, bearing, download):
    item = next(f for f in metadata['files'] if f['key'] == bearing+'.rar')
    path = RAW/item['key']
    if not path.exists() and download:
        url = item['links']['self']
        assert url.startswith('https://zenodo.org/api/records/15845309/files/')
        temporary = path.with_suffix('.part')
        with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as stream:
            while chunk := response.read(1024*1024):
                stream.write(chunk)
        temporary.replace(path)
    assert path.stat().st_size == item['size']
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    assert 'md5:'+digest.hexdigest() == item['checksum']
    names = subprocess.check_output(['bsdtar', '-tf', str(path)], text=True).splitlines()
    for name in names:
        p = PurePosixPath(name)
        assert not p.is_absolute() and '..' not in p.parts
        assert p.parts[0] == bearing
        assert len(p.parts) <= 2
        if len(p.parts) == 2:
            assert p.suffix in ['.pdf', '.mat']
    verbose = subprocess.check_output(['bsdtar', '-tvf', str(path)], text=True).splitlines()
    assert all(line[0] in ['-', 'd'] for line in verbose), 'Non-regular archive member'
    subprocess.run(['bsdtar', '-xf', str(path), '-C', str(RAW)], check=True)
    return {'bearing_id': bearing, 'mirror_download_url': item['links']['self'],
            'mirror_md5_verified': item['checksum'], 'archive_sha256': file_sha256(path),
            'archive_bytes': path.stat().st_size, 'members': names,
            'official_byte_identity_verified': False}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--download', action='store_true')
    args = parser.parse_args(); RAW.mkdir(exist_ok=True)
    metadata = json.loads((DATA/'zenodo_15845309.json').read_text())
    prior = json.loads((ROOT/'experiments/results/paderborn_feasibility.json').read_text())
    contract = json.loads((ROOT/'experiments/PADERBORN_SPLIT_CONTRACT.json').read_text())
    assert set(SELECTED) <= set(contract['partitions']['training'])
    conditions = {c['condition']: c for c in prior['conditions']}
    archive_ids = {f['key'].removesuffix('.rar') for f in metadata['files']}
    missing = sorted({r['bearing_id'] for r in prior['bearings']}-archive_ids)
    archives = [prepare(metadata, bearing, args.download) for bearing in SELECTED]
    profiles, documents, records = {}, {}, []
    for bearing in SELECTED:
        folder = RAW/bearing
        profiles[bearing] = profile_metadata(folder/(bearing+'.pdf'))
        documents[bearing] = {}
        for doc in [folder/(bearing+'.pdf'), folder/('measuring_log_'+bearing+'.pdf')]:
            text = subprocess.check_output(['pdftotext', '-layout', str(doc), '-'], text=True)
            documents[bearing][doc.name] = {'sha256': file_sha256(doc),
                'date_field': re.search(r'(?m)^date:\s*(\S+)', text).group(1) if re.search(r'(?m)^date:\s*(\S+)', text) else None,
                'sample_KA04_17_replacement_documented': 'N09_M07_F10_KA04_17.mat' in text and 'Kopie' in text and 'N09_M07_F10_KA04_18.mat' in text,
                'disabled_sensor_remarks_present': 'ohne Zuschaltung' in text and 'Kistler' in text,
                'sampling_64_kHz_documented': 'vibration sampling frequency: 64 kHz' in text}
        orders = ball_pass_orders(profiles[bearing]['geometry'])
        paths = sorted(folder.glob('*.mat')); assert len(paths) == 80
        for path in paths:
            parsed = parse_capture_name(path.name, SELECTED, conditions)
            native, vibration, time = native_capture(path)
            assert parsed['bearing_id'] == bearing
            c = conditions[parsed['condition']]
            native.update(parsed)
            native['spectral_rpm_source'] = 'nominal operating condition; native unit fields are empty'
            native['nominal_setting'] = c
            native['primary_sampling_Hz'] = 64000
            native['physical_components_from_profile'] = profiles[bearing]['components']
            native['primary_anchors'] = spectral_anchors(vibration, 64000, c['speed_rpm'], orders)
            resampled = resample_native_time(vibration, time)
            native['timestamp_resampled_n'] = len(resampled)
            native['timestamp_anchors'] = spectral_anchors(resampled, 64000, c['speed_rpm'], orders)
            records.append(native)
        print('COMPLETE:', bearing, '80 captures; physical profile, native metadata, both spectral variants', flush=True)
    summary = []
    for bearing in SELECTED:
        for condition in sorted(conditions):
            rows = [r for r in records if (r['bearing_id'], r['condition']) == (bearing, condition)]
            assert len(rows) == 20 and {r['capture_index'] for r in rows} == set(range(1, 21))
            s = {'bearing_id': bearing, 'condition': condition, 'capture_n': len(rows),
                 'unique_vibration_payloads': len({r['channel_sha256']['vibration_1'] for r in rows}),
                 'unique_all_Y_payloads': len({r['all_Y_sha256'] for r in rows}),
                 'screen_counts': {},
                 'native_means': {name: float(np.mean([r['channel_summaries'][name]['mean_native_value'] for r in rows])) for name in ['speed', 'force', 'torque']},
                 'time_span_rate_range': [min(r['channel_summaries']['vibration_1']['span_rate_Hz'] for r in rows), max(r['channel_summaries']['vibration_1']['span_rate_Hz'] for r in rows)]}
            for variant in ['primary_anchors', 'timestamp_anchors']:
                s['screen_counts'][variant] = {location: sum(r[variant][location]['two_of_three_screen'] for r in rows) for location in ['IR', 'OR']}
            summary.append(s)
    overlay = {'status': 'physical-document overlay; summary-table split IDs unchanged',
               'source': 'Zenodo mirror of bearing-specific university profiles; original-byte identity unverified',
               'scope': SELECTED, 'prior_summary_component': {'KA04': 'OR', 'KI04': 'IR'},
               'profile_components': {b: profiles[b]['components'] for b in SELECTED},
               'pure_IR_interpretation_invalidated': ['KI04'],
               'new_training_with_old_pure_labels_permitted': False,
               'profiles': profiles,
               'frozen_role_ids_changed': False,
               'protected_calibration_or_test_signals_inspected': False}
    overlay_path = ROOT/'experiments/PADERBORN_PHYSICAL_LABEL_OVERLAY.json'
    overlay_path.write_text(json.dumps(overlay, indent=2)+'\n')
    files = ['experiments/PADERBORN_PHYSICAL_ACQUISITION_PROTOCOL.md',
             'experiments/33_paderborn_physical_acquisition.py', 'src/pdm_audit/physical_acquisition.py',
             'src/pdm_audit/paderborn.py', 'experiments/34_verify_physical_acquisition.py',
             'experiments/35_physical_acquisition_assets.py',
             'IEEE-conference-template-062824/PADERBORN_PHYSICAL_ACQUISITION.md',
             'tests/test_physical_acquisition.py', 'data/paderborn/zenodo_15845309.json', 'data/paderborn/test_rig.html',
             'experiments/PADERBORN_SPLIT_CONTRACT.json', 'experiments/PADERBORN_PHYSICAL_LABEL_OVERLAY.json']
    result = {'complete': len(records) == 160, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': SELECTED, 'training_bearing_only': True, 'fitted_models': 0,
              'protected_calibration_or_test_signals_inspected': False,
              'mirror_record': 'https://zenodo.org/records/15845309',
              'mirror_creator': metadata['metadata']['creators'],
              'mirror_archive_count': len(metadata['files']), 'mirror_missing_documented_bearings': missing,
              'official_byte_identity_verified': False, 'archives': archives,
              'profiles': profiles, 'documents': documents, 'records': records, 'summary': summary,
              'duplicate_vibration_groups': duplicate_groups(records, 'vibration'),
              'duplicate_all_Y_groups': duplicate_groups(records, 'all'),
              'unique_vibration_payloads': len({r['channel_sha256']['vibration_1'] for r in records}),
              'native_dates': dict(Counter(r['native_DateTime'].split()[0] for r in records)),
              'acquisition_independence_verified': False, 'bearing_exchangeability_verified': False,
              'exhaustive_physical_alignment_verified': False,
              'pure_IR_interpretation_invalidated': ['KI04'],
              'genuine_new_bearings_observed': 2, 'calibration_gate_reopened': False,
              'source_sha256': {p: file_sha256(ROOT/p) for p in files}}
    (ROOT/'experiments/results/paderborn_physical_acquisition.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('RESULT: KI04 pure-IR interpretation invalidated by detailed IR+OR damage profile', flush=True)
    print('RESULT: duplicate vibration groups', result['duplicate_vibration_groups'], flush=True)


if __name__ == '__main__':
    main()
