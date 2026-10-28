"""Authenticate native records, audit schema, and extract the frozen feature set."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from scipy.io import loadmat
from pdm_audit.kaist import CLASSES, features

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def strings(value):
    if isinstance(value,str):
        return [value]
    array = np.asarray(value)
    if array.dtype == object:
        return [''.join(np.asarray(x).ravel().tolist()) for x in array.ravel()]
    return [''.join(array.ravel().tolist())]


def main():
    archive = ROOT/'data/dataset_scout/kaist_vibration.zip'
    manifest = json.loads((OUT/'kaist_archive.json').read_text())
    assert hashlib.file_digest(archive.open('rb'),'sha256').hexdigest() == manifest['sha256']
    protocol = ROOT/'experiments/KAIST_PREDICTIVE_PROTOCOL.md'
    contract = json.loads((ROOT/'experiments/KAIST_LOAD_PILOT_CONTRACT.json').read_text())
    inventory, rows, seen = [], [], {}
    with zipfile.ZipFile(archive) as z:
        for item in contract['native_subset']:
            name = item['native_filename']
            # Reading each complete member verifies its native ZIP CRC as well.
            content = z.read(name)
            from io import BytesIO
            signal = loadmat(BytesIO(content),simplify_cells=True)['Signal']
            native_hash = hashlib.sha256(content).hexdigest()
            del content
            axis, y = signal['x_values'], signal['y_values']
            assert axis['quantity']['label'] == 's' and y['quantity']['label'] == 'g'
            assert axis['quantity']['unit_transformation']['factor'] == 1
            assert axis['quantity']['unit_transformation']['offset'] == 0
            increment = float(axis['increment'])
            fs = 1/increment
            assert abs(fs-25600) < 1e-6, (name,fs)
            values = np.asarray(y['values'])
            assert values.shape == (int(axis['number_of_values']),4)
            assert np.isfinite(values).all()
            # Test.Lab explicitly records MKS storage and the display-unit transform.
            transform = y['quantity']['unit_transformation']
            factor, offset = float(transform['factor']),float(transform['offset'])
            assert offset == 0 and abs(factor-1/9.80665) < 1e-12
            metadata = signal['function_record']
            window_size = int(round(fs))
            starts = np.linspace(0,len(values)-window_size,30).astype(int)
            assert np.diff(starts).min() >= window_size
            payload = hashlib.sha256()
            for j in range(4):
                payload.update(values[:,j].tobytes())
            payload_hash = payload.hexdigest()
            assert payload_hash not in {r['payload_sha256'] for r in inventory}, name
            for w,start in enumerate(starts):
                native_window = values[start:start+window_size]
                digest = hashlib.sha256(native_window.tobytes()).hexdigest()
                assert digest not in seen, (name,w,seen.get(digest))
                seen[digest] = (name,w)
                rows.append(dict(recording=name,load_Nm=item['load_Nm'],family=item['family'],
                    label=CLASSES.index(item['family']),window=w,start_sample=int(start),
                    window_sha256=digest,**features(native_window*factor,fs)))
            inventory.append(dict(**item,mat_sha256=native_hash,payload_sha256=payload_hash,
                sampling_Hz=fs,samples=len(values),duration_s=len(values)/fs,
                time_start_s=float(axis['start_value']),encoded_uniform_increment_s=increment,
                timing_scope='Uniform axis encoded in MAT metadata; no independent clock-jitter trace',
                stored_unit_info=y['quantity'].get('info'),display_unit='g',
                applied_multiplier=factor,applied_offset=offset,
                native_channel_names=strings(metadata['name']),
                native_creation_times=strings(metadata['creation_time']),
                channel_mapping='Publisher order: x_A, y_A, x_B, y_B; native names Point1--Point4',
                min_g=(values.min(axis=0)*factor).tolist(),max_g=(values.max(axis=0)*factor).tolist(),
                windows=30,starts=starts.tolist()))
            print(name,values.shape,'Hz',fs,flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'kaist_features.csv',index=False)
    record = dict(archive_sha256=manifest['sha256'],
        protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        feature_code_sha256=hashlib.sha256((ROOT/'src/pdm_audit/kaist.py').read_bytes()).hexdigest(),
        records=inventory,exact_duplicate_payloads=0,exact_duplicate_windows=0,
        unique_native_records=len(inventory),windows=len(frame),
        independence='Physical bearing/session independence not established; no new-bearing inference')
    (OUT/'kaist_signal_inventory.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print('PASS:',len(inventory),'records;',len(frame),'nonoverlapping windows; no exact duplicates',flush=True)


if __name__ == '__main__':
    main()
