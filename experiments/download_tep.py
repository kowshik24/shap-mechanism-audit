"""
Download the Tennessee Eastman runs used by stages 15 and 22.

Source: the extended TEP reference data of Reinartz, Kulahci and Ravn (2021), DTU Data,
https://doi.org/10.11583/DTU.13385936.v1. The 24 GB file TEP_Mode1.h5 is not downloaded in full: its
HDF5 metadata is read through cached HTTP range requests, and only the 400 complete 100-hour
trajectories of IDV4 and IDV5 (mode 1, single fault, runs 1-200) are fetched, about 350 MB in total.
Each run is written as a raw little-endian float64 array of shape (2001, 54), time in column 0, to
data/dtu_tep/prospective_runs/<fault>_<run>.bin, and data/dtu_tep/runs.json records their SHA-256.

Usage: python experiments/download_tep.py
"""
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np

from pdm_audit.range_hdf5 import RangeReader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'dtu_tep'
URL = 'https://ndownloader.figshare.com/files/26003087'  # TEP_Mode1.h5
SIZE = 23924710848
SHAPE, DTYPE = (2001, 54), '<f8'


def run_offsets():
    """HDF5 byte offset of every IDV4 and IDV5 trajectory, after checking the fault settings."""
    entries = []
    with h5py.File(RangeReader(URL, SIZE, DATA / 'mode1_ranges64k'), 'r') as f:
        labels = f['Processdata_Labels'].asstr()[()].tolist()
        assert labels[0] == 'Time' and len(labels) == SHAPE[1]
        for fault in (4, 5):
            branch = f[f'Mode1/SingleFault/SimulationCompleted/IDV{fault}/Mode1_IDVInfo_{fault}_100']
            for run in range(1, 201):
                group = branch[f'Run{run}']
                d, settings = group['processdata'], group['idv_init'][()]
                # fault `fault` switched on with magnitude 1 at 30 h, no other fault
                assert np.isclose(settings[1, fault - 1], 1) and np.isclose(settings[2, fault - 1], 30)
                assert np.isclose(settings[1].sum(), 1) and np.allclose(settings[0], 0)
                assert d.shape == SHAPE and d.chunks is None and d.compression is None
                entries.append({'fault': fault, 'run': run, 'offset': int(d.id.get_offset())})
    return entries


def fetch(entry):
    path = DATA / 'prospective_runs' / f"{entry['fault']}_{entry['run']:03d}.bin"
    length = int(np.prod(SHAPE)) * np.dtype(DTYPE).itemsize
    if path.exists():
        data = path.read_bytes()
    else:
        start, end = entry['offset'], entry['offset'] + length - 1
        request = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 206
            data = response.read(length + 1)
        path.write_bytes(data)
    assert len(data) == length
    full = np.frombuffer(data, dtype=DTYPE).reshape(SHAPE)
    assert np.isfinite(full).all() and np.all(np.diff(full[:, 0]) > 0)
    assert np.isclose(full[0, 0], 0) and np.isclose(full[-1, 0], 100)
    return {**entry, 'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    (DATA / 'prospective_runs').mkdir(parents=True, exist_ok=True)
    entries = run_offsets()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(fetch, entries))
    (DATA / 'runs.json').write_text(json.dumps({'source': 'https://doi.org/10.11583/DTU.13385936.v1',
                                                'file': 'TEP_Mode1.h5', 'runs': records}, indent=1) + '\n')
    print(f'{len(records)} runs in', DATA / 'prospective_runs')


if __name__ == '__main__':
    main()
