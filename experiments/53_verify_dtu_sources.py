"""Offline reconstruction of retained author-hosted DTU schemas and prefixes."""
import pdm_audit
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
from pdm_audit.range_hdf5 import RangeReader

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';DATA=ROOT/'data/dtu_tep'


def main():
    schema=json.loads((OUT/'dtu_preflight.json').read_text());r=json.loads((OUT/'dtu_prefix_check.json').read_text())
    for p,value in r['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==value,p
    for record in [schema,r]:
        for start,block in record['range_blocks'].items():
            data=(DATA/'mode1_ranges64k'/f'{int(start):016d}.bin').read_bytes()
            assert len(data)==block['length'] and hashlib.sha256(data).hexdigest()==block['sha256']
    item=schema['file'];reader=RangeReader(item['download_url'],item['size'],DATA/'mode1_ranges64k',block_size=65536,max_new_bytes=0)
    arrays=dict(np.load(OUT/'dtu_prefix_arrays.npz'))
    with h5py.File(reader,'r') as f:
        assert list(f.keys())==schema['root'] and f['Processdata_Labels'].asstr()[()].tolist()==r['labels']
        for fault in ['IDV1','IDV2','IDV4']:
            for name,record in schema[fault].items():
                branch=f['Mode1/SingleFault/SimulationCompleted'][fault][name]
                assert len(branch)==record['run_count']
                for key,value in record['first_run_schema'].items():assert list(branch['Run1'][key].shape)==value['shape']
        for record in r['records']:
            group=f[record['path']];values=arrays[record['array_key']]
            assert np.array_equal(group['processdata'][:64],values)
            assert hashlib.sha256(values.astype('<f8').tobytes()).hexdigest()==record['prefix_sha256']
            assert np.array_equal(group['time_info'][()],record['time_info'])
            assert np.array_equal(group['idv_init'][()],record['idv_init'])
    pairs=[];aligned=[]
    for i,a in enumerate(r['records']):
        for b in r['records'][i+1:]:
            x,y=arrays[a['array_key']],arrays[b['array_key']]
            if np.array_equal(x,y):pairs.append([a['array_key'],b['array_key']])
            times,ix,iy=np.intersect1d(x[:,0],y[:,0],return_indices=True)
            if len(times)>=32 and np.array_equal(x[ix],y[iy]):aligned.append({'pairs':[a['array_key'],b['array_key']],'shared_times':len(times)})
    assert pairs==r['equal_prefix_pairs'] and aligned==r['equal_time_aligned_pairs'] and reader.downloaded==0
    assert not any(r[k] for k in ['independent_rng_streams_verified','physical_acquisitions','full_object_checksum_verified'])
    files=[OUT/'dtu_preflight.json',OUT/'dtu_prefix_check.json',OUT/'dtu_prefix_arrays.npz',Path(__file__),ROOT/'experiments/51_dtu_preflight.py']
    manifest={'verified':True,'prefixes':len(r['records']),'new_network_bytes':0,'seed_independence_verified':False,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'dtu_source_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS author-hosted DTU range hashes, schemas, 15 prefixes and exact-time comparisons; no independence claim')


if __name__=='__main__':main()
