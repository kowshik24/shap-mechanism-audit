"""Check numerical identity in fixed DTU pilot runs, without fitting a model."""
import pdm_audit
import hashlib,json
from pathlib import Path
import h5py
import numpy as np
from pdm_audit.range_hdf5 import RangeReader

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/dtu_tep';OUT=ROOT/'experiments/results'


def main():
    metadata=json.loads((DATA/'metadata.json').read_text())
    item=next(x for x in metadata['files'] if x['name']=='TEP_Mode1.h5')
    reader=RangeReader(item['download_url'],item['size'],DATA/'mode1_ranges64k',block_size=65536,max_new_bytes=16*1024*1024)
    records=[];arrays={}
    with h5py.File(reader,'r') as f:
        labels=f['Processdata_Labels'].asstr()[()].tolist()
        for fault in ['IDV1','IDV2','IDV4']:
            branch=f['Mode1/SingleFault/SimulationCompleted'][fault]
            for magnitude in branch:
                runs=['Run1','Run2'] if magnitude.endswith('_100') else ['Run1']
                for run in runs:
                    if run not in branch[magnitude]:continue
                    group=branch[magnitude][run];name=f'{fault}_{magnitude}_{run}'
                    prefix=np.asarray(group['processdata'][:64]);assert prefix.shape==(64,54) and np.isfinite(prefix).all()
                    arrays[name]=prefix
                    record={'fault':fault,'magnitude':magnitude,'run':run,'array_key':name,
                        'path':group.name,'process_shape':list(group['processdata'].shape),
                        'prefix_sha256':hashlib.sha256(prefix.astype('<f8').tobytes()).hexdigest(),
                        'time_info':group['time_info'][()].tolist(),'idv_init':group['idv_init'][()].tolist()}
                    records.append(record);print('prefix',fault,magnitude,run,flush=True)
    pairs=[];aligned=[]
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['prefix_sha256']==b['prefix_sha256']:
                assert np.array_equal(arrays[a['array_key']],arrays[b['array_key']])
                pairs.append([a['array_key'],b['array_key']])
            x,y=arrays[a['array_key']],arrays[b['array_key']]
            times,ix,iy=np.intersect1d(x[:,0],y[:,0],return_indices=True)
            if len(times)>=32 and np.array_equal(x[ix],y[iy]):
                aligned.append({'pairs':[a['array_key'],b['array_key']],'shared_times':len(times)})
    path=OUT/'dtu_prefix_arrays.npz';np.savez_compressed(path,**arrays)
    files=[Path(__file__),ROOT/'src/pdm_audit/range_hdf5.py',ROOT/'experiments/DTU_PREFIX_CHECK_PROTOCOL.md',DATA/'metadata.json',path]
    out={'complete':True,'labels':labels,'records':records,'equal_prefix_pairs':pairs,'equal_time_aligned_pairs':aligned,
         'independent_rng_streams_verified':False,'physical_acquisitions':False,'full_object_checksum_verified':False,
         'new_downloaded_bytes':reader.downloaded,'range_blocks':reader.blocks,
         'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'dtu_prefix_check.json').write_text(json.dumps(out,indent=2)+'\n')
    print('COMPLETE prefix identity audit',len(records),'runs;',len(pairs),'equal-prefix pairs',flush=True)


if __name__=='__main__':main()
