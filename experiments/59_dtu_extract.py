"""Extract prespecified complete process arrays from an author-hosted HDF5 file."""
import pdm_audit
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path
import h5py
import numpy as np
from pdm_audit.range_hdf5 import RangeReader

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/dtu_tep';OUT=ROOT/'experiments/results'


def download(entry,url,size):
    path=DATA/'prospective_runs'/f"{entry['fault']}_{entry['run']:03d}.bin";path.parent.mkdir(exist_ok=True)
    length=int(np.prod(entry['shape']))*np.dtype(entry['dtype']).itemsize
    if path.exists():data=path.read_bytes()
    else:
        start=entry['offset'];end=start+length-1
        for attempt in range(3):
            try:
                request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(request,timeout=60) as response:
                    assert response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{size}'
                    data=response.read(length+1);assert len(data)==length
                path.write_bytes(data);break
            except Exception:
                if attempt==2:raise
                time.sleep(attempt+1)
    assert len(data)==length
    full=np.frombuffer(data,dtype=np.dtype(entry['dtype'])).reshape(entry['shape'])
    assert np.isfinite(full).all() and np.all(np.diff(full[:,0])>0)
    assert np.isclose(full[0,0],0) and np.isclose(full[-1,0],100)
    targets=np.r_[np.linspace(10,25,16),np.linspace(40,95,48)]
    indices=abs(full[:,0,None]-targets).argmin(axis=0);assert len(set(indices))==64
    selected=full[indices].copy();assert (selected[:16,0]<30).all() and (selected[16:,0]>30).all()
    return {**entry,'raw_path':str(path.relative_to(ROOT)),'raw_sha256':hashlib.sha256(data).hexdigest(),
            'indices':indices.tolist(),'selected_sha256':hashlib.sha256(selected.astype('<f8').tobytes()).hexdigest()},selected


def main():
    began=time.perf_counter();metadata=json.loads((DATA/'metadata.json').read_text())
    item=next(x for x in metadata['files'] if x['name']=='TEP_Mode1.h5')
    reader=RangeReader(item['download_url'],item['size'],DATA/'mode1_ranges64k',block_size=65536,max_new_bytes=256*1024*1024)
    entries=[]
    with h5py.File(reader,'r') as f:
        labels=f['Processdata_Labels'].asstr()[()].tolist();assert labels[0]=='Time' and len(labels)==54
        for fault in [4,5]:
            branch=f[f'Mode1/SingleFault/SimulationCompleted/IDV{fault}/Mode1_IDVInfo_{fault}_100']
            assert len(branch)==200
            for run in range(1,201):
                group=branch[f'Run{run}'];d=group['processdata'];settings=group['idv_init'][()]
                assert settings.shape==(3,28) and np.isclose(settings[1,fault-1],1) and np.isclose(settings[2,fault-1],30)
                assert np.isclose(settings[1].sum(),1) and np.allclose(settings[0],0)
                assert d.shape==(2001,54) and d.chunks is None and d.compression is None
                offset=d.id.get_offset();assert offset is not None
                entries.append({'fault':fault,'run':run,'path':group.name,'offset':int(offset),
                    'shape':list(d.shape),'dtype':d.dtype.str,'idv_init':settings.tolist()})
                if run%40==0:print('DTU source offsets',fault,run,flush=True)
    records=[];arrays=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i,(record,values) in enumerate(pool.map(lambda x:download(x,item['download_url'],item['size']),entries)):
            records.append(record);arrays.append(values)
            if (i+1)%40==0:print('DTU process arrays',i+1,'/400',flush=True)
    seen={}
    for record in records:
        role='train' if record['run']<=40 else ('calibration' if record['run']<=140 else 'evaluation')
        if record['raw_sha256'] in seen:assert seen[record['raw_sha256']]==role,'Duplicate full trajectory crosses roles'
        seen[record['raw_sha256']]=role
    path=OUT/'dtu_prospective_data.npz';np.savez_compressed(path,values=np.asarray(arrays))
    files=[Path(__file__),ROOT/'experiments/DTU_PROSPECTIVE_PROTOCOL.md',DATA/'metadata.json',path]
    result={'complete':True,'doi':metadata['doi'],'file':item,'labels':labels,'records':records,
        'roles':{'train':[1,40],'calibration':[41,140],'evaluation':[141,200]},
        'full_object_checksum_verified':False,'physical_acquisitions':False,'replicate_exchangeability':'publisher-described simulation-seed assumption',
        'range_blocks':reader.blocks,'new_metadata_bytes':reader.downloaded,'seconds':time.perf_counter()-began,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'dtu_prospective_source.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE 400 source trajectories, 200 grouped run packets',round(result['seconds'],1),flush=True)


if __name__=='__main__':main()
