"""Inspect author-hosted HDF5 metadata before selecting or fitting an audit."""
import pdm_audit
import json
from pathlib import Path
import urllib.request
import h5py
from pdm_audit.range_hdf5 import RangeReader

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/dtu_tep';OUT=ROOT/'experiments/results'


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    url='https://api.figshare.com/v2/articles/13385936'
    metadata=json.load(urllib.request.urlopen(url,timeout=30))
    (DATA/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    item=next(x for x in metadata['files'] if x['name']=='TEP_Mode1.h5')
    reader=RangeReader(item['download_url'],item['size'],DATA/'mode1_ranges64k',block_size=65536,max_new_bytes=32*1024*1024)
    result={'source_doi':metadata['doi'],'source_api':url,'h5py_version':h5py.__version__,
            'file':item,'entire_object_checksum_verified':False,'field_acquisitions':False}
    with h5py.File(reader,'r') as f:
        result['root']=list(f.keys());print('root',result['root'],flush=True)
        for name in ['Processdata_Labels','Additional_Meas_Labels']:
            if name in f:
                node=f[name];print(name,node.shape,node.dtype,flush=True)
                if node.size<1000:
                    try:result[name]=node.asstr()[()].tolist()
                    except Exception:result[name]=repr(node[()].tolist())
        mode=f['Mode1'];print('Mode1',list(mode.keys()),flush=True)
        fault=mode['SingleFault'];print('SingleFault',list(fault.keys()),flush=True)
        result['fault_tree']={}
        for status in fault:
            ids=list(fault[status].keys());result['fault_tree'][status]=ids
            print(status,'faults',ids,flush=True)
        group=fault['SimulationCompleted']
        for name in ['IDV1','IDV2','IDV4']:
            if name not in group:continue
            node=group[name];print(name,list(node.keys()),flush=True)
            result[name]={}
            for magnitude in node:
                branch=node[magnitude];count=len(branch)
                result[name][magnitude]={'run_count':count,'sampled_run_name':'Run1' if 'Run1' in branch else None}
                print(magnitude,'runs',count,flush=True)
                if 'Run1' in branch:
                    run=branch['Run1'];info={}
                    for key in run:
                        d=run[key];info[key]={'shape':list(d.shape),'dtype':str(d.dtype),'chunks':d.chunks}
                    result[name][magnitude]['first_run_schema']=info
                    print('schema',info,flush=True)
    result['downloaded_bytes']=reader.downloaded;result['range_blocks']=reader.blocks
    (OUT/'dtu_preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PREFLIGHT metadata complete; bytes fetched',reader.downloaded,flush=True)


if __name__=='__main__':main()
