"""Inspect a pinned larger field release without equating events with assets."""
import pdm_audit
import hashlib,io,json,urllib.request,zipfile
from pathlib import Path
import pandas as pd
from pdm_audit.range_hdf5 import RangeReader

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/care';OUT=ROOT/'experiments/results'


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    api='https://zenodo.org/api/records/15846963'
    metadata=json.load(urllib.request.urlopen(api,timeout=45))
    (DATA/'official_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    item=next(x for x in metadata['files'] if x['key']=='CARE_To_Compare.zip')
    reader=RangeReader(item['links']['self'],item['size'],DATA/'range_cache',block_size=65536,max_new_bytes=32*1024*1024)
    members=[];selected=[];tables=[]
    with zipfile.ZipFile(reader) as z:
        for entry in z.infolist():
            members.append({'name':entry.filename,'size':entry.file_size,'CRC32':entry.CRC})
            lower=entry.filename.lower();name=Path(lower).name
            wanted=('readme' in name or 'event_info' in name or 'feature' in name and ('descr' in name or 'info' in name))
            if not wanted or entry.file_size>1024*1024 or Path(lower).suffix not in ['.txt','.md','.csv','.html']:continue
            data=z.read(entry);local=DATA/'metadata_members'/f'{len(selected):03d}_{Path(entry.filename).name}'
            local.parent.mkdir(exist_ok=True);local.write_bytes(data)
            selected.append({'member':entry.filename,'local':str(local.relative_to(ROOT)),'sha256':hashlib.sha256(data).hexdigest()})
            if lower.endswith('.csv'):
                table=pd.read_csv(io.BytesIO(data),sep=None,engine='python')
                tables.append({'member':entry.filename,'columns':list(table.columns),'rows':len(table),
                    'records':json.loads(table.to_json(orient='records'))})
            print('CARE metadata',entry.filename,len(data),flush=True)
    files=[Path(__file__),DATA/'official_metadata.json']+[ROOT/x['local'] for x in selected]
    result={'complete':True,'record':metadata['id'],'doi':metadata['doi'],'file':item,
        'inventory':members,'selected_metadata':selected,'tables':tables,'range_blocks':reader.blocks,
        'new_bytes':reader.downloaded,'full_zip_checksum_verified':False,
        'physical_mechanism_alignment_verified':False,'independent_assets_exchangeable_verified':False,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'care_preflight.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('COMPLETE CARE metadata audit',len(members),'members;',len(selected),'metadata files;',reader.downloaded,'new bytes',flush=True)


if __name__=='__main__':main()
