"""Replay pinned field metadata and preserve farm-qualified asset identities."""
import pdm_audit
import hashlib,io,json,zipfile
from collections import Counter
from pathlib import Path
import pandas as pd
from pdm_audit.range_hdf5 import RangeReader

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';DATA=ROOT/'data/care'


def main():
    r=json.loads((OUT/'care_preflight.json').read_text());assert r['complete'] and r['record']==15846963
    for path,digest in r['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    for start,block in r['range_blocks'].items():
        data=(DATA/'range_cache'/f'{int(start):016d}.bin').read_bytes()
        assert len(data)==block['length'] and hashlib.sha256(data).hexdigest()==block['sha256']
    item=r['file'];reader=RangeReader(item['links']['self'],item['size'],DATA/'range_cache',block_size=65536,max_new_bytes=0)
    events=[];farms=[]
    with zipfile.ZipFile(reader) as z:
        assert len(z.infolist())==len(r['inventory'])
        for selected in r['selected_metadata']:
            data=z.read(selected['member']);assert hashlib.sha256(data).hexdigest()==selected['sha256']
            assert data==(ROOT/selected['local']).read_bytes()
        for table in r['tables']:
            data=z.read(table['member']);frame=pd.read_csv(io.BytesIO(data),sep=None,engine='python')
            assert list(frame.columns)==table['columns'] and len(frame)==table['rows']
            assert json.loads(frame.to_json(orient='records'))==table['records']
            if 'event_info' not in table['member']:continue
            farm=table['member'].split('/')[1];column='asset_id' if 'asset_id' in frame else 'asset'
            assert column==('asset' if farm=='Wind Farm A' else 'asset_id')
            rows=table['records'];assets={str(x[column]) for x in rows}
            farms.append({'farm':farm,'event_datasets':len(rows),'assets':len(assets),'native_asset_column':column,
                          'labels':dict(Counter(x['event_label'] for x in rows))})
            for row in rows:events.append({'farm':farm,'asset':str(row[column]),'event_id':row['event_id'],'label':row['event_label']})
    assert len(events)==95 and len({(x['farm'],x['asset']) for x in events})==36
    assert Counter(x['label'] for x in events)=={'normal':50,'anomaly':45}
    assert reader.downloaded==0 and not r['full_zip_checksum_verified']
    files=[OUT/'care_preflight.json',Path(__file__)]
    out={'verified':True,'farms':farms,'events':events,'event_datasets':95,'assets':36,'labels':{'anomaly':45,'normal':50},
        'full_zip_checksum_verified':False,'mechanism_alignment_verified':False,'asset_exchangeability_verified':False,
        'scope':'Pinned version-6 metadata and asset-count audit; no waveform or fitted-model evaluation',
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'care_preflight_verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print('PASS pinned CARE metadata: 95 event datasets, 36 farm-qualified assets, 45 anomalies/50 normal; offline range/CRC replay')


if __name__=='__main__':main()
