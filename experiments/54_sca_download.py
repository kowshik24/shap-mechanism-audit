"""Download and authenticate the original SCA release through its public API."""
import pdm_audit
import concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/sca';OUT=ROOT/'experiments/results'
API='https://data.mendeley.com/public-api/datasets/tdn96mkkpt'


def fetch_json(url):
    request=urllib.request.Request(url,headers={'User-Agent':'pdm-audit/1.0 research dataset verification'})
    with urllib.request.urlopen(request,timeout=45) as response:return json.load(response)


def download(item):
    path=DATA/item['name'];path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        temporary=path.with_suffix('.partial')
        request=urllib.request.Request(item['url'],headers={'User-Agent':'pdm-audit/1.0 research dataset verification'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request,timeout=60) as response,temporary.open('wb') as f:
                    while block:=response.read(1024*1024):f.write(block)
                assert temporary.stat().st_size==item['size']
                assert hashlib.file_digest(temporary.open('rb'),'sha256').hexdigest()==item['sha256']
                temporary.replace(path);break
            except Exception:
                if attempt==2:raise
    assert path.stat().st_size==item['size']
    with path.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==item['sha256']
    print('verified SCA',item['name'],item['size'],flush=True)
    return item


def main():
    start=time.perf_counter();DATA.mkdir(parents=True,exist_ok=True)
    folders=fetch_json(API+'/folders/2');assert {x['name'] for x in folders}=={str(i) for i in range(1,12)}
    items=[];metadata=[]
    for folder in sorted(folders,key=lambda x:int(x['name'])):
        files=fetch_json(API+f"/files?folder_id={folder['id']}&version=2")
        assert {x['filename'] for x in files}=={'train.mat','test.mat'}
        metadata.append({'folder':folder,'files':files})
        for f in files:
            d=f['content_details'];items.append({'name':folder['name']+'/'+f['filename'],'size':d['size'],
                'sha256':d['sha256_hash'],'url':d['download_url']})
    (DATA/'official_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(download,items))
    files=[Path(__file__),DATA/'official_metadata.json']
    result={'complete':True,'doi':'10.17632/tdn96mkkpt.2','source_api':API,'files':records,
        'raw_signals_opened':False,'case_independence_verified':False,'physical_alignment_verified':False,
        'total_bytes':sum(x['size'] for x in items),'seconds':time.perf_counter()-start,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'sca_download.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPLETE authenticated SCA download',len(records),'files',result['total_bytes'],'bytes',flush=True)


if __name__=='__main__':main()
