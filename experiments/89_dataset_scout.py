"""Cache public source metadata and inspect a bounded UCI archive, without fitting."""
import pdm_audit
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT/'data/dataset_scout'
MAX_BYTES = 100_000_000


def fetch(url, path):
    if path.is_file():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        chunks, count = [], 0
        while True:
            block = response.read(1024*1024)
            if not block:
                break
            count += len(block)
            if count > MAX_BYTES:
                raise ValueError('Download exceeds 100 MB scouting limit')
            chunks.append(block)
    data = b''.join(chunks)
    path.write_bytes(data)
    return data


class RemoteZip(io.RawIOBase):
    """Read ZIP metadata by bounded HTTP ranges; never silently fetch a whole large file."""
    def __init__(self,url,size):
        self.url,self.size,self.position,self.fetched = url,size,0,0

    def seekable(self):
        return True

    def seek(self,offset,whence=0):
        self.position = offset if whence == 0 else self.position+offset if whence == 1 else self.size+offset
        return self.position

    def tell(self):
        return self.position

    def read(self,size=-1):
        size = self.size-self.position if size < 0 else min(size,self.size-self.position)
        if size == 0:
            return b''
        if size > 1_000_000:
            raise ValueError('ZIP metadata range exceeds 1 MB limit')
        request = urllib.request.Request(self.url,headers={'Range':f'bytes={self.position}-{self.position+size-1}','User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(request,timeout=45) as response:
            if response.status != 206:
                raise ValueError(f'Server ignored range: status {response.status}')
            content = response.read(size+1)
        if len(content) != size:
            raise ValueError('Range length mismatch')
        self.position += size
        self.fetched += size
        return content


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    result = {'sources':[], 'scope':'Metadata feasibility only; no new model fitting or paper results'}
    urls = {
        'kaist_load':'https://data.mendeley.com/datasets/ztmf3m7h5x/6',
        'ottawa_speed':'https://data.mendeley.com/datasets/v43hmbwxpm/2',
        'uci_hydraulic':'https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems',
        'mafaulda':'https://www02.smt.ufrj.br/~offshore/mfs/page_01.html'}
    for name,url in urls.items():
        path = DEST/(name+'.html')
        try:
            content = fetch(url,path)
            text = content.decode('utf-8',errors='replace')
            scripts = re.findall(r'<script[^>]*>(.*?)</script>',text,re.S)
            (DEST/(name+'_scripts.txt')).write_text('\n\n'.join(scripts))
            result['sources'].append(dict(name=name,url=url,bytes=len(content),
                sha256=hashlib.sha256(content).hexdigest(),status='downloaded',
                links=re.findall(r'href=["\']([^"\']+)',text)))
            print(name,len(content),'metadata bytes',flush=True)
            if name in ['kaist_load','ottawa_speed']:
                # Public dataset-page hydration contains its published metadata.
                state_script = next(s for s in scripts if s.startswith('window.INITIAL_STATE = '))
                state = json.JSONDecoder().raw_decode(state_script[state_script.index('{'):])[0]
                (DEST/(name+'_state.json')).write_text(json.dumps(state,indent=2)+'\n')
                dataset_id = state['dataset']['snapshot']['id']
                version = state['dataset']['snapshot']['version']
                # This public endpoint was verified against the repository's file listing.
                endpoint = f'https://data.mendeley.com/public-api/datasets/{dataset_id}/files?folder_id=root&version={version}'
                try:
                    listing = fetch(endpoint,DEST/(name+'_files.json'))
                    result['sources'][-1]['file_listing_url'] = endpoint
                    result['sources'][-1]['file_listing_bytes'] = len(listing)
                    print(name,'file listing',len(listing),'bytes',flush=True)
                    if name == 'ottawa_speed':
                        folders_url = f'https://data.mendeley.com/public-api/datasets/{dataset_id}/folders?folder_id=root&version={version}'
                        folders = fetch(folders_url,DEST/(name+'_folders.json'))
                        print(name,'root folders',len(folders),'bytes',flush=True)
                    if name == 'kaist_load':
                        records = json.loads(listing)
                        for record in records:
                            details = record['content_details']
                            if record['filename'] == 'acoustic.zip':
                                path = DEST/'kaist_acoustic.zip'
                                binary = fetch(details['download_url'],path)
                                assert hashlib.sha256(binary).hexdigest() == details['sha256_hash']
                                with ZipFile(path) as source:
                                    entries = [dict(name=i.filename,bytes=i.file_size,crc=i.CRC) for i in source.infolist()]
                                (DEST/'kaist_acoustic_inventory.json').write_text(json.dumps(dict(
                                    verified_full_sha256=True,sha256=details['sha256_hash'],members=entries),indent=2)+'\n')
                                print('KAIST acoustic archive authenticated:',len(entries),'members',flush=True)
                            elif record['filename'] == 'vibration.zip':
                                try:
                                    remote = RemoteZip(details['download_url'],details['size'])
                                    with ZipFile(remote) as source:
                                        entries = [dict(name=i.filename,bytes=i.file_size,crc=i.CRC) for i in source.infolist()]
                                    (DEST/'kaist_vibration_inventory.json').write_text(json.dumps(dict(
                                        verified_full_sha256=False,published_sha256=details['sha256_hash'],
                                        metadata_bytes_fetched=remote.fetched,members=entries),indent=2)+'\n')
                                    print('KAIST vibration:',len(entries),'members via',remote.fetched,'range bytes',flush=True)
                                except Exception as exc:
                                    print('KAIST vibration metadata unavailable:',str(exc),flush=True)
                except Exception as exc:
                    result['sources'][-1]['file_listing_error'] = str(exc)
        except Exception as exc:
            result['sources'].append(dict(name=name,url=url,status='unavailable',reason=str(exc)))
            print(name,type(exc).__name__,str(exc),flush=True)
    archive = DEST/'uci_hydraulic.zip'
    url = 'https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip'
    try:
        content = fetch(url,archive)
        with ZipFile(archive) as source:
            entries = [dict(name=i.filename,bytes=i.file_size,compressed=i.compress_size,crc=i.CRC) for i in source.infolist()]
            for entry in entries:
                name = entry['name']
                # Read only metadata/profile members. Sensor arrays remain unopened.
                if name.lower().endswith(('description.txt','profile.txt','readme.txt')):
                    data = source.read(name)
                    assert len(data) < 1000000
                    (DEST/Path(name).name).write_bytes(data)
            result['uci_archive'] = dict(url=url,sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content),members=entries)
        print('UCI archive',len(content),'bytes; profile/description only',flush=True)
    except Exception as exc:
        result['uci_archive'] = dict(url=url,error=str(exc))
        print('UCI archive',type(exc).__name__,str(exc),flush=True)
    (ROOT/'experiments/results/dataset_scout.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
