"""Retain and authenticate author-hosted physical intervention measurements."""
import pdm_audit
import hashlib,json,re,urllib.request,zipfile
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/causal_chamber';OUT=ROOT/'experiments/results'
DATA_COMMIT='0c8a77ea14bd4437b1fdd79adc1aa91ac2342569'
GRAPH_COMMIT='9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6'


def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response:path.write_bytes(response.read())
    return path.read_bytes()


def main():
    DATA.mkdir(exist_ok=True);files=[];datasets=[]
    for dataset,config in [('lt_validate_v1','lt_validation_configs.csv'),('wt_validate_v1','wt_standard_validation_configs.csv')]:
        base=f'https://raw.githubusercontent.com/juangamella/causal-chamber/{DATA_COMMIT}/datasets/{dataset}/'
        for name in ['README.md','variables.csv',config,'generators/binary_interventions.py','LICENSE_DATASETS.txt','LICENSE_SOFTWARE.txt']:
            path=DATA/dataset/'source'/name;fetch(base+name,path);files.append(path)
        readme=(DATA/dataset/'source/README.md').read_text(encoding='utf-8-sig')
        url,md5=re.search(r'\[ZIP file\]\(([^)]+)\)\s*\|\s*([a-f0-9]{32})',readme).groups()
        archive=DATA/(dataset+'.zip');payload=fetch(url,archive);assert hashlib.md5(payload).hexdigest()==md5
        files.append(archive);members=[]
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            for entry in z.infolist():
                if entry.is_dir():continue
                relative=Path(entry.filename);assert not relative.is_absolute() and '..' not in relative.parts
                path=DATA/dataset/'members'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(entry))
                files.append(path);members.append({'member':entry.filename,'bytes':entry.file_size,'CRC':entry.CRC,
                    'zip_modified_unverified':list(entry.date_time),'path':str(path.relative_to(ROOT))})
        graph='adjacency_'+dataset[:2]+'_standard.csv';url=f'https://raw.githubusercontent.com/juangamella/causal-chamber-package/{GRAPH_COMMIT}/causalchamber/ground_truth/adjacencies/{graph}'
        path=DATA/dataset/'source'/graph;fetch(url,path);files.append(path)
        datasets.append({'dataset':dataset,'configuration':config,'download_url':re.search(r'\[ZIP file\]\(([^)]+)\)',readme).group(1),
                         'archive':str(archive.relative_to(ROOT)),'published_md5':md5,'bytes':len(payload),'members':members})
        print('Authenticated physical source',dataset,len(payload),'bytes',len(members),'members',flush=True)
    files += [Path(__file__),ROOT/'experiments/PHYSICAL_INTERVENTION_PROTOCOL.md']
    result={'complete':True,'retrieved_utc':datetime.now(timezone.utc).isoformat(),'data_commit':DATA_COMMIT,'graph_commit':GRAPH_COMMIT,
            'physical_systems':['Light tunnel Mk1','Wind tunnel Mk1'],'datasets':datasets,
            'physical_acquisition_provenance':'Author-hosted measurements authenticated against pinned published archive MD5s',
            'new_machine_independence_verified':False,'causal_map_exhaustive_verified':False,
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'chamber_sources.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
