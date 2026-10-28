"""Check one legacy RNG runtime in isolation from the project environment."""
import pdm_audit
import hashlib,json,re,shutil,subprocess,sys,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';DATA=ROOT/'data/causal_chamber'


def main():
    source=json.loads((OUT/'chamber_sources.json').read_text());current=json.loads((OUT/'publisher_protocol_replay.json').read_text())
    assert current['verified'];directory=DATA/'legacy_numpy126/wt_validate_v1';(directory/'protocols').mkdir(parents=True,exist_ok=True)
    generator=DATA/'wt_validate_v1/source/generators/binary_interventions.py'
    assert hashlib.sha256(generator.read_bytes()).hexdigest()==source['sha256'][str(generator.relative_to(ROOT))]
    url=f"https://raw.githubusercontent.com/juangamella/causal-chamber/{source['data_commit']}/datasets/wt_validate_v1/generators/requirements.txt"
    requirements=directory/'published_requirements.txt'
    if not requirements.exists():
        with urllib.request.urlopen(url,timeout=30) as response:requirements.write_bytes(response.read())
    assert requirements.read_text().strip()=='numpy>=1.18.0'
    uv=shutil.which('uv');assert uv,'This optional compatibility check requires the uv CLI'
    wrapper="import sys,numpy,pandas,runpy; print('RUNTIME',sys.version.split()[0],numpy.__version__,pandas.__version__); runpy.run_path(sys.argv[1],run_name='__main__')"
    command=[uv,'run','--no-project','--isolated','--python','3.11','--with','numpy==1.26.3','--with','pandas==2.1.4','python','-c',wrapper,str(generator)]
    output=subprocess.check_output(command,cwd=directory,text=True);log=directory/'generator_stdout.txt';log.write_text(output)
    runtime=next(line for line in output.splitlines() if line.startswith('RUNTIME ')).split()[1:];assert runtime[1:]==['1.26.3','2.1.4']
    ledger=json.loads((OUT/'chamber_acquisitions.json').read_text());records=[]
    files=[Path(__file__),generator,requirements,log,OUT/'publisher_protocol_replay.json',OUT/'chamber_sources.json']
    for record in [x for x in current['records'] if x['dataset']=='wt_validate_v1']:
        path=directory/'protocols'/(record['experiment']+'.txt');files.append(path)
        assert path.read_bytes()==(ROOT/record['protocol']).read_bytes()
        frame=pd.read_csv(ROOT/next(x['path'] for x in ledger if x['dataset']=='wt_validate_v1' and x['experiment']==record['experiment']))
        flags=[int(v) for v in re.findall(r'^SET,flag,([01])$',path.read_text(),re.M)]
        match=np.array_equal(frame.flag,flags);assert match==record['flag_match']
        records.append({'experiment':record['experiment'],'identical_protocol_text':True,'flag_match':match})
    assert len(records)==28 and sum(r['flag_match'] for r in records)==3
    result={'verified':True,'current_runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__},
        'legacy_runtime':{'python':runtime[0],'numpy':runtime[1],'pandas':runtime[2]},'records':records,
        'physical_commands_executed':False,'original_author_runtime_known':False,'requirements_url':url,
        'scope':'Current/NumPy-1.26.3 comparison gives identical protocol text; does not resolve original acquisition provenance',
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'publisher_runtime_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS isolated NumPy 1.26.3 check: all 28 wind protocols identical; 3 exact record matches')


if __name__=='__main__':main()
