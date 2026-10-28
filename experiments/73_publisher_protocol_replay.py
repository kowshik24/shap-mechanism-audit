"""Compile the original publisher generators locally; never actuate hardware."""
import pdm_audit
import hashlib,json,re,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';DATA=ROOT/'data/causal_chamber'


def main():
    source=json.loads((OUT/'chamber_sources.json').read_text());ledger=json.loads((OUT/'chamber_acquisitions.json').read_text());records=[];files=[]
    for p,h in source['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    for dataset in ['lt_validate_v1','wt_validate_v1']:
        directory=DATA/'compiled_protocols'/dataset;(directory/'protocols').mkdir(parents=True,exist_ok=True)
        generator=DATA/dataset/'source/generators/binary_interventions.py'
        output=subprocess.check_output([sys.executable,str(generator)],cwd=directory,text=True)
        (directory/'generator_stdout.txt').write_text(output);files.extend([generator,directory/'generator_stdout.txt'])
        for acquisition in [x for x in ledger if x['dataset']==dataset]:
            path=directory/'protocols'/(acquisition['experiment']+'.txt');text=path.read_text();files.append(path)
            flags=np.array([int(x) for x in re.findall(r'^SET,flag,([01])$',text,re.M)])
            settings=np.array([float(x) for x in re.findall(r'^SET,'+re.escape(acquisition['target'])+r',([^\n]+)$',text,re.M)])
            frame=pd.read_csv(ROOT/acquisition['path']);matches=np.array_equal(frame.flag,flags)
            assert matches==acquisition['checks']['treatment_replay']
            assert len(flags)==len(settings)==len(re.findall(r'^MSR,1,0$',text,re.M))
            records.append({'dataset':dataset,'experiment':acquisition['experiment'],'protocol':str(path.relative_to(ROOT)),
                            'compiled_measurements':len(flags),'recorded_measurements':len(frame),'flag_match':matches,
                            'target_sequence_match':bool(len(settings)==len(frame) and np.allclose(settings,frame[acquisition['target']],rtol=1e-5,atol=1e-6)),
                            'protocol_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    assert len(records)==57 and sum(x['flag_match'] for x in records)==32
    files += [Path(__file__),OUT/'chamber_sources.json',OUT/'chamber_acquisitions.json']
    result={'verified':True,'records':records,'physical_commands_executed':False,
            'scope':'Source-authenticated original generators compile to local text only; confirms 32/57 exact record associations, not physical acquisition independence',
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'publisher_protocol_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS original publisher generators independently reproduce 32/57 record schedules; no hardware connection')


if __name__=='__main__':main()
