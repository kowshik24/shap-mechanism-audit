"""Independent share reconstruction, map enumeration and rank-event checks."""
import pdm_audit
import hashlib,itertools,json,math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def point(a,b,e):
    outside=list(set(range(len(a)))-set(e));r=float(a[list(e)].sum())
    return [r,None,None] if b is None else [r,float((a-b)[list(e)].sum()),float(max((a-b)[outside],default=-np.inf))]


def main():
    result=json.loads((OUT/'partial_map.json').read_text());assert result['complete']
    for p,h in result['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    saved=np.load(OUT/'partial_map_draws.npz');cw=json.loads((OUT/'cwru_attribution_audit.json').read_text());cr=np.load(OUT/'cwru_attribution_draws.npz')
    field=json.loads((OUT/'sca_field.json').read_text());fr=np.load(OUT/'sca_field_arrays.npz');df=pd.read_csv(OUT/'sca_features.csv')
    dtu=json.loads((OUT/'dtu_prospective.json').read_text());dr=np.load(OUT/'dtu_prospective_draws.npz');meta=pd.DataFrame(cw['recording_metadata'])
    vectors={}
    for row in result['rows']:
        study=row['study'];features=row['features'];fault=row['fault'];e=row['bounds']['required']
        if study=='CWRU':
            c=0 if row['condition']=='original' else 1;v=0 if row['view']=='matched' else 1;j=['B','IR','OR'].index(fault)
            weights=[];labels=[]
            for recording in sorted(meta.recording.unique()):
                mask=meta.recording.to_numpy()==recording;weights.append(abs(cr['signed_shap'][c,v,mask,:,j]).mean(axis=0));labels.append(meta.loc[mask,'fault'].iloc[0])
            weights=np.asarray(weights);labels=np.asarray(labels);a=weights[labels==fault].mean(axis=0);b=weights[labels!=fault].mean(axis=0)
            assert set(e)=={i for i,name in enumerate(features) if name.startswith(('raw_'+fault+'_','envelope_'+fault+'_'))}
        elif study.startswith('SCA'):
            phi=abs(fr[row['view']+'_shap']);label={'IR':1,'B':2,'OR':3}[fault]
            if study=='SCA within case':
                use=(df.case.to_numpy()==row['case'])&(df.label.to_numpy()==label);a=phi[use].mean(axis=0);b=None
                assert not ((df.case==row['case'])&(df.label>0)&(df.label!=label)).any()
            else:
                own=[];other=[]
                for case in range(1,11):
                    for candidate in [1,2,3]:
                        use=(df.case.to_numpy()==case)&(df.label.to_numpy()==candidate)
                        if use.any():(own if candidate==label else other).append(phi[use].mean(axis=0))
                a=np.mean(own,axis=0);b=np.mean(other,axis=0)
            assert set(e)=={i for i,name in enumerate(features) if '_'+fault+'_' in name}
        else:
            assert study=='DTU';prefix=f"{row['role']}_{row['packet']:03d}_{row['view']}";phi=abs(dr[prefix+'_shap'])
            own=np.arange(16,64) if fault==4 else np.arange(80,128);rest=np.arange(80,128) if fault==4 else np.arange(16,64)
            a=phi[own].mean(axis=0);b=phi[rest].mean(axis=0)
            assert e==dtu['expected_feature_indices'][str(fault)]
        a=a/a.sum();b=b/b.sum() if b is not None else None
        index=row['array_index'];assert np.allclose(saved[f'own_{index}'],a,atol=1e-14)
        assert np.isnan(saved[f'comparison_{index}']).all() if b is None else np.allclose(saved[f'comparison_{index}'],b,atol=1e-14)
        assert not row['bounds']['excluded'] and not row['universal_outside_warning_available']
        u=sorted(set(range(len(a)))-set(e));assert u==row['bounds']['unknown']
        actual=point(a,b,e)
        for k,key in enumerate(['R','D','S']):
            if actual[k] is not None:assert np.isclose(actual[k],row['point'][key],atol=1e-12)
        vectors[index]=(a,b,e,u)
    enumerated=0
    for row in result['rows']:
        a,b,e,u=vectors[row['array_index']];bounds=row['bounds']
        if row['study']!='DTU':
            size=1<<len(u);membership=np.zeros((size,len(a)),dtype=bool);membership[:,e]=True
            membership[:,u]=((np.arange(size)[:,None]>>np.arange(len(u)))&1).astype(bool)
            outcomes=[membership@a]
            if b is not None:outcomes += [membership@(a-b),np.where(~membership,(a-b)[None,:],-np.inf).max(axis=1)]
            enumerated+=size
            for k,key in enumerate(['R','D','S']):
                if k>=len(outcomes):assert bounds[key] is None;continue
                assert np.allclose([outcomes[k].min(),outcomes[k].max()],np.array(bounds[key],dtype=float),atol=1e-12)
        else:
            delta=a-b;witnesses={'R':[e,e+u],
                'D':[e+[j for j in u if delta[j]<0],e+[j for j in u if delta[j]>0]],'S':[e+u,e]}
            for k,key in enumerate(['R','D','S']):
                attained=[point(a,b,expected)[k] for expected in witnesses[key]]
                assert np.allclose(attained,np.array(bounds[key],dtype=float),atol=1e-12)
    assert enumerated==2002176
    calibration={};evaluation={}
    for row in result['rows']:
        if row['study']!='DTU':continue
        key=row['role'],row['packet'],row['view'];interval=np.array(row['bounds']['S'],dtype=float)
        if key not in evaluation:evaluation[key]=[-np.inf,-np.inf]
        evaluation[key]=np.maximum(evaluation[key],interval)
    bank=[evaluation['calibration',i,'unaltered'][1] for i in range(41,141)];k=math.ceil(101*.95);cut=sorted(bank)[k-1]
    assert k==96 and cut==result['calibration_upper_cut']==dtu['cuts'][0]
    for row in result['rank_summary']:
        values=[evaluation[row['role'],i,row['view']] for i in range(141,201)]
        assert row['conservative_exceedances']==sum(v[0]>cut for v in values)==0
        assert row['causal_warning_abstentions']==row['units']==60
        assert row['closed_map_recovery_warnings']==sum(v[1]>cut for v in values)
    warnings=0;trials=0
    for permutation in itertools.permutations(range(5)):
        exact=np.asarray(permutation,dtype=float);upper=exact[:-1]+.2;lower=exact[-1]-.2;k=math.ceil(5*.6)
        robust=lower>np.sort(upper)[k-1];ordinary=exact[-1]>np.sort(exact[:-1])[k-1]
        assert not robust or ordinary;warnings+=robust;trials+=1
    assert trials==120 and warnings/trials<=.4
    assert not result['physical_exclusions_invented'] and not result['physical_acquisition_independence_verified']
    files=[Path(__file__),OUT/'partial_map.json',OUT/'partial_map_draws.npz',OUT/'partial_map_ranges.csv']
    (OUT/'partial_map_verification.json').write_text(json.dumps({'verified':True,'identified_ranges':len(result['rows']),
        'exact_map_completions_enumerated':enumerated,'rank_event_permutations':trials,
        'scope':'Identification arithmetic and conservative event inclusion; no field acquisition/causal certificate',
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    print('PASS',len(result['rows']),'share reconstructions,',enumerated,'map completions and conservative rank inclusion')


if __name__=='__main__':main()
