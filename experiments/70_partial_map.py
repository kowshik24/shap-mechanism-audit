"""Carry incomplete physical feature maps through attribution and rank decisions."""
import pdm_audit
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pdm_audit.partial_map import identification_ranges,conservative_rank_cut
from pdm_audit.cwru import EXPECTED as CWRU_EXPECTED
from pdm_audit.sca import EXPECTED as SCA_EXPECTED,CLASSES

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,list):return [safe(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return 'Infinity' if x>0 else '-Infinity'
    return x


def main():
    inputs=[]
    for name in ['reference_transfer_verification','sca_field_verification','dtu_prospective_verification','chamber_verification']:
        path=OUT/(name+'.json');proof=json.loads(path.read_text());assert proof['verified'];inputs.append(path)
        for p,h in proof['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    rows=[];arrays={}
    def add(study,identifier,features,required,a,b,point,scope,**metadata):
        bounds=identification_ranges(a,b,required,())
        assert np.isclose(point['R'],bounds['R'][0],atol=1e-12)
        if b is not None:
            assert bounds['D'][0]-1e-12<=point['D']<=bounds['D'][1]+1e-12
            assert np.isclose(point['S'],bounds['S'][1],atol=1e-12)
        index=len(rows);arrays[f'own_{index}']=a;arrays[f'comparison_{index}']=b if b is not None else np.full(len(a),np.nan)
        rows.append({'study':study,'identifier':identifier,'features':features,'point':point,'bounds':bounds,'array_index':index,
                     'scope':scope,'universal_outside_warning_available':False,**metadata})

    cw=json.loads((OUT/'cwru_attribution_audit.json').read_text());raw=np.load(OUT/'cwru_attribution_draws.npz');meta=pd.DataFrame(cw['recording_metadata']);features=cw['features']
    for c,condition in enumerate(['original','spectral']):
        for v,view in enumerate(['matched','filter_removed']):
            if c==0 and v==1:continue
            for j,fault in enumerate(['B','IR','OR']):
                weights=pd.DataFrame(abs(raw['signed_shap'][c,v,:,:,j]),columns=features).groupby(meta.recording.to_numpy()).mean()
                labels=meta.groupby('recording').fault.first();own=weights.loc[labels==fault].mean().to_numpy();rest=weights.loc[labels!=fault].mean().to_numpy()
                a,b=own/own.sum(),rest/rest.sum();expected=[features.index(x) for x in CWRU_EXPECTED[fault]]
                saved=next(x for x in cw['records'] if x['condition']==condition and x['view']==view and x['fault']==fault)
                add('CWRU',f'{condition}/{view}/{fault}',features,expected,a,b,{k:saved[k] for k in ['R','D','S']},'Equal recording weights; partial map, no physical exclusions',fault=fault,condition=condition,view=view)
    field=json.loads((OUT/'sca_field.json').read_text());data=np.load(OUT/'sca_field_arrays.npz');df=pd.read_csv(OUT/'sca_features.csv')
    for item in field['summary']:
        view=item['view'];features=item['columns'];phi=abs(data[view+'_shap']);use=df.case<=10
        t=pd.DataFrame(phi[use],columns=features);t['case']=df.loc[use,'case'].to_numpy();t['label']=df.loc[use,'label'].to_numpy()
        grouped=t[t.label>0].groupby(['case','label'])[features].mean()
        for label,fault in CLASSES.items():
            own=grouped.index.get_level_values('label')==label;left=grouped.loc[own].mean().to_numpy();right=grouped.loc[~own].mean().to_numpy()
            saved=next(x for x in item['audit'] if x['fault']==fault)
            add('SCA pooled',f'{view}/{fault}',features,[features.index(x) for x in SCA_EXPECTED[fault]],left/left.sum(),right/right.sum(),
                {k:saved[k] for k in ['R','D','S']},'Case/type weighted cross-fitted summary; not one fixed-function audit',view=view,fault=fault)
        for case in range(1,11):
            selected=(df.case==case)&(df.label>0);labels=df.loc[selected,'label'].unique();assert len(labels)==1
            fault=CLASSES[int(labels[0])];left=phi[selected].mean(axis=0);a=left/left.sum()
            saved=next(z for x in item['within_case_audit'] if x['case']==case for z in x['audit'] if z['fault']==fault)
            assert not saved['available'] and saved['reason']=='missing_other_failure_cohort'
            add('SCA within case',f'{view}/{case}/{fault}',features,[features.index(x) for x in SCA_EXPECTED[fault]],a,None,
                {'R':saved['R']},'Missing other-failure cohort: relevance range only',view=view,case=case,fault=fault)
    dtu=json.loads((OUT/'dtu_prospective.json').read_text());draws=np.load(OUT/'dtu_prospective_draws.npz');features=dtu['feature_names'];flags=np.r_[np.zeros(16,int),np.full(48,4),np.zeros(16,int),np.full(48,5)]
    expected={int(k):v for k,v in dtu['expected_feature_indices'].items()};scores={}
    for record in dtu['records']:
        prefix=f"{record['role']}_{record['packet']:03d}"
        for item in record['results']:
            phi=abs(draws[prefix+'_'+item['view']+'_shap']);upper=[];lower=[]
            for fault in [4,5]:
                own=phi[flags==fault].mean(axis=0);rest=phi[(flags>0)&(flags!=fault)].mean(axis=0);a,b=own/own.sum(),rest/rest.sum()
                saved=next(x for x in item['scores'][0]['statistics'] if x['mechanism']==fault)
                add('DTU',f'{prefix}/{item["view"]}/{fault}',features,expected[fault],a,b,{k:saved[k] for k in ['R','D','S']},
                    'Frozen fitting/simulation law; incomplete component map, no physical exclusions',role=record['role'],packet=record['packet'],view=item['view'],fault=fault)
                lower.append(rows[-1]['bounds']['S'][0]);upper.append(rows[-1]['bounds']['S'][1])
            scores[prefix,item['view']]=(max(lower),max(upper))
    bank=[scores[f'calibration_{i:03d}','unaltered'][1] for i in range(41,141)];cut,rank=conservative_rank_cut(bank)
    assert cut==dtu['cuts'][0] and rank==96
    rank_summary=[]
    for role,views in [('evaluation',['unaltered']),('biased',['matched','removed','reversed'])]:
        for view in views:
            selected=[scores[f'{role}_{i:03d}',view] for i in range(141,201)]
            old=next(x for x in dtu['summary'] if x['score']=='shap' and x['role']==role and x['view']==view)
            assert sum(high>cut for low,high in selected)==old['warning_count']
            rank_summary.append({'role':role,'view':view,'units':60,'conservative_exceedances':sum(low>cut for low,high in selected),
                                 'causal_warning_abstentions':60,'closed_map_recovery_warnings':old['warning_count']})
    path=OUT/'partial_map_draws.npz';np.savez_compressed(path,**arrays)
    flat=[]
    for row in rows:
        item={k:row[k] for k in ['study','identifier','scope']};item['unknown_features']=len(row['bounds']['unknown'])
        for score in ['R','D','S']:
            interval=row['bounds'][score];item[score+'_lower']=interval[0] if interval is not None else None;item[score+'_upper']=interval[1] if interval is not None else None
        flat.append(item)
    csv=OUT/'partial_map_ranges.csv';pd.DataFrame(flat).to_csv(csv,index=False)
    files=inputs+[Path(__file__),ROOT/'src/pdm_audit/partial_map.py',ROOT/'experiments/PARTIAL_MAP_PROTOCOL.md',
        OUT/'cwru_attribution_audit.json',OUT/'cwru_attribution_draws.npz',OUT/'sca_field.json',OUT/'sca_field_arrays.npz',OUT/'sca_features.csv',
        OUT/'dtu_prospective.json',OUT/'dtu_prospective_draws.npz',path,csv]
    result={'complete':True,'rows':rows,'rank_summary':rank_summary,'calibration_upper_cut':cut,'calibration_rank':rank,
            'scope':'Post hoc identification ranges conditional on supplied map constraints and observed shares; not causal or sampling confidence intervals',
            'physical_exclusions_invented':False,'physical_acquisition_independence_verified':False,
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'partial_map.json').write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n')
    print('Computed',len(rows),'partial-map identification ranges; original point scores and DTU cut preserved')


if __name__=='__main__':main()
