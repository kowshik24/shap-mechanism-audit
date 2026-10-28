"""Verify physical protocols and assess every registered intervention/channel pair."""
import pdm_audit
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from pdm_audit.physical_map import randomized_response,holm_adjust,path_status,map_coverage

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/causal_chamber';OUT=ROOT/'experiments/results'
FEATURES={'lt_validate_v1':['current','angle_1','angle_2','ir_1','vis_1','ir_2','vis_2','ir_3','vis_3','v_board','v_reg'],
          'wt_validate_v1':['current_in','current_out','rpm_in','rpm_out','pressure_upwind','pressure_downwind','pressure_ambient','pressure_intake','mic','signal_1','signal_2']}


def main():
    source=json.loads((OUT/'chamber_sources.json').read_text());assert source['complete']
    for p,h in source['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    ledger=[];pairs=[];hypotheses=[];payloads={};summary=[]
    for item in source['datasets']:
        dataset=item['dataset'];directory=DATA/dataset/'source';configs=pd.read_csv(directory/item['configuration'])
        graph=pd.read_csv(directory/f'adjacency_{dataset[:2]}_standard.csv',index_col=0)
        assert list(graph.index)==list(graph.columns) and np.isin(graph.to_numpy(),[0,1]).all()
        rng=np.random.default_rng(56192837);target_column='target' if dataset.startswith('lt') else 'from'
        controls=list(configs[target_column].unique());features=FEATURES[dataset]
        for _,row in configs.iterrows():
            target=row[target_column];suffix='' if dataset.startswith('lt') or pd.isna(row['to']) else '_'+row['to']
            experiment='validate_'+target+suffix
            member=next(x for x in item['members'] if Path(x['member']).name==experiment+'.csv');df=pd.read_csv(ROOT/member['path'])
            assert set(df.columns)-set(controls)-{'timestamp','config','counter','flag','intervention','camera'}==set(features)
            n=int(row.N);waiting=rng.integers(1,1000,endpoint=True,size=n);flags=np.array([rng.choice([0,1]) for _ in range(n)])
            checks={'row_count':len(df)==n,'counter':np.array_equal(df.counter,np.arange(len(df))),
                    'treatment_replay':np.array_equal(df.flag,flags),'target_setting':np.allclose(df[target],np.where(df.flag==0,row.xA,row.xB),rtol=1e-5,atol=1e-6),
                    'clock_increases':bool((np.diff(df.timestamp)>0).all()),'configuration':bool((df.config=='standard').all()),
                    'finite_features':bool(np.isfinite(df[features].to_numpy()).all()),
                    'fixed_controls':all(np.allclose(df[c],row[c],rtol=1e-5,atol=1e-6) for c in controls if c!=target),
                    'after_set':bool((df.intervention==1).all()),
                    'camera_disabled_when_applicable':bool((df.camera==0).all()) if 'camera' in df else True}
            numeric=df.drop(columns=['timestamp','config','counter','flag','intervention']).to_numpy(dtype='<f8')
            digest=hashlib.sha256(numeric.tobytes()).hexdigest();duplicate_of=payloads.get((dataset,digest));payloads[dataset,digest]=experiment
            record={'dataset':dataset,'experiment':experiment,'target':target,'path':member['path'],'rows':len(df),
                    'treatment_counts':[int((df.flag==k).sum()) for k in [0,1]],'source_settling_wait_ms':int(row['T']),
                    'clock_range':[float(df.timestamp.min()),float(df.timestamp.max())],
                    'zip_modified_unverified':member['zip_modified_unverified'],
                    'clock_reference_verified':False,'device_serial_or_session_id_available':False,
                    'numerical_payload_sha256':digest,'duplicate_of':duplicate_of,'checks':checks,
                    'passed':all(checks.values()) and duplicate_of is None}
            ledger.append(record)
            for feature in features:
                response=randomized_response(df[feature],df.flag)
                pair={'dataset':dataset,'experiment':experiment,'target':target,'feature':feature,
                      'source_graph':path_status(graph.to_numpy(),list(graph.index),target,feature),
                      'response_testing_eligible':record['passed'],**response}
                pairs.append(pair)
                for test in response['tests']:
                    test['family_input_p']=test['p'] if record['passed'] else 1.
                    hypotheses.append(test)
        summary.append({'dataset':dataset,'source_configurations':len(configs),'measurement_channels':len(features),
                        'expected_pairs':len(configs)*len(features),'rows':sum(x['rows'] for x in ledger if x['dataset']==dataset)})
    adjusted=holm_adjust([x['family_input_p'] for x in hypotheses])
    for test,p in zip(hypotheses,adjusted):test['holm_p']=float(p)
    for pair in pairs:
        pair['randomized_response_supported']=any(x['holm_p']<=.05 for x in pair['tests'])
        pair['state']='supported' if pair['randomized_response_supported'] or pair['source_graph'] in ['direct','path'] else 'unknown'
        pair['support_basis']=('response_and_graph' if pair['source_graph'] in ['direct','path'] else 'response_only') if pair['randomized_response_supported'] else ('graph_only' if pair['source_graph'] in ['direct','path'] else 'unknown')
        pair['causally_excluded']=False
    flat=[]
    for pair in pairs:
        row={k:v for k,v in pair.items() if k!='tests'}
        for test in pair['tests']:
            name=test['endpoint'];row.update({name+'_'+k:v for k,v in test.items() if k!='endpoint'})
        flat.append(row)
    csv=OUT/'chamber_physical_map.csv';pd.DataFrame(flat).to_csv(csv,index=False)
    acquisition=OUT/'chamber_acquisitions.json';acquisition.write_text(json.dumps(ledger,indent=2)+'\n')
    for item in summary:
        these=[x for x in pairs if x['dataset']==item['dataset']]
        item.update(passed_acquisitions=sum(x['passed'] for x in ledger if x['dataset']==item['dataset']),
                    assessed_pairs=len(these),supported_pairs=sum(x['state']=='supported' for x in these),
                    response_supported_pairs=sum(x['randomized_response_supported'] for x in these),
                    unknown_pairs=sum(x['state']=='unknown' for x in these),excluded_pairs=0,
                    unmodeled_pairs=sum(x['source_graph']=='unmodeled' for x in these))
    coverage=[]
    for record in ledger:
        these=[x for x in pairs if x['dataset']==record['dataset'] and x['experiment']==record['experiment']]
        coverage.append({'dataset':record['dataset'],'experiment':record['experiment'],
                         **map_coverage(FEATURES[record['dataset']],[x['feature'] for x in these if x['state']=='supported'],[])})
    files=[Path(__file__),ROOT/'src/pdm_audit/physical_map.py',ROOT/'experiments/PHYSICAL_INTERVENTION_PROTOCOL.md',ROOT/'experiments/PHYSICAL_INTERVENTION_RUN_NOTE.md',OUT/'chamber_sources.json',csv,acquisition]
    result={'complete':True,'summary':summary,'pairs':pairs,'coverage':coverage,'hypothesis_family_size':len(hypotheses),
            'all_acquisitions_passed':all(x['passed'] for x in ledger),'all_pairs_assessed':len(pairs)==sum(x['expected_pairs'] for x in summary),
            'new_machine_independence_verified':False,'acquisition_exchangeability_verified':False,
            'exhaustive_causal_map_verified':False,'calibrated_physical_warning_supported':False,
            'scope':'Authenticated randomized physical intervention evidence; finite-grid assessment with explicit unknowns, not an exhaustive causal truth map',
            'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'chamber_physical_map.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':summary,'all_acquisitions_passed':result['all_acquisitions_passed'],'family':len(hypotheses)},indent=2))


if __name__=='__main__':main()
