"""Offline independent replay of source integrity, acquisitions and effect-map arithmetic."""
import pdm_audit
import hashlib,json,re,zipfile,zlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from scipy.special import logsumexp

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';DATA=ROOT/'data/causal_chamber'


def exact_table_probability(table):
    a=np.asarray(table);n=a.sum();r=a[0].sum();success=a[:,1].sum();observed=a[0,1]
    support=np.arange(max(0,r+success-n),min(r,success)+1)
    logs=hypergeom.logpmf(support,n,success,r);obs=hypergeom.logpmf(observed,n,success,r)
    return min(1.,float(np.exp(logsumexp(logs[logs<=obs+1e-8]))))


def main():
    source=json.loads((OUT/'chamber_sources.json').read_text());result=json.loads((OUT/'chamber_physical_map.json').read_text())
    for artifact in [source,result]:
        for p,h in artifact['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    saved_ledger=json.loads((OUT/'chamber_acquisitions.json').read_text());pairs={(x['dataset'],x['experiment'],x['feature']):x for x in result['pairs']}
    all_tests=[];checks=[];seen={};rows=0;channel_total=0
    for item in source['datasets']:
        dataset=item['dataset'];base=DATA/dataset/'source';readme=(base/'README.md').read_text(encoding='utf-8-sig')
        published=re.search(r'\[ZIP file\]\([^)]+\)\s*\|\s*([a-f0-9]{32})',readme).group(1)
        archive=ROOT/item['archive'];assert hashlib.md5(archive.read_bytes()).hexdigest()==published==item['published_md5']
        with zipfile.ZipFile(archive) as z:
            assert {i.filename for i in z.infolist() if not i.is_dir()}=={x['member'] for x in item['members']}
            for member in item['members']:
                payload=(ROOT/member['path']).read_bytes();assert payload==z.read(member['member'])
                assert len(payload)==member['bytes'] and zlib.crc32(payload)==member['CRC']
                assert member['zip_modified_unverified']==list(z.getinfo(member['member']).date_time)
        config=pd.read_csv(base/item['configuration']);target_key='target' if dataset.startswith('lt') else 'from'
        fields=(['current','angle_1','angle_2','ir_1','vis_1','ir_2','vis_2','ir_3','vis_3','v_board','v_reg'] if dataset.startswith('lt') else
                ['current_in','current_out','rpm_in','rpm_out','pressure_upwind','pressure_downwind','pressure_ambient','pressure_intake','mic','signal_1','signal_2'])
        graph=pd.read_csv(base/f'adjacency_{dataset[:2]}_standard.csv',index_col=0);reach=graph.to_numpy().astype(bool).copy()
        for k in range(len(reach)):reach |= reach[:,k,None]&reach[None,k,:]
        names=list(graph.index);rng=np.random.default_rng(56192837)
        for _,setting in config.iterrows():
            target=setting[target_key];suffix='' if dataset.startswith('lt') or pd.isna(setting['to']) else '_'+setting['to']
            experiment='validate_'+target+suffix;record=next(x for x in saved_ledger if x['dataset']==dataset and x['experiment']==experiment)
            df=pd.read_csv(ROOT/record['path']);rows+=len(df);n=int(setting['N'])
            assert set(df.columns)-set(config[target_key])-{'timestamp','config','counter','flag','intervention','camera'}==set(fields)
            rng.integers(1,1000,endpoint=True,size=n);flags=np.array([rng.choice([0,1]) for _ in range(n)])
            actual={'row_count':len(df)==n,'counter':np.array_equal(df.counter,np.arange(len(df))),
                    'treatment_replay':np.array_equal(df.flag,flags),'target_setting':np.allclose(df[target],np.where(df.flag==0,setting.xA,setting.xB),rtol=1e-5,atol=1e-6),
                    'clock_increases':bool((np.diff(df.timestamp)>0).all()),'configuration':bool((df.config=='standard').all()),
                    'finite_features':bool(np.isfinite(df[fields]).all().all()),
                    'fixed_controls':all(np.allclose(df[c],setting[c],rtol=1e-5,atol=1e-6) for c in config[target_key].unique() if c!=target),
                    'after_set':bool((df.intervention==1).all()),
                    'camera_disabled_when_applicable':bool((df.camera==0).all()) if 'camera' in df else True}
            assert actual==record['checks']
            payload=df.drop(columns=['timestamp','config','counter','flag','intervention']).to_numpy(dtype='<f8').tobytes()
            sha=hashlib.sha256(payload).hexdigest();duplicate=seen.get((dataset,sha));seen[dataset,sha]=experiment
            assert record['numerical_payload_sha256']==sha and record['duplicate_of']==duplicate
            assert record['passed']==(all(actual.values()) and duplicate is None)
            assert record['treatment_counts']==[int((df.flag==k).sum()) for k in [0,1]]
            assert record['source_settling_wait_ms']==int(setting['T']) and record['rows']==len(df)
            member=next(x for x in item['members'] if x['path']==record['path'])
            assert record['zip_modified_unverified']==member['zip_modified_unverified']
            assert np.allclose(record['clock_range'],[df.timestamp.min(),df.timestamp.max()])
            assert not record['clock_reference_verified'] and not record['device_serial_or_session_id_available']
            for feature in fields:
                channel_total+=1;p=pairs[dataset,experiment,feature];values=df[feature].to_numpy();center=np.median(values);dev=abs(values-center)
                assert p['response_testing_eligible']==record['passed']
                if target not in names or feature not in names:status='unmodeled'
                else:
                    i,j=names.index(target),names.index(feature)
                    status='direct' if graph.iloc[i,j] else ('path' if reach[i,j] else 'no_listed_path')
                assert p['source_graph']==status
                assert np.isclose(p['median_A'],np.median(values[df.flag==0])) and np.isclose(p['median_B'],np.median(values[df.flag==1]))
                assert np.isclose(p['mean_difference'],values[df.flag==1].mean()-values[df.flag==0].mean())
                for test,indicator,cut in zip(p['tests'],[values>center,dev>np.median(dev)],[center,np.median(dev)]):
                    counts=[[int(np.count_nonzero((df.flag.to_numpy()==t)&(indicator==v))) for v in [0,1]] for t in [0,1]]
                    assert counts==test['counts'] and np.isclose(test['threshold'],cut)
                    actual_p=exact_table_probability(counts);assert np.isclose(actual_p,test['p'],rtol=1e-7,atol=1e-11)
                    assert np.isclose(test['risk_difference'],counts[1][1]/sum(counts[1])-counts[0][1]/sum(counts[0]))
                    assert test['family_input_p']==(test['p'] if record['passed'] else 1.)
                    all_tests.append(test)
            checks.append(record['passed'])
    assert rows==sum(x['rows'] for x in result['summary']) and channel_total==len(pairs)==627
    assert len(all_tests)==result['hypothesis_family_size']==1254
    order=sorted(range(len(all_tests)),key=lambda i:all_tests[i]['family_input_p']);running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1,(len(order)-rank)*all_tests[i]['family_input_p']))
        assert np.isclose(all_tests[i]['holm_p'],running)
    for pair in pairs.values():
        supported=any(t['holm_p']<=.05 for t in pair['tests']);graph=pair['source_graph'] in ['direct','path']
        assert pair['randomized_response_supported']==supported and not pair['causally_excluded']
        assert pair['state']==('supported' if supported or graph else 'unknown')
    for row in result['summary']:
        these=[p for p in pairs.values() if p['dataset']==row['dataset']]
        assert row['assessed_pairs']==row['expected_pairs']==len(these)
        assert row['supported_pairs']==sum(p['state']=='supported' for p in these)
        assert row['response_supported_pairs']==sum(p['randomized_response_supported'] for p in these)
        assert row['unknown_pairs']==sum(p['state']=='unknown' for p in these) and row['excluded_pairs']==0
    for row in result['coverage']:
        these=[p for p in pairs.values() if p['dataset']==row['dataset'] and p['experiment']==row['experiment']]
        assert set(row['supported'])=={p['feature'] for p in these if p['state']=='supported'}
        assert set(row['unknown'])=={p['feature'] for p in these if p['state']=='unknown'}
        assert not row['excluded'] and row['complete_relative_to_universe']==(not row['unknown'])
    assert result['all_pairs_assessed'] and result['all_acquisitions_passed']==all(checks)
    assert not any(result[k] for k in ['new_machine_independence_verified','acquisition_exchangeability_verified','exhaustive_causal_map_verified','calibrated_physical_warning_supported'])
    files=[Path(__file__),OUT/'chamber_sources.json',OUT/'chamber_physical_map.json',OUT/'chamber_acquisitions.json',OUT/'chamber_physical_map.csv']
    proof={'verified':True,'source_acquisitions':len(checks),'source_replays_passed':sum(checks),'measurement_rows':rows,'assessed_pairs':len(pairs),
           'endpoints':len(all_tests),'empirically_supported_pairs':sum(p['randomized_response_supported'] for p in pairs.values()),
           'scope':'Offline reconstruction verifies evidence and failures; no new-machine or exhaustive causal-truth certificate',
           'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'chamber_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('PASS archive integrity, every acquisition gate, all 627 pairs/1254 endpoints and retained unknowns')


if __name__=='__main__':main()
