"""Reconstruct source packets and every prospective prediction/audit decision."""
import pdm_audit
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import beta,binomtest
from sklearn.ensemble import RandomForestClassifier
import shap
from pdm_audit.tree_precision import probability_explainer

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def verify_source():
    r=json.loads((OUT/'dtu_prospective_source.json').read_text());assert r['complete']
    for path,digest in r['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    data=np.load(OUT/'dtu_prospective_data.npz')['values'];assert data.shape==(400,64,54)
    lookup={};prefixes={};all_prefixes=set()
    for i,x in enumerate(r['records']):
        raw=(ROOT/x['raw_path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==x['raw_sha256']
        a=np.frombuffer(raw,dtype=np.dtype(x['dtype'])).reshape(x['shape'])
        assert a.shape==(2001,54) and np.isfinite(a).all() and np.all(np.diff(a[:,0])>0)
        targets=np.r_[np.linspace(10,25,16),np.linspace(40,95,48)]
        ids=abs(a[:,0,None]-targets).argmin(axis=0)
        assert np.array_equal(ids,x['indices']) and np.array_equal(a[ids],data[i])
        assert hashlib.sha256(data[i].astype('<f8').tobytes()).hexdigest()==x['selected_sha256']
        settings=np.asarray(x['idv_init']);assert settings.shape==(3,28)
        assert settings[1,x['fault']-1]==1 and settings[2,x['fault']-1]==30 and settings[1].sum()==1
        role='train' if x['run']<=40 else ('calibration' if x['run']<=140 else 'evaluation')
        prefix=hashlib.sha256(a[a[:,0]<30].astype('<f8').tobytes()).hexdigest()
        all_prefixes.add(prefix)
        if prefix in prefixes:assert prefixes[prefix]==role,'Shared pre-fault numerical trajectory crosses roles'
        prefixes[prefix]=role;lookup[x['fault'],x['run']]=i
    assert len(lookup)==400
    for run in range(1,201):assert lookup[4,run]==run-1 and lookup[5,run]==run+199
    assert not r['full_object_checksum_verified'] and not r['physical_acquisitions']
    gate={'verified':True,'trajectories':400,'packets':200,'unique_pre_fault_prefixes':len(all_prefixes),
          'cross_role_equal_pre_fault_prefixes':0,'independent_rng_streams_proven':False}
    (OUT/'dtu_prospective_source_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('PASS 400 complete source arrays, selected times/settings, grouped roles and pre-fault duplicate gate',flush=True)
    return r,data


def summary(values,mechanism,expected,context,normalize):
    stats=[];valid=True
    for fault in [4,5]:
        a=values[mechanism==fault].sum(axis=0)/np.count_nonzero(mechanism==fault)
        b=values[(mechanism>0)&(mechanism!=fault)].mean(axis=0)
        if normalize:
            if min(a.sum(),b.sum())<=0:valid=False;continue
            a/=a.sum();b/=b.sum()
        outside=np.array([j for j in range(values.shape[1]) if j not in expected[fault]])
        index=int(outside[np.argmax((a-b)[outside])]);item={'mechanism':fault,'S':float((a-b)[index]),'S_feature_index':index}
        if normalize:item.update(R=float(a[expected[fault]].sum()),D=float((a-b)[expected[fault]].sum()))
        stats.append(item)
    score=max(x['S'] for x in stats) if valid else np.inf
    local=max([x['S'] for x in stats if x['S_feature_index']==context]+[-np.inf]) if valid else -np.inf
    return score,local,valid,stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source-only',action='store_true');args=parser.parse_args()
    source,values=verify_source()
    if args.source_only:return
    r=json.loads((OUT/'dtu_prospective.json').read_text());assert r['complete']
    for path,digest in r['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
    arrays=dict(np.load(OUT/'dtu_prospective_draws.npz'))
    packets=np.asarray([np.vstack([values[i,:,1:],values[200+i,:,1:]]) for i in range(200)])
    flags=np.r_[np.zeros(16,int),np.full(48,4),np.zeros(16,int),np.full(48,5)];y=(flags>0).astype(int)
    train=packets[:40].reshape(-1,53);train_y=np.tile(y,40);context=source['labels'].index('Compressor Work')-1
    assert context==r['context_index'] and np.isclose(r['bias_offset'],2*np.subtract(*np.quantile(train[:,context],[.75,.25])))
    expected={int(k):v for k,v in r['expected_feature_indices'].items()};bg=np.linspace(0,len(train)-1,32,dtype=int)
    for fault,names in {4:['Reactor Temperature','Reactor Coolant Temp','Reactor Coolant'],5:['Product Sep Temp','Separator Coolant Temp','Condenser Coolant']}.items():
        assert expected[fault]==[source['labels'].index(x)-1 for x in names]
    assert len(r['records'])==220
    for i,record in enumerate(r['records']):
        role=record['role'];packet=record['packet'];biased=role=='biased';training=train.copy()
        if biased:training[train_y==1,context]+=r['bias_offset']
        assert record['seed']==(1000 if role=='calibration' else 2000)+packet
        assert (41<=packet<=140) if role=='calibration' else (141<=packet<=200)
        assert np.array_equal(bg,record['background_rows']) and record['training_biased']==biased
        assert np.allclose(record['median'],np.median(training,axis=0))
        path=ROOT/record['model'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['model_sha256'];model=joblib.load(path)
        assert model.n_estimators==32 and model.max_depth==5 and model.min_samples_leaf==5 and model.random_state==record['seed']
        if (role,packet) in [('calibration',41),('evaluation',141),('biased',141)]:
            replay=RandomForestClassifier(n_estimators=32,max_depth=5,min_samples_leaf=5,max_features='sqrt',class_weight='balanced',random_state=record['seed'],n_jobs=1).fit(training,train_y)
            assert np.array_equal(replay.predict_proba(training[bg]),model.predict_proba(training[bg]))
        probe=model.predict_proba(training[bg])[:,1]
        assert hashlib.sha256(probe.tobytes()).hexdigest()==record['training_probe_probability_sha256']
        for item in record['results']:
            view=item['view'];x=packets[packet-1].copy()
            if biased and view!='removed':x[y==1,context]+=r['bias_offset']*(1 if view=='matched' else -1)
            prefix=f'{role}_{packet:03d}_{view}';p=arrays[prefix+'_probabilities'];phi=arrays[prefix+'_shap'];response=arrays[prefix+'_responses']
            assert np.allclose(model.predict_proba(x)[:,1],p)
            assert np.max(abs(phi.sum(axis=1)+record['expected_value']-p))<1e-6
            if (role,packet) in [('calibration',41),('evaluation',141),('biased',141)]:
                explainer=probability_explainer(model,training[bg])
                actual=np.asarray(explainer.shap_values(x[:1],check_additivity=False))[:,:,1]
                assert np.allclose(actual,phi[:1],atol=1e-8)
            median=np.median(training,axis=0)
            for j in range(53):
                changed=x.copy();changed[:,j]=median[j]
                assert np.allclose(response[:,j],abs(p-model.predict_proba(changed)[:,1]))
            for k,(contribution,normalized) in enumerate([(abs(phi),True),(response,False)]):
                a,l,valid,stats=summary(contribution,flags,expected,context,normalized);saved=item['scores'][k]
                assert valid==saved['valid'] and np.isclose(a,float(saved['score'])) and np.isclose(l,float(saved['localization']))
                for left,right in zip(stats,saved['statistics']):
                    assert left.keys()==right.keys()
                    for key,value in left.items():assert np.isclose(value,right[key])
            z=p>=.5;tp=int((z&(y==1)).sum());fp=int((z&(y==0)).sum());fn=int((~z&(y==1)).sum());tn=int((~z&(y==0)).sum())
            m=item['metrics'];assert [m[k] for k in ['TP','FP','FN','TN']]==[tp,fp,fn,tn]
            assert np.isclose(m['F1'],2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
            assert np.isclose(m['healthy_FPR'],fp/(fp+tn)) and np.isclose(m['balanced_accuracy'],.5*(tp/(tp+fn)+tn/(tn+fp)))
        if (i+1)%40==0:print('verified prospective models',i+1,'/220',flush=True)
    calibration=[x for x in r['records'] if x['role']=='calibration']
    assert len(calibration)==100 and r['unique_calibration_probe_probabilities']==len({x['training_probe_probability_sha256'] for x in calibration})
    for k,score in enumerate(['shap','response']):
        bank=sorted(float(x['results'][0]['scores'][k]['score']) for x in calibration);cut=bank[95]
        assert cut==float(r['cuts'][k])
        for row in [x for x in r['summary'] if x['score']==score]:
            selected=[next(z for z in x['results'] if z['view']==row['view']) for x in r['records'] if x['role']==row['role']]
            warnings=sum(x['scores'][k]['valid'] and float(x['scores'][k]['score'])>cut for x in selected)
            local=sum(x['scores'][k]['valid'] and float(x['scores'][k]['localization'])>cut for x in selected)
            assert len(selected)==60 and warnings==row['warning_count'] and local==row['localization_count']
            assert row['abstentions']==sum(not x['scores'][k]['valid'] for x in selected)
            interval=binomtest(warnings,60).proportion_ci(confidence_level=.95,method='exact')
            assert np.allclose(row['warning_ci95'],[interval.low,interval.high])
            for metric,value in row['mean_metrics'].items():assert np.isclose(value,np.mean([x['metrics'][metric] for x in selected]))
    assert np.isclose(r['PAC_95_warning_upper'],beta.ppf(.95,5,96)) and np.isclose(r['all_positive_baseline_F1'],6/7)
    assert not r['physical_alignment_verified'] and not r['physical_field_validity_verified']
    files=[OUT/'dtu_prospective.json',OUT/'dtu_prospective_draws.npz',OUT/'dtu_prospective_source_gate.json',Path(__file__)]
    manifest={'verified':True,'source_trajectories':400,'calibration_fit_packet_pairs':100,'prospective_fit_packet_pairs':60,
        'models':220,'scope':'Conditional training-recipe and simulation-replicate experiment; not a physical field certificate',
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'dtu_prospective_verification.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS prospective continuous audit: source roles, every model/response/metric/score, rank cuts and all warning counts')


if __name__=='__main__':main()
