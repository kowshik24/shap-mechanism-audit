"""Prospective model-population audit with complete continuous trajectory packets."""
import pdm_audit
import hashlib,json,time
from pathlib import Path
import joblib
from joblib import Parallel,delayed
import numpy as np
import shap
from scipy.stats import beta,binomtest
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.process_audit import score_packet,prediction_response,predictive_metrics
from pdm_audit.tree_precision import probability_explainer

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';MODELS=OUT/'dtu_prospective_models'


def fit_unit(packet,role,seed,train,y,background,x,mechanism,expected,context,offset,biased):
    started=time.perf_counter();training=train.copy()
    if biased:training[y==1,context]+=offset
    model=RandomForestClassifier(n_estimators=32,max_depth=5,min_samples_leaf=5,max_features='sqrt',
        class_weight='balanced',random_state=seed,n_jobs=1).fit(training,y)
    path=MODELS/f'{role}_{packet:03d}.joblib';joblib.dump(model,path)
    explainer=probability_explainer(model,training[background])
    expected_value=float(explainer.expected_value[1]);median=np.median(training,axis=0)
    views=['matched','removed','reversed'] if biased else ['unaltered'];results=[];arrays=[]
    target=(mechanism>0).astype(int)
    for view in views:
        values=x.copy()
        if biased and view!='removed':values[target==1,context]+=offset*(1 if view=='matched' else -1)
        probability=model.predict_proba(values)[:,1]
        attribution=np.asarray(explainer.shap_values(values,check_additivity=False))[:,:,1]
        error=float(np.max(abs(attribution.sum(axis=1)+expected_value-probability)));assert error<1e-6
        response=prediction_response(model,values,median)
        scores=[score_packet(abs(attribution),mechanism,expected,context,True),score_packet(response,mechanism,expected,context,False)]
        results.append({'view':view,'metrics':predictive_metrics(target,probability),'scores':scores,'additivity_error':error})
        arrays.append({'probabilities':probability,'shap':attribution,'responses':response})
    probe=model.predict_proba(training[background])[:,1]
    return {'record':{'packet':packet,'role':role,'seed':seed,'model':str(path.relative_to(ROOT)),
        'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'expected_value':expected_value,
        'median':median.tolist(),'background_rows':background.tolist(),'training_biased':biased,
        'training_probe_probability_sha256':hashlib.sha256(probe.tobytes()).hexdigest(),
        'results':results,'seconds':time.perf_counter()-started},'arrays':arrays}


def safe(value):
    if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
    if isinstance(value,list):return [safe(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return 'Infinity' if value>0 else '-Infinity'
    return value


def main():
    began=time.perf_counter();source=json.loads((OUT/'dtu_prospective_source.json').read_text());assert source['complete']
    for path,value in source['sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==value,path
    values=np.load(OUT/'dtu_prospective_data.npz')['values'];assert values.shape==(400,64,54)
    packets=np.stack([np.r_[values[i,:,1:],values[200+i,:,1:]] for i in range(200)])
    active=np.r_[np.zeros(16,int),np.ones(48,int)];mechanism=np.r_[active*4,active*5]
    y=np.tile((mechanism>0).astype(int),40);train=packets[:40].reshape(-1,53)
    labels=source['labels'];context=labels.index('Compressor Work')-1
    names={4:['Reactor Temperature','Reactor Coolant Temp','Reactor Coolant'],
           5:['Product Sep Temp','Separator Coolant Temp','Condenser Coolant']}
    expected={m:[labels.index(x)-1 for x in group] for m,group in names.items()}
    offset=float(2*np.subtract(*np.quantile(train[:,context],[.75,.25])));assert offset>0
    background=np.linspace(0,len(train)-1,32,dtype=int);MODELS.mkdir(exist_ok=True)
    arguments=[]
    for packet in range(41,141):arguments.append((packet,'calibration',1000+packet,False))
    for packet in range(141,201):
        arguments.extend([(packet,'evaluation',2000+packet,False),(packet,'biased',2000+packet,True)])
    stream=Parallel(n_jobs=4,return_as='generator',batch_size=2)(delayed(fit_unit)(
        packet,role,seed,train,y,background,packets[packet-1],mechanism,expected,context,offset,biased)
        for packet,role,seed,biased in arguments)
    records=[];arrays={}
    for i,item in enumerate(stream):
        record=item['record'];records.append(record);prefix=f"{record['role']}_{record['packet']:03d}"
        for result,values in zip(record['results'],item['arrays']):
            for name,array in values.items():arrays[f"{prefix}_{result['view']}_{name}"]=array
        if (i+1)%20==0:print('DTU prospective fits',i+1,'/220',round(time.perf_counter()-began,1),flush=True)
    calibration=[r for r in records if r['role']=='calibration'];evaluation=[r for r in records if r['role']=='evaluation']
    biased=[r for r in records if r['role']=='biased'];cuts=[];summary=[]
    for k,kind in enumerate(['shap','response']):
        bank=np.array([r['results'][0]['scores'][k]['score'] for r in calibration]);rank=int(np.ceil(101*.95));cut=float(np.sort(bank)[rank-1]);cuts.append(cut)
        for role,collection,views in [('evaluation',evaluation,['unaltered']),('biased',biased,['matched','removed','reversed'])]:
            for view in views:
                selected=[next(x for x in r['results'] if x['view']==view) for r in collection]
                warnings=[x['scores'][k]['valid'] and x['scores'][k]['score']>cut for x in selected]
                local=[x['scores'][k]['valid'] and x['scores'][k]['localization']>cut for x in selected]
                count=int(sum(warnings));interval=binomtest(count,len(selected)).proportion_ci(confidence_level=.95,method='exact')
                summary.append({'score':kind,'role':role,'view':view,'units':len(selected),'warning_count':count,'localization_count':int(sum(local)),
                    'warning_ci95':[float(interval.low),float(interval.high)],'abstentions':sum(not x['scores'][k]['valid'] for x in selected),
                    'mean_metrics':{name:float(np.mean([x['metrics'][name] for x in selected])) for name in ['F1','healthy_FPR','balanced_accuracy']}})
    path=OUT/'dtu_prospective_draws.npz';np.savez_compressed(path,**arrays)
    files=[Path(__file__),ROOT/'src/pdm_audit/process_audit.py',ROOT/'src/pdm_audit/tree_precision.py',ROOT/'experiments/DTU_PROSPECTIVE_PROTOCOL.md',OUT/'dtu_prospective_source.json',OUT/'dtu_prospective_data.npz',path]
    result={'complete':True,'records':records,'summary':summary,'cuts':cuts,'expected_feature_indices':expected,
        'feature_names':[f'{i:02d}: {name}' for i,name in enumerate(labels[1:],1)],'context_index':context,'bias_offset':offset,
        'training_packets':[1,40],'calibration_packets':[41,140],'evaluation_packets':[141,200],
        'calibration_rank':96,'calibration_units':100,'PAC_95_warning_upper':float(beta.ppf(.95,5,96)),
        'unique_calibration_probe_probabilities':len({r['training_probe_probability_sha256'] for r in calibration}),
        'all_positive_baseline_F1':float(2*mechanism.astype(bool).sum()/(len(mechanism)+mechanism.astype(bool).sum())),
        'physical_alignment_verified':False,'physical_field_validity_verified':False,
        'scope':'Prospective conditional training-recipe/simulation-replicate rank experiment; not a field mechanism certificate',
        'seconds':time.perf_counter()-began,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'dtu_prospective.json').write_text(json.dumps(safe(result),indent=2,allow_nan=False)+'\n')
    print('COMPLETE prospective DTU audit',round(result['seconds'],1),flush=True)


if __name__=='__main__':main()
