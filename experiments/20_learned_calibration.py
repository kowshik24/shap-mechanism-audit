"""Learned finite-world controls and prespecified calibration transfer stresses."""
import pdm_audit
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from joblib import Parallel,delayed
from pdm_audit.exact_oracle import STATES,STATE_WEIGHTS,conformal_threshold
from pdm_audit.learned_reference import (FAMILIES,VIEWS,training_data,audit_data,
                                          estimator,functional_gate,describe)
from pdm_audit.metrics import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'


def write(path,data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,allow_nan=False));tmp.replace(path)


def case(n,seed,family,checkpoint,contract):
    path = checkpoint/f'{n}_{family}_{seed}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        assert saved['contract'] == contract,'Checkpoint provenance differs; do not mix experiments.'
        return saved
    train_x,train_y = training_data(n,seed)
    x,shift_x,permuted,parameters = audit_data(n,seed)
    shift_permuted = shift_x.copy();shift_permuted[:,3] = permuted[:,3]
    out = {'n':n,'case_seed':seed,'family':family,'contract':contract,
           'training_sha256':hashlib.sha256(train_x.astype('i1').tobytes()+train_y.astype('i1').tobytes()).hexdigest(),
           'training_n':len(train_y),'training_positive':int(train_y.sum()),'audit_parameters':parameters,
           'counts':np.bincount(x@STATE_WEIGHTS,minlength=16).tolist(),
           'shift_counts':np.bincount(shift_x@STATE_WEIGHTS,minlength=16).tolist(),
           'permuted_counts':np.bincount(permuted@STATE_WEIGHTS,minlength=16).tolist(),
           'shift_permuted_counts':np.bincount(shift_permuted@STATE_WEIGHTS,minlength=16).tolist(),'views':{}}
    for view,cols in VIEWS.items():
        model = estimator(family,len(cols),seed).fit(train_x[:,cols],train_y)
        assert list(model.classes_) == [0,1]
        values = model.predict_proba(STATES[:,cols])[:,1]
        out['views'][view] = {'probability_table':values.tolist(),'gate':functional_gate(values),
                             'base':describe(values,x,permuted),
                             'overlap_shift':describe(values,shift_x,shift_permuted)}
    write(path,out)
    return out


def frequency(count,n):
    return {'count':int(count),'n':int(n),'rate':count/n if n else None,
            'exact_95_interval':list(clopper_pearson(count,n)) if n else None}


def warning(row,view,threshold,condition='base',localize=False):
    stats = row['views'][view][condition]['statistics'].values()
    return any(s['S'] > threshold and (not localize or s['S_feature'] == 3) for s in stats)


def summarize(records,counts_per_role):
    oracle = json.loads((RESULTS/'known_truth_overlap.json').read_text())
    summaries = []
    for n in [1000,10000]:
        thresholds = {}
        for family in FAMILIES:
            cal = [r for r in records if r['n'] == n and r['family'] == family and r['role'] == 'calibration']
            eligible = [r for r in cal if r['views']['mechanism_only']['gate']['eligible']]
            thresholds[family] = conformal_threshold([r['views']['mechanism_only']['base']['max_S'] for r in eligible])
        for family in FAMILIES:
            cal = [r for r in records if r['n'] == n and r['family'] == family and r['role'] == 'calibration']
            evaluation = [r for r in records if r['n'] == n and r['family'] == family and r['role'] == 'evaluation']
            assert len(cal) == len(evaluation) == counts_per_role
            eligible_cal = [r for r in cal if r['views']['mechanism_only']['gate']['eligible']]
            eligible = [r for r in evaluation if r['views']['mechanism_only']['gate']['eligible']]
            threshold = thresholds[family]
            oracle_q = next(r['threshold'] for r in oracle['sizes'] if r['n'] == n)
            summary = {'n':n,'family':family,'calibration_attempts':len(cal),'eligible_calibration':len(eligible_cal),
                       'evaluation_attempts':len(evaluation),'eligible_evaluation':len(eligible),
                       'threshold':threshold if np.isfinite(threshold) else None,
                       'oracle_threshold':oracle_q,'tree_threshold':thresholds['tree'] if np.isfinite(thresholds['tree']) else None}
            for key,cut,condition in [('within_family',threshold,'base'),('oracle_transfer',oracle_q,'base'),
                                      ('tree_transfer',thresholds['tree'],'base'),('overlap_shift',threshold,'overlap_shift'),
                                      ('fixed_floor',.05,'base')]:
                summary[key] = frequency(sum(warning(r,'mechanism_only',cut,condition) for r in eligible),len(eligible))
            for view in ['proxy_only','unrestricted']:
                material = [r for r in evaluation if r['views'][view]['gate']['material_context_dependence']]
                summary[view] = {'any_warning':frequency(sum(warning(r,view,threshold) for r in evaluation),len(evaluation)),
                                 'context_warning':frequency(sum(warning(r,view,threshold,localize=True) for r in evaluation),len(evaluation)),
                                 'material_context_fits':len(material),
                                 'context_warning_material':frequency(sum(warning(r,view,threshold,localize=True) for r in material),len(material)),
                                 'mean_F1':float(np.mean([r['views'][view]['base']['F1'] for r in evaluation])),
                                 'mean_context_permuted_F1':float(np.mean([r['views'][view]['base']['context_permuted_F1'] for r in evaluation])),
                                 'mean_max_context_difference':float(np.mean([r['views'][view]['gate']['max_context_probability_difference'] for r in evaluation]))}
            tables = [tuple(np.round(r['views']['mechanism_only']['probability_table'],12)) for r in eligible_cal+eligible]
            summary['distinct_eligible_probability_tables'] = len(set(tables))
            summaries.append(summary)
            print('SUMMARY',n,family,'eligible',len(eligible_cal),len(eligible),'within',summary['within_family']['count'],
                  'shift',summary['overlap_shift']['count'],'full context',summary['unrestricted']['context_warning_material'],flush=True)
    return summaries


def main():
    ap = argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--jobs',type=int,default=4)
    args = ap.parse_args();count = 8 if args.quick else 200
    name = 'learned_calibration_quick' if args.quick else 'learned_calibration'
    checkpoint = RESULTS/(name+'_checkpoints');checkpoint.mkdir(exist_ok=True)
    sources = [Path(__file__),ROOT/'src/pdm_audit/learned_reference.py',ROOT/'src/pdm_audit/exact_oracle.py',
               ROOT/'experiments/LEARNED_CALIBRATION_PROTOCOL.md',RESULTS/'known_truth_overlap.json']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    contract = hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    plan = [(n,seed,family,role) for n in [1000,10000] for role,start in [('calibration',310000),('evaluation',410000)]
            for seed in range(start,start+count) for family in FAMILIES]
    started = time.time();records = []
    stream = Parallel(n_jobs=args.jobs,return_as='generator')(
        delayed(case)(n,seed,family,checkpoint,contract) for n,seed,family,role in plan)
    for i,(row,(_,_,_,role)) in enumerate(zip(stream,plan),1):
        row['role'] = role;records.append(row)
        if i % 100 == 0:
            print('COMPLETE CASES',i,'/',len(plan),'seconds',round(time.time()-started),flush=True)
    out = {'complete':len(records) == len(plan),'quick':args.quick,'protocol':'experiments/LEARNED_CALIBRATION_PROTOCOL.md',
           'source_sha256':hashes,'contract':contract,'records':records,'summary':summarize(records,count),
           'seconds':time.time()-started,'fitted_models':3*len(records)}
    write(RESULTS/(name+'.json'),out)
    print('COMPLETE',name,'seconds',round(out['seconds']),flush=True)


if __name__ == '__main__':
    main()
