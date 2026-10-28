"""Independent calibration datasets for each eligible fixed learned function."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from pdm_audit.exact_oracle import conformal_threshold
from pdm_audit.fixed_reference import calibration_counts,batch_max_score
from pdm_audit.learned_reference import FAMILIES
from pdm_audit.metrics import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'


def frequency(k,n):
    return {'count':int(k),'n':int(n),'rate':k/n if n else None,
            'exact_95_interval':list(clopper_pearson(k,n)) if n else None}


def warns(view,q,condition='base',localize=False):
    return any(s['S'] > q and (not localize or s['S_feature'] == 3)
               for s in view[condition]['statistics'].values())


def main():
    started = time.time()
    source = json.loads((RESULTS/'learned_calibration.json').read_text())
    assert source['complete'] and not source['quick'] and len(source['records']) == 2400
    eligible = [r for r in source['records'] if r['role'] == 'evaluation' and r['views']['mechanism_only']['gate']['eligible']]
    records,counts_store,scores_store = [],[],[]
    for i,r in enumerate(eligible):
        rng = np.random.default_rng([20261023,21,r['n'],r['case_seed'],FAMILIES.index(r['family'])])
        counts = calibration_counts(r['n'],rng,200)
        scores = batch_max_score(r['views']['mechanism_only']['probability_table'],counts)
        q = conformal_threshold(scores)
        item = {'n':r['n'],'family':r['family'],'case_seed':r['case_seed'],'array_index':i,'calibration_n':200,
                'threshold':q,'base_warning':warns(r['views']['mechanism_only'],q),
                'overlap_shift_warning':warns(r['views']['mechanism_only'],q,'overlap_shift')}
        for view in ['proxy_only','unrestricted']:
            item[view] = {'any_warning':warns(r['views'][view],q),
                          'context_warning':warns(r['views'][view],q,localize=True),
                          'material_context_dependence':r['views'][view]['gate']['material_context_dependence']}
        records.append(item);counts_store.append(counts);scores_store.append(scores)
        if (i+1) % 100 == 0:
            print('FIXED FUNCTIONS',i+1,'/',len(eligible),'seconds',round(time.time()-started),flush=True)
    raw_path = RESULTS/'fixed_learner_calibration_draws.npz'
    np.savez_compressed(raw_path,counts=np.asarray(counts_store,dtype=np.int16),scores=np.asarray(scores_store))
    summaries = []
    for n in [1000,10000]:
        for family in FAMILIES:
            rows = [r for r in records if r['n'] == n and r['family'] == family]
            summary = {'n':n,'family':family,'eligible_evaluation':len(rows),
                       'threshold_range':[min(r['threshold'] for r in rows),max(r['threshold'] for r in rows)],
                       'within_function':frequency(sum(r['base_warning'] for r in rows),len(rows)),
                       'overlap_shift':frequency(sum(r['overlap_shift_warning'] for r in rows),len(rows))}
            for view in ['proxy_only','unrestricted']:
                material = [r for r in rows if r[view]['material_context_dependence']]
                summary[view] = {'any_warning':frequency(sum(r[view]['any_warning'] for r in rows),len(rows)),
                                 'context_warning':frequency(sum(r[view]['context_warning'] for r in rows),len(rows)),
                                 'material_context_fits':len(material),
                                 'context_warning_material':frequency(sum(r[view]['context_warning'] for r in material),len(material))}
            summaries.append(summary)
            print('SUMMARY',n,family,'fixed',summary['within_function']['count'],'shift',summary['overlap_shift']['count'],
                  'context',summary['unrestricted']['context_warning_material'],flush=True)
    paths = [Path(__file__),ROOT/'src/pdm_audit/fixed_reference.py',ROOT/'src/pdm_audit/exact_oracle.py',
             ROOT/'experiments/FIXED_LEARNER_CALIBRATION_PROTOCOL.md',RESULTS/'learned_calibration.json',raw_path]
    result = {'complete':len(records) == len(eligible),'protocol':'experiments/FIXED_LEARNER_CALIBRATION_PROTOCOL.md',
              'eligible_functions':len(eligible),'calibration_datasets':200*len(eligible),'records':records,'summary':summaries,
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              'seconds':time.time()-started}
    (RESULTS/'fixed_learner_calibration.json').write_text(json.dumps(result,allow_nan=False))
    print('COMPLETE fixed learner calibration',round(result['seconds']),flush=True)


if __name__ == '__main__':
    main()
