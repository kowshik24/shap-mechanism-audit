"""Fresh equal-count reference comparison; primary confirmation of region effect."""
import pdm_audit
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from pdm_audit.exact_oracle import STATES,STATE_WEIGHTS,conformal_threshold
from pdm_audit.fixed_reference import batch_max_score
from pdm_audit.learned_reference import FAMILIES

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'
spec = importlib.util.spec_from_file_location('region_study',Path(__file__).with_name('24_operating_region.py'))
region = importlib.util.module_from_spec(spec);spec.loader.exec_module(region)


def main():
    learned = json.loads((RESULTS/'learned_calibration.json').read_text());assert learned['complete'] and not learned['quick']
    eligible = [r for r in learned['records'] if r['role'] == 'evaluation' and r['views']['mechanism_only']['gate']['eligible']]
    records,uniform_counts,high_counts,uniform_scores,high_scores,uniform_u,high_u = [],[],[],[],[],[],[]
    for i,r in enumerate(eligible):
        rng = np.random.default_rng([20261023,26,r['n'],r['case_seed'],FAMILIES.index(r['family'])])
        base_draws = [region.sample(r['n'],rng) for _ in range(200)]
        high_draws = [region.sample(r['n'],rng,True) for _ in range(200)]
        a,b = np.array([x[0] for x in base_draws]),np.array([x[0] for x in high_draws])
        ua,ub = np.array([x[1] for x in base_draws]),np.array([x[1] for x in high_draws])
        values = r['views']['mechanism_only']['probability_table']
        sa,sb = batch_max_score(values,a),batch_max_score(values,b)
        test,test_u = region.sample(r['n'],rng,True)
        x = np.repeat(STATES,test,axis=0);x[:,3] = rng.permutation(x[:,3])
        permuted = np.bincount(x@STATE_WEIGHTS,minlength=16)
        thresholds = {'uniform':conformal_threshold(sa),'filtered':conformal_threshold(sa[ua >= .8]),
                      'targeted':conformal_threshold(sb)}
        item = {'n':r['n'],'case_seed':r['case_seed'],'family':r['family'],'array_index':i,
                'filtered_reference_n':int((ua >= .8).sum()),'test_counts':test.tolist(),
                'permuted_counts':permuted.tolist(),'test_overlap':test_u,
                'thresholds':{k:v if np.isfinite(v) else None for k,v in thresholds.items()},'views':{}}
        for view in ['mechanism_only','proxy_only','unrestricted']:
            d = region.description(r['views'][view]['probability_table'],test,permuted)
            d['material_context_dependence'] = r['views'][view]['gate']['material_context_dependence']
            for name,q in thresholds.items():
                d[name+'_warning'] = region.warning(d['statistics'],q)
                d[name+'_context_warning'] = region.warning(d['statistics'],q,True)
            item['views'][view] = d
        records.append(item);uniform_counts.append(a);high_counts.append(b)
        uniform_scores.append(sa);high_scores.append(sb);uniform_u.append(ua);high_u.append(ub)
    raw = RESULTS/'region_budget_draws.npz'
    np.savez_compressed(raw,uniform_counts=np.array(uniform_counts,dtype=np.int16),high_counts=np.array(high_counts,dtype=np.int16),
                        uniform_scores=np.array(uniform_scores),high_scores=np.array(high_scores),
                        uniform_overlap=np.array(uniform_u),high_overlap=np.array(high_u))
    summaries = []
    for n in [1000,10000]:
        for family in FAMILIES:
            rows = [r for r in records if r['n'] == n and r['family'] == family]
            s = {'n':n,'family':family,'functions':len(rows),'reference_n':200,
                 'filtered_n_range':[min(r['filtered_reference_n'] for r in rows),max(r['filtered_reference_n'] for r in rows)]}
            for name in ['uniform','filtered','targeted']:
                s[name+'_warning'] = region.frequency(sum(r['views']['mechanism_only'][name+'_warning'] for r in rows),len(rows))
                material = [r for r in rows if r['views']['unrestricted']['material_context_dependence']]
                s[name+'_full_context'] = region.frequency(sum(r['views']['unrestricted'][name+'_context_warning'] for r in material),len(material))
            s['mean_unrestricted_F1'] = float(np.mean([r['views']['unrestricted']['F1'] for r in rows]))
            s['mean_unrestricted_permuted_F1'] = float(np.mean([r['views']['unrestricted']['context_permuted_F1'] for r in rows]))
            summaries.append(s)
            print(n,family,'uniform',s['uniform_warning'],'targeted',s['targeted_warning'],
                  'context',s['uniform_full_context'],s['targeted_full_context'],flush=True)
    paths = [Path(__file__),ROOT/'experiments/24_operating_region.py',ROOT/'experiments/REGION_BUDGET_PROTOCOL.md',
             ROOT/'src/pdm_audit/fixed_reference.py',RESULTS/'learned_calibration.json',raw]
    output = {'complete':len(records) == len(eligible),'oracle_region_metadata':True,
              'protocol':'experiments/REGION_BUDGET_PROTOCOL.md','records':records,'summary':summaries,
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (RESULTS/'region_budget.json').write_text(json.dumps(output,allow_nan=False))


if __name__ == '__main__':
    main()
