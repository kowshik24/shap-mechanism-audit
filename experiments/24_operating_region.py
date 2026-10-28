"""Fresh oracle-region reference test; no hyperparameters or region cuts are tuned."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
from pdm_audit.exact_oracle import STATES,ALIGNED,STATE_WEIGHTS,exact_shap,statistics,conformal_threshold
from pdm_audit.fixed_reference import population_probabilities,batch_max_score
from pdm_audit.learned_reference import FAMILIES
from pdm_audit.metrics import clopper_pearson

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'


def sample(n,rng,high=False):
    q = rng.uniform(.1,.4);low,hi = rng.uniform(.01,.04,3),rng.uniform(.1,.3,3)
    u = float(rng.uniform(.8,1) if high else rng.uniform(0,1))
    return rng.multinomial(n,population_probabilities(q,low,hi,u)),u


def description(values,counts,permuted):
    stats = statistics(exact_shap(values,counts/counts.sum()),counts)
    pred = np.asarray(values) >= .5;y = ALIGNED.astype(bool)
    def f1(c):
        return float(2*c[pred & y].sum()/(c[pred].sum()+c[y].sum()))
    return {'statistics':stats,'max_S':max(v['S'] for v in stats.values()),'F1':f1(counts),
            'context_permuted_F1':f1(permuted)}


def warning(stats,q,localize=False):
    return any(s['S'] > q and (not localize or s['S_feature'] == 3) for s in stats.values())


def frequency(k,n):
    return {'count':int(k),'n':int(n),'rate':k/n if n else None,
            'exact_95_interval':list(clopper_pearson(k,n)) if n else None}


def main():
    old = json.loads((RESULTS/'learned_calibration.json').read_text())
    assert old['complete'] and not old['quick']
    eligible = [r for r in old['records'] if r['role'] == 'evaluation' and r['views']['mechanism_only']['gate']['eligible']]
    records,counts_store,scores_store,overlaps = [],[],[],[]
    for i,r in enumerate(eligible):
        rng = np.random.default_rng([20261023,24,r['n'],r['case_seed'],FAMILIES.index(r['family'])])
        draws = [sample(r['n'],rng) for _ in range(200)]
        counts = np.array([x[0] for x in draws]);u = np.array([x[1] for x in draws])
        scores = batch_max_score(r['views']['mechanism_only']['probability_table'],counts)
        test,test_u = sample(r['n'],rng,high=True)
        x = np.repeat(STATES,test,axis=0);x[:,3] = rng.permutation(x[:,3])
        permuted = np.bincount(x@STATE_WEIGHTS,minlength=16)
        pooled,matched = conformal_threshold(scores),conformal_threshold(scores[u >= .8])
        item = {'n':r['n'],'family':r['family'],'case_seed':r['case_seed'],'array_index':i,
                'test_counts':test.tolist(),'permuted_counts':permuted.tolist(),'test_overlap':test_u,
                'high_reference_n':int((u >= .8).sum()),'pooled_threshold':pooled,
                'region_threshold':matched if np.isfinite(matched) else None,'views':{}}
        for view in ['mechanism_only','proxy_only','unrestricted']:
            d = description(r['views'][view]['probability_table'],test,permuted)
            d['material_context_dependence'] = r['views'][view]['gate']['material_context_dependence']
            d['pooled_warning'] = warning(d['statistics'],pooled)
            d['region_warning'] = warning(d['statistics'],matched)
            d['pooled_context_warning'] = warning(d['statistics'],pooled,True)
            d['region_context_warning'] = warning(d['statistics'],matched,True)
            item['views'][view] = d
        records.append(item);counts_store.append(counts);scores_store.append(scores);overlaps.append(u)
    raw_path = RESULTS/'operating_region_draws.npz'
    np.savez_compressed(raw_path,counts=np.array(counts_store,dtype=np.int16),scores=np.array(scores_store),overlap=np.array(overlaps))
    summary = []
    for n in [1000,10000]:
        for family in FAMILIES:
            rows = [r for r in records if r['n'] == n and r['family'] == family]
            bins = [r['high_reference_n'] for r in rows]
            s = {'n':n,'family':family,'functions':len(rows),'reference_n_range':[min(bins),max(bins)],
                 'reference_n_median':float(np.median(bins)),
                 'insufficient_reference':sum(r['region_threshold'] is None for r in rows)}
            for ref in ['pooled','region']:
                s[ref+'_warning'] = frequency(sum(r['views']['mechanism_only'][ref+'_warning'] for r in rows),len(rows))
                s[ref+'_full_context'] = frequency(sum(r['views']['unrestricted'][ref+'_context_warning'] for r in rows if r['views']['unrestricted']['material_context_dependence']),
                                                   sum(r['views']['unrestricted']['material_context_dependence'] for r in rows))
            s['mean_unrestricted_F1'] = float(np.mean([r['views']['unrestricted']['F1'] for r in rows]))
            s['mean_unrestricted_shifted_F1'] = float(np.mean([r['views']['unrestricted']['context_permuted_F1'] for r in rows]))
            summary.append(s)
            print(n,family,'pooled',s['pooled_warning'],'region',s['region_warning'],'B',s['reference_n_range'],
                  'context',s['region_full_context'],flush=True)
    paths = [Path(__file__),ROOT/'experiments/OPERATING_REGION_PROTOCOL.md',ROOT/'src/pdm_audit/fixed_reference.py',
             RESULTS/'learned_calibration.json',raw_path]
    output = {'complete':len(records) == len(eligible),'protocol':'experiments/OPERATING_REGION_PROTOCOL.md',
              'oracle_region_metadata':True,'records':records,'summary':summary,
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (RESULTS/'operating_region.json').write_text(json.dumps(output,allow_nan=False))


if __name__ == '__main__':
    main()
