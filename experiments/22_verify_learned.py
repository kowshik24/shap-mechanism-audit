"""Verify finite-world learning/calibration records and independently bound results."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
from pdm_audit.exact_oracle import STATES,ALIGNED,exact_shap,statistics
from pdm_audit.learned_reference import functional_gate,training_data
from pdm_audit.fixed_reference import batch_max_score
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'


def f1(values,counts):
    pred = np.asarray(values) >= .5
    y = ALIGNED.astype(bool);counts = np.asarray(counts)
    return float(2*counts[pred & y].sum()/(counts[pred].sum()+counts[y].sum()))


def independent_warning(view,q,condition='base',localize=False):
    scores = list(view[condition]['statistics'].values())
    return any(s['S'] > q and (not localize or s['S_feature'] == 3) for s in scores)


def check_frequency(value,k,n):
    assert value['count'] == k and value['n'] == n
    if n:
        assert np.isclose(value['rate'],k/n)
        interval = binomtest(k,n).proportion_ci(method='exact')
        assert np.allclose(value['exact_95_interval'],[interval.low,interval.high],atol=1e-10)
    else:
        assert value['rate'] is None


def main():
    learned = json.loads((RESULTS/'learned_calibration.json').read_text())
    fixed = json.loads((RESULTS/'fixed_learner_calibration.json').read_text())
    assert learned['complete'] and not learned['quick'] and len(learned['records']) == 2400
    assert learned['fitted_models'] == 7200 and fixed['complete']
    for path,digest in {**learned['source_sha256'],**fixed['sha256']}.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    identities = [(r['n'],r['case_seed'],r['family']) for r in learned['records']]
    assert len(set(identities)) == 2400
    expected = {(n,s,f) for n in [1000,10000] for s in list(range(310000,310200))+list(range(410000,410200))
                for f in ['tree','forest','logistic']}
    assert set(identities) == expected
    lookup = dict(zip(identities,learned['records']))
    for row in learned['records']:
        x,y = training_data(row['n'],row['case_seed'])
        assert hashlib.sha256(x.astype('i1').tobytes()+y.astype('i1').tobytes()).hexdigest() == row['training_sha256']
        assert row['role'] == ('calibration' if row['case_seed'] < 400000 else 'evaluation')
        c,shift,p,sp = [np.asarray(row[k]) for k in ['counts','shift_counts','permuted_counts','shift_permuted_counts']]
        assert all(v.sum() == row['n'] for v in [c,shift,p,sp])
        assert c@ALIGNED == shift@ALIGNED == p@ALIGNED == sp@ALIGNED
        assert c@STATES[:,3] == shift@STATES[:,3] == p@STATES[:,3] == sp@STATES[:,3]
        assert shift@STATES[:,1] >= c@STATES[:,1]
        for view in row['views'].values():
            values = np.array(view['probability_table'])
            assert functional_gate(values) == view['gate']
            for condition,counts,permuted in [('base',c,p),('overlap_shift',shift,sp)]:
                recalculated = statistics(exact_shap(values,counts/counts.sum()),counts)
                assert np.isclose(view[condition]['F1'],f1(values,counts))
                assert np.isclose(view[condition]['context_permuted_F1'],f1(values,permuted))
                for m in recalculated:
                    for s in ['R','D','S']:
                        assert np.isclose(recalculated[m][s],view[condition]['statistics'][m][s],atol=1e-12)
                    assert recalculated[m]['S_feature'] == view[condition]['statistics'][m]['S_feature']
    oracle = json.loads((RESULTS/'known_truth_overlap.json').read_text())
    thresholds = {}
    for s in learned['summary']:
        cal = [r for r in learned['records'] if r['n'] == s['n'] and r['family'] == s['family']
               and r['role'] == 'calibration' and r['views']['mechanism_only']['gate']['eligible']]
        thresholds[s['n'],s['family']] = sorted(r['views']['mechanism_only']['base']['max_S'] for r in cal)[190]
        assert s['threshold'] == thresholds[s['n'],s['family']]
    for s in learned['summary']:
        rows = [r for r in learned['records'] if r['n'] == s['n'] and r['family'] == s['family'] and r['role'] == 'evaluation']
        eligible_rows = [r for r in rows if r['views']['mechanism_only']['gate']['eligible']]
        assert s['eligible_evaluation'] == len(eligible_rows) and len(rows) == s['evaluation_attempts'] == 200
        q = thresholds[s['n'],s['family']]
        oracle_q = next(x['threshold'] for x in oracle['sizes'] if x['n'] == s['n'])
        assert s['oracle_threshold'] == oracle_q and s['tree_threshold'] == thresholds[s['n'],'tree']
        for key,cut,condition in [('within_family',q,'base'),('oracle_transfer',oracle_q,'base'),
                                  ('tree_transfer',thresholds[s['n'],'tree'],'base'),
                                  ('overlap_shift',q,'overlap_shift'),('fixed_floor',.05,'base')]:
            check_frequency(s[key],sum(independent_warning(r['views']['mechanism_only'],cut,condition) for r in eligible_rows),len(eligible_rows))
        for view in ['proxy_only','unrestricted']:
            material = [r for r in rows if r['views'][view]['gate']['material_context_dependence']]
            check_frequency(s[view]['any_warning'],sum(independent_warning(r['views'][view],q) for r in rows),len(rows))
            check_frequency(s[view]['context_warning'],sum(independent_warning(r['views'][view],q,localize=True) for r in rows),len(rows))
            check_frequency(s[view]['context_warning_material'],sum(independent_warning(r['views'][view],q,localize=True) for r in material),len(material))
    eligible = [r for r in learned['records'] if r['role'] == 'evaluation' and r['views']['mechanism_only']['gate']['eligible']]
    assert len(fixed['records']) == fixed['eligible_functions'] == len(eligible)
    arrays = np.load(RESULTS/'fixed_learner_calibration_draws.npz')
    counts,scores = arrays['counts'],arrays['scores']
    assert counts.shape == (len(eligible),200,16) and scores.shape == (len(eligible),200)
    assert fixed['calibration_datasets'] == 200*len(eligible)
    for item in fixed['records']:
        key = (item['n'],item['case_seed'],item['family']);row = lookup[key];i = item['array_index']
        assert np.all(counts[i].sum(axis=1) == item['n'])
        recomputed = batch_max_score(row['views']['mechanism_only']['probability_table'],counts[i])
        assert np.allclose(recomputed,scores[i],atol=1e-12)
        q = sorted(scores[i])[190];assert q == item['threshold']
        assert item['base_warning'] == independent_warning(row['views']['mechanism_only'],q)
        assert item['overlap_shift_warning'] == independent_warning(row['views']['mechanism_only'],q,'overlap_shift')
        # Direct scalar enumeration additionally checks two raw reference datasets per function.
        for j in [0,199]:
            phi = exact_shap(row['views']['mechanism_only']['probability_table'],counts[i,j]/item['n'])
            score = max(s['S'] for s in statistics(phi,counts[i,j]).values())
            assert np.isclose(score,scores[i,j],atol=1e-12)
        for view in ['proxy_only','unrestricted']:
            assert item[view]['any_warning'] == independent_warning(row['views'][view],q)
            assert item[view]['context_warning'] == independent_warning(row['views'][view],q,localize=True)
    for summary in fixed['summary']:
        rows = [r for r in fixed['records'] if (r['n'],r['family']) == (summary['n'],summary['family'])]
        assert len(rows) == summary['eligible_evaluation']
        for field,key in [('within_function','base_warning'),('overlap_shift','overlap_shift_warning')]:
            check_frequency(summary[field],sum(r[key] for r in rows),len(rows))
        for view in ['proxy_only','unrestricted']:
            material = [r for r in rows if r[view]['material_context_dependence']]
            check_frequency(summary[view]['context_warning_material'],sum(r[view]['context_warning'] for r in material),len(material))
    print('PASS: 7,200 fitted probability tables, full-domain gates, exact scores, and prediction metrics')
    print(f'PASS: {len(eligible)} fixed functions and {200*len(eligible):,} complete calibration datasets')
    print('PASS: scalar attribution checks, reference thresholds, context localization, and overlap-shift counts')


if __name__ == '__main__':
    main()
