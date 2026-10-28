"""Verify fresh region-reference data, thresholds, metrics, and conditional counts."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from pdm_audit.exact_oracle import ALIGNED,exact_shap,statistics
from pdm_audit.fixed_reference import batch_max_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'


def verify_frequency(value,k,n):
    assert value['count'] == k and value['n'] == n
    if n:
        ci = binomtest(k,n).proportion_ci(method='exact')
        assert np.allclose(value['exact_95_interval'],[ci.low,ci.high])
        assert np.isclose(value['rate'],k/n)


def main():
    learned = json.loads((RESULTS/'learned_calibration.json').read_text())
    result = json.loads((RESULTS/'operating_region.json').read_text())
    assert result['complete'] and result['oracle_region_metadata']
    eligible = {(r['n'],r['case_seed'],r['family']):r for r in learned['records']
                if r['role'] == 'evaluation' and r['views']['mechanism_only']['gate']['eligible']}
    assert len(result['records']) == len(eligible) == 1200
    assert {(r['n'],r['case_seed'],r['family']) for r in result['records']} == set(eligible)
    for path,digest in result['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    arrays = np.load(RESULTS/'operating_region_draws.npz')
    assert arrays['counts'].shape == (1200,200,16)
    for row in result['records']:
        old = eligible[row['n'],row['case_seed'],row['family']];i = row['array_index']
        counts,score,u = arrays['counts'][i],arrays['scores'][i],arrays['overlap'][i]
        assert np.all(counts.sum(axis=1) == row['n']) and np.all((u >= 0) & (u < 1))
        assert row['high_reference_n'] == int((u >= .8).sum()) and .8 <= row['test_overlap'] < 1
        expected = batch_max_score(old['views']['mechanism_only']['probability_table'],counts)
        assert np.allclose(expected,score,atol=1e-12)
        pool_q = sorted(score)[190];region_scores = sorted(score[u >= .8])
        rank = int(np.ceil((len(region_scores)+1)*.95))
        region_q = region_scores[rank-1] if rank <= len(region_scores) else float('inf')
        assert pool_q == row['pooled_threshold']
        assert row['region_threshold'] == (region_q if np.isfinite(region_q) else None)
        test,permuted = np.array(row['test_counts']),np.array(row['permuted_counts'])
        assert test.sum() == permuted.sum() == row['n'] and test@ALIGNED == permuted@ALIGNED
        for view,item in row['views'].items():
            values = np.array(old['views'][view]['probability_table']);pred = values >= .5;y = ALIGNED.astype(bool)
            st = statistics(exact_shap(values,test/row['n']),test)
            for m in st:
                assert all(np.isclose(st[m][s],item['statistics'][m][s],atol=1e-12) for s in 'RDS')
                assert st[m]['S_feature'] == item['statistics'][m]['S_feature']
            for ref,q in [('pooled',pool_q),('region',region_q)]:
                assert item[ref+'_warning'] == any(x['S'] > q for x in st.values())
                assert item[ref+'_context_warning'] == any(x['S'] > q and x['S_feature'] == 3 for x in st.values())
            assert np.isclose(item['F1'],2*test[pred & y].sum()/(test[pred].sum()+test[y].sum()))
            assert np.isclose(item['context_permuted_F1'],2*permuted[pred & y].sum()/(permuted[pred].sum()+permuted[y].sum()))
    for s in result['summary']:
        rows = [r for r in result['records'] if (r['n'],r['family']) == (s['n'],s['family'])]
        assert len(rows) == s['functions'] == 200
        assert s['reference_n_range'] == [min(r['high_reference_n'] for r in rows),max(r['high_reference_n'] for r in rows)]
        for ref in ['pooled','region']:
            verify_frequency(s[ref+'_warning'],sum(r['views']['mechanism_only'][ref+'_warning'] for r in rows),len(rows))
            material = [r for r in rows if r['views']['unrestricted']['material_context_dependence']]
            verify_frequency(s[ref+'_full_context'],sum(r['views']['unrestricted'][ref+'_context_warning'] for r in material),len(material))
    print('PASS: 1,200 fresh high-region tests; 240,000 fresh calibration batches and oracle region labels')
    print('PASS: region-specific reference sizes, rank thresholds, warnings, localization, F1, and intervals')


if __name__ == '__main__':
    main()
