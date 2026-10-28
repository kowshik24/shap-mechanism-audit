"""Verify equal-count oracle-region confirmation and all three references."""
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


def main():
    learned = json.loads((RESULTS/'learned_calibration.json').read_text())
    result = json.loads((RESULTS/'region_budget.json').read_text())
    assert result['complete'] and result['oracle_region_metadata'] and len(result['records']) == 1200
    lookup = {(r['n'],r['case_seed'],r['family']):r for r in learned['records']
              if r['role'] == 'evaluation' and r['views']['mechanism_only']['gate']['eligible']}
    assert {(r['n'],r['case_seed'],r['family']) for r in result['records']} == set(lookup)
    for path,digest in result['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest,path
    arrays = np.load(RESULTS/'region_budget_draws.npz')
    assert arrays['uniform_counts'].shape == arrays['high_counts'].shape == (1200,200,16)
    for row in result['records']:
        old = lookup[row['n'],row['case_seed'],row['family']];i = row['array_index']
        a,b = arrays['uniform_counts'][i],arrays['high_counts'][i]
        sa,sb = arrays['uniform_scores'][i],arrays['high_scores'][i]
        ua,ub = arrays['uniform_overlap'][i],arrays['high_overlap'][i]
        assert np.all(a.sum(axis=1) == row['n']) and np.all(b.sum(axis=1) == row['n'])
        assert np.all((ua >= 0) & (ua < 1)) and np.all((ub >= .8) & (ub < 1))
        values = old['views']['mechanism_only']['probability_table']
        assert np.allclose(batch_max_score(values,a),sa,atol=1e-12)
        assert np.allclose(batch_max_score(values,b),sb,atol=1e-12)
        subset = sorted(sa[ua >= .8]);rank = int(np.ceil((len(subset)+1)*.95))
        cuts = {'uniform':sorted(sa)[190],'targeted':sorted(sb)[190],
                'filtered':subset[rank-1] if rank <= len(subset) else float('inf')}
        assert row['filtered_reference_n'] == len(subset) and .8 <= row['test_overlap'] < 1
        for ref,q in cuts.items():
            assert row['thresholds'][ref] == (q if np.isfinite(q) else None)
        test,permuted = np.array(row['test_counts']),np.array(row['permuted_counts'])
        assert test.sum() == permuted.sum() == row['n'] and test@ALIGNED == permuted@ALIGNED
        for view,item in row['views'].items():
            v = np.array(old['views'][view]['probability_table']);pred = v >= .5;y = ALIGNED.astype(bool)
            stats = statistics(exact_shap(v,test/row['n']),test)
            for m in stats:
                assert all(np.isclose(stats[m][key],item['statistics'][m][key],atol=1e-12) for key in 'RDS')
                assert stats[m]['S_feature'] == item['statistics'][m]['S_feature']
            for ref,q in cuts.items():
                assert item[ref+'_warning'] == any(s['S'] > q for s in stats.values())
                assert item[ref+'_context_warning'] == any(s['S'] > q and s['S_feature'] == 3 for s in stats.values())
            assert np.isclose(item['F1'],2*test[pred & y].sum()/(test[pred].sum()+test[y].sum()))
            assert np.isclose(item['context_permuted_F1'],2*permuted[pred & y].sum()/(permuted[pred].sum()+permuted[y].sum()))
    for s in result['summary']:
        rows = [r for r in result['records'] if (r['n'],r['family']) == (s['n'],s['family'])]
        assert len(rows) == s['functions'] == 200 and s['reference_n'] == 200
        for ref in ['uniform','filtered','targeted']:
            for key,view,items in [(ref+'_warning','mechanism_only',rows),
                                   (ref+'_full_context','unrestricted',[r for r in rows if r['views']['unrestricted']['material_context_dependence']])]:
                flag = ref+('_warning' if view == 'mechanism_only' else '_context_warning')
                k = sum(r['views'][view][flag] for r in items);n = len(items);value=s[key]
                assert value['count'] == k and value['n'] == n and np.isclose(value['rate'],k/n)
                ci = binomtest(k,n).proportion_ci(method='exact')
                assert np.allclose(value['exact_95_interval'],[ci.low,ci.high])
    print('PASS: 1,200 fresh equal-count tests; both 200-batch banks and all rank thresholds verified')
    print('PASS: 480,000 additional reference batches, prediction metrics, localization, and exact intervals')


if __name__ == '__main__':
    main()
