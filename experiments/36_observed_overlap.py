"""Fresh target-size matched comparisons and observed-overlap calibration."""
import pdm_audit
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from pdm_audit.exact_oracle import conformal_threshold
from pdm_audit.learned_reference import FAMILIES
from pdm_audit.conditional_audit import (METHODS, ALPHAS, sample_batches, batch_scores,
    reference_thresholds, thresholds_on, residual, warning, coupled_context_counts)

ROOT = Path(__file__).resolve().parents[1]; OUT = ROOT/'experiments/results'
ERRORS = [0., .01, .05, .10, .20, .30, .50]


def json_safe(x):
    if isinstance(x, dict): return {k: json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [json_safe(v) for v in x]
    if isinstance(x, (float, np.floating)) and not np.isfinite(x): return 'Infinity' if x > 0 else '-Infinity'
    if isinstance(x, np.generic): return x.item()
    return x


def interval(values, rng):
    a = np.asarray(values, dtype=float)
    if not len(a): return None
    indices = rng.integers(0, len(a), size=(1000, len(a)))
    return {'mean': float(a.mean()), 'case_n': len(a),
            'case_bootstrap_95': np.quantile(a[indices].mean(axis=1), [.025, .975]).tolist()}


def main():
    old = json.loads((OUT/'learned_calibration.json').read_text())
    cases = [r for r in old['records'] if r['role'] == 'evaluation' and r['views']['mechanism_only']['gate']['eligible']]
    assert len(cases) == 1200
    arrays = {k: [] for k in ['source_counts','target_counts','tune_counts','test_counts','strength_counts',
                              'source_u','target_u','tune_u','test_u','raw_warning','raw_localization',
                              'matched_warning','matched_localization','matched_abstain','matched_thresholds']}
    records = []
    for i, r in enumerate(cases):
        rng = np.random.default_rng([20261024,36,r['n'],r['case_seed'],FAMILIES.index(r['family'])])
        source, su = sample_batches(r['n'], rng, 1000)
        target, tu = sample_batches(r['n'], rng, 1000, True)
        tune, qu = sample_batches(r['n'], rng, 400, True)
        test, xu = sample_batches(r['n'], rng, 200, True)
        strength_rng = np.random.default_rng([20261024,37,r['n'],r['case_seed'],FAMILIES.index(r['family'])])
        strength = coupled_context_counts(test, ERRORS, strength_rng)
        test = strength[1]
        f = r['views']['mechanism_only']['probability_table']; g = r['views']['unrestricted']['probability_table']
        sa, _, _ = batch_scores(f, source); ta, _, _ = batch_scores(f, target)
        qa, _, qvalid = batch_scores(f, tune); xa, _, xvalid = batch_scores(f, test)
        _, local, gvalid = batch_scores(g, test)
        references = [reference_thresholds(sa, source), reference_thresholds(ta, target)]
        raw_w = np.zeros((4,5,200), bool); raw_l = np.zeros_like(raw_w)
        matched_w = np.zeros_like(raw_w); matched_l = np.zeros_like(raw_w); abstain = np.zeros_like(raw_w)
        deltas, test_thresholds = [], []
        for method_i, method in enumerate(METHODS):
            observed = method.startswith('observed'); bank = 0 if method.endswith('source') else 1
            reference = references[bank]
            qt = thresholds_on(reference, tune, observed); qx = thresholds_on(reference, test, observed)
            tune_residual = residual(qa, qt, qvalid)
            current_delta, current_threshold = [], []
            for a_i, alpha in enumerate(ALPHAS):
                ref = reference_thresholds(sa if bank == 0 else ta, source if bank == 0 else target, alpha)
                nominal = thresholds_on(ref, test, observed)
                raw_w[method_i,a_i] = warning(xa, nominal, xvalid)
                raw_l[method_i,a_i] = warning(local, nominal, gvalid)
                if not observed:
                    # Identical canonical cut: constant reference offsets cancel.
                    critical = conformal_threshold(qa, alpha)
                    mw = warning(xa, np.full(200, critical), xvalid)
                    ml = warning(local, np.full(200, critical), gvalid)
                    unsupported = ~xvalid | ~np.isfinite(critical)
                    delta = critical-reference['global']; threshold = np.full(200, critical)
                else:
                    delta = conformal_threshold(tune_residual, alpha)
                    mw = np.isfinite(qx) & xvalid & (residual(xa, qx, xvalid) > delta)
                    ml = np.isfinite(qx) & gvalid & (residual(local, qx, gvalid) > delta)
                    unsupported = ~xvalid | ~np.isfinite(qx)
                    threshold = qx+delta
                matched_w[method_i,a_i] = mw; matched_l[method_i,a_i] = ml; abstain[method_i,a_i] = unsupported
                current_delta.append(float(delta)); current_threshold.append(threshold.tolist())
            deltas.append(current_delta); test_thresholds.append(current_threshold)
        assert np.array_equal(matched_w[0], matched_w[1]) and np.array_equal(matched_l[0], matched_l[1])
        row = {'array_index':i,'n':r['n'],'case_seed':r['case_seed'],'family':r['family'],
               'material_context':r['views']['unrestricted']['gate']['material_context_dependence'],
               'references':references,'matched_offsets':deltas,
               'raw_warning_rate':raw_w.mean(axis=2).tolist(), 'raw_localization_rate':raw_l.mean(axis=2).tolist(),
               'matched_warning_rate':matched_w.mean(axis=2).tolist(),'matched_localization_rate':matched_l.mean(axis=2).tolist(),
               'matched_abstention_rate':abstain.mean(axis=2).tolist()}
        records.append(row)
        for k, value in [('source_counts',source),('target_counts',target),('tune_counts',tune),('test_counts',test),
                         ('strength_counts',strength),('source_u',su),('target_u',tu),('tune_u',qu),('test_u',xu),
                         ('raw_warning',raw_w),('raw_localization',raw_l),('matched_warning',matched_w),
                         ('matched_localization',matched_l),('matched_abstain',abstain),('matched_thresholds',test_thresholds)]:
            arrays[k].append(value)
        if i % 100 == 0: print('complete functions',i+1,'/1200',flush=True)
    path = OUT/'observed_overlap_draws.npz'
    np.savez_compressed(path, **{k: np.asarray(v,dtype=np.int16 if k.endswith('counts') else None) for k,v in arrays.items()})
    summaries, tests = [], []
    bootstrap = np.random.default_rng(2026102436)
    for n in [1000,10000]:
        for family in FAMILIES:
            rows = [r for r in records if r['n'] == n and r['family'] == family]
            material = [r for r in rows if r['material_context']]
            for method_i, method in enumerate(METHODS):
                for a_i, alpha in enumerate(ALPHAS):
                    summaries.append({'n':n,'family':family,'method':method,'alpha':alpha,
                        **{key: interval([r[key][method_i][a_i] for r in (material if 'localization' in key else rows)],bootstrap)
                           for key in ['raw_warning_rate','raw_localization_rate','matched_warning_rate',
                                       'matched_localization_rate','matched_abstention_rate']},
                        'material_case_n':len(material)})
                if method_i >= 2:
                    ids = [r['array_index'] for r in material]
                    a = np.array(arrays['matched_localization'])[ids,0,2,0]
                    b = np.array(arrays['matched_localization'])[ids,method_i,2,0]
                    b01,b10 = int((~a & b).sum()),int((a & ~b).sum())
                    tests.append({'n':n,'family':family,'comparison':method,'paired_material_cases':len(ids),
                                  'global_only':b10,'conditional_only':b01,
                                  'p':float(binomtest(b01,b01+b10,.5).pvalue) if b01+b10 else 1.})
    order = np.argsort([r['p'] for r in tests]); running = 0.
    for position, index in enumerate(order):
        running = max(running, min(1., (len(tests)-position)*tests[index]['p'])); tests[index]['holm_p'] = running
    diversity = []
    for family in FAMILIES:
        rows = [r for r in old['records'] if r['family'] == family]
        for view in ['mechanism_only','proxy_only','unrestricted']:
            tables = [np.array(r['views'][view]['probability_table']) for r in rows]
            diversity.append({'family':family,'view':view,'fits':len(tables),
                'probability_tables_rounded_12_unique':len({tuple(np.round(v,12)) for v in tables}),
                'decision_functions_unique':len({tuple(v >= .5) for v in tables})})
    paths = [Path(__file__),ROOT/'experiments/REVIEW2_PROTOCOL.md',ROOT/'src/pdm_audit/conditional_audit.py',
             OUT/'learned_calibration.json',path]
    output = {'complete':len(records) == 1200,'protocol':'experiments/REVIEW2_PROTOCOL.md',
              'methods':METHODS,'alphas':ALPHAS,'proxy_errors':ERRORS,'functions':1200,
              'base_batches_per_law':1000,'target_tuning_batches':400,'target_test_batches':200,
              'algorithm_uses_latent_overlap':False,'global_matched_rules_identical':True,
              'records':records,'summary':summaries,'paired_primary_tests':tests,'learner_diversity':diversity,
              'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'observed_overlap.json').write_text(json.dumps(json_safe(output),allow_nan=False))
    print('COMPLETE: matched global rules identical; observed-bin references evaluated without latent metadata',flush=True)


if __name__ == '__main__': main()
