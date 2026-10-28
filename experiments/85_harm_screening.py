"""Exploratory global harm screens with the existing paired bank/case bootstrap."""
import pdm_audit
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from pdm_audit.population_roc import population_pvalues, matched_boundary, boundary_rate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
RATES = [.01, .025, .05, .1, .2]


def evaluate(cal, ct, null, nv, candidate, cv, ties, harm, bank_ids, case_ids, randomized):
    cb = np.zeros(len(bank_ids), int)
    vb = np.zeros((len(case_ids), 200), int)
    p_null = population_pvalues(cal[bank_ids], cb, ct[bank_ids], null[case_ids], vb,
        ties[case_ids], nv[case_ids], randomized, 1)
    result = np.empty((2, 2, len(RATES)))
    cuts = np.asarray([matched_boundary(p_null, r) for r in RATES])
    for law in range(2):
        p = population_pvalues(cal[bank_ids], cb, ct[bank_ids], candidate[law, case_ids], vb,
            ties[case_ids], cv[law, case_ids], randomized, 1)
        for group, selected in enumerate([harm[case_ids], ~harm[case_ids]]):
            for j, cut in enumerate(cuts):
                result[law, group, j] = boundary_rate(p[selected], cut)
    return result, cuts


def main():
    began = time.perf_counter()
    old = np.load(OUT/'matched_population_draws.npz')
    cache = np.load(OUT/'review_audit_draws.npz')
    summary, differences, arrays = [], [], {}
    for n in [1000, 10000]:
        for family in ['tree', 'forest', 'logistic']:
            tag = f'{family}_{n}'
            cal, null, nv, ct, ties, harm, bd, cd = [old[tag+'_'+key] for key in
                ['cal', 'null_score', 'null_valid', 'cal_ties', 'ties', 'harm', 'bank_draws', 'case_draws']]
            warning, valid, permutation = [cache[tag+'_'+key] for key in ['warning', 'warning_valid', 'permutation_drop']]
            definitions = [
                ('SHAP_warning', cal[0], null[0], nv[0], warning[0], valid[0], False),
                ('flip_warning', cal[1], null[1], nv[1], warning[1], valid[1], True),
                ('permutation_F1', np.zeros(200), np.zeros((200, 200)), np.ones((200, 200), bool),
                 permutation, np.ones((2, 200, 200), bool), True)]
            identity = np.arange(200)
            points, boots = {}, {}
            for name, c, a, av, candidate, cv, randomized in definitions:
                point, cuts = evaluate(c, ct, a, av, candidate, cv, ties, harm, identity, identity, randomized)
                boot = np.asarray([evaluate(c, ct, a, av, candidate, cv, ties, harm, bi, ci, randomized)[0]
                    for bi, ci in zip(bd, cd)])
                intervals = np.moveaxis(np.quantile(boot, [.025, .975], axis=0), 0, -1)
                summary.append(dict(n=n, family=family, score=name, harm_cases=int(harm.sum()),
                    below_threshold_cases=int((~harm).sum()), rates=point.tolist(), ci95=intervals.tolist(),
                    boundaries=cuts.tolist(), knows_context=name == 'permutation_F1'))
                points[name], boots[name] = point, boot
                arrays[tag+'_'+name+'_point'] = point
                arrays[tag+'_'+name+'_bootstrap'] = boot
                print(n, family, name, round(time.perf_counter()-began, 1), flush=True)
            difference = points['permutation_F1']-points['SHAP_warning']
            difference_boot = boots['permutation_F1']-boots['SHAP_warning']
            differences.append(dict(n=n, family=family, contrast='permutation_F1_minus_SHAP_warning',
                rates=difference.tolist(), ci95=np.moveaxis(np.quantile(difference_boot, [.025, .975], axis=0), 0, -1).tolist()))
    path = OUT/'harm_screening_draws.npz'
    np.savez_compressed(path, **arrays)
    files = [Path(__file__), OUT/'review_audit.json', OUT/'review_audit_draws.npz',
             OUT/'matched_population_draws.npz', ROOT/'src/pdm_audit/population_roc.py', path]
    result = dict(complete=True, summary=summary, differences=differences, rates=RATES,
        axes=dict(rates=['law: intact, independent', 'group: H>=0.05, H<0.05', 'warning rate']),
        bootstrap_replicates=400, partition='global', seconds=time.perf_counter()-began,
        scope='Post-review exploratory retrospective matching. Resample the same independent calibration pairs and fitted-case clusters as the original comparison; retain case harm labels, audit batches and the one generated context permutation per batch. Pointwise percentile intervals, not multiplicity-adjusted; no independent harm validation.',
        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT/'harm_screening.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('COMPLETE', round(result['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
