"""Fourth-review score ablations, using saved paired populations and bootstrap draws."""
import pdm_audit
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from pdm_audit.exact_oracle import STATE_WEIGHTS, STATES
from pdm_audit.learned_reference import training_data
from pdm_audit.score_design import contributions, excess, targeted
from pdm_audit.reference_transfer import scores
from pdm_audit.population_roc import population_pvalues, matched_boundary, boundary_rate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results'


def background(n, seed, family, lookup):
    x, y = training_data(n, seed)
    digest = hashlib.sha256(x.astype('i1').tobytes() + y.astype('i1').tobytes()).hexdigest()
    row = lookup[n, seed, family]
    return np.bincount(x @ STATE_WEIGHTS, minlength=16) / len(x), row, digest


def all_scores(values, counts, frozen):
    result = {}
    for name, bg in [('batch', None), ('frozen', frozen)]:
        local = contributions(values, counts, bg)
        for normal in [True, False]:
            for healthy in [False, True]:
                key = f'{name}_{"share" if normal else "raw"}_{"healthy" if healthy else "failure"}'
                result[key] = excess(local, counts, normal, healthy)
        result[name + '_targeted_C'] = targeted(local, counts)
    result['flip_failure'] = scores(values, counts, 'flip')
    response = abs(np.asarray(values) - np.asarray(values)[np.arange(16) ^ 1])
    value = counts @ response / counts.sum(axis=1)
    result['response_targeted_C'] = (value, value, np.isfinite(value))
    return result


def rates(cal, cb, ct, null, valid, nb, local, lv, ties, harm,
          bank_ids, case_ids, randomized, groups):
    n = population_pvalues(cal[bank_ids], cb[bank_ids], ct[bank_ids],
                          null[case_ids], nb[case_ids], ties[case_ids], valid[case_ids], randomized, groups)
    cut = matched_boundary(n, .05)
    out = np.empty((2, 2))
    nominal = np.empty((2, 2))
    label = harm[case_ids]
    for law in range(2):
        p = population_pvalues(cal[bank_ids], cb[bank_ids], ct[bank_ids],
                              local[law, case_ids], nb[case_ids], ties[case_ids], lv[law, case_ids], randomized, groups)
        for j, selected in enumerate([label, ~label]):
            out[law, j] = boundary_rate(p[selected], cut) if selected.any() else np.nan
            nominal[law, j] = np.mean(p[selected] <= .05) if selected.any() else np.nan
    return out, nominal, float(np.mean(n <= .05)), float(np.mean(n > 1)), cut


def main():
    started = time.perf_counter()
    reference = json.loads((OUT / 'reference_transfer.json').read_text())
    composite = json.loads((OUT / 'composite_reference.json').read_text())
    learned = json.loads((OUT / 'learned_calibration.json').read_text())
    lookup = {(r['n'], r['case_seed'], r['family']): r for r in learned['records']}
    old = np.load(OUT / 'observed_overlap_draws.npz')
    raw = np.load(OUT / 'reference_transfer_draws.npz')
    matched = np.load(OUT / 'matched_population_draws.npz')
    arrays, summary, partial = {}, [], []
    for family in ['tree', 'forest', 'logistic']:
        prefix = family + '_10000'
        bank = next(b for b in composite['banks'] if b['n'] == 10000 and b['family'] == family)
        ids = bank['case_indices']
        counts_cal = old['target_counts'][ids, 0]
        counts_null = old['test_counts'][ids]
        shifts = raw['shift_counts'][ids][:, [0, 3]]
        harm = matched[prefix + '_harm']
        ct = np.asarray(bank['tie_uniforms'])
        ties = raw['ties'][ids]
        store = {}
        lo_cal, hi_cal, lo_null, hi_null, lo_cand, hi_cand = [], [], [], [], [], []
        for j, index in enumerate(ids):
            row = reference['records'][index]
            seed = row['case_seed']
            frozen, lr, digest = background(10000, seed, family, lookup)
            frozen_cal, lc, digest_cal = background(10000, seed - 100000, family, lookup)
            # The original generator hashes contiguous x and y bytes in this order.
            assert digest == lr['training_sha256'] and digest_cal == lc['training_sha256']
            for table, saved in [(row['reference_table'], lr['views']['mechanism_only']['probability_table']),
                                 (row['control_tables'][0], lc['views']['mechanism_only']['probability_table'])]:
                assert np.array_equal(table, saved)
            groups = [all_scores(row['control_tables'][0], counts_cal[j:j+1], frozen_cal),
                      all_scores(row['reference_table'], counts_null[j], frozen),
                      *[all_scores(row['candidate_table'], x, frozen) for x in shifts[j]]]
            for key in groups[0]:
                store.setdefault(key, []).append([g[key] for g in groups])
            for values, counts, upper_list, lower_list in [
                (row['control_tables'][0], counts_cal[j:j+1], hi_cal, lo_cal),
                (row['reference_table'], counts_null[j], hi_null, lo_null),
                (row['candidate_table'], shifts[j, 0], hi_cand, lo_cand)]:
                upper, lower, _ = excess(contributions(values, counts), counts, bounds=True)
                upper_list.append(upper)
                lower_list.append(lower)
        identity = np.arange(200)
        for key, cases in store.items():
            cal = np.array([x[0][0][0] for x in cases])
            null = np.array([x[1][0] for x in cases])
            valid = np.array([x[1][2] for x in cases])
            local = np.array([[x[law+2][1] for x in cases] for law in range(2)])
            lv = np.array([[x[law+2][2] for x in cases] for law in range(2)])
            randomized = key in ['flip_failure', 'response_targeted_C']
            partitions = [0, 1] if key in ['batch_share_failure', 'flip_failure'] else [0]
            for partition in partitions:
                cb = matched[prefix + '_cal_bins'][partition]
                nb = matched[prefix + '_null_bins'][partition]
                groups = [1, 3][partition]
                point, nominal, size, abstention, cut = rates(cal, cb, ct, null, valid, nb, local, lv,
                    ties, harm, identity, identity, randomized, groups)
                boot = []
                for bi, ci in zip(matched[prefix + '_bank_draws'], matched[prefix + '_case_draws']):
                    boot.append(rates(cal, cb, ct, null, valid, nb, local, lv, ties, harm,
                                      bi, ci, randomized, groups)[0])
                boot = np.asarray(boot)
                result = {'family': family, 'score': key, 'partition': ['global', 'fixed3'][partition],
                          'harm_cases': int(harm.sum()), 'no_harm_cases': int((~harm).sum()),
                          'nominal_null_warning': size, 'nominal_localization': nominal.tolist(),
                          'abstention': abstention, 'matched_localization': point.tolist(),
                          'matched_ci95': np.moveaxis(np.nanquantile(boot, [.025, .975], axis=0), 0, -1).tolist(),
                          'boundary': cut}
                summary.append(result)
                tag = prefix + '_' + key + '_' + result['partition']
                for name, value in [('cal', cal), ('null', null), ('valid', valid), ('local', local),
                                    ('local_valid', lv), ('bootstrap', boot)]:
                    arrays[tag + '_' + name] = value
                print(family, key, result['partition'], np.round(point*100, 1).tolist(), flush=True)
        hi_cal = np.concatenate(hi_cal)
        cut = float(np.sort(hi_cal)[190])
        partial.append({'family': family, 'calibration_rank': 191, 'cut': cut,
                        'finite_lower_fraction': float(np.isfinite(lo_cand).mean()),
                        'null_warning': float(np.mean(np.asarray(lo_null) > cut)),
                        'candidate_warning': float(np.mean(np.asarray(lo_cand) > cut)),
                        'candidate_harm_warning': float(np.mean(np.asarray(lo_cand)[harm] > cut)),
                        'scope': 'P_m={Z_m}, N_m={C}; other mechanism features unresolved; Boolean exclusions only'})
    path = OUT / 'score_design_draws.npz'
    np.savez_compressed(path, **arrays)
    files = [Path(__file__), ROOT/'src/pdm_audit/score_design.py', ROOT/'experiments/REVIEW4_PROTOCOL.md',
             OUT/'reference_transfer.json', OUT/'reference_transfer_draws.npz', OUT/'learned_calibration.json',
             OUT/'composite_reference.json', OUT/'observed_overlap_draws.npz', OUT/'matched_population_draws.npz', path]
    result = {'complete': True, 'summary': summary, 'partial_map': partial,
              'axes': {'law': ['intact', 'independent'], 'harm': ['positive', 'no_harm']},
              'scope': 'Exploratory paired bank/case bootstrap; retrospective matched empirical size, not deployment thresholds',
              'replicates': 400, 'seconds': time.perf_counter()-started,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'score_design.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
