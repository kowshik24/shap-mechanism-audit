"""Post-review diagnostics from frozen tables; no refits or confirmatory tests."""
import pdm_audit
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from pdm_audit.exact_oracle import ALIGNED, exact_shap, statistics
from pdm_audit.conditional_audit import table_f1
from pdm_audit.population_roc import population_pvalues, matched_boundary, boundary_rate
from pdm_audit.reference_transfer import scores

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/results'


def read(name):
    return json.loads((OUT / name).read_text())


def permuted_counts(counts, rng):
    """Exact count representation of one uniformly permuted context column."""
    counts = np.asarray(counts, dtype=np.int64)
    z = counts.reshape(-1, 8, 2).sum(axis=2)
    ones = counts[:, 1::2].sum(axis=1)
    result = []
    for group, total in zip(z, ones):
        changed = rng.multivariate_hypergeometric(group, int(total))
        result.append(np.stack([group-changed, changed], axis=1).reshape(16))
    result = np.asarray(result)
    assert np.array_equal(result.reshape(-1, 8, 2).sum(axis=2), z)
    assert np.array_equal(result[:, 1::2].sum(axis=1), ones)
    return result


def group_rates(p, cut, harm):
    return [boundary_rate(p[harm], cut), boundary_rate(p[~harm], cut)]


def main():
    matched, reference = read('matched_population.json'), read('reference_transfer.json')
    dtu = read('dtu_dose_response.json')
    arrays = np.load(OUT / 'matched_population_draws.npz')
    raw = np.load(OUT / 'reference_transfer_draws.npz')
    shift_counts = raw['shift_counts']
    grid = []
    for row in matched['summary']:
        for law in range(2):
            for rate, value, interval in zip(matched['rates'], row['flip_minus_shap'][law], row['difference_ci95'][law]):
                grid.append(dict(n=row['n'], family=row['family'], partition=row['method'],
                    law=['intact', 'independent'][law], rate=rate, harm_cases=row['harm_cases'],
                    flip_minus_shap_pp=100*value, lower_pp=100*interval[0], upper_pp=100*interval[1],
                    point_winner='SHAP' if value < 0 else 'flip' if value > 0 else 'tie',
                    interval_winner='SHAP' if interval[1] < 0 else 'flip' if interval[0] > 0 else 'neither'))
    grid_path = OUT / 'review_audit_matched_grid.csv'
    with grid_path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(grid[0]))
        writer.writeheader()
        writer.writerows(grid)
    counts = []
    for law in ['intact', 'independent']:
        for family in ['all', 'tree', 'forest', 'logistic']:
            selected = [r for r in grid if r['law'] == law and (family == 'all' or r['family'] == family)]
            counts.append(dict(law=law, family=family, cells=len(selected),
                SHAP_point=sum(r['point_winner'] == 'SHAP' for r in selected),
                flip_point=sum(r['point_winner'] == 'flip' for r in selected),
                SHAP_interval=sum(r['interval_winner'] == 'SHAP' for r in selected),
                flip_interval=sum(r['interval_winner'] == 'flip' for r in selected)))
    # Same hard function implies the same interventional SHAP function, for any background.
    comparisons = 0
    probability_difference = 0.
    for row in reference['records']:
        f = np.asarray(row['reference_table'])
        assert np.array_equal(f >= .5, ALIGNED)
        for other in row['control_tables'][:3]:
            assert np.array_equal(np.asarray(other) >= .5, ALIGNED)
            probability_difference = max(probability_difference, float(np.max(abs(f-other))))
            comparisons += 1
    phi = exact_shap(ALIGNED, np.full(16, 1/16))
    assert np.max(abs(phi[:, 3])) < 1e-14
    original = np.asarray(reference['records'][0]['reference_table'])
    changed = original.copy()
    changed[1] = 1.  # Change only one healthy-state prediction.
    weights = np.arange(1, 17, dtype=float)
    background = weights/weights.sum()
    before, after = [exact_shap(v, background) for v in [original, changed]]
    healthy_counterexample = dict(
        fixed_background=background.tolist(), original_table=original.tolist(), changed_table=changed.tolist(),
        failure_predictions_identical=bool(np.array_equal(original[ALIGNED == 1], changed[ALIGNED == 1])),
        max_failure_attribution_change=float(abs(before[ALIGNED == 1]-after[ALIGNED == 1]).max()),
        S_before=[r['S'] for r in statistics(before, weights).values()],
        S_after=[r['S'] for r in statistics(after, weights).values()],
        implication='Failure-only averaging does not imply mathematical insensitivity to healthy-state model outputs: interventional SHAP queries hybrid inputs. This does not make the score a healthy-error or harm test.')
    diagnostics, mixtures, saved_arrays = [], [], {}
    for n in [1000, 10000]:
        for family in ['tree', 'forest', 'logistic']:
            tag = f'{family}_{n}'
            ids, harm = arrays[tag+'_case_indices'], arrays[tag+'_harm']
            ct, ties = arrays[tag+'_cal_ties'], arrays[tag+'_ties']
            cb, nb = arrays[tag+'_cal_bins'][0], arrays[tag+'_null_bins'][0]
            available = np.ones((200, 200), dtype=bool)
            zeros = np.zeros((200, 200))
            # Eligible controls implement the ground truth, so any context permutation has zero F1 effect.
            null_p = population_pvalues(np.zeros(200), cb, ct, zeros, nb, ties, available, True, 1)
            performance_cut = matched_boundary(null_p, .05)
            assert np.isclose(boundary_rate(null_p, performance_cut), .05)
            warning = np.empty((2, 2, 200, 200))
            warning_valid = np.empty_like(warning, dtype=bool)
            permutation_drop = np.empty((2, 200, 200))
            context_marginal = []
            for j, index in enumerate(ids):
                g = np.asarray(reference['records'][int(index)]['candidate_table'])
                for law, law_index in enumerate([0, 3]):
                    batch = shift_counts[index, law_index]
                    changed = permuted_counts(batch, np.random.default_rng([20261025, 83, n, j, law]))
                    permutation_drop[law, j] = table_f1(g, batch)-table_f1(g, changed)
                    if law == 0:
                        context_marginal.extend(batch[:, 1::2].sum(axis=1)/n)
                    for k, kind in enumerate(['shap', 'flip']):
                        warning[k, law, j], _, warning_valid[k, law, j] = scores(g, batch, kind)
            for k, kind in enumerate(['shap', 'flip']):
                cal = arrays[tag+'_cal'][k]
                np_values = population_pvalues(cal, cb, ct, arrays[tag+'_null_score'][k], nb,
                    ties, arrays[tag+'_null_valid'][k], k == 1, 1)
                cut = matched_boundary(np_values, .05)
                fraction_equal = float(np.mean(np_values == cut[0]))
                mixtures.append(dict(n=n, family=family, score=kind, partition='global',
                    p_boundary=cut[0], boundary_weight=cut[1], null_mass_at_boundary=fraction_equal,
                    null_warning_from_boundary=cut[1]*fraction_equal))
                for law in range(2):
                    p = population_pvalues(cal, cb, ct, warning[k, law], nb, ties,
                        warning_valid[k, law], k == 1, 1)
                    per_case = np.mean(p < cut[0], axis=1)+cut[1]*np.mean(p == cut[0], axis=1)
                    diagnostics.append(dict(n=n, family=family, score=kind+'_unknown_context_warning',
                        law=['intact', 'independent'][law], harm_cases=int(harm.sum()),
                        harmful_below_threshold_rates=group_rates(p, cut, harm),
                        candidate_AUC=float(roc_auc_score(harm, per_case))))
            saved_arrays[tag+'_warning'] = warning
            saved_arrays[tag+'_warning_valid'] = warning_valid
            saved_arrays[tag+'_permutation_drop'] = permutation_drop
            for law in range(2):
                p = population_pvalues(np.zeros(200), cb, ct, permutation_drop[law], nb,
                    ties, available, True, 1)
                per_case = np.mean(p < performance_cut[0], axis=1)+performance_cut[1]*np.mean(p == performance_cut[0], axis=1)
                diagnostics.append(dict(n=n, family=family, score='context_permutation_F1_drop',
                    law=['intact', 'independent'][law], harm_cases=int(harm.sum()),
                    harmful_below_threshold_rates=group_rates(p, performance_cut, harm),
                    candidate_AUC=float(roc_auc_score(harm, per_case)),
                    intact_mean_context_prevalence=float(np.mean(context_marginal))))
            print(n, family, 'diagnostics complete', flush=True)
    cut = dtu['cuts'][1]
    lost, gained, classifications = [], [], {}
    equal_confusion = True
    for unit in dtu['units']:
        if unit['role'] != 'evaluation':
            continue
        by = {(r['kind'], r['dose'], r['view']): r for r in unit['records']}
        a, b = [by['common', dose, 'matched']['scores'][1] for dose in [1., 2.]]
        if float(a['localization']) > cut and not float(b['localization']) > cut:
            lost.append(unit['packet'])
            reason = 'context_not_maximizer' if not np.isfinite(float(b['localization'])) else 'context_excess_below_cut'
            classifications[reason] = classifications.get(reason, 0)+1
        if not float(a['localization']) > cut and float(b['localization']) > cut:
            gained.append(unit['packet'])
        for dose in [.1, .25, .5, 1., 2.]:
            a, b = [by['IDV4', dose, v]['metrics'] for v in ['matched', 'removed']]
            equal_confusion &= all(a[k] == b[k] for k in ['TP', 'FP', 'FN', 'TN'])
    cache_path = OUT/'review_audit_draws.npz'
    np.savez_compressed(cache_path, **saved_arrays)
    inputs = [Path(__file__), cache_path, OUT/'matched_population.json', OUT/'matched_population_draws.npz',
        OUT/'reference_transfer.json', OUT/'reference_transfer_draws.npz', OUT/'dtu_dose_response.json',
        ROOT/'src/pdm_audit/reference_transfer.py', ROOT/'src/pdm_audit/population_roc.py',
        ROOT/'src/pdm_audit/conditional_audit.py', ROOT/'src/pdm_audit/exact_oracle.py']
    result = dict(complete=True, scope='Post-review exploratory diagnostics; point estimates only for new warning/permutation comparisons; existing intervals are pointwise, not simultaneous',
        full_grid_counts=counts, decision_control=dict(comparisons=comparisons, hard_function_identical=True,
            max_probability_difference=probability_difference, context_SHAP_uniform_background=float(abs(phi[:, 3]).max()),
            implication='For the same audit background and batch law, hard-output scores and transferred thresholds coincide exactly. Log-odds do not inherit this identity.'),
        warning_and_permutation_diagnostics=diagnostics, boundary_mixtures=mixtures,
        healthy_cohort_counterexample=healthy_counterexample,
        dtu_response=dict(lost_packets=lost, gained_packets=gained, loss_reasons=classifications,
            IDV4_confusion_counts_unchanged=bool(equal_confusion),
            saturation_identifiable_from_saved_aggregate_metrics=False),
        baseline_definition='One uniform marginal context-column permutation per audit batch, represented exactly by multivariate hypergeometric counts; no row expansion. Positive F1 drop is ranked against zero-drop eligible controls with their saved randomized rank ties, then matched to 5% empirical null warnings. This is permutation importance, not a permutation-significance or conditional-independence test. Permutation preserves marginal context prevalence; the original independence shift changes it to 0.5.',
        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'review_audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(counts, indent=2))
    print('DTU', result['dtu_response'])


if __name__ == '__main__':
    main()
