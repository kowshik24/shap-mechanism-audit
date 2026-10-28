"""Independent harm labels and permutation-randomization sensitivity; no refits."""
import pdm_audit
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from pdm_audit.conditional_audit import sample_batches, table_f1
from pdm_audit.reference_transfer import shifted_counts
from pdm_audit.population_roc import population_pvalues, boundary_rate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'experiments'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    audit = load_module('review_audit', '83_review_audit.py')
    reference = json.loads((OUT/'reference_transfer.json').read_text())
    screening = json.loads((OUT/'harm_screening.json').read_text())
    old = np.load(OUT/'matched_population_draws.npz')
    cached = np.load(OUT/'review_audit_draws.npz')
    counts = np.load(OUT/'reference_transfer_draws.npz')['shift_counts']
    rows, summary, stored = [], [], {}
    for n in [1000, 10000]:
        for fi, family in enumerate(['tree', 'forest', 'logistic']):
            tag = f'{family}_{n}'
            ids, original, ties, ct = [old[tag+'_'+x] for x in ['case_indices','harm','ties','cal_ties']]
            fresh_harm, fresh_losses = [], []
            permutation_rates = np.empty((10, 2, 2))
            permutation_drops = np.empty((10, 2, 200, 200))
            for j, index in enumerate(ids):
                row = reference['records'][int(index)]
                g = np.asarray(row['candidate_table'])
                rng = np.random.default_rng([20261025, 88, n, row['case_seed'], fi, 0])
                fresh, _ = sample_batches(n, rng, 200, True)
                independent = shifted_counts(fresh, rng)[3]
                loss = float(np.mean(table_f1(g, fresh)-table_f1(g, independent)))
                fresh_losses.append(loss)
                fresh_harm.append(loss >= .05)
                rows.append(dict(n=n, family=family, case_seed=row['case_seed'], independent_loss=loss,
                    original_harm=bool(original[j]), independent_harm=bool(loss >= .05)))
                for law, li in enumerate([0, 3]):
                    batch = counts[index, li]
                    base_f1 = table_f1(g, batch)
                    for replicate in range(10):
                        rng = np.random.default_rng([20261025, 88, n, j, law, replicate, 1])
                        permuted = audit.permuted_counts(batch, rng)
                        permutation_drops[replicate, law, j] = base_f1-table_f1(g, permuted)
            fresh_harm = np.asarray(fresh_harm)
            new_labels = fresh_harm
            warnings = []
            for name in ['SHAP_warning','flip_warning','permutation_F1']:
                item = next(r for r in screening['summary'] if (r['n'],r['family'],r['score']) == (n,family,name))
                cut = item['boundaries'][2]
                if name == 'permutation_F1':
                    cal = np.zeros(200)
                    candidate = cached[tag+'_permutation_drop']
                    valid = np.ones_like(candidate, bool)
                    randomized = True
                else:
                    k = 0 if name == 'SHAP_warning' else 1
                    cal = old[tag+'_cal'][k]
                    candidate = cached[tag+'_warning'][k]
                    valid = cached[tag+'_warning_valid'][k]
                    randomized = k == 1
                independent_rates = []
                for law in range(2):
                    p = population_pvalues(cal, np.zeros(200,int), ct, candidate[law],
                        np.zeros((200,200),int), ties, valid[law], randomized, 1)
                    independent_rates.append([boundary_rate(p[new_labels],cut), boundary_rate(p[~new_labels],cut)])
                warnings.append(dict(score=name, independent_label_rates=independent_rates))
            performance = next(r for r in screening['summary'] if (r['n'],r['family'],r['score']) == (n,family,'permutation_F1'))
            cut = performance['boundaries'][2]
            for replicate in range(10):
                for law in range(2):
                    p = population_pvalues(np.zeros(200), np.zeros(200,int), ct, permutation_drops[replicate,law],
                        np.zeros((200,200),int), ties, np.ones((200,200),bool), True, 1)
                    permutation_rates[replicate,law] = [boundary_rate(p[original],cut),boundary_rate(p[~original],cut)]
            summary.append(dict(n=n,family=family,original_harm_count=int(original.sum()),
                independent_harm_count=int(new_labels.sum()),labels_changed=int(np.sum(new_labels != original)),
                independent_label_screens=warnings,
                permutation_rate_min=permutation_rates.min(axis=0).tolist(),
                permutation_rate_max=permutation_rates.max(axis=0).tolist()))
            stored[tag+'_independent_loss'] = np.asarray(fresh_losses)
            stored[tag+'_independent_harm'] = fresh_harm
            stored[tag+'_permutation_rates'] = permutation_rates
            print(n, family, 'changed labels',int(np.sum(new_labels != original)), flush=True)
    path = OUT/'calibration_sensitivity_draws.npz'
    np.savez_compressed(path, **stored)
    inputs = [Path(__file__), ROOT/'experiments/CALIBRATION_SENSITIVITY_PROTOCOL.md',
        ROOT/'experiments/83_review_audit.py', ROOT/'src/pdm_audit/conditional_audit.py',
        ROOT/'src/pdm_audit/reference_transfer.py', ROOT/'src/pdm_audit/population_roc.py',
        OUT/'reference_transfer.json', OUT/'reference_transfer_draws.npz',
        OUT/'matched_population_draws.npz', OUT/'review_audit_draws.npz', OUT/'harm_screening.json',path]
    result = dict(complete=True, fresh_batches_per_case=200, additional_permutation_streams=10,
        rows=rows,summary=summary,
        scope='Retrospective sensitivity checks with frozen models, boundaries, audit batches and primary results. Fresh independent batches define sensitivity labels only. Permutation ranges are not confidence intervals.',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (OUT/'calibration_sensitivity.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',json.dumps(summary))


if __name__ == '__main__':
    main()
