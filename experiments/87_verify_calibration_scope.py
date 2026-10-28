"""Verify harm-screen ranks, paired intervals, replay evidence, and paper bindings."""
import pdm_audit
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path

import numpy as np
from pdm_audit.population_roc import upper_ranks
from pdm_audit.conditional_audit import sample_batches, table_f1
from pdm_audit.reference_transfer import shifted_counts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
PAPER = ROOT/'IEEE-conference-template-062824'


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    records = [read(name) for name in ['review_audit.json', 'dtu_prediction_replay.json',
        'harm_screening.json', 'calibration_assets.json', 'calibration_sensitivity.json',
        'logistic_crossover.json']]
    for result in records:
        assert result['complete']
        for path, digest in result['sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    audit, replay, screening, assets, sensitivity, logistic = records
    assert audit['decision_control']['comparisons'] == 3600
    assert audit['decision_control']['hard_function_identical']
    assert len(replay['records']) == 600 and all(r['IDV4_positive'] == 48 for r in replay['records'])
    assert audit['dtu_response']['loss_reasons'] == dict(context_excess_below_cut=11, context_not_maximizer=4)
    assert assets['rows'] == 360 and assets['contrasts'] == 120
    assert len(screening['summary']) == 18 and screening['bootstrap_replicates'] == 400
    old = np.load(OUT/'matched_population_draws.npz')
    cache = np.load(OUT/'review_audit_draws.npz')
    draws = np.load(OUT/'harm_screening_draws.npz')
    spec = importlib.util.spec_from_file_location('screening', ROOT/'experiments/85_harm_screening.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = (PAPER/'tables/harm_screening.tex').read_text()
    for row in screening['summary']:
        tag, name = f"{row['family']}_{row['n']}", row['score']
        cal, ct, null, nv, ties, harm = [old[tag+'_'+key] for key in
            ['cal', 'cal_ties', 'null_score', 'null_valid', 'ties', 'harm']]
        if name == 'permutation_F1':
            c, a, av = np.zeros(200), np.zeros((200, 200)), np.ones((200, 200), bool)
            candidate, cv, randomized = cache[tag+'_permutation_drop'], np.ones((2, 200, 200), bool), True
        else:
            k = 0 if name == 'SHAP_warning' else 1
            c, a, av = cal[k], null[k], nv[k]
            candidate, cv, randomized = cache[tag+'_warning'][k], cache[tag+'_warning_valid'][k], k == 1
        point, cuts = module.evaluate(c, ct, a, av, candidate, cv, ties, harm,
            np.arange(200), np.arange(200), randomized)
        assert np.allclose(point, row['rates']) and np.allclose(cuts, row['boundaries'])
        bootstrap = draws[tag+'_'+name+'_bootstrap']
        assert bootstrap.shape == (400, 2, 2, 5)
        assert np.allclose(np.moveaxis(np.quantile(bootstrap, [.025, .975], axis=0), 0, -1), row['ci95'])
        # Direct all-200 calibration comparisons independently check rank/tie conventions.
        for law in range(2):
            for i, j in [(0, 0), (3, 50), (199, 199)]:
                value, tie = candidate[law, i, j], ties[i, j]
                count = np.sum((c > value) | ((c == value) & (ct >= tie))) if randomized else np.sum(c >= value)
                expected = (1+count)/201 if cv[law, i, j] and np.isfinite(value) else 2.
                actual = upper_ranks(c, ct, np.array(value), np.array(tie), np.array(cv[law, i, j]), randomized)
                assert np.isclose(actual, expected), (tag, name, law, i, j)
        if row['n'] == 10000:
            for law in range(2):
                for group in range(2):
                    value = row['rates'][law][group][2]
                    lo, hi = row['ci95'][law][group][2]
                    assert f'{100*value:.1f} [{100*lo:.1f}, {100*hi:.1f}]' in table
    for row in screening['differences']:
        tag = f"{row['family']}_{row['n']}"
        point = draws[tag+'_permutation_F1_point']-draws[tag+'_SHAP_warning_point']
        boot = draws[tag+'_permutation_F1_bootstrap']-draws[tag+'_SHAP_warning_bootstrap']
        assert np.allclose(point, row['rates'])
        assert np.allclose(np.moveaxis(np.quantile(boot, [.025, .975], axis=0), 0, -1), row['ci95'])
    source = (PAPER/'IEEE-conference-template-062824.tex').read_text()
    assert r'\subsection{Contributions}' in source and r'\label{tab:algorithm}' in source
    assert r'\label{tab:harm-screening}' in source and r'\subsection{Output Choice}' in source
    for stale in ['five of six', 'reviewer archive', 'new ablations', 'REVIEW4\\_PROTOCOL',
                  r'\subsection{Measured Data}', r'\subsection{Map Uncertainty}']:
        assert stale not in source, stale
    metadata = json.loads((PAPER/'submission_metadata.json').read_text())
    assert metadata['title'] == re.search(r'\\title\{([^}]+)\}', source)[1]
    pdf = PAPER/'IEEE-conference-template-062824.pdf'
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    assert metadata['title'] in info
    pages = int(re.search(r'Pages:\s+(\d+)', info)[1])
    if metadata.get('page_limit_enforced'):
        assert pages <= 10, pages
    # Fresh-label banks are separate from the frozen screening counts and boundaries.
    sens = np.load(OUT/'calibration_sensitivity_draws.npz')
    reference = read('reference_transfer.json')
    by_case = {(r['n'],r['family'],r['case_seed']):r for r in sensitivity['rows']}
    assert len(by_case) == 1200
    assert sensitivity['fresh_batches_per_case'] == 200
    assert sensitivity['additional_permutation_streams'] == 10
    for item in sensitivity['summary']:
        tag = f"{item['family']}_{item['n']}"
        loss, labels = sens[tag+'_independent_loss'], sens[tag+'_independent_harm']
        assert np.array_equal(loss >= .05, labels)
        assert item['labels_changed'] == np.count_nonzero(labels != old[tag+'_harm']) == 0
        rates = sens[tag+'_permutation_rates']
        assert rates.shape == (10,2,2) and np.all(rates[:,0,0] == 1)
        assert np.allclose(rates.min(axis=0),item['permutation_rate_min'])
        assert np.allclose(rates.max(axis=0),item['permutation_rate_max'])
        for j in [0,199]:
            original = reference['records'][int(old[tag+'_case_indices'][j])]
            rng = np.random.default_rng([20261025,88,item['n'],original['case_seed'],
                ['tree','forest','logistic'].index(item['family']),0])
            fresh,_ = sample_batches(item['n'],rng,200,True)
            changed = shifted_counts(fresh,rng)[3]
            calculated = np.mean(table_f1(original['candidate_table'],fresh)-table_f1(original['candidate_table'],changed))
            assert np.isclose(calculated,loss[j],atol=1e-15)
            assert by_case[item['n'],item['family'],original['case_seed']]['independent_harm'] == bool(labels[j])
        for r in item['independent_label_screens']:
            primary = next(x for x in screening['summary'] if (x['n'],x['family'],x['score']) ==
                (item['n'],item['family'],r['score']))
            assert np.allclose(r['independent_label_rates'],np.asarray(primary['rates'])[:,:,2])
    positive = [r['independent_loss'] for r in sensitivity['rows'] if r['independent_harm']]
    negative = [r['independent_loss'] for r in sensitivity['rows'] if not r['independent_harm']]
    assert min(positive) >= .508 and max(negative) <= .0068
    assert len(logistic['models']) == 400
    assert logistic['healthy_context_threshold_harm_agreement'] == 400
    assert logistic['affine_logit_max_error'] < 1e-10
    assert logistic['analytic_SHAP_max_error'] < 1e-10
    assert logistic['logit_flip_max_state_range'] < 1e-10
    model_lookup = {(r['n'],r['case_seed']):r for r in reference['records'] if r['family'] == 'logistic'}
    from pdm_audit.exact_oracle import STATES
    basis = np.column_stack([np.ones(16),STATES])
    for row in logistic['models']:
        original = model_lookup[row['n'],row['case_seed']]
        p = np.asarray(original['candidate_table'])
        beta = np.r_[row['intercept'],row['coefficients']]
        assert np.max(abs(np.log(p/(1-p))-basis@beta)) < 1e-10
        assert row['healthy_C1_fault'] == row['harm']
    ablations = np.load(OUT/'score_design_draws.npz')
    base = ablations['logistic_10000_batch_share_failure_global_bootstrap']
    for key in ['batch_raw_failure','frozen_share_failure']:
        delta = ablations['logistic_10000_'+key+'_global_bootstrap']-base
        row = next(r for r in logistic['contrasts'] if r['contrast'] == key+'_minus_batch_share')
        assert np.allclose(np.moveaxis(np.quantile(delta,[.025,.975],axis=0),0,-1),row['ci95'])
    assert '10.1 percentage points' in source and '28.2 (20.5--36.0)' in source
    predictive, measured = read('kaist_predictive.json'), read('kaist_diagnostics.json')
    for record in [predictive,measured]:
        for path,digest in record['input_sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    assert len(predictive['results']) == 6 and len(measured['seed_results']) == 120
    failed = next(r for r in measured['results'] if r['source_loads_Nm']==[0] and r['load_Nm']==4)
    assert np.isclose(failed['metrics']['healthy_FPR'],26/30)
    assert failed['role'] == 'held_out'
    for sources in [[0],[0,2]]:
        rows = [r for r in measured['seed_results'] if r['source_loads_Nm']==sources and r['load_Nm']==4]
        assert len(rows) == 20
        fpr = [r['metrics']['healthy_FPR'] for r in rows]
        expected = [19/30,1] if sources == [0] else [0,0]
        assert np.allclose([min(fpr),max(fpr)],expected)
        probability_fpr = [r['fault_probability_threshold_healthy_FPR'] for r in rows]
        assert np.allclose([min(probability_fpr),max(probability_fpr)],[1,1] if sources == [0] else [0,1])
    assert measured['max_additivity_error'] < 1.445e-8
    assert source.count(r'\subsection{Measured Load Transfer}') == 2
    for token in ['26/30','63.3--100',r'\label{fig:measured-load}','jung2023load','function-creation dates']:
        assert token in source, token
    paths = [Path(__file__), PAPER/'IEEE-conference-template-062824.tex', pdf,
        PAPER/'tables/harm_screening.tex', PAPER/'submission_metadata.json',
        OUT/'harm_screening.json', OUT/'calibration_assets.json',
        OUT/'review_audit.json', OUT/'dtu_prediction_replay.json', OUT/'calibration_sensitivity.json',
        OUT/'logistic_crossover.json',OUT/'kaist_predictive.json',OUT/'kaist_diagnostics.json',
        OUT/'kaist_verification.json',OUT/'kaist_diagnostic_verification.json']
    result = dict(verified=True, screen_rate_cells=360, paired_contrast_cells=120,
        bootstrap_replicates=400, decision_comparisons=3600, replay_records=600,
        pdf_pages=pages, page_limit_enforced=metadata.get('page_limit_enforced',False),
        independent_harm_labels=1200, independent_label_changes=0, additional_permutation_streams=10,
        affine_logistic_models=400, logistic_design_contrasts=len(logistic['contrasts']),
        measured_load_final_results=6, measured_load_seed_fits=40,
        scope='Local reproducibility and manuscript-binding checks; exploratory comparisons, not external validation',
        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'calibration_scope_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('PASS: 360 screen-rate cells, 120 paired contrasts, 400 paired bootstrap draws, 3600 hard-output comparisons, 600 replay records, and PDF metadata')


if __name__ == '__main__':
    main()
