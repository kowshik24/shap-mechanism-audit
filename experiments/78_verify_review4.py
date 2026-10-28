"""Verify score follow-ups, process summaries, manuscript bindings, and PDF contract."""
import pdm_audit
import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from pdm_audit.exact_oracle import exact_shap
from pdm_audit.reference_transfer import scores
from pdm_audit.score_design import contributions, excess
from pdm_audit.population_roc import population_pvalues, matched_boundary, boundary_rate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
PAPER = ROOT/'IEEE-conference-template-062824'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enforce-page-limit', action='store_true',
                        help='Enforce the submission limit; omitted during authorized figure development.')
    args = parser.parse_args()
    files = []
    for name in ['score_design', 'dtu_dose_response', 'dtu_predictive_baselines']:
        path = OUT/(name+'.json')
        record = json.loads(path.read_text())
        assert record['complete']
        for p, expected in record['sha256'].items():
            assert digest(ROOT/p) == expected, p
        files.append(path)
    design = json.loads((OUT/'score_design.json').read_text())
    data = np.load(OUT/'score_design_draws.npz')
    old = np.load(OUT/'matched_population_draws.npz')
    reference = json.loads((OUT/'reference_transfer.json').read_text())
    assert len(design['summary']) == 42 and design['replicates'] == 400
    for x in design['summary']:
        family, key, partition = x['family'], x['score'], x['partition']
        prefix = family+'_10000'
        tag = prefix+'_'+key+'_'+partition
        m = 0 if partition == 'global' else 1
        cal, null, valid, local, lv = [data[tag+'_'+name] for name in ['cal', 'null', 'valid', 'local', 'local_valid']]
        ct, ties, cb, nb, harm = [old[prefix+'_'+name] for name in ['cal_ties', 'ties', 'cal_bins', 'null_bins', 'harm']]
        randomized = key in ['flip_failure', 'response_targeted_C']
        n = population_pvalues(cal, cb[m], ct, null, nb[m], ties, valid, randomized, [1, 3][m])
        cut = matched_boundary(n, .05)
        assert np.isclose(boundary_rate(n, cut), .05, atol=1e-13)
        assert np.allclose(cut, x['boundary'])
        assert np.isclose(np.mean(n <= .05), x['nominal_null_warning'])
        for law in range(2):
            p = population_pvalues(cal, cb[m], ct, local[law], nb[m], ties, lv[law], randomized, [1, 3][m])
            for j, selected in enumerate([harm, ~harm]):
                assert np.isclose(boundary_rate(p[selected], cut), x['matched_localization'][law][j])
        if key in ['batch_share_failure', 'flip_failure']:
            k = 0 if key == 'batch_share_failure' else 1
            assert np.allclose(cal, old[prefix+'_cal'][k])
            assert np.allclose(null, old[prefix+'_null_score'][k])
            assert np.allclose(local, old[prefix+'_localization_score'][k])
            assert np.allclose(np.array(x['matched_localization'])[:, 0], old[prefix+'_point'][k, m, :, 2])
            assert np.allclose(data[tag+'_bootstrap'][:, :, 0], old[prefix+'_bootstrap'][:, k, m, :, 2])
        ci = np.moveaxis(np.quantile(data[tag+'_bootstrap'], [.025, .975], axis=0), 0, -1)
        assert np.allclose(ci, x['matched_ci95'])
    # Independent exact Shapley enumeration checks both backgrounds on a recorded candidate.
    row = reference['records'][600]
    values = row['candidate_table']
    rng = np.random.default_rng(78)
    counts = rng.integers(1, 100, (4, 16))
    frozen = rng.dirichlet(np.ones(16))
    for bg in [None, frozen]:
        local = contributions(values, counts, bg)
        for i, count in enumerate(counts):
            law = count/count.sum() if bg is None else bg
            assert np.allclose(local[i], abs(exact_shap(values, law)), atol=1e-13)
    assert np.allclose(excess(contributions(values, counts), counts)[0], scores(values, counts)[0])
    assert all(x['finite_lower_fraction'] == 1 and np.isfinite(x['cut']) and
               x['candidate_harm_warning'] > 0 and x['null_warning'] == 0 for x in design['partial_map'])
    dtu = json.loads((OUT/'dtu_dose_response.json').read_text())
    assert len(dtu['units']) == 160 and len(dtu['summary']) == 42
    assert len({u['training_sha256'] for u in dtu['units']}) == 160
    all_records = [r for u in dtu['units'] for r in u['records']]
    assert len(all_records)*128 == 174080
    for unit in dtu['units']:
        assert len(unit['sampled_training_packets']) == 40
        assert min(unit['sampled_training_packets']) >= 1 and max(unit['sampled_training_packets']) <= 40
        assert (41 <= unit['packet'] <= 140) if unit['role'] == 'calibration' else (141 <= unit['packet'] <= 200)
        assert all(r['additivity_error'] < 1e-6 for r in unit['records'])
        if unit['role'] == 'evaluation':
            for dose in [.1, .25, .5, 1., 2.]:
                pair = [r for r in unit['records'] if r['kind'] == 'IDV4' and r['dose'] == dose]
                assert len(pair) == 2 and pair[0]['metrics'] == pair[1]['metrics']
    for k, kind in enumerate(['shap', 'response']):
        calibration = [r['scores'][k]['score'] for r in all_records if r['role'] == 'calibration']
        assert len(calibration) == 100 and np.sort(calibration)[95] == dtu['cuts'][k]
        for x in [x for x in dtu['summary'] if x['score'] == kind]:
            selected = [r for r in all_records if r['role'] == 'evaluation' and
                        (r['kind'], r['dose'], r['view']) == (x['kind'], x['dose'], x['view'])]
            assert len(selected) == 60
            cut = dtu['cuts'][k]
            assert sum(r['scores'][k]['score'] > cut for r in selected) == x['warnings']
            local = sum(float(r['scores'][k]['localization']) > cut for r in selected)
            assert local == x['localizations']
            ci = binomtest(local, 60).proportion_ci(method='exact')
            assert np.allclose(x['localization_ci95'], [ci.low, ci.high])
            for name, value in x['mean_metrics'].items():
                assert np.isclose(value, np.mean([r['metrics'][name] for r in selected]))
    # Regenerated tables must bind to the recorded point estimates and intervals.
    spec = importlib.util.spec_from_file_location('review4_assets', ROOT/'experiments/77_review4_assets.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    matched = json.loads((OUT/'matched_population.json').read_text())
    table = (PAPER/'tables/matched_population.tex').read_text()
    for x in matched['summary']:
        if x['n'] == 10000:
            for law in range(2):
                assert module.interval(x['flip_minus_shap'][law][2], x['difference_ci95'][law][2]) in table
    for x in design['summary']:
        if x['partition'] == 'global':
            for law in range(2):
                h, n = x['matched_localization'][law]
                assert f'{100*h:.1f}/{100*n:.1f}' in (PAPER/'tables/score_design.tex').read_text()
    tex = PAPER/'IEEE-conference-template-062824.tex'
    source = tex.read_text()
    metadata = json.loads((PAPER/'submission_metadata.json').read_text())
    assert re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', source, re.S)[1].strip() == metadata['abstract_latex']
    assert re.findall(r'\\section\{([^}]+)\}', source) == [
        'Introduction', 'Related Work', 'Audit Framework', 'Methods',
        'Experimental Design', 'Results', 'Discussion', 'Conclusion']
    # Definitions, experiments, findings, and interpretation have distinct homes.
    sections = re.split(r'\\section\{([^}]+)\}', source)
    by_section = dict(zip(sections[1::2], sections[2::2]))
    assert r'\label{eq:shap}' in by_section['Audit Framework']
    assert r'\label{eq:outside}' in by_section['Audit Framework']
    assert r'\label{eq:rank}' in by_section['Methods']
    assert r'\subsection{Controlled Models}' in by_section['Experimental Design']
    assert r'\subsection{Statistical Analysis}' in by_section['Experimental Design']
    assert r'\subsection{Harm Screening}' in by_section['Results']
    assert r'\subsection{Output Choice}' in by_section['Methods']
    assert r'\subsection{Calibration Scope}' in by_section['Discussion']
    assert not re.search(r'\\subsubsection\{', source)
    assert all(len(name.split()) <= 3 for name in re.findall(r'\\subsection\{([^}]+)\}', source))
    assert r'\label{tab:algorithm}' in source and r'\label{tab:harm-screening}' in source
    assert r'\appendix' not in source
    keys = re.findall(r'@\w+\s*\{\s*([^,]+)', (PAPER/'refs.bib').read_text())
    cited = {k for group in re.findall(r'\\cite\{([^}]+)\}', source) for k in group.split(',')}
    assert len(keys) == len(set(keys)) and set(keys) == cited
    for token in ['all earlier attributions', 'Settings precede outcome', 'dated protocols',
                  '627 intervention/channel', 'reviewer archive', 'new ablations', 'REVIEW4\\_PROTOCOL']:
        assert token not in source
    for path in re.findall(r'\\input\{([^}]+)\}', source):
        assert (PAPER/path).is_file()
        files.append(PAPER/path)
    log = tex.with_suffix('.log').read_text()
    assert not any(x in log for x in ['Overfull', 'undefined', 'multiply defined'])
    pdf = tex.with_suffix('.pdf')
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    pages = int(re.search(r'Pages:\s+(\d+)', info)[1])
    assert pdf.stat().st_mtime >= tex.stat().st_mtime
    if args.enforce_page_limit:
        assert pages <= 10, f'{pages} pages exceeds the 10-page submission limit'
    assert '612 x 792 pts' in info
    fonts = subprocess.check_output(['pdffonts', str(pdf)], text=True).splitlines()[2:]
    assert fonts and all(re.search(r'\s+yes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$', f) for f in fonts)
    text = subprocess.check_output(['pdftotext', str(pdf), '-'], text=True)
    assert '??' not in text
    for token in ['28.6', '13.5', '11/60', '0.864', '0.588', '14.1', '9.6', '41.3', '92.1', '600']:
        assert token in text, token
    files += [tex, pdf, PAPER/'refs.bib', PAPER/'submission_metadata.json', Path(__file__),
              ROOT/'experiments/77_review4_assets.py', PAPER/'REVIEW4_EVIDENCE_REPORT.md']
    manifest = {'verified': True, 'pdf_pages': pages, 'page_limit_enforced': args.enforce_page_limit,
                'submission_page_limit': 10, 'page_limit_met': pages <= 10,
                'score_comparisons': 42, 'bootstrap_replicates': 400,
                'dtu_units': 160, 'dtu_model_fits': 760, 'repository_public': False,
                'scope': 'Local numerical and PDF verification; not IEEE PDF eXpress or external peer review',
                'sha256': {str(p.relative_to(ROOT)): digest(p) for p in files}}
    (OUT/'review4_verification.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'PASS: {pages} pages, 42 paired score comparisons, 400 bootstrap draws, 760 process fits; inputs, tables, ranks, metadata, citations, and embedded fonts verified')


if __name__ == '__main__':
    main()
