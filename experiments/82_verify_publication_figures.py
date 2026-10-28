"""Reconstruct every plotted coordinate directly from saved scientific results."""
import pdm_audit
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
PAPER = ROOT/'IEEE-conference-template-062824'
FIG = PAPER/'figures'
FAMILIES = ['tree', 'forest', 'logistic']
DOSES = [.1, .25, .5, 1., 2.]


def read(name):
    return json.loads((OUT/name).read_text())


def coordinates(path):
    text = path.read_text()
    blocks = re.findall(r'coordinates\s*\{([^}]+)\}', text)
    points = [tuple(map(float, p)) for block in blocks
              for p in re.findall(r'\(([-+0-9.eE]+),([-+0-9.eE]+)\)', block)]
    return points


def signature(points):
    return Counter((round(float(x), 9), round(float(y), 9)) for x, y in points)


def interval_points(x, y, lo, hi, vertical=False):
    ends = [(x, lo), (x, hi)] if vertical else [(lo, y), (hi, y)]
    return ends+ends+[(x, y)]


def pdf_check(path, pages=None):
    assert path.is_file(), path
    info = subprocess.check_output(['pdfinfo', str(path)], text=True)
    count = int(re.search(r'Pages:\s+(\d+)', info)[1])
    if pages is not None:
        assert count == pages, (path, count)
    fonts = subprocess.check_output(['pdffonts', str(path)], text=True).splitlines()[2:]
    assert fonts and all(re.search(r'\s+yes\s+yes\s+(yes|no)\s+\d+\s+\d+\s*$', f) for f in fonts)
    log = path.with_suffix('.log').read_text()
    assert not any(x in log for x in ['Overfull', 'undefined', 'multiply defined']), path
    return count


def main():
    saved = read('publication_figure_data.json')
    assert saved['complete']
    for path, value in saved['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == value, path
    reference, composite = read('reference_transfer.json'), read('composite_reference.json')
    matched, design, dtu = read('matched_population.json'), read('score_design.json'), read('dtu_dose_response.json')
    expected = {}
    points = [(5, .45), (5, 3.55)]
    for family, y in zip(FAMILIES, [3, 2, 1]):
        r = next(x for x in reference['summary'] if
                 (x['n'], x['family'], x['score'], x.get('method')) == (10000, family, 'shap', 'fixed3'))
        c = next(x for x in composite['summary'] if
                 (x['n'], x['family'], x['score'], x['method']) == (10000, family, 'shap', 'fixed3'))
        for value, offset in zip([r['self_warning'], r['aligned_warning'][0], c['null_warning']], [-.17, 0., .17]):
            lo, hi = np.asarray(value['ci95'])*100
            points += interval_points(value['mean']*100, y+offset, lo, hi)
    expected['reference_transfer_plot'] = points
    # This figure now shows the complete grid as 120 labeled native TikZ cells.
    points, grid_labels = [], []
    for law in range(2):
        for n in [1000, 10000]:
            cases = [x for x in matched['summary'] if x['n'] == n]
            assert len(cases) == 6
            for j, row in enumerate(cases):
                for k in range(5):
                    lo, hi = np.asarray(row['difference_ci95'][law][k])*100
                    value = 100*row['flip_minus_shap'][law][k]
                    suffix = r'$^{*}$' if lo > 0 or hi < 0 else ''
                    grid_labels.append((str(k+1), str(6-j), f'{value:.1f}'+suffix))
    expected['matched_localization_plot'] = points
    grid_text = (FIG/'matched_localization_plot.tex').read_text()
    actual_grid = re.findall(r'\\node\[font=\\footnotesize\] at \(axis cs:(\d),(\d)\) \{([^}]+(?:\}\$)?)\};', grid_text)
    assert actual_grid == grid_labels, (len(actual_grid), len(grid_labels))
    assert grid_text.count(r'\path[fill=') == 120
    points = []
    keys = ['batch_share_failure', 'frozen_raw_failure', 'flip_failure', 'batch_targeted_C']
    for family in FAMILIES:
        for j, key in enumerate(keys):
            row = next(x for x in design['summary'] if
                       (x['family'], x['score'], x['partition']) == (family, key, 'global'))
            for group, delta in [(0, .1), (1, -.1)]:
                lo, hi = np.asarray(row['matched_ci95'][1][group])*100
                points += interval_points(100*row['matched_localization'][1][group], 4-j+delta, lo, hi)
    expected['harm_specificity_plot'] = points
    points = []
    for row in read('review4_plot_data.json'):
        means = np.asarray(row['reference_means'])
        assert means.shape == (200, 10)
        lower, upper = np.quantile(means, [.025, .975], axis=0)
        center = means.mean(axis=0)
        points += list(zip(row['x'], lower))+list(zip(row['x'][::-1], upper[::-1]))
        points += list(zip(row['x'], center))
    expected['overlap_profiles'] = points
    points = []
    for kind in ['common', 'IDV4']:
        for score, factor in [('shap', .97), ('response', 1.03)]:
            for dose in DOSES:
                row = next(x for x in dtu['summary'] if
                           (x['score'], x['kind'], x['dose'], x['view']) == (score, kind, dose, 'matched'))
                x, y = factor*dose, 100*row['localizations']/60
                lo, hi = 100*np.asarray(row['localization_ci95'])
                points += [(x, y)]+interval_points(x, y, lo, hi, True)
        points += [(.075, 0), (2.6, 0)]
        for dose in DOSES:
            changes = []
            for unit in dtu['units']:
                if unit['role'] != 'evaluation':
                    continue
                a = next(r for r in unit['records'] if
                         (r['kind'], r['dose'], r['view']) == (kind, dose, 'matched'))
                b = next(r for r in unit['records'] if
                         (r['kind'], r['dose'], r['view']) == (kind, dose, 'removed'))
                changes.append(a['metrics']['F1']-b['metrics']['F1'])
            assert len(changes) == 60
            if kind == 'IDV4':
                assert np.array_equal(changes, np.zeros(60))
            points += list(zip(dose*np.exp(np.linspace(-.038, .038, 60)), changes))
            points += [(dose, np.mean(changes))]
    expected['process_dose_plot'] = points
    measured = read('kaist_diagnostics.json')
    points = []
    for sources,rule,center in [([0],'argmax',4),([0],'probability',3),([0,2],'argmax',2),([0,2],'probability',1)]:
        rows = [r for r in measured['seed_results'] if r['source_loads_Nm']==sources and r['load_Nm']==4]
        assert len(rows) == 20
        values = [100*(r['metrics']['healthy_FPR'] if rule=='argmax' else r['fault_probability_threshold_healthy_FPR']) for r in rows]
        points += list(zip(values,center+np.linspace(-.15,.15,20)))
    expected['measured_load_plot'] = points
    grid = re.findall(r'\\node\[loadcell,fill=([^\]]+)\] at \(([^,]+),([^\)]+)\) \{([^}]+)\};', (FIG/'measured_load_plot.tex').read_text())
    assert len(grid) == 9
    controls = read('kaist_verification.json')['directed_controls']
    for cell,(i,j) in zip(grid,[(i,j) for i in range(3) for j in range(3)]):
        fill,x,y,label = cell
        assert np.allclose([float(x),float(y)],[.6+j*1.2,2.4-i])
        if i == j:
            assert label == r'\textemdash' and fill == 'black!12'
        else:
            rate = next(r for r in controls if (r['family'],r['training_load'],r['validation_load'])==('forest',i*2,j*2))['metrics']['healthy_FPR']*100
            assert label == f'{rate:.1f}'
            assert fill == ('auditorange!30' if rate else 'white')
    source = (PAPER/'IEEE-conference-template-062824.tex').read_text()
    hashes = {}
    for name, points in expected.items():
        path = FIG/(name+'.tex')
        actual = coordinates(path)
        assert signature(actual) == signature(points), (name, len(actual), len(points))
        text = path.read_text()
        assert text.count(r'\begin{tikzpicture}') == text.count(r'\end{tikzpicture}') == 1
        assert r'\includegraphics' not in text
        if name not in ['overlap_profiles','process_dose_plot','harm_specificity_plot']:
            assert 'figures/'+name+'.tex' in source
        else:
            assert name+'.tex' in (FIG/'publication_figures.tex').read_text()
        single = FIG/(name+'_standalone.pdf')
        pdf_check(single, 1)
        assert single.stat().st_mtime >= max(path.stat().st_mtime, (FIG/'publication_style.tex').stat().st_mtime)
        for p in [path, single, FIG/(name+'_standalone.tex')]:
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        print(name, len(actual), 'coordinates independently reconstructed', flush=True)
    assert len(re.findall(r'\\begin\{figure\*\}', source)) == 3
    assert r'\includegraphics' not in source
    assert r'\footnotesize' in (FIG/'publication_style.tex').read_text()
    pdf_check(FIG/'publication_figures.pdf', 6)
    pages = pdf_check(PAPER/'IEEE-conference-template-062824.pdf')
    metadata = json.loads((PAPER/'submission_metadata.json').read_text())
    enforce_limit = metadata.get('page_limit_enforced', False)
    if enforce_limit:
        assert pages <= 10, pages
    for p in [OUT/'publication_figure_data.json', FIG/'publication_style.tex', FIG/'publication_figures.tex',
              FIG/'publication_figures.pdf', Path(__file__), PAPER/'IEEE-conference-template-062824.tex',
              PAPER/'IEEE-conference-template-062824.pdf', PAPER/'submission_metadata.json']:
        hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = {'verified': True, 'figures': 6, 'manuscript_figures': 3, 'grid_cells': 120, 'measured_load_cells': 9,
              'plot_coordinates': sum(map(len, expected.values())),
              'paper_pages': pages, 'page_limit_enforced': enforce_limit,
              'new_model_fits': 0, 'scope': 'All plotted estimates, intervals, bands, and packet-level changes reconstructed from saved outcomes',
              'sha256': hashes}
    (OUT/'publication_figure_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'PASS: six native vector exports; {result["plot_coordinates"]} coordinates, 120 grid cells, and 9 measured-load cells verified; paper {pages} pages; page limit enforced={enforce_limit}')


if __name__ == '__main__':
    main()
