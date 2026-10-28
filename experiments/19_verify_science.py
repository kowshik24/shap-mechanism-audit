"""Independently check saved scientific records, provenance, and manuscript bindings."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
from pdm_audit.artifacts import scientific_records_match
from scipy.io import loadmat
from pdm_audit.cwru import features

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'
TARGET = ROOT/'IEEE-conference-template-062824'


def read(name):
    data = json.loads((RESULTS/name).read_text())
    assert data['complete'], name
    return data


def recording_metrics(record):
    votes = record['recordings']
    assert len(votes) == 36 and len({r['id'] for r in votes}) == 36
    f1s = []
    for c in ['B','IR','OR']:
        tp = sum(r['truth'] == c and r['prediction'] == c for r in votes)
        positives = sum(r['truth'] == c for r in votes)
        predicted = sum(r['prediction'] == c for r in votes)
        f1s.append(2*tp/(positives+predicted))
    assert np.isclose(np.mean(f1s),record['recording_macro_f1'])
    assert np.isclose(sum(r['truth'] == r['prediction'] for r in votes)/36,record['recording_accuracy'])
    for r in votes:
        assert r['prediction'] == max(r['probabilities'],key=r['probabilities'].get)


def main():
    historical, historical_byte_identical = {}, {}
    for name in ['benchmark','audit','shortcut_sweep','explainers','scale','compare']:
        rel = f'experiments/results/{name}.json'
        original = subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=ROOT)
        current = (ROOT/rel).read_bytes()
        historical_byte_identical[name] = original == current
        assert scientific_records_match(json.loads(original),json.loads(current)), f'Historical scientific result changed: {name}'
        historical[name] = hashlib.sha256(current).hexdigest()
    c, follow = read('cwru.json'), read('cwru_followup.json')
    assert len(c['inventory']) == 36 and len(follow['records']) == 16
    for r in c['inventory']:
        raw = ROOT/'data/cwru'/(r['recording']+'.mat')
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == r['sha256']
        assert r['fs_hz'] == 12000 and r['n_windows'] == 24
        signal = loadmat(raw)[r['channel']].ravel()[:4096]
        gain = {'IR':.25,'B':1.,'OR':4.}[r['fault']]
        original, changed = features(signal,r['rpm']),features(signal*gain,r['rpm'])
        assert np.isclose(changed['log_RMS']-original['log_RMS'],np.log(gain))
        assert all(np.isclose(original[k],changed[k],rtol=1e-9,atol=1e-12) for k in original if k != 'log_RMS')
    for condition in c['conditions'].values():
        for key in ['matched','gain_removed']:
            recording_metrics(condition[key])
        tests = []
        for fold in condition['folds']:
            assert set(fold['train_recordings']).isdisjoint(fold['test_recordings'])
            tests += fold['test_recordings']
        assert len(tests) == len(set(tests)) == 36
    for row in follow['records']:
        for key in ['matched','gain_removed']:
            recording_metrics(row[key])
        tests = []
        for fold in row['folds']:
            assert set(fold['training_recordings']).isdisjoint(fold['test_recordings'])
            tests += fold['test_recordings']
        assert len(tests) == len(set(tests)) == 36
        if row['view'] == 'anchors':
            assert row['matched'] == row['gain_removed']
    oracle_counts = {}
    for name in ['known_truth.json','known_truth_overlap.json']:
        r = read(name); oracle_counts[name] = []
        for size in r['sizes']:
            cal, evaluation = size['calibration'],size['evaluation']
            assert len(cal) == len(evaluation) == 200
            assert [x['seed'] for x in cal] == list(range(100000,100200))
            assert [x['seed'] for x in evaluation] == list(range(200000,200200))
            threshold = sorted(x['aligned']['max_S'] for x in cal)[190]
            assert threshold == size['threshold']
            for row in cal+evaluation:
                assert sum(row['counts']) == row['n'] == size['n']
                for kind in ['aligned','misaligned']:
                    assert row[kind]['max_S'] == max(x['S'] for x in row[kind]['statistics'].values())
            for kind in ['aligned','misaligned']:
                for rule, cut in [('calibrated',threshold),('fixed_floor',.05)]:
                    count = sum(x[kind]['max_S'] > cut for x in evaluation)
                    summary = size[kind+'_'+rule+'_warning']
                    assert count == summary['count'] and summary['n'] == 200
                    assert summary['rate'] == count/200
            oracle_counts[name].append({'n':size['n'], 'aligned_warnings':size['aligned_calibrated_warning']['count'],
                                       'proxy_warnings':size['misaligned_calibrated_warning']['count']})
    temporal = read('temporal.json'); assert len(temporal['records']) == 11
    for row in temporal['records']:
        assert len(set(row['test_rows'])) == len(row['test_rows'])
        y, twf = np.array(row['test_labels']),np.array(row['test_TWF'])
        for model in row['models'].values():
            pred = np.array(model['predictions'])
            assert len(pred) == len(y)
            tp = int((pred*y).sum()); denominator = int(pred.sum()+y.sum())
            assert np.isclose(2*tp/denominator,model['test_F1'])
            assert int((pred*twf).sum()) == model['mechanism_recall']['TWF']['detected']
            assert int(twf.sum()) == model['mechanism_recall']['TWF']['total']
    effects = read('effect_replication.json'); assert len(effects['records']) == 40
    identifiers = {(r['n'],r['generation_seed'],r['sigma']) for r in effects['records']}
    assert identifiers == {(n,s,sigma) for n in [10000,30000,100000,300000] for s in range(5) for sigma in [None,.25]}
    for row in effects['records']:
        assert row['model_fold_seed'] == 42
        assert not any('p_holm' in key for key in row)
    industrial = read('industrial_alignment.json')
    assert len(industrial['records']) == 42 and industrial['metadata_consistent']
    subprocess.run([sys.executable,str(ROOT/'experiments/29_verify_industrial.py')],check=True)
    learned = read('learned_calibration.json');fixed = read('fixed_learner_calibration.json')
    assert not learned['quick'] and learned['fitted_models'] == 7200 and fixed['calibration_datasets'] == 240000
    subprocess.run([sys.executable,str(ROOT/'experiments/22_verify_learned.py')],check=True)
    region = read('operating_region.json')
    subprocess.run([sys.executable,str(ROOT/'experiments/25_verify_region.py')],check=True)
    budget = read('region_budget.json')
    subprocess.run([sys.executable,str(ROOT/'experiments/27_verify_region_budget.py')],check=True)
    paderborn = read('paderborn_physical_acquisition.json')
    assert len(paderborn['records']) == 160 and paderborn['unique_vibration_payloads'] == 156
    assert len(paderborn['duplicate_all_Y_groups']) == 3 and paderborn['scope'] == ['KA04','KI04']
    assert not paderborn['protected_calibration_or_test_signals_inspected']
    assert not paderborn['exhaustive_physical_alignment_verified'] and not paderborn['acquisition_independence_verified']
    subprocess.run([sys.executable,str(ROOT/'experiments/32_verify_paderborn.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'experiments/34_verify_physical_acquisition.py')],check=True)
    source = (TARGET/'IEEE-conference-template-062824.tex').read_text()
    bibliography = (TARGET/'refs.bib').read_text()
    cited = set(k for group in re.findall(r'\\cite\{([^}]+)\}',source) for k in group.split(','))
    defined = set(re.findall(r'@\w+\{([^,]+),',bibliography))
    assert cited <= defined, cited-defined
    assert not re.search(r'(?<!\\)%',source), 'Author-facing source comment'
    log = (TARGET/'IEEE-conference-template-062824.log').read_text()
    assert 'Overfull' not in log and 'undefined' not in log
    pdf = TARGET/'IEEE-conference-template-062824.pdf'
    assert pdf.stat().st_mtime >= (TARGET/'IEEE-conference-template-062824.tex').stat().st_mtime, 'Rebuild the manuscript before verification.'
    info = subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages = int(re.search(r'Pages:\s+(\d+)',info).group(1)); assert pages <= 10
    text = subprocess.check_output(['pdftotext',str(pdf),'-'],text=True)
    assert '??' not in text and not re.search(r'\b(?:TODO|PENDING|PLACEHOLDER)\b',text)
    subprocess.run([sys.executable,str(ROOT/'experiments/41_verify_review2_paper.py')],check=True)
    checkpoints = list((RESULTS/'scale_replication_checkpoints').rglob('draw_*.json'))
    manifest = {'verified':True,'pdf_pages':pages,'historical_result_sha256':historical,
                'historical_results_byte_identical':historical_byte_identical,
                'historical_scientific_values_match':True,
                'cwru_recordings':36,'gain_signal_invariance_checked_recordings':36,
                'cwru_followup_conditions':16,'industrial_alignment_models':42,
                'industrial_metadata_records':36,
                'industrial_causal_alignment_validated':False,
                'industrial_five_percent_calibration_validated':False,
                'paderborn_raw_files':160,'paderborn_training_bearings':2,
                'paderborn_unique_numerical_payloads':156,'paderborn_duplicate_groups':3,
                'paderborn_physical_alignment_verified':False,'paderborn_acquisition_independence_verified':False,
                'oracle_evaluation':oracle_counts,
                'temporal_sequences':11,'effect_audits':40,
                'learned_classifier_fits':learned['fitted_models'],'eligible_fixed_functions':fixed['eligible_functions'],
                'fixed_function_calibration_batches':fixed['calibration_datasets'],
                'fresh_region_tests':len(region['records']),'fresh_region_calibration_batches':200*len(region['records']),
                'equal_count_region_tests':len(budget['records']),'equal_count_reference_batches':400*len(budget['records']),
                'optional_reference':{'complete':False,'completed_draws_first_condition':len(checkpoints),
                                      'reported_in_paper':False,'status':'interrupted; checkpoints preserved'},
                'output_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
    inputs = list((ROOT/'src/pdm_audit').glob('*.py'))+list((ROOT/'experiments').glob('*.py'))+list((ROOT/'experiments').glob('*PROTOCOL.md'))
    manifest['source_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    manifest['input_sha256'] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [ROOT/'dataset/ai4i2020.csv', ROOT/'models/xgboost_params.json', ROOT/'data/cwru/manifest.json',
                                         TARGET/'IEEE-conference-template-062824.tex', TARGET/'refs.bib',
                                         ROOT/'experiments/PADERBORN_SPLIT_CONTRACT.json',
                                         ROOT/'experiments/PADERBORN_PHYSICAL_LABEL_OVERLAY.json',
                                         ROOT/'data/paderborn/zenodo_15845309.json']}
    manifest['result_sha256'] = {name:hashlib.sha256((RESULTS/name).read_bytes()).hexdigest() for name in
                                ['cwru.json','cwru_followup.json','known_truth.json','known_truth_overlap.json','temporal.json','effect_replication.json',
                                 'learned_calibration.json','fixed_learner_calibration.json','fixed_learner_calibration_draws.npz',
                                 'operating_region.json','operating_region_draws.npz','region_budget.json','region_budget_draws.npz',
                                 'industrial_alignment.json','industrial_alignment_summary.csv',
                                 'paderborn_feasibility.json','paderborn_physical_acquisition.json',
                                 'paderborn_capture_metadata.csv','paderborn_physical_spectra.csv',
                                 'paderborn_physical_verification_manifest.json']}
    (RESULTS/'scientific_run_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('PASS: historical scientific values match; runtime-only differences permitted and byte identity recorded')
    print('PASS: 36 checked recordings; 16 grouped follow-up conditions')
    print('PASS: independent oracle calibration/evaluation counts, 11 temporal sequences, 40 effect audits')
    print('PASS: two Paderborn training bearings; 160 raw files; 156 distinct payloads; physical/independence limits retained')
    print(f'PASS: references and PDF bindings; {pages} pages; {len(checkpoints)} incomplete optional draws excluded')


if __name__ == '__main__':
    main()
