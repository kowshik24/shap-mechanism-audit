"""Compare an isolated frozen-input replay with the submitted scientific assets."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
VOLATILE = {'sha256','input_sha256','seconds','runtime_seconds'}


def compare(left, right, path=''):
    if isinstance(left,dict):
        keys = set(left)-VOLATILE
        assert keys == set(right)-VOLATILE, path
        for key in keys:
            compare(left[key],right[key],path+'/'+key)
    elif isinstance(left,list):
        assert len(left) == len(right), path
        for i,(a,b) in enumerate(zip(left,right)):
            compare(a,b,path+f'/{i}')
    elif isinstance(left,(int,float)) and not isinstance(left,bool):
        assert np.isclose(left,right,rtol=1e-10,atol=1e-12,equal_nan=True), (path,left,right)
    else:
        assert left == right, (path,left,right)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--isolated-project',required=True,type=Path)
    parser.add_argument('--tested-archive-sha256',required=True)
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{64}',args.tested_archive_sha256)
    other = args.isolated_project.resolve()
    assert other != ROOT and ROOT not in other.parents
    environment = json.loads((other.parent/'environment.json').read_text())
    assert Path(environment['package_file']).resolve().is_relative_to(other)
    assert not any(str(ROOT) in p for p in environment['sys_path'])
    json_files = ['review_audit.json','dtu_prediction_replay.json','harm_screening.json',
        'calibration_assets.json','calibration_sensitivity.json','logistic_crossover.json']
    for name in json_files:
        compare(json.loads((OUT/name).read_text()),json.loads((other/'experiments/results'/name).read_text()),name)
    array_files = ['review_audit_draws.npz','harm_screening_draws.npz','calibration_sensitivity_draws.npz']
    arrays_checked = 0
    for name in array_files:
        left,right = np.load(OUT/name),np.load(other/'experiments/results'/name)
        assert set(left.files) == set(right.files)
        for key in left.files:
            a,b = left[key],right[key]
            assert a.shape == b.shape and a.dtype == b.dtype, (name,key)
            assert np.allclose(a,b,rtol=1e-10,atol=1e-12,equal_nan=True), (name,key)
            arrays_checked += 1
    missing,wrong,dependencies = [],[],0
    for p in (other/'experiments/results').glob('*.json'):
        record = json.loads(p.read_text())
        if not isinstance(record,dict):
            continue
        for key in ['sha256','input_sha256']:
            hashes = record.get(key,{})
            if not isinstance(hashes,dict):
                continue
            for path,digest in hashes.items():
                target = other/path
                dependencies += 1
                if not target.is_file():
                    missing.append([p.name,path])
                elif hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                    wrong.append([p.name,path])
    assert not missing and not wrong, dict(missing=missing,hash_mismatches=wrong)
    verifiers = {}
    for name in ['review4_verification.json','publication_figure_verification.json','calibration_scope_verification.json']:
        record = json.loads((other/'experiments/results'/name).read_text())
        assert record['verified']
        verifiers[name] = record
    assert verifiers['review4_verification.json']['pdf_pages'] == 10
    assert verifiers['publication_figure_verification.json']['manuscript_figures'] == 3
    assert verifiers['publication_figure_verification.json']['plot_coordinates'] == 1071
    test_log = (other.parent/'tests.log').read_text()
    assert '9 passed' in test_log and 'failed' not in test_log
    result = dict(verified=True,date='2026-10-26',
        scope='Fresh Python environment outside checkout; frozen-input diagnostic regeneration, DTU prediction replay, and TeX rebuild. Original Boolean training banks and raw-waveform extraction are not regenerated.',
        environment=environment,semantic_json_comparisons=len(json_files),
        array_files_compared=len(array_files),arrays_compared=arrays_checked,
        declared_dependency_hashes_checked=dependencies,missing_dependencies=missing,
        hash_mismatches=wrong,isolated_unit_tests_passed=9,pdf_pages=10,
        manuscript_figures=3,figure_atlas_pages=6,plot_coordinates=1071,
        cold_build_fix='Makefile invokes latexmk for prebuilt figure PDFs so omitted logs are regenerated.',
        missing_provenance_fixed=['data/dtu_tep/metadata.json','experiments/results/observed_overlap.json'],
        initial_pipeline_failure='Initial archived PDFs suppressed standalone log generation; numerical stages completed, build/verifiers resumed after the Makefile fix.',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),ROOT/'requirements-reproduce.txt',ROOT/'IEEE-conference-template-062824/Makefile']},
        original_archive_sha256=args.tested_archive_sha256)
    (OUT/'submission_readiness.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {len(json_files)} scientific JSONs, {arrays_checked} arrays, {dependencies} declared dependencies, nine isolated tests, and ten-page PDF')


if __name__ == '__main__':
    main()
