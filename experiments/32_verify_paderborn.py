"""Verify the source audit, frozen bearing partition, and conservative stop gate."""
import pdm_audit
from collections import Counter
import csv
import hashlib
import json
from math import ceil
from pathlib import Path
import subprocess
from pdm_audit.paderborn import parse_capture_name

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/paderborn'


def main():
    result = json.loads((ROOT/'experiments/results/paderborn_feasibility.json').read_text())
    contract = json.loads((ROOT/'experiments/PADERBORN_SPLIT_CONTRACT.json').read_text())
    assert result['audit_complete'] and result['fitted_models'] == 0
    assert not result['calibration_experiment_executed']
    assert not result['raw_archives_inspected'] and result['verified_signal_files'] == 0
    assert not result['expected_inventory_is_observed']
    assert not contract['new_bearing_five_percent_S_calibration_permitted']
    for path, digest in result['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    # Independent transcription from Tables 4/5/7, checked against their rendered pages.
    pure_ir = set('KI01 KI03 KI05 KI07 KI08 KI04 KI14 KI16 KI17 KI18 KI21'.split())
    pure_or = set('KA01 KA03 KA05 KA06 KA07 KA08 KA09 KA04 KA15 KA16 KA22 KA30'.split())
    healthy = set('K001 K002 K003 K004 K005 K006'.split())
    mixed = {'KB23', 'KB24', 'KB27'}
    artificial = set('KA01 KA03 KA05 KA06 KA07 KA08 KA09 KI01 KI03 KI05 KI07 KI08'.split())
    rows = result['bearings']; ids = {r['bearing_id'] for r in rows}
    assert len(rows) == len(ids) == 32 and ids == pure_ir | pure_or | healthy | mixed
    groups = {'IR': pure_ir, 'OR': pure_or, 'healthy': healthy, 'mixed_or_secondary': mixed}
    for name, members in groups.items():
        assert {r['bearing_id'] for r in rows if r['diagnosis_class'] == name} == members
    for row in rows:
        assert row['origin'] == ('artificial' if row['bearing_id'] in artificial else
                                 'healthy' if row['bearing_id'] in healthy else 'accelerated_lifetime')
        if row['bearing_id'] in mixed:
            assert row['component'] == ('OR + IR' if row['bearing_id'] == 'KB27' else 'IR (+OR)')
    assert result['bearing_counts'] == {'healthy': 6, 'OR': 12, 'mixed_or_secondary': 3, 'IR': 11}
    assert result['origin_counts'] == {'healthy': 6, 'artificial': 12, 'accelerated_lifetime': 14}
    narrative = subprocess.check_output(['pdftotext', '-f', '9', '-l', '9', '-raw',
                                          str(DATA/'original_benchmark.pdf'), '-'], text=True)
    narrative = ' '.join(narrative.split())
    assert 'inner ring (KA04)' in narrative and 'outer ring (KI18)' in narrative
    assert len(result['source_inconsistencies']) == 1
    with (DATA/'bearings.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
    for actual, expected in zip(csv_rows, rows):
        assert actual == {k: '' if v is None else str(v) for k, v in expected.items()}
    assert len(csv_rows) == len(rows)
    for page, codes in [(5, artificial), (6, ids-artificial-healthy), (8, healthy)]:
        text = subprocess.check_output(['pdftotext', '-f', str(page), '-l', str(page),
                                        '-raw', str(DATA/'original_benchmark.pdf'), '-'], text=True)
        assert all(code in text for code in codes)
    expected_conditions = {'N15_M07_F10': (1500, .7, 1000), 'N09_M07_F10': (900, .7, 1000),
                           'N15_M01_F10': (1500, .1, 1000), 'N15_M07_F04': (1500, .7, 400)}
    for c in result['conditions']:
        assert (c['speed_rpm'], c['torque_Nm'], c['radial_force_N']) == expected_conditions[c['condition']]
    assert {c['condition'] for c in result['conditions']} == set(expected_conditions)
    partition = {b: k for k, values in contract['partitions'].items() for b in values}
    assert len(partition) == sum(map(len, contract['partitions'].values())) == 32
    assert set(partition) == ids and set(contract['partitions']['mixed_challenge']) == mixed
    lookup = {r['bearing_id']: r for r in rows}
    for role, codes in contract['partitions'].items():
        assert contract['partition_class_counts'][role] == dict(Counter(lookup[c]['diagnosis_class'] for c in codes))
        if role != 'mixed_challenge':
            assert {lookup[c]['diagnosis_class'] for c in codes} == {'IR', 'OR', 'healthy'}
    with (DATA/'expected_recordings.csv').open() as f:
        inventory = list(csv.DictReader(f))
    assert len(inventory) == result['advertised_expected_signal_files'] == 32*4*20
    assert len({r['filename'] for r in inventory}) == len(inventory)
    for r in inventory:
        parsed = parse_capture_name(r['filename'], ids, expected_conditions)
        assert parsed['bearing_id'] == r['bearing_id'] and parsed['condition'] == r['condition']
        assert parsed['capture_index'] == int(r['capture_index'])
        assert r['role'] == partition[r['bearing_id']] and r['signal_file_verified'] == 'False'
    assert Counter(r['bearing_id'] for r in inventory) == {b: 80 for b in ids}
    c = result['capacity']
    assert c['damaged_bearings'] == 26 and c['optimistic_all_damaged_batches_before_training_or_test'] == 13
    assert c['pure_IR_OR_batches_before_training_or_test'] == 11
    for name, b in [('optimistic_after_one_test', 12), ('pure_IR_OR_after_one_test', 10)]:
        value = c[name]
        assert value['calibration_batches'] == b and value['rank'] == ceil((b+1)*.95) > b
        assert not value['finite_threshold_available']
    assert c['minimum_calibration_batches'] == 19 and ceil(20*.95) == 19
    assert c['minimum_damaged_bearings_for_calibration_plus_one_test_before_training'] == 2*(19+1)
    assert not c['finite_five_percent_S_gate_passes']
    assert c['frozen_reserved_calibration_pure_pairs'] == 2
    print('PASS: 32 source-bearing IDs, components, provenance tables, 4 documented operating regions')
    print('PASS: 2,560 expected filenames stay with 32 bearing clusters; advertised/observed distinction retained')
    print('PASS: frozen five-way bearing split and challenge labels; no calibration/test leakage')
    print('PASS: optimistic and pure-pair capacity bounds; no unexecuted model or industrial-calibration claim')
    print('PASS: source-document audit is independent of the manuscript build')


if __name__ == '__main__':
    main()
