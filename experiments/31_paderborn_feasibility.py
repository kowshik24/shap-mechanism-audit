"""Audit official Paderborn metadata before any model fitting or signal selection."""
import pdm_audit
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import urllib.request
from pdm_audit.paderborn import bearing_registry, condition_registry, capacity, FUTURE_SPLIT

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/paderborn'
SOURCES = {
    'original_benchmark.pdf': 'https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf',
    'dataset.html': 'https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download',
    'conditions.html': 'https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/operating-conditions',
    'damage.html': 'https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/bearing-damage',
}


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-documents', action='store_true', help='Fetch missing official documentation, never raw archives.')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        path = DATA/name
        if not path.exists() and args.download_documents:
            payload = urllib.request.urlopen(url, timeout=45).read()
            path.write_bytes(payload)
        assert path.exists(), f'Missing official source {name}; rerun with --download-documents when network access is available'
    bearings = bearing_registry(DATA/'original_benchmark.pdf')
    conditions = condition_registry((DATA/'conditions.html').read_text())
    classes = dict(Counter(r['diagnosis_class'] for r in bearings))
    origins = dict(Counter(r['origin'] for r in bearings))
    write_csv(DATA/'bearings.csv', bearings); write_csv(DATA/'conditions.csv', conditions)
    mapping = {b: role for role, ids in FUTURE_SPLIT.items() for b in ids}
    assert len(mapping) == 32 and set(mapping) == {r['bearing_id'] for r in bearings}
    expected = [{'filename': f"{c['condition']}_{b['bearing_id']}_{i}.mat",
                 'bearing_id': b['bearing_id'], 'condition': c['condition'], 'capture_index': i,
                 'role': mapping[b['bearing_id']], 'signal_file_verified': False}
                for b, c, i in itertools.product(bearings, conditions, range(1, 21))]
    write_csv(DATA/'expected_recordings.csv', expected)
    lookup = {b['bearing_id']: b for b in bearings}
    contract = {'status': 'unexecuted diagnosis split; not a valid S-calibration study',
                'target': 'pure healthy/IR/OR diagnosis on previously unseen bearings',
                'hierarchy': ['bearing_id', 'operating_condition', 'capture_index', 'window'],
                'all_captures_and_conditions_follow_bearing': True,
                'partitions': {k: sorted(v) for k, v in FUTURE_SPLIT.items()},
                'partition_class_counts': {role: dict(Counter(lookup[b]['diagnosis_class'] for b in ids))
                                           for role, ids in FUTURE_SPLIT.items()},
                'operating_conditions': conditions,
                'cross_condition_rule': 'Fit on each of the four conditions in turn; evaluate each condition only on disjoint reserved/test bearings. All sixteen condition pairs are prespecified.',
                'selection': 'Training and validation bearings in the source/training operating condition only; reserved calibration and test never select features, models, thresholds, regions, or stopping.',
                'mixed_rule': 'Preserve secondary/mixed component labels; do not relabel KB23/KB24/KB27 as pure IR or OR.',
                'session_and_lifetime_cohort_verified': False,
                'physical_alignment_reference_verified': False,
                'new_bearing_five_percent_S_calibration_permitted': False}
    contract_path = ROOT/'experiments/PADERBORN_SPLIT_CONTRACT.json'
    contract_path.write_text(json.dumps(contract, indent=2)+'\n')
    count = capacity(bearings)
    count['frozen_reserved_calibration_pure_pairs'] = min(
        contract['partition_class_counts']['reserved_calibration']['IR'],
        contract['partition_class_counts']['reserved_calibration']['OR'])
    assert not count['finite_five_percent_S_gate_passes']
    source_files = [str((DATA/name).relative_to(ROOT)) for name in SOURCES]
    generated = ['data/paderborn/bearings.csv', 'data/paderborn/conditions.csv',
                 'data/paderborn/expected_recordings.csv', 'experiments/PADERBORN_SPLIT_CONTRACT.json']
    code = ['experiments/PADERBORN_FEASIBILITY_PROTOCOL.md', 'experiments/31_paderborn_feasibility.py',
            'src/pdm_audit/paderborn.py', 'src/pdm_audit/industrial.py',
            'experiments/32_verify_paderborn.py', 'tests/test_paderborn.py',
            'IEEE-conference-template-062824/PADERBORN_FEASIBILITY.md']
    result = {'audit_complete': True, 'audited_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'documentation-only source audit; raw inspection is a separate follow-up',
              'summary_classes_certify_physical_purity': False,
              'physical_followup': 'experiments/results/paderborn_physical_acquisition.json',
              'study_status': 'NO_GO for independent-new-bearing 5% S-calibration',
              'documentation_sources': SOURCES, 'bearing_counts': classes, 'origin_counts': origins,
              'bearings': bearings, 'conditions': conditions,
              'advertised_expected_signal_files': len(expected), 'expected_inventory_is_observed': False,
              'verified_signal_files': 0, 'raw_archives_inspected': False,
              'acquisition_sessions_verified': False, 'lifetime_cohorts_verified': False,
              'physical_alignment_reference_verified': False,
              'source_inconsistencies': [{'source_pdf_page': 9, 'section': 6,
                  'issue': 'Narrative interchanges the KA04 and KI18 inner/outer component descriptions.',
                  'resolution': 'Tables and corresponding spectrum figures take precedence; raw fact sheets remain unverified.'}],
              'fitted_models': 0, 'calibration_experiment_executed': False,
              'capacity': count,
              'gates': {'documented_bearing_and_region_ids': 'PASS at documentation level',
                        'raw_metadata_and_capture_headers': 'UNVERIFIED; original archive access failed from this environment',
                        'independence_and_exchangeability': 'UNVERIFIED; bearing IDs alone do not prove either',
                        'finite_five_percent_new_bearing_S_capacity': 'FAIL even under optimistic cluster counts',
                        'independent_physical_alignment_reference': 'NOT ESTABLISHED; physical anchors are partial'},
              'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files+generated+code}}
    (ROOT/'experiments/results/paderborn_feasibility.json').write_text(json.dumps(result, indent=2)+'\n')
    print('COMPLETE: 32 source-derived bearing entries, 4 regions, 2,560 expected (not observed) capture names')
    print('NO_GO: at most 13 damaged-bearing batches before reservation, 12 after one test; 19 calibration batches required')
    print('NOT RUN: no signal-level study, fitted model, or 5% mechanism-calibration result')


if __name__ == '__main__':
    main()
