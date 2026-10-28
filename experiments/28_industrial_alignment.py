"""Measured CWRU alignment gates and honest operating-region capacity."""
import pdm_audit
import hashlib
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.cwru import EXPECTED, load_features
from pdm_audit.industrial import table_metadata, split_support, recording_binary, rank_capacity

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results/industrial_alignment.json'
MODELS = ROOT/'experiments/results/industrial_alignment_models'
DEPTHS = [3, 6, None]


def model(depth):
    return RandomForestClassifier(n_estimators=100, max_depth=depth, min_samples_leaf=3,
                                  class_weight='balanced', random_state=2026, n_jobs=1)


def main():
    df, inventory = load_features(ROOT/'data/cwru')
    manifest = json.loads((ROOT/'data/cwru/manifest.json').read_text())
    official = table_metadata((ROOT/'data/cwru/source_table.html').read_text(), manifest['source'])
    assert len(official) == 36
    metadata = []
    for r in inventory:
        source = official[r['recording']]
        for key in ['load_hp', 'fault', 'diameter_in', 'url']:
            assert r[key] == source[key], (r['recording'], key)
        delta = r['rpm']-source['approximate_rpm']
        metadata.append({'recording': r['recording'], **source, 'measured_rpm': r['rpm'],
                         'rpm_residual': delta, 'rpm_consistent_within_50': abs(delta) <= 50,
                         'raw_sha256': r['sha256']})
    df['diameter_in'] = df.recording.map({r['recording']: r['diameter_in'] for r in inventory})
    features = [c for c in df if c not in ['recording', 'fault', 'load_hp', 'window', 'diameter_in']]
    MODELS.mkdir(exist_ok=True)
    records = []
    for axis in ['load_hp', 'diameter_in']:
        for held in sorted(df[axis].unique()):
            train, test = df.loc[df[axis] != held], df.loc[df[axis] == held]
            for fault, view in itertools.product(['B', 'IR', 'OR'], ['own_anchors', 'all']):
                cols = EXPECTED[fault] if view == 'own_anchors' else features
                scores = []
                for depth in DEPTHS:
                    inner = []
                    for group in sorted(train[axis].unique()):
                        a, b = train.loc[train[axis] != group], train.loc[train[axis] == group]
                        fit = model(depth).fit(a[cols], a.fault == fault)
                        inner.append(recording_binary(b, fit.predict_proba(b[cols])[:, 1], fault)['F1'])
                    scores.append({'depth': depth, 'fold_F1': inner, 'mean_F1': float(np.mean(inner))})
                depth = DEPTHS[int(np.argmax([s['mean_F1'] for s in scores]))]
                fit = model(depth).fit(train[cols], train.fault == fault)
                base = fit.predict_proba(test[cols])[:, 1]
                medians = train[EXPECTED[fault]].median().to_dict()
                masked = test[cols].copy()
                for col in EXPECTED[fault]:
                    masked[col] = medians[col]
                mask_p = fit.predict_proba(masked)[:, 1]
                interventions = {}
                for gain in [.25, 4.]:
                    x = test[cols].copy()
                    if 'log_RMS' in cols:
                        x['log_RMS'] += np.log(gain)
                    p = fit.predict_proba(x)[:, 1]
                    interventions[str(gain)] = {'metrics': recording_binary(test, p, fault),
                                                'max_window_probability_change': float(np.max(np.abs(p-base)))}
                name = f'{axis}-{held:g}-{fault}-{view}.joblib'
                joblib.dump(fit, MODELS/name, compress=3)
                record = {'axis': axis, 'held_out': float(held), 'fault': fault, 'view': view,
                          'features': cols, 'depth': depth, 'inner_scores': scores,
                          'train_recordings': sorted(train.recording.unique()),
                          'test_recordings': sorted(test.recording.unique()),
                          'split_gate': split_support(fit, cols, EXPECTED[fault]),
                          'original': recording_binary(test, base, fault), 'gain': interventions,
                          'anchor_mask': recording_binary(test, mask_p, fault),
                          'anchor_mask_max_window_probability_change': float(np.max(np.abs(mask_p-base))),
                          'training_anchor_medians': medians,
                          'model_path': str((MODELS/name).relative_to(ROOT)),
                          'model_sha256': hashlib.sha256((MODELS/name).read_bytes()).hexdigest()}
                records.append(record)
                print(axis, held, fault, view, record['original']['F1'], flush=True)
    summary = []
    for axis, fault, view in itertools.product(['load_hp', 'diameter_in'], ['B', 'IR', 'OR'], ['own_anchors', 'all']):
        rows = [r for r in records if (r['axis'], r['fault'], r['view']) == (axis, fault, view)]
        total = sum(r['original']['positive_n'] for r in rows)
        negatives = sum(r['original']['negative_n'] for r in rows)
        tp, tn = sum(r['original']['tp'] for r in rows), sum(r['original']['tn'] for r in rows)
        def f1(key):
            t = sum(r[key]['tp'] for r in rows)
            return 2*t/(2*t+sum(r[key]['positive_n']-r[key]['tp']+r[key]['negative_n']-r[key]['tn'] for r in rows))
        summary.append({'axis': axis, 'fault': fault, 'view': view, 'pooled_recording_F1': f1('original'),
                        'sensitivity': tp/total, 'specificity': tn/negatives,
                        'masked_F1': f1('anchor_mask'),
                        'coordinate_invariant_fits': sum(r['split_gate']['coordinate_invariant'] for r in rows),
                        'fits': len(rows),
                        'max_gain_probability_change': max(g['max_window_probability_change'] for r in rows for g in r['gain'].values())})
    batches = [{'load_hp': load, 'diameter_in': diameter,
                'recordings': sorted(r['recording'] for r in inventory if r['load_hp'] == load and r['diameter_in'] == diameter)}
               for load, diameter in itertools.product(range(4), [.007, .014, .021])]
    assert all(len(b['recordings']) == 3 for b in batches)
    provenance = ['experiments/INDUSTRIAL_ALIGNMENT_PROTOCOL.md', 'experiments/28_industrial_alignment.py',
                  'src/pdm_audit/industrial.py', 'src/pdm_audit/cwru.py', 'data/cwru/manifest.json', 'data/cwru/source_table.html']
    result = {'complete': len(records) == 42, 'protocol': provenance[0], 'metadata': metadata,
              'metadata_consistent': all(m['rpm_consistent_within_50'] for m in metadata),
              'records': records, 'summary': summary, 'nonoverlapping_inventory_batches': batches,
              'optimistic_capacity_before_training': {'matched_load_after_test': rank_capacity(2),
                                                        'pooled_after_test': rank_capacity(11),
                                                        'minimum_finite_reference': rank_capacity(19)},
              'industrial_causal_alignment_validated': False,
              'nominal_five_percent_warning_validated': False,
              'bearing_identity_known': False,
              'sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in provenance}}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
