"""Replay every saved measured-system model and audit reference capacity."""
import pdm_audit
import hashlib
import itertools
import json
from pathlib import Path
import joblib
import numpy as np
from pdm_audit.cwru import EXPECTED, load_features
from pdm_audit.industrial import table_metadata, split_support, recording_binary

ROOT = Path(__file__).resolve().parents[1]


def main():
    result = json.loads((ROOT/'experiments/results/industrial_alignment.json').read_text())
    assert result['complete'] and len(result['records']) == 42
    assert not result['industrial_causal_alignment_validated']
    assert not result['nominal_five_percent_warning_validated'] and not result['bearing_identity_known']
    for path, digest in result['sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    df, inventory = load_features(ROOT/'data/cwru')
    df['diameter_in'] = df.recording.map({r['recording']: r['diameter_in'] for r in inventory})
    manifest = json.loads((ROOT/'data/cwru/manifest.json').read_text())
    source = table_metadata((ROOT/'data/cwru/source_table.html').read_text(), manifest['source'])
    assert {m['recording'] for m in result['metadata']} == set(source)
    lookup = {r['recording']: r for r in inventory}
    for item in result['metadata']:
        label = item['recording']; raw = lookup[label]
        for key, value in source[label].items():
            assert item[key] == value
        assert item['load_hp'] == raw['load_hp'] == int(label[-1])
        assert item['diameter_in'] == raw['diameter_in'] == int(label.split('@')[0].split('_')[0][-3:])/1000
        assert item['measured_rpm'] == raw['rpm']
        assert item['rpm_residual'] == raw['rpm']-item['approximate_rpm']
        assert item['rpm_consistent_within_50'] == (abs(item['rpm_residual']) <= 50)
    assert result['metadata_consistent'] == all(m['rpm_consistent_within_50'] for m in result['metadata'])
    keys = {(r['axis'], r['held_out'], r['fault'], r['view']) for r in result['records']}
    expected = {(a, float(g), m, v) for a in ['load_hp', 'diameter_in'] for g in df[a].unique()
                for m, v in itertools.product(['B', 'IR', 'OR'], ['own_anchors', 'all'])}
    assert keys == expected
    for r in result['records']:
        train, test = df.loc[df[r['axis']] != r['held_out']], df.loc[df[r['axis']] == r['held_out']]
        assert set(r['train_recordings']) == set(train.recording)
        assert set(r['test_recordings']) == set(test.recording)
        assert not set(r['train_recordings']) & set(r['test_recordings'])
        path = ROOT/r['model_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == r['model_sha256']
        fit = joblib.load(path)
        assert fit.random_state == 2026 and fit.n_estimators == 100 and fit.min_samples_leaf == 3
        assert fit.max_depth == r['depth'] == [3, 6, None][np.argmax([s['mean_F1'] for s in r['inner_scores']])]
        for s in r['inner_scores']:
            assert np.isclose(s['mean_F1'], np.mean(s['fold_F1']))
        assert list(fit.feature_names_in_) == r['features']
        assert split_support(fit, r['features'], EXPECTED[r['fault']]) == r['split_gate']
        x = test[r['features']].copy(); base = fit.predict_proba(x)[:, 1]
        assert recording_binary(test, base, r['fault']) == r['original']
        median = train[EXPECTED[r['fault']]].median().to_dict()
        assert median == r['training_anchor_medians']
        for col in EXPECTED[r['fault']]:
            x[col] = median[col]
        p = fit.predict_proba(x)[:, 1]
        assert recording_binary(test, p, r['fault']) == r['anchor_mask']
        assert float(np.max(np.abs(p-base))) == r['anchor_mask_max_window_probability_change']
        for gain in [.25, 4.]:
            x = test[r['features']].copy()
            if 'log_RMS' in x:
                x['log_RMS'] += np.log(gain)
            p = fit.predict_proba(x)[:, 1]; item = r['gain'][str(gain)]
            assert recording_binary(test, p, r['fault']) == item['metrics']
            assert float(np.max(np.abs(p-base))) == item['max_window_probability_change']
            if r['view'] == 'own_anchors':
                assert np.array_equal(base, p) and r['split_gate']['coordinate_invariant']
    for s in result['summary']:
        rows = [r for r in result['records'] if (r['axis'], r['fault'], r['view']) == (s['axis'], s['fault'], s['view'])]
        for key, target in [('original', 'pooled_recording_F1'), ('anchor_mask', 'masked_F1')]:
            predictions = [p for r in rows for p in r[key]['recordings']]
            assert len(predictions) == len({p['id'] for p in predictions}) == 36
            tp = sum(p['prediction'] and p['truth'] for p in predictions)
            denominator = sum(int(p['prediction'])+int(p['truth']) for p in predictions)
            assert np.isclose(2*tp/denominator, s[target])
        assert s['coordinate_invariant_fits'] == sum(r['split_gate']['coordinate_invariant'] for r in rows)
        assert s['max_gain_probability_change'] == max(g['max_window_probability_change'] for r in rows for g in r['gain'].values())
    batches = result['nonoverlapping_inventory_batches']
    assert len(batches) == 12
    assert len(set(x for b in batches for x in b['recordings'])) == 36
    for b in batches:
        rows = [lookup[k] for k in b['recordings']]
        assert {r['fault'] for r in rows} == {'IR', 'B', 'OR'}
        assert {r['load_hp'] for r in rows} == {b['load_hp']}
        assert {r['diameter_in'] for r in rows} == {b['diameter_in']}
    capacities = result['optimistic_capacity_before_training']
    for key, b in [('matched_load_after_test', 2), ('pooled_after_test', 11), ('minimum_finite_reference', 19)]:
        c = capacities[key]; rank = int(np.ceil((b+1)*.95))
        assert c['calibration_batches'] == b and c['rank'] == rank
        assert c['finite_threshold_available'] == (rank <= b)
        assert c['minimum_tail_resolution'] == 1/(b+1)
    print('PASS: 36 official metadata rows, raw checksums, load/diameter/fault groups, RPM residuals')
    print('PASS: all 42 saved forests replayed; nested selections, recording predictions, gain and mask interventions')
    print('PASS: exhaustive split-support gates; 12 non-overlapping inventory batches; sparse-reference abstention')


if __name__ == '__main__':
    main()
