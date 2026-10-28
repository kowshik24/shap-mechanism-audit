"""Replay frozen IDV4 fits to inspect mechanism-specific decision saturation."""
import pdm_audit
import hashlib
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.process_audit import predictive_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def replay(unit, config, packets, mechanism, context):
    train = packets[np.asarray(unit['sampled_training_packets'])-1].reshape(-1, 53)
    assert hashlib.sha256(train.tobytes()).hexdigest() == unit['training_sha256']
    y = mechanism > 0
    train_mechanism = np.tile(mechanism, 40)
    result = []
    for dose in [.1, .25, .5, 1., 2.]:
        training = train.copy()
        training[train_mechanism == 4, context] += dose*unit['iqr']
        trees, depth, leaf, weight = config
        model = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
            class_weight=weight, random_state=unit['seed'], n_jobs=1).fit(training, np.tile(y, 40))
        for view in ['matched', 'removed']:
            x = packets[unit['packet']-1].copy()
            if view == 'matched':
                x[mechanism == 4, context] += dose*unit['iqr']
            probability = model.predict_proba(x)[:, 1]
            saved = next(r for r in unit['records'] if (r['kind'], r['dose'], r['view']) == ('IDV4', dose, view))
            assert predictive_metrics(y, probability) == saved['metrics']
            result.append(dict(packet=unit['packet'], dose=dose, view=view,
                IDV4_positive=int(np.sum(probability[mechanism == 4] >= .5)), IDV4_rows=48,
                IDV5_positive=int(np.sum(probability[mechanism == 5] >= .5)),
                IDV4_probability_range=[float(probability[mechanism == 4].min()), float(probability[mechanism == 4].max())]))
    return result


def main():
    dtu = json.loads((OUT/'dtu_dose_response.json').read_text())
    source = json.loads((OUT/'dtu_prospective_source.json').read_text())
    values = np.load(OUT/'dtu_prospective_data.npz')['values']
    packets = np.stack([np.r_[values[i, :, 1:], values[200+i, :, 1:]] for i in range(200)])
    active = np.r_[np.zeros(16, int), np.ones(48, int)]
    mechanism = np.r_[active*4, active*5]
    context = source['labels'].index('Compressor Work')-1
    stream = Parallel(n_jobs=4, return_as='generator', batch_size=1)(delayed(replay)(
        u, dtu['chosen_config'], packets, mechanism, context) for u in dtu['units'] if u['role'] == 'evaluation')
    records = []
    for i, result in enumerate(stream):
        records.extend(result)
        if (i+1) % 10 == 0:
            print(i+1, '/60 units replayed', flush=True)
    assert len(records) == 600
    summary = []
    for dose in [.1, .25, .5, 1., 2.]:
        for view in ['matched', 'removed']:
            rows = [r for r in records if (r['dose'], r['view']) == (dose, view)]
            summary.append(dict(dose=dose, view=view, units=len(rows),
                fully_positive_IDV4_units=sum(r['IDV4_positive'] == 48 for r in rows),
                IDV4_positive_range=[min(r['IDV4_positive'] for r in rows), max(r['IDV4_positive'] for r in rows)],
                IDV4_minimum_probability=min(r['IDV4_probability_range'][0] for r in rows)))
    paths = [Path(__file__), OUT/'dtu_dose_response.json', OUT/'dtu_prospective_data.npz',
             OUT/'dtu_prospective_source.json', ROOT/'src/pdm_audit/process_audit.py']
    result = dict(complete=True, scope='Prediction-only replay of 300 previously specified fits; all 600 aggregate metric records reproduced exactly; no tuning, new study units, or SHAP recomputation',
        records=records, summary=summary,
        sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/'dtu_prediction_replay.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
