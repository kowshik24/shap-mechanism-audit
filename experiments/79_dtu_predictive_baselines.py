"""Training-only F1-selected predictive comparator for the BA-selected audit model."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.process_audit import predictive_metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def main():
    followup = json.loads((OUT/'dtu_dose_response.json').read_text())
    selected = max(followup['selection'], key=lambda x: x['F1'])
    trees, depth, leaf, weight = selected['config']
    values = np.load(OUT/'dtu_prospective_data.npz')['values']
    packets = np.stack([np.r_[values[i, :, 1:], values[200+i, :, 1:]] for i in range(200)])
    y = np.tile(np.r_[np.zeros(16, int), np.ones(48, int)], 2)
    records = []
    for unit in followup['units']:
        if unit['role'] != 'evaluation':
            continue
        sampled = np.asarray(unit['sampled_training_packets'])-1
        train = packets[sampled].reshape(-1, 53)
        assert hashlib.sha256(train.tobytes()).hexdigest() == unit['training_sha256']
        model = RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                                       class_weight=weight, random_state=unit['seed'], n_jobs=1).fit(train, np.tile(y, 40))
        probability = model.predict_proba(packets[unit['packet']-1])[:, 1]
        records.append({'packet': unit['packet'], 'metrics': predictive_metrics(y, probability)})
    assert len(records) == 60
    result = {'complete': True, 'selection': selected, 'records': records,
              'mean_metrics': {key: float(np.mean([r['metrics'][key] for r in records]))
                               for key in ['F1', 'balanced_accuracy', 'healthy_FPR']},
              'all_positive': predictive_metrics(y, np.ones(128)),
              'scope': 'Unaltered predictive comparator selected only using training-held-out F1; no audit thresholds fitted',
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                         [Path(__file__), ROOT/'experiments/REVIEW4_PREDICTIVE_BASELINE_AMENDMENT.md',
                          OUT/'dtu_dose_response.json', OUT/'dtu_prospective_data.npz']}}
    (OUT/'dtu_predictive_baselines.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['mean_metrics'])


if __name__ == '__main__':
    main()
