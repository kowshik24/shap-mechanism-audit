"""Check diagnostic output scale, signed attribution reconstruction, and roles."""
import pdm_audit
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pdm_audit.kaist import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def main():
    record = json.loads((OUT/'kaist_diagnostics.json').read_text())
    for path,digest in record['input_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
    arrays = np.load(OUT/'kaist_diagnostic_arrays.npz')
    frame = pd.read_csv(OUT/'kaist_features.csv')
    seeds = pd.read_csv(OUT/'kaist_seed_predictions.csv')
    top = []
    for sources in [[0],[0,2]]:
        tag = 'source_'+'_'.join(map(str,sources))
        phi, probability, prediction = [arrays[tag+'_'+key] for key in ['fault_phi','fault_probability','multiclass_prediction']]
        assert phi.shape == (630,40)
        rows = [r for r in record['results'] if r['source_loads_Nm']==sources]
        base = rows[0]['background_fault_probability']
        assert max(abs(base+phi.sum(axis=1)-probability)) < 1e-6
        for row in rows:
            mask = frame.load_Nm==row['load_Nm']
            assert row['role'] == ('in_sample' if row['load_Nm'] in sources else 'held_out')
            assert all(np.allclose(v,row['metrics'][k]) for k,v in metrics(frame[mask],prediction[mask]).items())
            healthy = mask & (frame.label==0)
            assert np.isclose(np.mean(probability[healthy]>=.5),row['fault_probability_threshold_healthy_FPR'])
        for profile in [p for p in record['profiles'] if p['source_loads_Nm']==sources]:
            mask = (frame.load_Nm==profile['load_Nm'])&(frame.label==profile['label'])
            assert np.allclose(phi[mask].mean(axis=0),profile['mean_signed'])
            assert np.allclose(abs(phi[mask]).mean(axis=0),profile['mean_absolute'])
        fp = (frame.load_Nm==4)&(frame.label==0)&(prediction!=0)
        if fp.any():
            mean = phi[fp].mean(axis=0)
            top = [dict(feature=record['features'][j],mean_signed_fault_probability_contribution=float(mean[j]))
                   for j in np.argsort(-mean)[:5]]
    assert len(record['seed_results']) == 120
    for row in record['seed_results']:
        tag = 'source_'+'_'.join(map(str,row['source_loads_Nm']))
        saved = seeds[(seeds.source_tag==tag)&(seeds.seed==row['seed'])&(seeds.load_Nm==row['load_Nm'])]
        assert len(saved) == 210
        assert row['role'] == ('in_sample' if row['load_Nm'] in row['source_loads_Nm'] else 'held_out')
        for key,value in metrics(saved,saved.prediction).items():
            assert np.allclose(value,row['metrics'][key])
        healthy = saved[saved.label==0]
        assert np.isclose(np.mean(healthy.fault_probability>=.5),row['fault_probability_threshold_healthy_FPR'])
    result = dict(passed=True,reconstructed_rows=1260,cohort_profiles=18,seed_model_fits=40,seed_load_results=120,
        max_additivity_error=record['max_additivity_error'],
        top_false_positive_features=top,
        scope='Single-rig exploratory prediction/explanation replay; no warning calibration or causal identification',
        diagnostic_sha256=hashlib.sha256((OUT/'kaist_diagnostics.json').read_bytes()).hexdigest(),
        arrays_sha256=hashlib.sha256((OUT/'kaist_diagnostic_arrays.npz').read_bytes()).hexdigest(),
        seed_predictions_sha256=hashlib.sha256((OUT/'kaist_seed_predictions.csv').read_bytes()).hexdigest())
    (OUT/'kaist_diagnostic_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: 1260 probability reconstructions, 18 cohort profiles, 40 seed fits / 120 load results; output-rule and held-out-role checks')


if __name__ == '__main__':
    main()
