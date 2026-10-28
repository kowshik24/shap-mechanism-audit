"""Training-packet resampling and fault-common/fault-specific dose response."""
import pdm_audit
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.process_audit import score_packet, prediction_response, predictive_metrics
from pdm_audit.tree_precision import probability_explainer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
CONFIGS = [(64, 8, 2, 'balanced'), (64, None, 2, 'balanced'), (64, None, 1, None)]
DOSES = [.1, .25, .5, 1., 2.]


def model(config, seed):
    trees, depth, leaf, weight = config
    return RandomForestClassifier(n_estimators=trees, max_depth=depth, min_samples_leaf=leaf,
                                  class_weight=weight, random_state=seed, n_jobs=1)


def unit(packet, role, packets, mechanism, expected, context, config):
    rng = np.random.default_rng([20261025, 76, packet])
    sampled = rng.integers(0, 40, 40)
    train = packets[sampled].reshape(-1, 53)
    target = (mechanism > 0).astype(int)
    y = np.tile(target, 40)
    train_mechanism = np.tile(mechanism, 40)
    background = np.linspace(0, len(train)-1, 32, dtype=int)
    iqr = float(np.subtract(*np.quantile(train[:, context], [.75, .25])))
    records = []
    designs = [('unaltered', 0.)] if role == 'calibration' else [('unaltered', 0.)] + [
        (kind, dose) for kind in ['common', 'IDV4'] for dose in DOSES]
    for kind, dose in designs:
        training = train.copy()
        train_fault = train_mechanism > 0 if kind == 'common' else train_mechanism == 4
        training[train_fault, context] += dose*iqr
        fitted = model(config, 760000+packet).fit(training, y)
        explainer = probability_explainer(fitted, training[background])
        expected_value = float(explainer.expected_value[1])
        median = np.median(training, axis=0)
        for view in (['unaltered'] if dose == 0 else ['matched', 'removed']):
            x = packets[packet-1].copy()
            if view == 'matched':
                fault = mechanism > 0 if kind == 'common' else mechanism == 4
                x[fault, context] += dose*iqr
            probability = fitted.predict_proba(x)[:, 1]
            phi = np.asarray(explainer.shap_values(x, check_additivity=False))[:, :, 1]
            error = float(np.max(abs(phi.sum(axis=1)+expected_value-probability)))
            assert error < 1e-6
            response = prediction_response(fitted, x, median)
            scores = [score_packet(abs(phi), mechanism, expected, context, True),
                      score_packet(response, mechanism, expected, context, False)]
            records.append({'packet': packet, 'role': role, 'kind': kind, 'dose': dose,
                            'view': view, 'metrics': predictive_metrics(target, probability),
                            'scores': scores, 'additivity_error': error})
    return {'packet': packet, 'role': role, 'sampled_training_packets': (sampled+1).tolist(),
            'training_sha256': hashlib.sha256(train.tobytes()).hexdigest(),
            'iqr': iqr, 'seed': 760000+packet, 'records': records}


def safe(x):
    if isinstance(x, dict):
        return {k: safe(v) for k, v in x.items()}
    if isinstance(x, list):
        return [safe(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return 'Infinity' if x > 0 else '-Infinity'
    return x


def main():
    began = time.perf_counter()
    source = json.loads((OUT/'dtu_prospective_source.json').read_text())
    values = np.load(OUT/'dtu_prospective_data.npz')['values']
    packets = np.stack([np.r_[values[i, :, 1:], values[200+i, :, 1:]] for i in range(200)])
    active = np.r_[np.zeros(16, int), np.ones(48, int)]
    mechanism = np.r_[active*4, active*5]
    target = (mechanism > 0).astype(int)
    labels = source['labels']
    context = labels.index('Compressor Work')-1
    names = {4: ['Reactor Temperature', 'Reactor Coolant Temp', 'Reactor Coolant'],
             5: ['Product Sep Temp', 'Separator Coolant Temp', 'Condenser Coolant']}
    expected = {m: [labels.index(x)-1 for x in group] for m, group in names.items()}
    validation = []
    for config in CONFIGS:
        fitted = model(config, 762026).fit(packets[:30].reshape(-1, 53), np.tile(target, 30))
        metrics = [predictive_metrics(target, fitted.predict_proba(x)[:, 1]) for x in packets[30:40]]
        validation.append({'config': config, 'balanced_accuracy': float(np.mean([x['balanced_accuracy'] for x in metrics])),
                           'F1': float(np.mean([x['F1'] for x in metrics]))})
    chosen = int(np.argmax([r['balanced_accuracy'] for r in validation]))
    config = CONFIGS[chosen]
    print('training-only selection', validation, 'chosen', chosen, flush=True)
    units = []
    stream = Parallel(n_jobs=4, return_as='generator', batch_size=1)(delayed(unit)(
        p, 'calibration' if p <= 140 else 'evaluation', packets, mechanism, expected, context, config)
        for p in range(41, 201))
    for i, item in enumerate(stream):
        units.append(item)
        if (i+1) % 10 == 0:
            print('DTU units', i+1, '/160', round(time.perf_counter()-began, 1), flush=True)
    records = [r for u in units for r in u['records']]
    cuts, summary = [], []
    for k, score in enumerate(['shap', 'response']):
        cal = np.array([r['scores'][k]['score'] for r in records if r['role'] == 'calibration'])
        cut = float(np.sort(cal)[95])
        cuts.append(cut)
        designs = [('unaltered', 0., 'unaltered')] + [
            (kind, dose, view) for kind in ['common', 'IDV4'] for dose in DOSES for view in ['matched', 'removed']]
        for kind, dose, view in designs:
            selected = [r for r in records if r['role'] == 'evaluation' and
                        (r['kind'], r['dose'], r['view']) == (kind, dose, view)]
            assert len(selected) == 60
            warning = sum(r['scores'][k]['score'] > cut for r in selected)
            local = sum(float(r['scores'][k]['localization']) > cut for r in selected)
            by_mechanism = {str(m): sum(any(s['mechanism'] == m and s['S_feature_index'] == context and s['S'] > cut
                                             for s in r['scores'][k]['statistics']) for r in selected) for m in [4, 5]}
            ci = binomtest(local, 60).proportion_ci(method='exact')
            summary.append({'score': score, 'kind': kind, 'dose': dose, 'view': view,
                            'warnings': warning, 'localizations': local, 'localization_ci95': [ci.low, ci.high],
                            'localization_by_mechanism': by_mechanism,
                            'mean_metrics': {name: float(np.mean([r['metrics'][name] for r in selected]))
                                             for name in ['F1', 'balanced_accuracy', 'healthy_FPR']}})
    files = [Path(__file__), ROOT/'experiments/REVIEW4_PROTOCOL.md', ROOT/'src/pdm_audit/process_audit.py',
             ROOT/'src/pdm_audit/tree_precision.py', OUT/'dtu_prospective_data.npz', OUT/'dtu_prospective_source.json']
    result = {'complete': True, 'selection': validation, 'chosen_config': config, 'units': units,
              'summary': summary, 'cuts': cuts, 'calibration_rank': 96, 'calibration_units': 100,
              'scope': 'Exploratory empirical training-packet bootstrap plus seed population; simulated process, label-assigned offsets',
              'seconds': time.perf_counter()-began,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (OUT/'dtu_dose_response.json').write_text(json.dumps(safe(result), indent=2, allow_nan=False)+'\n')
    print('COMPLETE DTU follow-up', round(result['seconds'], 1), flush=True)


if __name__ == '__main__':
    main()
