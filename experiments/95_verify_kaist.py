"""Replay saved predictions, source-only choices, and native unit boundaries."""
import pdm_audit
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from pdm_audit.kaist import metrics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def main():
    result = json.loads((OUT/'kaist_predictive.json').read_text())
    for p,digest in result['input_sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, p
    inventory = json.loads((OUT/'kaist_signal_inventory.json').read_text())
    assert inventory['protocol_sha256'] == hashlib.sha256((ROOT/'experiments/KAIST_PREDICTIVE_PROTOCOL.md').read_bytes()).hexdigest()
    assert inventory['feature_code_sha256'] == hashlib.sha256((ROOT/'src/pdm_audit/kaist.py').read_bytes()).hexdigest()
    assert len(inventory['records']) == 21
    assert inventory['windows'] == 630
    assert inventory['exact_duplicate_windows'] == inventory['exact_duplicate_payloads'] == 0
    frame = pd.read_csv(OUT/'kaist_features.csv')
    predictions = pd.read_csv(OUT/'kaist_predictions.csv')
    dates = {}
    for item in inventory['records']:
        days = {x.split()[0] for x in item['native_creation_times']}
        assert len(days) == 1
        dates.setdefault(item['family'],set()).update(days)
        assert item['sampling_Hz'] == 25600
        assert item['applied_offset'] == 0
        assert abs(item['applied_multiplier']-1/9.80665) < 1e-12
    assert all(len(x) == 1 for x in dates.values())
    assert len(set.union(*dates.values())) == 3
    assert frame.groupby('recording').size().eq(30).all()
    plot = (ROOT/'IEEE-conference-template-062824/figures/kaist_load_pilot.tex').read_text()
    plotted = re.findall(r'coordinates\s*\{([^}]+)\}',plot)
    assert len(plotted) == 2
    for family,coordinates in zip(['logistic','forest'],plotted):
        values = [(float(x),float(y)) for x,y in re.findall(r'\(([^,]+),([^\)]+)\)',coordinates)]
        expected = [(r['target_load_Nm'],r['metrics']['macro_F1']) for r in result['results'] if r['family'] == family]
        assert len(values) == len(expected) == 3
        assert np.allclose([y for _,y in values],[y for _,y in expected])
        assert np.allclose([x for x,_ in values],[x+(-.1 if family == 'logistic' else .1) for x,_ in expected])
    source_replays = 0
    controls = {}
    for row in result['results']:
        target,family = row['target_load_Nm'], row['family']
        assert target not in row['source_loads_Nm']
        saved = predictions[(predictions.outer_target_load == target)&(predictions.family == family)]
        selected = max(row['source_validation'],key=lambda x:x['mean_macro_F1'])
        assert selected['parameter'] == row['selected_parameter']
        for directed in selected['directed']:
            key = (family,directed['training_load'],directed['validation_load'])
            control = dict(family=family,parameter=selected['parameter'],**directed)
            if key in controls:
                assert controls[key] == control
            controls[key] = control
        for variant in row['source_validation']:
            source = saved[saved.role == 'source_validation']
            source = source[source.parameter.isna()] if variant['parameter'] is None else source[source.parameter == variant['parameter']]
            values = []
            for directed in variant['directed']:
                test = source[source.load_Nm == directed['validation_load']]
                assert len(test) == 210
                assert set(test.training_load) == {directed['training_load']}
                assert not set(test.recording)&set(frame[frame.load_Nm==directed['training_load']].recording)
                replay = metrics(test,test.prediction)
                assert np.isclose(replay['macro_F1'],directed['metrics']['macro_F1'])
                values.append(replay['macro_F1'])
                source_replays += 1
            assert np.isclose(np.mean(values),variant['mean_macro_F1'])
        test = saved[saved.role == 'target']
        assert len(test) == 210 and set(test.load_Nm) == {target}
        assert not set(test.recording)&set(frame[frame.load_Nm.isin(row['source_loads_Nm'])].recording)
        replay = metrics(test,test.prediction)
        for key in row['metrics']:
            assert np.allclose(replay[key],row['metrics'][key]), key
        vote = test.groupby('recording').agg(label=('label','first'),prediction=('prediction',lambda x:int(np.bincount(x,minlength=3).argmax()))).reset_index()
        for key,value in metrics(vote,vote.prediction).items():
            assert np.allclose(value,row['record_vote_metrics'][key])
    record = dict(passed=True,outer_model_results=6,source_validation_replays=source_replays,
        native_records=21,windows=630,feature_count=40,
        creation_dates_by_family={k:sorted(v) for k,v in dates.items()},
        creation_date_scope='Native function creation timestamps; not authenticated acquisition dates or asset IDs',
        directed_controls=list(controls.values()),
        independent_warning_sensitivity_validation=False,
        observed_single_source_healthy_errors=True,
        source_sha256=hashlib.sha256((OUT/'kaist_predictive.json').read_bytes()).hexdigest())
    (OUT/'kaist_verification.json').write_text(json.dumps(record,indent=2)+'\n')
    print('PASS: 6 target results, 36 source-selection replays, 21 native records; creation-date/class coupling documented')


if __name__ == '__main__':
    main()
