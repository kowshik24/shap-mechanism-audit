"""Offline checks of cached native inventories; prepare a bounded sensor-study plan."""
import pdm_audit
import hashlib
import json
from collections import Counter
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/dataset_scout'
OUT = ROOT/'experiments/results'


def main():
    vibration = json.loads((DATA/'kaist_vibration_inventory.json').read_text())
    acoustic = json.loads((DATA/'kaist_acoustic_inventory.json').read_text())
    files, loads, conditions, subset = [], Counter(), Counter(), []
    for item in vibration['members']:
        match = re.fullmatch(r'([024])Nm_(.+)\.mat',item['name'])
        assert match,item['name']
        load,case = int(match[1]),match[2]
        canonical = case.replace('Unbalalnce','Unbalance')
        condition = canonical.split('_')[0]
        assert condition in ['Normal','BPFI','BPFO','Misalign','Unbalance']
        files.append(dict(native_filename=item['name'],load_Nm=load,case=canonical,
            family=condition,bytes=item['bytes'],crc=item['crc'],
            filename_alias_applied=case != canonical))
        loads[load] += 1
        conditions[condition] += 1
        if condition in ['Normal','BPFI','BPFO']:
            subset.append(files[-1])
    assert len(files) == 45 and dict(loads) == {0:15,2:15,4:15}
    assert len(subset) == 21
    assert len(acoustic['members']) == 5 and all(x['name'].startswith('0Nm_') for x in acoustic['members'])
    profiles = np.loadtxt(DATA/'profile.txt',dtype=int)
    assert profiles.shape == (2205,5)
    normal = np.all(profiles[:,:4] == [100,100,0,130],axis=1)
    stable = profiles[:,4] == 0
    native = dict(cycles=len(profiles),unique_component_profiles=len(np.unique(profiles[:,:4],axis=0)),
        stable_cycles=int(stable.sum()),fully_normal_cycles=int(normal.sum()),
        stable_fully_normal_cycles=int((normal&stable).sum()),
        fully_normal_contiguous_episodes=int(np.sum(np.diff(np.r_[False,normal].astype(int)) == 1)))
    assert native == dict(cycles=2205,unique_component_profiles=144,stable_cycles=1449,
        fully_normal_cycles=21,stable_fully_normal_cycles=10,fully_normal_contiguous_episodes=3)
    plan = dict(stage='Metadata-ready; vibration values, acquisition independence and predictive outcomes not validated',
        source='https://data.mendeley.com/datasets/ztmf3m7h5x/6',
        primary_task='Within-rig diagnosis under held-out measured torque load',
        native_subset=subset,all_vibration_records=files,
        folds=[dict(test_load_Nm=test,source_loads_Nm=[x for x in [0,2,4] if x != test]) for test in [0,2,4]],
        targets=['three-class: normal / inner race / outer race','binary: normal / any bearing fault'],
        primary_metrics=['macro F1 over diagnosis classes','balanced accuracy for normal/fault','healthy false-positive rate'],
        pooling='Each native recording receives equal weight within its diagnosis class; windows are not independent assets',
        acquisition_bias='Optional prespecified training-sample selection by native load and label; no label-assigned offsets. Such selection is a designed acquisition-bias challenge, not proof of a naturally occurring production confound.',
        prohibitions=['No random window splits across the same native recording',
            'No target-load-based feature, background, model or threshold selection',
            'No claim that 45 records or their windows are 45 independent bearings',
            'No new-asset 5% conformal claim from this inventory',
            'No exhaustive physical feature map without verified geometry and acquisition metadata'],
        next_gates=['Download and authenticate the full 2.66 GB vibration archive, or explicitly distinguish authenticated range objects from full-archive authentication',
            'Verify native time axes, sampling rates, channel identities, repeated payloads and acquisition/asset IDs',
            'Fix feature extraction and source-only model-selection rules before inspecting target signals',
            'Run all three held-out-load folds and retain negative or zero-loss outcomes',
            'Attempt mechanism-level auditing only after expected-feature and null-population assumptions are justified'])
    (ROOT/'experiments/KAIST_LOAD_PILOT_CONTRACT.json').write_text(json.dumps(plan,indent=2)+'\n')
    files_to_hash = [Path(__file__),DATA/'kaist_vibration_inventory.json',DATA/'kaist_acoustic_inventory.json',
        DATA/'kaist_load_files.json',DATA/'profile.txt',DATA/'description.txt',
        ROOT/'experiments/KAIST_LOAD_PILOT_CONTRACT.json']
    result = dict(verified=True,kaist_load_counts=dict(loads),kaist_family_counts=dict(conditions),
        kaist_vibration_records=45,kaist_vibration_whole_archive_authenticated=False,
        kaist_vibration_metadata_bytes=vibration['metadata_bytes_fetched'],
        kaist_acoustic_records=5,kaist_acoustic_loads=[0],kaist_acoustic_whole_archive_authenticated=True,
        kaist_filename_aliases=sum(x['filename_alias_applied'] for x in files),
        uci_native_profile=native,
        scope='Offline native metadata/profile feasibility; no classifier outcomes or field calibration claims',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files_to_hash})
    (OUT/'dataset_feasibility.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: KAIST 45 vibration records, five zero-load acoustic records; UCI 2205 profiles, 21 fully normal cycles; bounded pilot contract saved')


if __name__ == '__main__':
    main()
