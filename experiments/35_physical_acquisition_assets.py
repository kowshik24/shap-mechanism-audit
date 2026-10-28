"""Export complete physical/acquisition checks without selecting favorable captures."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    r = json.loads((ROOT/'experiments/results/paderborn_physical_acquisition.json').read_text())
    assert r['complete'] and len(r['records']) == 160
    spectra, captures = [], []
    for x in r['records']:
        captures.append({'file': x['file'], 'bearing_id': x['bearing_id'], 'condition': x['condition'],
                         'native_DateTime': x['native_DateTime'], 'measurement_id': x['measurement_id'],
                         'vibration_sha256': x['channel_sha256']['vibration_1'], 'all_Y_sha256': x['all_Y_sha256'],
                         'nominal_rpm': x['nominal_setting']['speed_rpm'],
                         'native_speed_mean_unverified_unit': x['channel_summaries']['speed']['mean_native_value'],
                         'native_force_mean_unverified_unit': x['channel_summaries']['force']['mean_native_value'],
                         'native_torque_mean_unverified_unit': x['channel_summaries']['torque']['mean_native_value'],
                         'vibration_span_rate_Hz': x['channel_summaries']['vibration_1']['span_rate_Hz'],
                         'vibration_n': x['channel_summaries']['vibration_1']['n'],
                         'physical_components': '+'.join(x['physical_components_from_profile'])})
        for variant in ['primary_anchors', 'timestamp_anchors']:
            for loc in ['IR', 'OR']:
                for line in x[variant][loc]['harmonics']:
                    spectra.append({'file': x['file'], 'bearing_id': x['bearing_id'], 'condition': x['condition'],
                                    'variant': variant, 'candidate_location': loc,
                                    'component_recorded_in_profile': loc in x['physical_components_from_profile'], **line})
    for name, rows in [('paderborn_capture_metadata.csv', captures), ('paderborn_physical_spectra.csv', spectra)]:
        with (ROOT/'experiments/results'/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    assert len(spectra) == 1920
    print('EXPORT: all 160 capture rows and 1,920 harmonic rows; no outcome-based filtering')


if __name__ == '__main__':
    main()
