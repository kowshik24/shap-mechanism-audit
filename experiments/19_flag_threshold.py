"""
Stage 19: every significant shortcut flag in the stored results, with its excess share, and which
flags a minimum-excess rule (flag only when p_C < alpha and C_m >= threshold) would keep.

Reads experiments/results/*.json from earlier stages; no models are fitted.

Usage: python experiments/19_flag_threshold.py [--threshold 0.05]
"""
import argparse
import json

import pdm_audit  # noqa: F401

from pdm_audit.config import RESULTS_DIR

ap = argparse.ArgumentParser()
ap.add_argument('--threshold', type=float, default=0.05)
args = ap.parse_args()
ALPHA = 0.05

flags = []
for r in json.loads((RESULTS_DIR / 'scale.json').read_text())['rows']:
    for m, p in r['p_holm_S'].items():
        if p < ALPHA:
            planted = r['S_feature'][m] == 'Shortcut sensor'
            flags.append({'source': f'scale n={r["n"]} seed={r["seed"]} sigma={r["sigma"]}', 'mechanism': m,
                          'excess': r['S'][m], 'feature': r['S_feature'][m],
                          'kind': 'planted shortcut' if planted else 'false flag'})


def walk(source, node, path=''):
    """run_audit-style results keep p-values under 'all' and statistics under observed['all']."""
    if not isinstance(node, dict):
        return
    if isinstance(node.get('all'), dict) and 'p_holm_S' in node['all'] and 'observed' in node:
        observed = node['observed']['all']
        for m, p in node['all']['p_holm_S'].items():
            if p < ALPHA:
                flags.append({'source': f'{source}{path}', 'mechanism': m, 'excess': observed[m]['S'],
                              'feature': observed[m]['S_feature'], 'kind': 'audit variant'})
    elif 'p_holm_S' in node and 'observed' in node:  # stage 8, 14 and 15 layout
        for m, p in node['p_holm_S'].items():
            if p < ALPHA:
                flags.append({'source': f'{source}{path}', 'mechanism': m, 'excess': node['observed'][m]['S'],
                              'feature': node['observed'][m]['S_feature'], 'kind': 'audit variant'})
    for key, value in node.items():
        walk(source, value, f'{path}/{key}')


for name in ('compare', 'cwru_audit', 'tep_audit', 'final_checks', 'more_checks', 'twf_diagnostics'):
    path = RESULTS_DIR / f'{name}.json'
    if path.exists():
        walk(name, json.loads(path.read_text()))

for f in flags:
    f['kept'] = f['excess'] >= args.threshold
for f in sorted(flags, key=lambda f: f['excess']):
    print(f'{f["excess"]:.3f} {"kept   " if f["kept"] else "dropped"} {f["mechanism"]:<5} {f["feature"]:<26} '
          f'{f["kind"]:<17} {f["source"]}')
out = {'threshold': args.threshold, 'flags': flags,
       'max_false_flag': max(f['excess'] for f in flags if f['kind'] == 'false flag'),
       'min_planted': min(f['excess'] for f in flags if f['kind'] == 'planted shortcut')}
(RESULTS_DIR / 'flag_threshold.json').write_text(json.dumps(out, indent=1))
print('saved', RESULTS_DIR / 'flag_threshold.json')
