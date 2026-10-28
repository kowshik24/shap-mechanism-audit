"""Export frozen field results without treating them as calibrated warnings."""
import pdm_audit
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'experiments/results/sca_field.json').read_text())
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'View & F1 & Healthy FPR & Recall & Case 11 FPR \\',r'\midrule']
    for x in r['summary']:
        m=x['case_weighted_metrics'];external=next(v for v in x['external'] if v['native_split']=='test')
        row=['12 anchors' if x['view']=='anchors' else '21 features']+[f'{m[k]:.3f}' for k in ['F1','healthy_FPR','recall']]+[f"{external['healthy_FPR']:.3f}"]
        lines.append(' & '.join(row)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'IEEE-conference-template-062824/tables/sca_field.tex').write_text('\n'.join(lines)+'\n')
    source=json.loads((ROOT/'experiments/results/sca_download.json').read_text())
    native=pd.read_csv(ROOT/'experiments/results/sca_features.csv')
    report=['# SCA field study: results and claim boundaries','',
        f"The original version-2 release supplies {len(source['files'])} files totaling {source['total_bytes']:,} bytes. All match the SHA-256 values returned by the official Mendeley API. The extraction retains {r['records']:,} measurements containing {int(native.samples.sum()):,} scalar vibration samples; 198 off/missing-speed records are excluded. Both native files and all sensor positions remain grouped by case.",
        '', 'Sources: [dataset version 2](https://doi.org/10.17632/tdn96mkkpt.2), [source paper](https://doi.org/10.3390/data8070115).',
        '', '## Frozen evaluation','',
        'Cases 1–10 are held out in turn; case 11 never enters fitting or SHAP backgrounds. Two fixed forest views use twelve frequency anchors or all 21 features. Backgrounds contain 96 training-only rows. The released protocol is `experiments/SCA_FIELD_PROTOCOL.md`; experiment 56 verifies native labels and features, folds, fitted predictions, attributions, and all reported counts.',
        '', '## Prediction results','',
        '| View | Case-weighted F1 | Healthy prediction FPR | Fault recall | Case 11 healthy-period FPR | Case 11 test-period FPR |',
        '|---|---:|---:|---:|---:|---:|']
    for x in r['summary']:
        m=x['case_weighted_metrics'];external={a['native_split']:a for a in x['external']}
        report.append('| '+' | '.join([x['view']]+[f'{m[k]:.6f}' for k in ['F1','healthy_FPR','recall']]+[f"{external[s]['healthy_FPR']:.6f}" for s in ['train','test']])+' |')
    report.extend(['','These are classifier prediction errors at threshold .5, not explanation-warning frequencies. Case 11 has 254 retained native training-period and 38 test-period measurements. Both ensembles predict no bearing faults in its 38 test-period measurements. This is one observed non-bearing challenge, not a field false-alarm bound.',
        '', '## Cross-fitted descriptive attribution','',
        'The following pooled summaries combine different held-out fold models and backgrounds. Each case/type contributes equally before normalization. They must not be represented as a mechanism-specificity audit of one fitted function.',
        '', '| View | Recorded fault | Contributing case/types | R | D | S | Maximizer | RPM/fs share |',
        '|---|---|---:|---:|---:|---:|---|---:|'])
    for x in r['summary']:
        for a in x['audit']:
            report.append('| '+' | '.join([x['view'],a['fault'],str(a['own_case_types'])]+[f"{a[k]:.6f}" for k in ['R','D','S']]+[a['S_feature'],f"{a['acquisition_share']:.6f}"])+' |')
    report.extend(['','All ten held-out cases contain only one recorded fault type. Every within-case D/S comparison therefore lacks an other-failure cohort. R remains descriptive when its own cohort has nonzero attribution. Pooling other fold models supplies a different, cross-fitted procedure summary and does not repair the missing within-function comparison.',
        '', 'Even if case independence and complete labels were optimistically assumed, five inner-ring, one ball, and four outer-ring cases permit at most one disjoint three-type comparison batch before reserving any data for training. That is an upper bound, not a certified calibration unit. More measurements from these same cases cannot supply nineteen independent complete batches.',
        '', '## Limits','',
        'Case-held-out is not proof of independent physical bearings or acquisition sessions. Dominant fault labels were assigned from spectral analysis, with inspection when available; they are not exhaustive mechanism annotations. Extra spectral or acquisition features may be legitimate. Higher expected-set share after restricting the feature view does not independently validate physical alignment. There is no industrial calibrated-warning claim, no i.i.d.-measurement confidence interval, and no temporal-attribution result.',
        '', '## Reproduction','',
        'Run `PYTHON=.venv/bin/python bash reproduce_field_study.sh`. The first run downloads 835 MB of public files; subsequent runs verify and reuse them. To verify saved results without fitting or network access, run `.venv/bin/python experiments/56_verify_sca_field.py`.', ''])
    (ROOT/'IEEE-conference-template-062824/SCA_FIELD_REPORT.md').write_text('\n'.join(report))
    print('Exported descriptive field prediction table')


if __name__=='__main__':main()
