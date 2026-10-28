"""Native LaTeX table for retrospective matched acceptable-model warning rates."""
import pdm_audit
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    r=json.loads((ROOT/'experiments/results/matched_population.json').read_text())
    lines=[r'\begin{tabular}{llrrrrl}',r'\toprule',
           r'Learner & Calibration & SHAP, intact & Flip, intact & SHAP, shifted & Flip, shifted & Shifted difference [95\% interval] \\',r'\midrule']
    for x in r['summary']:
        if x['n']!=10000:continue
        a,b=x['scores'];delta=x['flip_minus_shap'][1][2];lo,hi=x['difference_ci95'][1][2]
        row=[x['family'].title(),'Global' if x['method']=='global' else 'Fixed bins']
        row += [f"{100*s['matched_localization'][law][2]:.1f}" for law in [0,1] for s in [a,b]]
        row += [f'{100*delta:.1f} [{100*lo:.1f}, {100*hi:.1f}]']
        lines.append(' & '.join(row)+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (ROOT/'IEEE-conference-template-062824/tables/matched_population.tex').write_text('\n'.join(lines)+'\n')
    print('Wrote matched-population table from saved joint-bootstrap outcomes')


if __name__=='__main__':main()
