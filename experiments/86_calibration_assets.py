"""Export complete harm-screen rates and the manuscript's compact comparison."""
import pdm_audit
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
PAPER = ROOT/'IEEE-conference-template-062824'


def cell(value, interval):
    return f'{100*value:.1f} [{100*interval[0]:.1f}, {100*interval[1]:.1f}]'


def main():
    data = json.loads((OUT/'harm_screening.json').read_text())
    labels = dict(SHAP_warning='SHAP warning', flip_warning='Flip warning', permutation_F1='Permutation F1')
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Learner & Screen & Intact $H^+$ & Intact $H^-$ & Shifted $H^+$ & Shifted $H^-$ \\', r'\midrule']
    exported = []
    for row in data['summary']:
        for law in range(2):
            for group in range(2):
                for j, rate in enumerate(data['rates']):
                    lo, hi = row['ci95'][law][group][j]
                    exported.append(dict(n=row['n'], family=row['family'], score=row['score'],
                        knows_context=row['knows_context'], law=['intact','independent'][law],
                        harm=['H>=0.05','H<0.05'][group], matched_warning_rate=rate,
                        frequency=row['rates'][law][group][j], lower=lo, upper=hi))
        if row['n'] == 10000:
            cells = [cell(row['rates'][law][group][2], row['ci95'][law][group][2])
                     for law in range(2) for group in range(2)]
            lines.append(row['family'].title()+' & '+labels[row['score']]+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    table = PAPER/'tables/harm_screening.tex'
    table.write_text('\n'.join(lines)+'\n')
    with (OUT/'harm_screening_rates.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    contrasts = []
    for row in data['differences']:
        for law in range(2):
            for group in range(2):
                for j, rate in enumerate(data['rates']):
                    lo, hi = row['ci95'][law][group][j]
                    contrasts.append(dict(n=row['n'], family=row['family'], law=['intact','independent'][law],
                        harm=['H>=0.05','H<0.05'][group], matched_warning_rate=rate,
                        permutation_minus_SHAP_pp=100*row['rates'][law][group][j],
                        lower_pp=100*lo, upper_pp=100*hi))
    with (OUT/'harm_screening_contrasts.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)
    files = [Path(__file__), OUT/'harm_screening.json', table, OUT/'harm_screening_rates.csv', OUT/'harm_screening_contrasts.csv']
    (OUT/'calibration_assets.json').write_text(json.dumps(dict(complete=True, rows=len(exported),
        contrasts=len(contrasts), sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
    print('Exported 360 screen-rate cells and 120 paired contrasts; generated manuscript table')


if __name__ == '__main__':
    main()
