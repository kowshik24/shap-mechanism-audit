"""Export every measured alignment result and a compact manuscript comparison."""
import pdm_audit
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT/'experiments/results/industrial_alignment.json').read_text())
    assert data['complete'] and len(data['records']) == 42
    summary = data['summary']
    path = ROOT/'experiments/results/industrial_alignment_summary.csv'
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    text = [r'\begin{tabular}{llrrrr}', r'\toprule',
            r'Held out & Fault & Full F1 & Anchor F1 & Mask F1 & $\Delta_g$ \\', r'\midrule']
    for axis in ['load_hp', 'diameter_in']:
        for fault in ['B', 'IR', 'OR']:
            a = next(r for r in summary if (r['axis'], r['fault'], r['view']) == (axis, fault, 'own_anchors'))
            b = next(r for r in summary if (r['axis'], r['fault'], r['view']) == (axis, fault, 'all'))
            label = 'Load' if axis == 'load_hp' else 'Diameter'
            text.append(f"{label} & {fault} & {b['pooled_recording_F1']:.3f} & {a['pooled_recording_F1']:.3f} & {a['masked_F1']:.3f} & {b['max_gain_probability_change']:.3f}" + r' \\')
    text += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'IEEE-conference-template-062824/tables/industrial_alignment.tex').write_text('\n'.join(text)+'\n')
    report = '''# Measured-system alignment and operating-region validation

This follow-up is complete as a bounded feasibility study. It does **not**
validate industrial causal alignment or a 5% mechanism-warning rate.

## Metadata evidence

All 36 cached official table rows agree with manifest load, defect location,
diameter, fixed outer-race position, and download URL. Raw-file SHA-256 values
match. MAT-file RPM differs by at most 9 RPM from the approximate official row
value, within the prospectively specified 50-RPM consistency tolerance.
The inventory is balanced: four loads, three locations, three diameters, one
recording per cell. No classifier receives load, fault labels, or diameter as
inputs; RPM only sets feature-band centers. Load is an observed operating label,
not an inferred latent regime. Calibration/test population exchangeability does
not follow from matching that one label.

Primary provenance:

- [Official recording table](https://engineering.case.edu/bearingdatacenter/12k-drive-end-bearing-fault-data)
- [Official bearing frequencies](https://engineering.case.edu/bearingdatacenter/bearing-information)
- [Apparatus and seeded-fault context](https://engineering.case.edu/bearingdatacenter/welcome)

## Learned-model evidence

Forty-two binary forests cover every combination of outer grouping (four loads
or three diameters), fault location, and feature view. Inner grouped selection
uses only training recordings. All 21 own-anchor fits use only the corresponding
four documented spectral coordinates; exhaustive inspection of their 2,100
trees confirms invariance to excluded coordinates. Gain changes of 0.25 and 4
leave their predictions exactly unchanged. All 21 full-view fits split on
outside-anchor coordinates; this does not make those coordinates spurious.

Load-held-out pooled recording F1 for restricted ball/inner/outer heads is
0.857/1.000/1.000, compared with 1.000 for each full-view head. Diameter-held-out
restricted F1 falls to 0.611/0.800/0.500 (full: 0.500/0.800/0.500).
Replacing own-anchor blocks with training medians changes restricted F1 to
0.500/0/0 for held-out loads and 0.444/0/0 for held-out diameters. This is
off-manifold model reliance, not an intervention on physical bearing faults.
Full-view maximum window-probability change under the two gain interventions
reaches 0.435. No outside-coordinate dependence or prediction change alone is
labelled a mechanism error.

These restricted controls define functional alignment by construction. They do
not independently establish complete physical alignment, and performance failures
are retained rather than filtered to create favorable reference models.

## Calibration-capacity decision

A non-overlapping S-audit batch uses one recording from each fault location at
a fixed load and diameter. The inventory contains 12 such batches, three per
load. Reserving one new test batch leaves at most two matched-load calibration
batches, before training consumes any recordings. Pooling all other loads gives
at most eleven. These are optimistic inventory counts, not demonstrated
independent or exchangeable acquisitions; physical bearing identity is unknown.

At 5%, rank ceil((B+1)*0.95) exceeds B for B=2 and B=11. Their minimum strict-rank
tail resolutions are 1/3 and 1/12. An honest threshold is infinity (abstain).
At least 19 calibration batches are numerically necessary for a finite 5% rank
threshold; that count is not sufficient without aligned fixed functions and
exchangeability. Window resampling, repeated seeds, and synthetic batches cannot
increase the independent measured acquisition inventory.

## Remaining acquisition gate

Full validation needs independently justified mechanism labels/reference
functions and disjoint training, calibration, and test acquisitions with bearing,
machine, load/speed, severity, sensor, and session identifiers. For each proposed
region, collect at least 19 exchangeable calibration batches plus held-out test
batches; substantially more are needed to estimate operating error rates with
useful precision. Predefine whether a batch contains all required mechanisms,
how class composition is controlled, and whether severity/population changes
preserve the calibration law. Do not describe this dataset as field maintenance.

## Reproduction and verification

`experiments/INDUSTRIAL_ALIGNMENT_PROTOCOL.md` precedes the new model outcomes.
Script 28 fits and saves every estimator; script 29 checks official rows,
checksums, partition grids, nested selections, split supports, and replays all
predictions and interventions. Script 30 exports all summary rows and the paper
table. Raw probabilities, metrics, source hashes, and model hashes are saved in
`experiments/results/industrial_alignment.json`; serialized models are retained
under `experiments/results/industrial_alignment_models/`. No historical result
is overwritten. Table values are descriptive for a shared recording inventory;
no independent-model significance test is claimed.
'''
    (ROOT/'IEEE-conference-template-062824/INDUSTRIAL_VALIDATION.md').write_text(report)


if __name__ == '__main__':
    main()
