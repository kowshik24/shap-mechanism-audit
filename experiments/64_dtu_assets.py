"""Export prospective continuous-study results without empirical size matching."""
import pdm_audit
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results';PAPER=ROOT/'IEEE-conference-template-062824'


def main():
    r=json.loads((OUT/'dtu_prospective.json').read_text())
    proof=json.loads((OUT/'dtu_prospective_verification.json').read_text());assert proof['verified']
    extraction=json.loads((OUT/'dtu_prospective_source.json').read_text())
    (PAPER/'tables/dtu_runtime.tex').write_text(
        '\\newcommand{\\dtuModelRuntime}{'+f"{r['seconds']:.1f}"+'}\n'+
        '\\newcommand{\\dtuSourceRuntime}{'+f"{extraction['seconds']:.0f}"+'}\n')
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Condition & F1 & BA & SHAP W/L & Response W/L \\',r'\midrule']
    report=['# Prospective continuous-process audit','',
        'Source: [DTU release, version 1](https://doi.org/10.11583/DTU.13385936.v1). Settings were fixed in `experiments/DTU_PROSPECTIVE_PROTOCOL.md` before extracting evaluation trajectories or fitting outcomes. This is a simulation experiment conditional on a frozen training corpus; it is not a physical field certificate.','',
        '## Units and calibration','',
        'The extraction retains 400 complete 100-hour IDV4/IDV5 trajectories, paired conservatively into 200 run-number packets. Each packet supplies 128 selected time points and 53 process variables. The entire run stays in one role. Forty packets train the models, 100 calibrate independently seeded fit/packet scores, and 60 evaluate new fits/packets. All 400 full pre-fault prefixes differ; unequal prefixes do not prove independence. Exchangeability relies on the specified simulation replicate law. The full 23.9-GB source file is not hashed locally; retained byte ranges and extracted arrays are.','',
        f"Each score uses rank 96 of 100 calibration pairs at 5% nominal level. Cuts: SHAP {r['cuts'][0]:.9f}; median-response {r['cuts'][1]:.9f}. The separate 95% calibration-bank tolerance upper bound is {100*r['PAC_95_warning_upper']:.2f}%. This is distinct from the marginal rank guarantee and from evaluation binomial intervals. All 100 calibration models have distinct training-probe probability vectors.",
        '', '## Prospective results','',
        '| Condition | F1 | Balanced accuracy | SHAP warnings/localizations | Response warnings/localizations |','|---|---:|---:|---:|---:|']
    for view,label in [('unaltered','Unaltered'),('matched','Biased, matched'),('removed','Bias removed'),('reversed','Bias reversed')]:
        a,b=[next(x for x in r['summary'] if x['score']==kind and x['view']==view) for kind in ['shap','response']]
        m=a['mean_metrics'];values=[label,f"{m['F1']:.3f}",f"{m['balanced_accuracy']:.3f}",f"{a['warning_count']}/{a['localization_count']}",f"{b['warning_count']}/{b['localization_count']}"]
        lines.append(' & '.join(values)+r' \\');report.append('| '+' | '.join(values)+' |')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (PAPER/'tables/dtu_prospective.tex').write_text('\n'.join(lines)+'\n')
    report.extend(['','All counts have denominator 60; localization identifies Compressor Work as a maximizing outside feature above the same cut. No thresholds are chosen to match observed evaluation warning rates. Zero SHAP warnings on unaltered fits has an exact 95% binomial interval [0, 5.96%]; 1/60 response warnings has [0.042%, 8.94%]. For 60/60, the interval is [94.04%, 100%]. These intervals condition on the realized bank and stipulated independent replicate pairs. There is no claim that the two scores have equal population size.','',
        'The label-assigned 2-IQR offset is a deliberate acquisition-information pressure test, not a natural anomaly in the released simulator. Matched, removed and reversed conditions use the same biased model. Healthy/fault prevalence is 25%/75%; an all-positive predictor has F1 0.857 and balanced accuracy 0.5. The unaltered model therefore has lower F1 than that trivial predictor, but higher balanced accuracy (0.763). The injected matched F1 0.965 is not a diagnostic improvement.','',
        '## Descriptive partial-map audit','',
        'Expected sets contain three reactor cooling variables for IDV4 and three separator/condenser cooling variables for IDV5. These maps are partial component correspondences, not exhaustive causal truth. Fault labels come from simulation injection metadata and time, rather than threshold rules on the model inputs. Faults are mutually exclusive, so overlap stratification is not exercised. The following medians aggregate the 60 fitted-model/packet pairs; no mechanism-truth interpretation is attached.','',
        '| Condition | Fault | R median | D median | S median |','|---|---|---:|---:|---:|'])
    for view in ['unaltered','matched','removed','reversed']:
        selected=[next(x for x in record['results'] if x['view']==view)['scores'][0]['statistics'] for record in r['records'] if record['role']==('evaluation' if view=='unaltered' else 'biased')]
        for fault in [4,5]:
            stats=[next(x for x in z if x['mechanism']==fault) for z in selected]
            report.append('| '+view+' | IDV'+str(fault)+' | '+' | '.join(f'{np.median([z[k] for z in stats]):.4f}' for k in ['R','D','S'])+' |')
    report.extend(['','## Numerical verification','',
        'The installed TreeSHAP interventional routine rounds split thresholds to float32. A double midpoint rounded upward produced a 0.00537 additivity discrepancy before the calibration bank completed. The implementation now explains a copy with each split floored to the float32 lattice. Because sklearn casts prediction inputs to float32, this preserves every split and every coalition prediction, without changing fitted models. Additivity tolerances remain 1e-6. See `experiments/DTU_PRECISION_NOTE.md`. Experiment 61 independently verifies all saved model predictions, all median-response interventions, scores, cuts, counts and intervals, plus representative refits and SHAP replays. Experiment 63 recomputes every earlier CWRU/SCA attribution using the same encoding; all are exactly unchanged.'])
    (PAPER/'DTU_PROSPECTIVE_REPORT.md').write_text('\n'.join(report)+'\n')
    print('Exported prospective table and report')


if __name__=='__main__':main()
