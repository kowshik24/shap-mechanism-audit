"""Evidence-bound manuscript tables and complete repository follow-up report."""
import pdm_audit
import json
import re
from pathlib import Path
import numpy as np
from pdm_audit.reference_transfer import scores
from pdm_audit.conditional_audit import observed_overlap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'
PAPER = ROOT/'IEEE-conference-template-062824'


def table(name, columns, header, rows):
    text = [r'\begin{tabular}{'+columns+'}', r'\toprule', header+r' \\', r'\midrule']
    text += [' & '.join(row)+r' \\' for row in rows]
    text += [r'\bottomrule', r'\end{tabular}']
    (PAPER/'tables'/name).write_text('\n'.join(text)+'\n')


def interval(value, ci):
    return f'{100*value:.1f} [{100*ci[0]:.1f}, {100*ci[1]:.1f}]'


def main():
    reference = json.loads((OUT/'reference_transfer.json').read_text())
    composite = json.loads((OUT/'composite_reference.json').read_text())
    rows = []
    for n in [1000, 10000]:
        for family in ['tree', 'forest', 'logistic']:
            x = next(x for x in reference['summary'] if
                     (x['n'], x['family'], x['score'], x.get('method')) == (n, family, 'shap', 'fixed3'))
            c = next(x for x in composite['summary'] if
                     (x['n'], x['family'], x['score'], x['method']) == (n, family, 'shap', 'fixed3'))
            cross = [a['mean']*100 for a in x['aligned_warning'][1:]]
            rows.append([f'{n:,}', family.title(), f"{100*x['self_warning']['mean']:.2f}",
                         f"{100*x['aligned_warning'][0]['mean']:.2f}", f'{min(cross):.2f}--{max(cross):.2f}',
                         f"{100*c['null_warning']['mean']:.2f}", f"{100*c['abstention']['mean']:.2f}"])
    table('transfer_warning.tex', 'llrrrrr',
          r'$n$ & Learner & Fixed $f$ & New fit, same family & New fit, other families & Population warning & Population abstain', rows)
    matched = json.loads((OUT/'matched_population.json').read_text())
    rows = []
    for x in matched['summary']:
        if x['n'] != 10000:
            continue
        a, b = x['scores']
        rows.append([x['family'].title(), 'Global' if x['method'] == 'global' else 'Fixed bins',
                     *[f"{100*s['matched_localization'][0][2]:.1f}" for s in [a, b]],
                     *[interval(x['flip_minus_shap'][law][2], x['difference_ci95'][law][2]) for law in [0, 1]]])
    table('matched_population.tex', 'llrrll',
          r'Learner & Calibration & SHAP, intact & Flip, intact & Intact difference [95\%] & Shifted difference [95\%]', rows)
    design = json.loads((OUT/'score_design.json').read_text())
    keys = ['batch_share_failure', 'batch_raw_failure', 'frozen_share_failure', 'frozen_raw_failure',
            'batch_share_healthy', 'batch_raw_healthy', 'frozen_share_healthy', 'frozen_raw_healthy',
            'batch_targeted_C', 'frozen_targeted_C', 'flip_failure', 'response_targeted_C']
    labels = ['Batch / share / other fault', 'Batch / raw / other fault', 'Frozen / share / other fault',
              'Frozen / raw / other fault', 'Batch / share / healthy', 'Batch / raw / healthy',
              'Frozen / share / healthy', 'Frozen / raw / healthy', 'Batch / targeted SHAP',
              'Frozen / targeted SHAP', 'Flip / other fault', 'Targeted flip']
    rows = []
    for key, label in zip(keys, labels):
        row = [label]
        for family in ['tree', 'forest', 'logistic']:
            x = next(x for x in design['summary'] if (x['family'], x['score'], x['partition']) == (family, key, 'global'))
            intact, shift = x['matched_localization']
            row += [f'{100*intact[0]:.1f}/{100*intact[1]:.1f}', f'{100*shift[0]:.1f}/{100*shift[1]:.1f}']
        rows.append(row)
    table('score_design.tex', 'lrrrrrr',
          r'Design & Tree, intact & Tree, shifted & Forest, intact & Forest, shifted & Logistic, intact & Logistic, shifted', rows)
    dtu = json.loads((OUT/'dtu_dose_response.json').read_text())
    baseline = json.loads((OUT/'dtu_predictive_baselines.json').read_text())
    rows = []
    for kind in ['common', 'IDV4']:
        for dose in [.1, .25, .5, 1., 2.]:
            selected = [next(x for x in dtu['summary'] if (x['score'], x['kind'], x['dose'], x['view']) ==
                             (score, kind, dose, view)) for view in ['matched', 'removed'] for score in ['shap', 'response']]
            f1 = [x['mean_metrics']['F1'] for x in selected[::2]]
            rows.append(['All faults' if kind == 'common' else 'IDV4 only', f'{dose:g}',
                         *[str(x['localizations']) for x in selected], f'{f1[0]:.3f}/{f1[1]:.3f}'])
    table('dtu_dose_response.tex', 'lrrrrrl',
          r'Offset cohort & IQR & SHAP, match & Resp., match & SHAP, removed & Resp., removed & F1 match/removed', rows)
    # Fix bottom-row denominator labels without altering numerical curves.
    figure = PAPER/'figures/transfer_harm.tex'
    text = figure.read_text()
    sections = text.split(r'\nextgroupplot')
    for i in range(4, 7):
        sections[i] = re.sub(r'\$N_H=\d+\$', '$N=200$', sections[i])
    figure.write_text(r'\nextgroupplot'.join(sections))
    # All-reference bands replace the single-reference diagnostic.
    reference = json.loads((OUT/'reference_transfer.json').read_text())
    archive = np.load(OUT/'observed_overlap_draws.npz')
    lines = [r'\begin{tikzpicture}', r'\begin{axis}[width=0.93\columnwidth,height=0.57\columnwidth,',
             r'xmin=0.8,xmax=1,xtick={0.8,0.9,1},ymin=0,ymax=0.35,grid=major,grid style={gray!15},',
             r'xlabel={Observed overlap $O$},ylabel={Mean null score $A_f$},',
             r'legend style={font=\scriptsize,at={(0.02,0.98)},anchor=north west,draw=none},',
             r'tick label style={font=\footnotesize,/pgf/number format/fixed},label style={font=\footnotesize}]']
    diagnostic = []
    edges = np.linspace(.8, 1, 11)
    for family, color, marker in zip(['tree', 'forest', 'logistic'], ['black', 'auditblue', 'auditorange'], ['*', 'square*', 'triangle*']):
        means = []
        for row in reference['records']:
            if (row['n'], row['family']) != (10000, family):
                continue
            counts = archive['target_counts'][row['array_index']]
            overlap = observed_overlap(counts)
            a = scores(row['reference_table'], counts)[0]
            means.append([a[(overlap >= lo) & (overlap < hi if hi < 1 else overlap <= hi)].mean()
                          for lo, hi in zip(edges[:-1], edges[1:])])
        means = np.asarray(means)
        center = means.mean(axis=0)
        lower, upper = np.quantile(means, [.025, .975], axis=0)
        x = (edges[1:]+edges[:-1])/2
        polygon = list(zip(x, lower))+list(zip(x[::-1], upper[::-1]))
        lines.append(r'\addplot[draw=none,fill='+color+r',fill opacity=0.12,forget plot] coordinates {'+
                     ' '.join(f'({a:.17g},{b:.17g})' for a, b in polygon)+r'} \closedcycle;')
        lines.append(r'\addplot+[color='+color+',mark='+marker+r',line width=0.8pt] coordinates {'+
                     ' '.join(f'({a:.17g},{b:.17g})' for a, b in zip(x, center))+'};')
        lines.append(r'\addlegendentry{'+family.title()+'}')
        diagnostic.append({'family': family, 'reference_means': means.tolist(), 'center': center.tolist(),
                           'lower': lower.tolist(), 'upper': upper.tolist(), 'x': x.tolist()})
    lines += [r'\end{axis}', r'\end{tikzpicture}']
    (PAPER/'figures/overlap_score.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'review4_plot_data.json').write_text(json.dumps(diagnostic, indent=2)+'\n')
    report = ['# Fourth-review evidence and revisions', '',
              'These follow-ups are exploratory. The protocol was recorded before their execution; it does not retroactively preregister earlier claims.', '',
              '## Review coverage', '',
              '- M1: Added both intact difference intervals and shifted intervals. Abstract and conclusion now report the five-of-six intact SHAP advantage. Defined the healthy-row blind spot at the score definition and tested healthy cohorts and targeted SHAP.',
              '- M2: Added harmful/no-harm localization for every score and law at matched empirical size, including both original partitions. Retained sensitivity is no longer presented as harm discrimination.',
              '- M3: Ran the complete background × normalization × cohort factorial on the existing 600 large-batch pairs, with the same disjoint calibration fits and paired bank/case bootstrap.',
              '- M4: Added primary-source positioning for underspecification, model class reliance, predictive multiplicity, explanatory multiplicity, explanation disagreement, and harmful-shift monitoring. Corrected reviewer bibliographic shorthand: Brunet et al. is NeurIPS 2022; Krishna et al. is TMLR 2024.',
              '- M5: Made reference scope the primary question, added a reproducible audit recipe, removed secondary measured-study tables/method detail and chamber material from the submission, and demonstrated non-vacuous bounds with known Boolean exclusions. No physical exclusions are asserted.',
              '- M6: Added all five offset doses, common-fault/IDV4-only comparisons, training-packet resampling, training-only model selection, an F1-selected predictive comparator, both null cuts, and localization by mechanism. Mild-bias sensitivity and field validity remain limited.',
              '- M7: Explicitly labeled all headline and follow-up analyses exploratory. The existing Holm family remains a secondary bin-effect test; no retrospective confirmatory designation was introduced.',
              '- Minor issues: Added same/cross-family ranges, tie-rule scope, composite abstention, missing notation, two-sided CWRU swap resolution, DTU cut values, all-reference plot bands, PdM/time-series surveys, and clearer AI-use attribution. Removed uncited bibliography entries and revision-history prose. Future work is folded into the conclusion.', '',
              '## Complete matched-size score comparisons', '',
              'All values are percentages. Each cell reports harmful / no-harm localization with separate 95% paired bank/case percentile intervals. The common empirical acceptable-model warning size is 5%. Harm labels and 400 paired resamples are reused from the existing study. Global and fixed-bin results are distinguished. No-harm means realized F1 loss below 0.05, not absence of any possible deployment harm.', '',
              '| Learner | Score | Partition | Intact harmful | Intact no-harm | Shifted harmful | Shifted no-harm | Nominal null warnings | Abstention |',
              '|---|---|---|---|---|---|---|---|---|']
    for x in design['summary']:
        cells = [interval(x['matched_localization'][law][group], x['matched_ci95'][law][group])
                 for law in range(2) for group in range(2)]
        report.append('| '+' | '.join([x['family'], x['score'], x['partition'], *cells,
                                       f"{100*x['nominal_null_warning']:.2f}%", f"{100*x['abstention']:.2f}%"] )+' |')
    report += ['', '## Incomplete maps with known Boolean exclusions', '',
               'P_m={Z_m}, N_m={C}, with the remaining mechanism inputs unresolved. Upper calibration bounds and lower candidate bounds use the same global population reference at rank 191/200. These are Boolean design facts, not certified physical exclusions.', '']
    for x in design['partial_map']:
        report.append(f"- {x['family']}: finite cut {x['cut']:.6f}; finite lower bounds {100*x['finite_lower_fraction']:.1f}%; null warnings {100*x['null_warning']:.2f}%; intact candidate warnings {100*x['candidate_warning']:.2f}%; harmful candidate warnings {100*x['candidate_harm_warning']:.2f}%.")
    report += ['', '## DTU training-only model selection', '',
               'Configurations are (trees, depth, minimum leaf, class weight). Selection maximizes packet-held-out balanced accuracy on training packets 31–40 after fitting packets 1–30. Evaluation packets do not choose configuration.', '']
    report += [f"- {x['config']}: validation BA {x['balanced_accuracy']:.4f}, F1 {x['F1']:.4f}." for x in dtu['selection']]
    report += [f"- Selected: {dtu['chosen_config']}. All-positive baseline F1=0.857143, BA=0.5.", '',
               f"- Training-only F1-selected unaltered comparator: F1={baseline['mean_metrics']['F1']:.6f}, BA={baseline['mean_metrics']['balanced_accuracy']:.6f}. This improves F1 while sacrificing healthy-row specificity; it is not the calibrated audit model.", '',
               f"Frozen unaltered rank-96 cuts: SHAP {dtu['cuts'][0]:.8f}; response {dtu['cuts'][1]:.8f}.", '',
               '| Score | Offset cohort | Dose (IQR) | View | Warnings /60 | Localization /60 [95%] | IDV4 / IDV5 localization | F1 | BA |',
               '|---|---|---|---|---|---|---|---|---|']
    for x in dtu['summary']:
        lo, hi = x['localization_ci95']
        report.append(f"| {x['score']} | {x['kind']} | {x['dose']} | {x['view']} | {x['warnings']} | {x['localizations']}/60 [{100*lo:.2f}%, {100*hi:.2f}%] | {x['localization_by_mechanism']['4']} / {x['localization_by_mechanism']['5']} | {x['mean_metrics']['F1']:.3f} | {x['mean_metrics']['balanced_accuracy']:.3f} |")
    report += ['', '## Scope and submission status', '',
               '- The original intact SHAP advantage is now reported alongside post-shift differences. A failure-only contrast cannot directly measure healthy-row errors.',
               '- Frozen backgrounds and raw SHAP change the outcome. This prevents attributing the original failure to SHAP as a method.',
               '- Targeted SHAP and targeted responses detect context use in many no-harm models. Healthy cohorts do not universally repair the score.',
               '- DTU remains a simulation with label-assigned offsets. Training-packet resampling expands the empirical training recipe; it does not supply independent plants.',
               '- IEEE BigData 2026 official main CFP specifies 10 pages including references, no appendix, and single-blind review: https://bigdataieee.org/BigData2026/calls/papers/',
               '- User supplied the named session CFP: submission October 30, decision November 10, camera-ready November 14. Its live portal could not be independently fetched.',
               '- Read-only GitHub API check on October 25: linked repository is private; pushed_at=2026-10-21T18:16:56Z. Untracked local review files are not in HEAD. No public release or remote write was performed.',
               '- Secondary measured-study details are retained in SECONDARY_MEASURED_STUDIES.tex.txt. They are not a submitted appendix.',
               '- An author-controlled public artifact release or reviewer-accessible archive remains necessary for the reproducibility claim. No change of repository visibility is authorized by these edits.', '']
    (PAPER/'REVIEW4_EVIDENCE_REPORT.md').write_text('\n'.join(report))
    print('Wrote fourth-review tables, all-reference bands, and complete evidence report')


if __name__ == '__main__':
    main()
