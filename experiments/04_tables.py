"""
Stage 4: LaTeX tables for the paper, generated from experiments/results/*.json so no number is
transcribed by hand. Writes paper/ieee_bigdata/tables/*.tex (tabular bodies, \\input by main.tex) and
the tables the TUAI workshop paper uses to big_data_workshop_TUAI/tables/. The shortcut statistic is
S_m in the IEEE BigData paper and C_m in the workshop paper (where S_m names the mechanism's rows), so
headers carry the placeholder @SC@ and each target substitutes its own symbol.
"""
import argparse
import json

import numpy as np

from pdm_audit.audit import holm
from pdm_audit.config import PHYSICAL, RESULTS_DIR, TABLES_DIR, WORKSHOP_TABLES_DIR

bench = json.loads((RESULTS_DIR / 'benchmark.json').read_text())
audit = json.loads((RESULTS_DIR / 'audit.json').read_text())
sweep = json.loads((RESULTS_DIR / 'shortcut_sweep.json').read_text())
explainers = json.loads((RESULTS_DIR / 'explainers.json').read_text())
scale = json.loads((RESULTS_DIR / 'scale.json').read_text())
retuned = json.loads((RESULTS_DIR / 'retuned_null.json').read_text())
effects = json.loads((RESULTS_DIR / 'effect_sizes.json').read_text())['mechanisms']
twf_diag = json.loads((RESULTS_DIR / 'twf_diagnostics.json').read_text())
other = json.loads((RESULTS_DIR / 'other_explainers.json').read_text())
review = json.loads((RESULTS_DIR / 'review_checks.json').read_text())
review_kernel_path = RESULTS_DIR / 'review_checks_kernel.json'
review_kernel = json.loads(review_kernel_path.read_text()) if review_kernel_path.exists() else None
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SHORT = {'Tool wear [min]': 'tool wear', 'Rotational speed [rpm]': 'speed', 'Power [W]': 'power',
         'Overstrain_ratio': 'overstrain ratio', 'Torque [Nm]': 'torque', 'Temp_diff [K]': 'temp.\\ difference',
         'Air temperature [K]': 'air temp.', 'Process temperature [K]': 'process temp.',
         'Overstrain [minNm]': 'overstrain', 'Shortcut sensor': 'shortcut sensor'}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--workshop-only', action='store_true', help='regenerate only TUAI paper and supplement tables')
WORKSHOP_ONLY = parser.parse_args().workshop_only

WORKSHOP_TABLES = {'models', 'config', 'audit', 'scale', 'cwru'}
WORKSHOP_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def write(name, lines, workshop_lines=None):
    """`workshop_lines` replaces `lines` in the workshop copy (for rows only that paper reports)."""
    text = '\n'.join(lines) + '\n'
    if not WORKSHOP_ONLY:
        (TABLES_DIR / f'{name}.tex').write_text(text.replace('@SC@', 'S'))
    if name in WORKSHOP_TABLES:
        text = text if workshop_lines is None else '\n'.join(workshop_lines) + '\n'
        (WORKSHOP_TABLES_DIR / f'{name}.tex').write_text(text.replace('@SC@', 'C'))
    print('wrote', name)


def write_ws(name, lines):
    """A table only the workshop paper or its supplement uses (nothing written to paper/)."""
    (WORKSHOP_TABLES_DIR / f'{name}.tex').write_text('\n'.join(lines).replace('@SC@', 'C') + '\n')
    print('wrote', name, '(workshop only)')


def verdict(pr, pd_, pc):
    return 'flagged' if pc < 0.05 else ('aligned' if max(pr, pd_) < 0.05 else 'not aligned')


def f2(x):
    return f'{x:.2f}'


def f3(x):
    return f'{x:.3f}'


def pv(p, fail_if_high=True):
    s = f'{p:.3f}'
    if s == '0.050' and p < 0.05:  # keep the pass/fail verdict visible after rounding
        s = f'{p:.4f}'
    return s + '$^{\\times}$' if (fail_if_high and p >= 0.05) else s


# ---- predictive performance
order = sorted(bench['test'], key=lambda k: bench['test'][k]['f1'], reverse=True)
best = {k: max(bench['test'][m][k] for m in order) for k in ('precision', 'recall', 'f1', 'roc_auc')}
lines = [r'\begin{tabular}{@{}lrrlr@{}}', r'\toprule', r'Model & Prec. & Rec. & F1 (95\% CI) & ROC-AUC \\', r'\midrule']
for m in order:
    t = bench['test'][m]
    cell = {k: (r'\textbf{' + (f3 if k == 'roc_auc' else f2)(t[k]) + '}') if t[k] == best[k]
            else (f3 if k == 'roc_auc' else f2)(t[k]) for k in best}
    lo, hi = t['ci_bootstrap']['f1']
    lines.append(f"{m} & {cell['precision']} & {cell['recall']} & {cell['f1']} ({f2(lo)}--{f2(hi)}) & {cell['roc_auc']} \\\\")
write('models', lines + [r'\bottomrule', r'\end{tabular}'])

# ---- comparison with prior work
tuned, cv = bench['test']['XGBoost (tuned)'], bench['cv5_default_threshold']
write('prior', [
    r'\begin{tabular}{@{}llr@{}}', r'\toprule', r'Study and protocol & Metric & Value \\', r'\midrule',
    r'Random forest, single split~\cite{physicsinformed2026} & pos.-class F1 & 0.722 \\',
    f"Ours, single split & pos.-class F1 & {f2(tuned['f1'])} \\\\",
    f"Ours, 5-fold CV & pos.-class F1 & ${f3(cv['f1'][0])} \\pm {f3(cv['f1'][1])}$ \\\\",
    r'\bottomrule', r'\end{tabular}'])

# ---- oversamplers
ov = bench['oversamplers']
bst = {k: max(v[k] for v in ov.values()) for k in ('precision', 'recall', 'f1', 'pr_auc')}
lines = [r'\begin{tabular}{@{}lrrrr@{}}', r'\toprule', r'Sampler & Prec. & Rec. & F1 & PR-AUC \\', r'\midrule']
for name in sorted(ov, key=lambda k: ov[k]['f1'], reverse=True):
    cells = [(r'\textbf{' + f3(ov[name][k]) + '}') if ov[name][k] == bst[k] else f3(ov[name][k]) for k in bst]
    lines.append(f"{name} & " + ' & '.join(cells) + r' \\')
write('resamplers', lines + [r'\bottomrule', r'\end{tabular}'])

# ---- model settings
p = bench['xgb_params']
write('config', [
    r'\begin{tabular}{@{}lp{5.6cm}@{}}', r'\toprule', r'Model & Settings \\', r'\midrule',
    r'Logistic regression & max.\ 1000 iterations \\',
    r'SVM & radial basis function (RBF) kernel, probability outputs \\',
    r'$k$-NN & $k = 7$ \\', r'Decision tree & max.\ depth 10 \\',
    r'Random forest & 300 trees, max.\ depth 12 \\', r'XGBoost & 300 trees, max.\ depth 6, learning rate 0.1 \\',
    f"XGBoost (tuned) & {p['n_estimators']} trees, max.\\ depth {p['max_depth']}, learning rate {p['learning_rate']}, "
    f"subsample {p['subsample']}, column sample {p['colsample_bytree']}, min.\\ child weight {p['min_child_weight']}, "
    f"$\\gamma = {p['gamma']}$ \\\\",
    r'MLP & 64 and 32 hidden units, ReLU, batch norm, dropout 0.3/0.2; Adam (learning rate $10^{-3}$, '
    r'weight decay $10^{-5}$), binary cross-entropy, batch 64, 40 epochs \\',
    r'\bottomrule', r'\end{tabular}'])

# ---- primary audit
prim = audit['primary']
counts = bench['label_facts']['mode_counts']
lines = [r'\begin{tabular}{@{}lrlrrrrrrrrr@{}}', r'\toprule',
         r'& $n$ & Top feature & $R_m$ & Null & $p$ & $D_m$ & Null & $p$ & $@SC@_m$ & Null & $p$ \\', r'\midrule']
for m in PHYSICAL:
    o, a = prim['observed']['all'][m], prim['all']
    lines.append(f"{m} & {counts[m]} & {SHORT.get(o['top_feature'], o['top_feature'])} & "
                 + ' & '.join(f"{f3(o[t])} & {f3(a['null_mean'][f'{m}.{t}'])} & "
                              + (pv(a[f'p_holm_{t}'][m]) if t != 'S' else pv(a['p_holm_S'][m], fail_if_high=False))
                              for t in ('R', 'D', 'S')) + r' \\')
o = o_rnf = prim['observed']['all']['RNF']
lines.append(f"RNF & {counts['RNF']} & {SHORT.get(o['top_feature'], o['top_feature'])} & {f3(o['top_share'])} & "
             f"{f3(prim['all']['null_mean']['RNF.top_share'])} & {f3(prim['all']['p_RNF_more_concentrated'])} & "
             r"-- & -- & -- & -- & -- & -- \\")

# workshop paper: R_m and D_m with 95% bootstrap CIs and null-standardized effect sizes z (stage 10)
ws = [r'\begin{tabular}{@{}lrlrrrrrrrrrrrl@{}}', r'\toprule',
      r'& $n$ & Top feature & $R_m$ [95\% CI] & Null & $z$ & $p$ & $D_m$ [95\% CI] & Null & $z$ & $p$ '
      r'& $@SC@_m$ & Null & $p$ & Verdict \\', r'\midrule']
for m in PHYSICAL:
    o, a = prim['observed']['all'][m], prim['all']
    cells = []
    for t in ('R', 'D'):
        e = effects[m][t]
        cells.append(f"{f3(o[t])} [{f2(e['ci95'][0])}, {f2(e['ci95'][1])}] & {f3(a['null_mean'][f'{m}.{t}'])} & "
                     f"{e['z']:.1f} & {pv(a[f'p_holm_{t}'][m])}")
    cells.append(f"{f3(o['S'])} & {f3(a['null_mean'][f'{m}.S'])} & {pv(a['p_holm_S'][m], fail_if_high=False)} & "
                 + verdict(a['p_holm_R'][m], a['p_holm_D'][m], a['p_holm_S'][m]))
    ws.append(f"{m} & {counts[m]} & {SHORT.get(o['top_feature'], o['top_feature'])} & " + ' & '.join(cells) + r' \\')
ws.append(f"RNF & {counts['RNF']} & {SHORT.get(o_rnf['top_feature'], o_rnf['top_feature'])} & {f3(o_rnf['top_share'])} & "
          f"{f3(prim['all']['null_mean']['RNF.top_share'])} & -- & {f3(prim['all']['p_RNF_more_concentrated'])} & "
          r"-- & -- & -- & -- & -- & -- & -- & -- \\")
write('audit', lines + [r'\bottomrule', r'\end{tabular}'], ws + [r'\bottomrule', r'\end{tabular}'])


def flag_s(p):
    """The shortcut statistic flags a shortcut when p < 0.05: mark those with a dagger."""
    return f'{p:.3f}' + ('$^{\\dagger}$' if p < 0.05 else '')


# ---- robustness across variants (Holm-adjusted p for R, D, S)
iv = explainers['interventional']
variants = [('Primary, seed 42', prim, 'all'), ('\\quad single-mechanism rows', prim, 'single')]
variants += [(f'Seed {s}', audit['seeds'][s], 'all') for s in sorted(audit['seeds'])]
variants += [('Raw features only', audit['raw'], 'all'), ('In-sample SHAP', audit['in_sample'], 'all')]
variants += [(f'Interventional, background {k.split("/")[0]}', iv[k], 'all') for k in iv if k.endswith('/0')]
variants += [('Random forest', explainers['random_forest'], 'all'),
             ('Permuted labels (sanity)', audit['permuted'], 'all')]
lines = [r'\begin{tabular}{@{}lr' + 'rrr' * len(PHYSICAL) + r'@{}}', r'\toprule',
         r'& & ' + ' & '.join(rf'\multicolumn{{3}}{{c}}{{{m}}}' for m in PHYSICAL) + r' \\',
         ''.join(rf'\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}' for i in range(len(PHYSICAL))),
         r'Variant & $K$ & ' + ' & '.join(['$R$ & $D$ & $@SC@$'] * len(PHYSICAL)) + r' \\', r'\midrule']


def variant_row(label, res, grp):
    a = res[grp]
    cells = [f"{pv(a['p_holm_R'][m])} & {pv(a['p_holm_D'][m])} & {flag_s(a['p_holm_S'][m])}" for m in PHYSICAL]
    return f"{label} & {res['K']} & " + ' & '.join(cells) + r' \\'


header = lines
lines = header + [variant_row(*v) for v in variants]
# ---- workshop paper only: every variant of the primary audit, grouped by what it changes
def optional(name):
    path = RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


final, more, dep = optional('final_checks.json'), optional('more_checks.json'), optional('dependence.json')


def row_of(label, src, key, K=None):
    """A table row for result `key` of a stage-13/16/17/18 file, if it exists."""
    if key not in src:
        return []
    res = src[key]
    a = res['all'] if 'all' in res else res
    return [variant_row(label, {'K': K or res.get('K') or src['K'], 'all': a}, 'all')]


def group_row(label, res):
    return f'{label} & {res["K"]} & ' + ' & '.join(
        f"{pv(res['p_holm_R'][m])} & {pv(res['p_holm_D'][m])} & {flag_s(res['p_holm_S'][m])}" for m in PHYSICAL) + r' \\'


seed_res = [audit['seeds'][k] for k in sorted(audit['seeds'])]
seeds_max = {'K': seed_res[0]['K'], 'all': {f'p_holm_{t}': {m: max(r['all'][f'p_holm_{t}'][m] for r in seed_res)
                                                             for m in PHYSICAL} for t in 'RDS'}}
by_label = {v[0]: v for v in variants}
workshop = header + [variant_row(*v) for v in variants[:2]]
workshop.append(variant_row('\\quad null re-tuned per permutation', {'K': retuned['K'], 'all': retuned['retuned']}, 'all'))
workshop += row_of('\\quad positive part of SHAP', final, 'positive')
workshop.append(variant_row('Seeds 1--4 (largest $p$)', seeds_max, 'all'))
workshop += [variant_row(*by_label['Raw features only']), variant_row(*by_label['In-sample SHAP'])]
workshop.append(r'\addlinespace')
workshop += [variant_row(f'{label}', {'K': review['K'], 'all': review[key]['all']}, 'all') for key, label in
             [('no_unflagged', 'Unflagged failures removed'), ('or_labels', 'Label = OR of flags'),
              ('class_weights', 'Class weights, no oversampling')]]
workshop += row_of('Focal loss, no oversampling', final, 'focal')
workshop += row_of('Borderline-SMOTE, 3 neighbours', final, 'bsmote_k3')
workshop += row_of('Borderline-SMOTE, 10 neighbours', final, 'bsmote_k10')
workshop += row_of('SMOTE', more, 'sampler_SMOTE')
workshop += row_of('ADASYN', more, 'sampler_ADASYN')
workshop += row_of('$E_m$ without derived features', more, 'triggers_no_derived')
workshop += row_of('$E_m$ without top trigger', more, 'triggers_drop_top')
workshop.append(r'\addlinespace')
workshop += [variant_row(*v) for v in variants if v[0] == 'Interventional, background 100']
workshop += [variant_row(f'\\quad {label}', twf_diag[key], 'all') for key, label in
             [('failures', 'failure-only background'), ('non_failures', 'non-failure background'),
              ('oversampled', 'oversampled background')]]
workshop += row_of('Pairwise (nearest normal reference)', final, 'pairwise')
workshop += [variant_row(*by_label['Random forest'])]
if dep:
    workshop += [group_row('Grouped TreeSHAP (4 groups)', {'K': dep['K'], **dep['grouped']}),
                 group_row('Conditional group Shapley', {'K': dep['K'], **dep['conditional']})]
workshop += row_of('LIME', final, 'lime')
workshop.append(r'\addlinespace')
workshop += [variant_row('MLP, KernelSHAP', other['mlp_kernel'], 'all')]
if review_kernel:
    workshop.append(variant_row('\\quad exact, background 100',
                                {'K': review_kernel['K'], 'all': review_kernel['kernel_exact']['all']}, 'all'))
workshop += row_of('\\quad exact, background 500', final, 'kernel_bg500')
workshop += row_of('\\quad k-means background', final, 'kernel_kmeans')
workshop += [variant_row('MLP, Integrated Gradients', other['mlp_ig'], 'all')]
workshop += row_of('MLP, DeepSHAP', final, 'mlp_deep') + row_of('MLP, SmoothGrad', final, 'mlp_smooth')
workshop.append(r'\addlinespace')
workshop += [variant_row(*by_label['Permuted labels (sanity)'])]
tail = [r'\midrule']
for label, res in [('Naive null, tuned model', prim), ('Naive null, permuted labels', audit['permuted'])]:
    naive = holm(res['naive_null_p'])
    tail.append(f'{label} & -- & ' + ' & '.join(f"{pv(naive[m])} & -- & --" for m in PHYSICAL) + r' \\')
tail += [r'\bottomrule', r'\end{tabular}']
write('robustness', lines + tail)
write_ws('robustness_full', workshop + tail[-2:])  # supplement; naive rows are in the text

# ---- workshop paper: verdict matrix of every variant (A aligned, x not aligned, ! flagged); the
# deciding Holm-adjusted p-value is shown where the verdict differs from the primary audit or is borderline
def ps(res):
    a = res['all'] if 'all' in res else res
    return {t: a[f'p_holm_{t}'] for t in 'RDS'}


primary_verdict = {m: verdict(prim['all']['p_holm_R'][m], prim['all']['p_holm_D'][m], prim['all']['p_holm_S'][m])
                   for m in PHYSICAL}


def cell(m, p):
    pr, pd_, pc = p['R'][m], p['D'][m], p['S'][m]
    v = verdict(pr, pd_, pc)
    fmt = lambda x: f'{x:.3f}'.lstrip('0') if f'{x:.3f}' != '0.050' else f'{x:.4f}'.lstrip('0')
    sym = {'aligned': r'\checkmark', 'not aligned': r'$\times$', 'flagged': r'\textbf{!}'}[v]
    note = ''
    if v == 'flagged':
        note = f'C {fmt(pc)}'
    elif v == 'not aligned' and v != primary_verdict[m]:
        fails = [(t, x) for t, x in (('R', pr), ('D', pd_)) if x >= 0.05]
        note = ' '.join(f'{t} {fmt(x)}' for t, x in fails) if len(fails) == 1 else ''
    elif v == 'aligned' and 0.03 <= max(pr, pd_) < 0.05:
        t, x = max((('R', pr), ('D', pd_)), key=lambda z: z[1])
        note = f'{t} {fmt(x)}'
    elif v == 'not aligned' and min(pr, pd_) < 0.05:
        t, x = (('R', pr) if pr >= 0.05 else ('D', pd_))
        note = f'{t} {fmt(x)}'
    return sym + (rf'\,{{\tiny {note}}}' if note else '')


GROUPS_V = [
    ('Primary and null', [('Primary audit', prim['K'] if 'K' in prim else audit['K_main'], ps(prim)),
                          ('Single-mechanism rows', audit['K_main'], ps(prim['single'])),
                          ('Seeds 1--4 (largest $p$)', seeds_max['K'], ps(seeds_max)),
                          ('Null re-tuned per permutation', retuned['K'], ps(retuned['retuned']))]),
    ('Data and expected sets', [('Raw features only', audit['raw']['K'], ps(audit['raw'])),
                                ('Unflagged failures removed', review['K'], ps(review['no_unflagged'])),
                                ('Label = OR of flags', review['K'], ps(review['or_labels'])),
                                ('$E_m$ without derived features', more.get('K', 200), ps(more['triggers_no_derived'])),
                                ('$E_m$ without top trigger', more.get('K', 200), ps(more['triggers_drop_top']))]),
    ('Imbalance treatment', [('Class weights', review['K'], ps(review['class_weights'])),
                             ('Focal loss', 200, ps(final['focal'])),
                             ('Borderline-SMOTE, 3 neighbours', 200, ps(final['bsmote_k3'])),
                             ('Borderline-SMOTE, 10 neighbours', 200, ps(final['bsmote_k10'])),
                             ('SMOTE', more.get('K', 200), ps(more['sampler_SMOTE'])),
                             ('ADASYN', more.get('K', 200), ps(more['sampler_ADASYN']))]),
    ('Attribution (XGBoost)', [('In-sample path TreeSHAP', audit['in_sample']['K'], ps(audit['in_sample'])),
                               ('Interventional, training bg.', iv['100/0']['K'], ps(iv['100/0'])),
                               ('\\quad non-failure bg.', twf_diag['non_failures']['K'], ps(twf_diag['non_failures'])),
                               ('\\quad oversampled bg.', twf_diag['oversampled']['K'], ps(twf_diag['oversampled'])),
                               ('\\quad failure-only bg.', twf_diag['failures']['K'], ps(twf_diag['failures'])),
                               ('Pairwise, nearest normal', 200, ps(final['pairwise'])),
                               ('Failure-pushing part only', 200, ps(final['positive'])),
                               ('Grouped path TreeSHAP', dep['K'], ps(dep['grouped'])),
                               ('Conditional group Shapley', dep['K'], ps(dep['conditional']))]
                              + ([('LIME', final['lime'].get('K', 200), ps(final['lime']))] if 'lime' in final else [])),
    ('Other models', [('Random forest, path TreeSHAP', explainers['random_forest']['K'], ps(explainers['random_forest'])),
                      ('MLP, KernelSHAP', other['mlp_kernel']['K'], ps(other['mlp_kernel'])),
                      ('\\quad exact, bg.\\ 100', review_kernel['K'], ps(review_kernel['kernel_exact'])),
                      ('\\quad exact, bg.\\ 500', 100, ps(final['kernel_bg500'])),
                      ('\\quad k-means bg.', 200, ps(final['kernel_kmeans'])),
                      ('MLP, Integrated Gradients', other['mlp_ig']['K'], ps(other['mlp_ig'])),
                      ('MLP, DeepSHAP', 200, ps(final['mlp_deep'])),
                      ('MLP, SmoothGrad', 200, ps(final['mlp_smooth']))]),
    ('Sanity check', [('Permuted labels', audit['permuted']['K'], ps(audit['permuted']))]),
]
# Keep the main-paper matrix focused on findings; implementation variants remain in the full table.
MAIN_VARIANTS = {
    'Primary and null': {'Primary audit', 'Seeds 1--4 (largest $p$)', 'Null re-tuned per permutation'},
    'Data and expected sets': {'Raw features only', 'Label = OR of flags', '$E_m$ without top trigger'},
    'Imbalance treatment': {'Class weights', 'Focal loss', 'SMOTE'},
    'Attribution (XGBoost)': {'Interventional, training bg.', '\\quad oversampled bg.',
                              '\\quad failure-only bg.', 'Pairwise, nearest normal',
                              'Conditional group Shapley', 'LIME'},
    'Other models': {'Random forest, path TreeSHAP', 'MLP, KernelSHAP', 'MLP, Integrated Gradients',
                     'MLP, SmoothGrad'},
    'Sanity check': {'Permuted labels'},
}
vm = [r'\begin{tabular}{@{}lrcccc@{}}', r'\toprule', r'Variant & $K$ & ' + ' & '.join(PHYSICAL) + r' \\', r'\midrule']
for g, (title, rows_v) in enumerate(GROUPS_V):
    if g:
        vm.append(r'\addlinespace[2pt]')
    vm.append(rf'\multicolumn{{6}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
    for label, K, p in rows_v:
        if label not in MAIN_VARIANTS[title]:
            continue
        symbols = {'aligned': r'\checkmark', 'not aligned': r'$\times$', 'flagged': r'\textbf{!}'}
        cells = [symbols[verdict(p['R'][m], p['D'][m], p['S'][m])] for m in PHYSICAL]
        vm.append(f'{label} & {K} & ' + ' & '.join(cells) + r' \\')
write_ws('verdicts', vm + [r'\bottomrule', r'\end{tabular}'])

# ---- shortcut sweep
passed_primary = [m for m in PHYSICAL if m not in ('HDF', 'TWF')
                  and max(prim['all']['p_holm_R'][m], prim['all']['p_holm_D'][m]) < 0.05]
lines = [r'\begin{tabular}{@{}rrrrrrrrl@{}}', r'\toprule',
         r'&&& \multicolumn{4}{c}{HDF} & \multicolumn{1}{c}{TWF} & \\', r'\cmidrule(lr){4-7}\cmidrule(lr){8-8}',
         r'$\sigma$ & F1 & Sensor & $R$ & $p_R$ & $p_D$ & $p_@SC@$ & $p_R$ / $p_D$ & ' + ', '.join(passed_primary) + r' \\',
         r'\midrule']
for r in sweep['rows']:
    others = [m for m in passed_primary if max(r['p_holm_R'][m], r['p_holm_D'][m]) >= 0.05
              or r['p_holm_S'][m] < 0.05]
    lines.append(f"{r['sigma']} & {f2(r['test_f1'])} & {f3(r['sensor_share_HDF'])} & {f3(r['R_HDF'])} & "
                 f"{pv(r['p_holm_R']['HDF'])} & {pv(r['p_holm_D']['HDF'])} & {flag_s(r['p_holm_S']['HDF'])} & "
                 f"{pv(r['p_holm_R']['TWF'])} / {pv(r['p_holm_D']['TWF'])} & "
                 f"{'pass' if not others else 'fail: ' + ', '.join(others)} \\\\")
write('sweep', lines + [r'\bottomrule', r'\end{tabular}'])

# ---- scale: generated AI4I at several sizes; fraction of seeds with each verdict
groups = {}
for r in scale['rows']:
    groups.setdefault((r['n'], r['sigma']), []).append(r)
lines = [r'\begin{tabular}{@{}rlrrrrrrr@{}}', r'\toprule',
         r'& & & & \multicolumn{2}{c}{HDF flagged by} & \multicolumn{2}{c}{TWF} & \\',
         r'\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
         r'$n$ & $\sigma$ & Seeds & Sensor & $R$ & $@SC@$ & $D$ pass & $@SC@$ flag & Min. \\', r'\midrule']
for (n, sigma), rs in groups.items():
    first = sigma is None
    frac = lambda cond: f"{sum(bool(cond(r)) for r in rs)}/{len(rs)}"
    minutes = np.mean([r['seconds'] for r in rs]) / 60
    lines.append(f"{f'{n:,}'.replace(',', '{,}') if first else ''} & {'--' if first else sigma} & {len(rs)} & "
                 f"{f3(np.mean([r['sensor_share_HDF'] or 0 for r in rs]))} & "
                 f"{frac(lambda r: r['p_holm_R']['HDF'] >= 0.05)} & {frac(lambda r: r['p_holm_S']['HDF'] < 0.05)} & "
                 f"{frac(lambda r: r['p_holm_D']['TWF'] < 0.05)} & {frac(lambda r: r['p_holm_S']['TWF'] < 0.05)} & "
                 f"{minutes:.0f} \\\\")

# workshop paper: grouped headers, centred counts, '--' for no sensor, a gap between dataset sizes
ws = [r'\begin{tabular}{@{}rcccccccccr@{}}', r'\toprule',
      r'& & & & \multicolumn{3}{c}{HDF verdicts} & \multicolumn{3}{c}{TWF verdicts} & \\',
      r'\cmidrule(lr){5-7}\cmidrule(lr){8-10}',
      r'$n$ & $\sigma$ & Runs & Share & A & N & F & A & N & F & Min. \\', r'\midrule']
for i, ((n, sigma), rs) in enumerate(groups.items()):
    first = sigma is None
    if first and i:
        ws.append(r'\addlinespace')
    frac = lambda cond: f"{sum(bool(cond(r)) for r in rs)}/{len(rs)}"
    ws.append(f"{f'{n:,}'.replace(',', '{,}') if first else ''} & {'none' if first else sigma} & {len(rs)} & "
              f"{'--' if first else f3(np.mean([r['sensor_share_HDF'] for r in rs]))} & "
              + ' & '.join(str(sum(verdict(r['p_holm_R'][m], r['p_holm_D'][m], r['p_holm_S'][m]) == v for r in rs))
                           for m in ('HDF', 'TWF') for v in ('aligned', 'not aligned', 'flagged'))
              + f" & {np.mean([r['seconds'] for r in rs]) / 60:.0f} \\\\")
write('scale', lines + [r'\bottomrule', r'\end{tabular}'], ws + [r'\bottomrule', r'\end{tabular}'])

# ---- measured CWRU bearing data (stage 14) and Tennessee Eastman (stage 15); workshop paper only
cwru = json.loads((RESULTS_DIR / 'cwru_audit.json').read_text())
tep = json.loads((RESULTS_DIR / 'tep_audit.json').read_text())
FAULT_NAME = {'IR': 'Inner', 'B': 'Ball', 'OR': 'Outer', 'IDV4': 'IDV4', 'IDV5': 'IDV5'}
ws = [r'\begin{tabular}{@{}llrrrrrrl@{}}', r'\toprule',
      r'Data & Fault & $R_m$ & $p$ & $D_m$ & $p$ & $C_m$ & $p$ & Verdict \\', r'\midrule']
for i, (r, label, mechs) in enumerate([(cwru['original'], 'CWRU', ['IR', 'B', 'OR']),
                                       (cwru['gain'], '+ gain', ['IR', 'B', 'OR']),
                                       (tep, 'TEP', ['IDV4', 'IDV5'])]):
    if i:
        ws.append(r'\addlinespace')
    for j, m in enumerate(mechs):
        o, pr, pd_, pc = r['observed'][m], r['p_holm_R'][m], r['p_holm_D'][m], r['p_holm_S'][m]
        verdict = 'flagged' if pc < 0.05 else ('aligned' if max(pr, pd_) < 0.05 else 'not aligned')
        ws.append(f"{label if j == 0 else ''} & {FAULT_NAME[m]} & {f3(o['R'])} & {pv(pr)} & {f3(o['D'])} & {pv(pd_)} & "
                  f"{f3(o['S'])} & {flag_s(pc)} & {verdict} \\\\")
ws += [r'\bottomrule', r'\end{tabular}']
(WORKSHOP_TABLES_DIR / 'cwru.tex').write_text('\n'.join(ws).replace('@SC@', 'C') + '\n')
print('wrote cwru')

