"""
Stage 4: LaTeX tables for the paper, generated from experiments/results/*.json so no number is
transcribed by hand. Writes paper/ieee_bigdata/tables/*.tex (tabular bodies, \\input by main.tex).
"""
import json

import numpy as np

from pdm_audit.audit import holm
from pdm_audit.config import PHYSICAL, RESULTS_DIR, TABLES_DIR

bench = json.loads((RESULTS_DIR / 'benchmark.json').read_text())
audit = json.loads((RESULTS_DIR / 'audit.json').read_text())
sweep = json.loads((RESULTS_DIR / 'shortcut_sweep.json').read_text())
explainers = json.loads((RESULTS_DIR / 'explainers.json').read_text())
scale = json.loads((RESULTS_DIR / 'scale.json').read_text())
TABLES_DIR.mkdir(parents=True, exist_ok=True)

SHORT = {'Tool wear [min]': 'tool wear', 'Rotational speed [rpm]': 'speed', 'Power [W]': 'power',
         'Overstrain_ratio': 'overstrain ratio', 'Torque [Nm]': 'torque', 'Temp_diff [K]': 'temp.\\ difference',
         'Air temperature [K]': 'air temp.', 'Process temperature [K]': 'process temp.',
         'Overstrain [minNm]': 'overstrain', 'Shortcut sensor': 'shortcut sensor'}


def write(name, lines):
    (TABLES_DIR / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    print('wrote', name)


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
         r'& $n$ & Top feature & $R_m$ & Null & $p$ & $D_m$ & Null & $p$ & $S_m$ & Null & $p$ \\', r'\midrule']
for m in PHYSICAL:
    o, a = prim['observed']['all'][m], prim['all']
    lines.append(f"{m} & {counts[m]} & {SHORT.get(o['top_feature'], o['top_feature'])} & "
                 + ' & '.join(f"{f3(o[t])} & {f3(a['null_mean'][f'{m}.{t}'])} & "
                              + (pv(a[f'p_holm_{t}'][m]) if t != 'S' else pv(a['p_holm_S'][m], fail_if_high=False))
                              for t in ('R', 'D', 'S')) + r' \\')
o = prim['observed']['all']['RNF']
lines.append(f"RNF & {counts['RNF']} & {SHORT.get(o['top_feature'], o['top_feature'])} & {f3(o['top_share'])} & "
             f"{f3(prim['all']['null_mean']['RNF.top_share'])} & {f3(prim['all']['p_RNF_more_concentrated'])} & "
             r"-- & -- & -- & -- & -- & -- \\")
write('audit', lines + [r'\bottomrule', r'\end{tabular}'])


def flag_s(p):
    """S flags a shortcut when p < 0.05: mark those with a dagger."""
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
         r'Variant & $K$ & ' + ' & '.join(['$R$ & $D$ & $S$'] * len(PHYSICAL)) + r' \\', r'\midrule']
for label, res, grp in variants:
    a = res[grp]
    cells = [f"{pv(a['p_holm_R'][m])} & {pv(a['p_holm_D'][m])} & {flag_s(a['p_holm_S'][m])}" for m in PHYSICAL]
    lines.append(f"{label} & {res['K']} & " + ' & '.join(cells) + r' \\')
lines.append(r'\midrule')
for label, res in [('Naive null, tuned model', prim), ('Naive null, permuted labels', audit['permuted'])]:
    naive = holm(res['naive_null_p'])
    lines.append(f'{label} & -- & ' + ' & '.join(f"{pv(naive[m])} & -- & --" for m in PHYSICAL) + r' \\')
write('robustness', lines + [r'\bottomrule', r'\end{tabular}'])

# ---- shortcut sweep
passed_primary = [m for m in PHYSICAL if m not in ('HDF', 'TWF')
                  and max(prim['all']['p_holm_R'][m], prim['all']['p_holm_D'][m]) < 0.05]
lines = [r'\begin{tabular}{@{}rrrrrrrrl@{}}', r'\toprule',
         r'&&& \multicolumn{4}{c}{HDF} & \multicolumn{1}{c}{TWF} & \\', r'\cmidrule(lr){4-7}\cmidrule(lr){8-8}',
         r'$\sigma$ & F1 & Sensor & $R$ & $p_R$ & $p_D$ & $p_S$ & $p_R$ / $p_D$ & ' + ', '.join(passed_primary) + r' \\',
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
         r'$n$ & $\sigma$ & Seeds & Sensor & $R$ & $S$ & $D$ pass & $S$ flag & Min. \\', r'\midrule']
for (n, sigma), rs in groups.items():
    first = sigma is None
    frac = lambda cond: f"{sum(bool(cond(r)) for r in rs)}/{len(rs)}"
    minutes = np.mean([r['seconds'] for r in rs]) / 60
    lines.append(f"{f'{n:,}'.replace(',', '{,}') if first else ''} & {'--' if first else sigma} & {len(rs)} & "
                 f"{f3(np.mean([r['sensor_share_HDF'] or 0 for r in rs]))} & "
                 f"{frac(lambda r: r['p_holm_R']['HDF'] >= 0.05)} & {frac(lambda r: r['p_holm_S']['HDF'] < 0.05)} & "
                 f"{frac(lambda r: r['p_holm_D']['TWF'] < 0.05)} & {frac(lambda r: r['p_holm_S']['TWF'] < 0.05)} & "
                 f"{minutes:.0f} \\\\")
write('scale', lines + [r'\bottomrule', r'\end{tabular}'])
