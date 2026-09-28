"""
Stage 4: LaTeX tables for the paper, generated from experiments/results/*.json so no number is
transcribed by hand. Writes paper/ieee_bigdata/tables/*.tex (tabular bodies, \\input by main.tex).
"""
import json

from pdm_audit.audit import holm
from pdm_audit.config import PHYSICAL, RESULTS_DIR, TABLES_DIR

bench = json.loads((RESULTS_DIR / 'benchmark.json').read_text())
audit = json.loads((RESULTS_DIR / 'audit.json').read_text())
sweep = json.loads((RESULTS_DIR / 'shortcut_sweep.json').read_text())
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
    r'LightGBM, 5-fold CV~\cite{connectedvehicles2026} & macro-F1 & $0.814 \pm 0.012$ \\',
    f"Ours, single split & pos.-class F1 & {f2(tuned['f1'])} \\\\",
    f"Ours, 5-fold CV & pos.-class F1 & ${f3(cv['f1'][0])} \\pm {f3(cv['f1'][1])}$ \\\\",
    f"Ours, 5-fold CV & macro-F1 & ${f3(cv['macro_f1'][0])} \\pm {f3(cv['macro_f1'][1])}$ \\\\",
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
lines = [r'\begin{tabular}{@{}lrlrrrrrr@{}}', r'\toprule',
         r'& $n$ & Top feature & $R_m$ & Null & $p$ & $D_m$ & Null & $p$ \\', r'\midrule']
for m in PHYSICAL:
    o = prim['observed']['all'][m]
    a = prim['all']
    lines.append(f"{m} & {counts[m]} & {SHORT.get(o['top_feature'], o['top_feature'])} & {f3(o['R'])} & "
                 f"{f3(a['null_mean'][m + '.R'])} & {pv(a['p_holm_R'][m])} & {f3(o['D'])} & "
                 f"{f3(a['null_mean'][m + '.D'])} & {pv(a['p_holm_D'][m])} \\\\")
o = prim['observed']['all']['RNF']
lines.append(f"RNF & {counts['RNF']} & {SHORT.get(o['top_feature'], o['top_feature'])} & {f3(o['top_share'])} & "
             f"{f3(prim['all']['null_mean']['RNF.top_share'])} & {f3(prim['all']['p_RNF_more_concentrated'])} & "
             r"-- & -- & -- \\")
write('audit', lines + [r'\bottomrule', r'\end{tabular}'])

# ---- robustness across variants (Holm-adjusted p for R and D)
variants = [('Primary, seed 42', prim, 'all'), ('\\quad single-mechanism rows', prim, 'single')]
variants += [(f'Seed {s}', audit['seeds'][s], 'all') for s in sorted(audit['seeds'])]
variants += [('Raw features only', audit['raw'], 'all'), ('In-sample SHAP', audit['in_sample'], 'all'),
             ('Permuted labels (sanity)', audit['permuted'], 'all')]
lines = [r'\begin{tabular}{@{}lr' + 'rr' * len(PHYSICAL) + r'@{}}', r'\toprule',
         r'& & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{m}}}' for m in PHYSICAL) + r' \\',
         ''.join(rf'\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}' for i in range(len(PHYSICAL))),
         r'Variant & $K$ & ' + ' & '.join(['$R$ & $D$'] * len(PHYSICAL)) + r' \\', r'\midrule']
for label, res, grp in variants:
    a = res[grp]
    cells = [f"{pv(a['p_holm_R'][m])} & {pv(a['p_holm_D'][m])}" for m in PHYSICAL]
    lines.append(f"{label} & {res['K']} & " + ' & '.join(cells) + r' \\')
naive = holm(audit['permuted']['naive_null_p'])
lines.append(r'\midrule')
lines.append(r'Permuted labels, naive null & -- & ' +
             ' & '.join(f"{pv(naive[m])} & --" for m in PHYSICAL) + r' \\')
write('robustness', lines + [r'\bottomrule', r'\end{tabular}'])

# ---- shortcut sweep
lines = [r'\begin{tabular}{@{}rrrrrrl@{}}', r'\toprule',
         r'$\sigma$ & F1 & Sensor share & $R_{\mathrm{HDF}}$ & $p_R$ & $p_D$ & Other mechanisms \\', r'\midrule']
for r in sweep['rows']:
    others = [m for m in PHYSICAL if m != 'HDF' and r['p_holm_R'][m] >= 0.05]
    lines.append(f"{r['sigma']} & {f2(r['test_f1'])} & {f3(r['sensor_share_HDF'])} & {f3(r['R_HDF'])} & "
                 f"{pv(r['p_holm_R']['HDF'])} & {pv(r['p_holm_D']['HDF'])} & "
                 f"{'all pass' if not others else 'fail: ' + ', '.join(others)} \\\\")
write('sweep', lines + [r'\bottomrule', r'\end{tabular}'])
