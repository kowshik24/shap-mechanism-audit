"""
Stage 26: camera-ready tables answering the TUAI 2026 reviews, generated from stored results.

  audit_review.tex   primary audit table with the raw-feature audit as a second block (bootstrap
                     intervals for the raw block are computed here exactly as in stage 10 and cached in
                     results/review_raw_effects.json)
  sensitivity.tex    each AI4I verdict reported with its prespecified sensitivity set (engineered/raw
                     inputs x path-dependent/normal-operation reference), explainer agreement, the two
                     mechanism-specificity references and the grouped outside-set test (stage 25)
  shortcut_stats.tex detection of C_m versus C3_m for shortcuts split over 1, 3, 5 sensors (stage 25)
                     and false-flag rates on independent generated data (stage 25 calib)
  Prints the console summary used in the text (TEP placebo, calibrated threshold).

Usage: python experiments/26_review_tables.py
"""
import json

import pdm_audit  # noqa: F401
import numpy as np

from pdm_audit.config import MODELS_DIR, PHYSICAL, RAW_NUMERIC, RESULTS_DIR, SEED, TABLES_DIR, TRIGGER_FEATURES

R = lambda name: json.loads((RESULTS_DIR / name).read_text())
audit, effects, bench = R('audit.json'), R('effect_sizes.json')['mechanisms'], R('benchmark.json')
explainers, twf_diag, final = R('explainers.json'), R('twf_diagnostics.json'), R('final_checks.json')
dep, other = R('dependence.json'), R('other_explainers.json')
rev = R('review_ai4i.json')
SHORT = {'Tool wear [min]': 'tool wear', 'Rotational speed [rpm]': 'speed', 'Power [W]': 'power',
         'Overstrain_ratio': 'overstrain ratio', 'Torque [Nm]': 'torque', 'Temp_diff [K]': 'temp.\\ difference',
         'Air temperature [K]': 'air temp.', 'Process temperature [K]': 'process temp.'}
f2, f3 = (lambda x: f'{x:.2f}'), (lambda x: f'{x:.3f}')


def pv(p, fail_if_high=True):
    s = f'{p:.3f}'
    return s + '$^{\\times}$' if (fail_if_high and p >= 0.05) else s


def verdict(a, m):
    a = a.get('all', a)
    if a['p_holm_S'][m] < 0.05:
        return 'flagged'
    return 'aligned' if max(a['p_holm_R'][m], a['p_holm_D'][m]) < 0.05 else 'not aligned'


def write(name, lines):
    (TABLES_DIR / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    print('wrote', name)


# ---------------------------------------------------------------- raw-feature bootstrap (stage 10 method)
cache = RESULTS_DIR / 'review_raw_effects.json'
if not cache.exists():
    from pdm_audit.audit import shap_cross_fitted, statistics
    from pdm_audit.data import load, mechanism_groups
    df = load()
    params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
    groups, failures = mechanism_groups(df), df.index[df['Machine failure'] == 1]
    A = shap_cross_fitted(df, RAW_NUMERIC, params, SEED)
    obs = statistics(A, df, groups)
    raw = audit['raw']
    rng, B, out = np.random.default_rng(SEED), 5000, {}
    for m in PHYSICAL:
        for s in ('R', 'D', 'S'):
            assert np.isclose(obs[m][s], raw['observed']['all'][m][s]), (m, s)
        own, rest = A.loc[groups[m]].to_numpy(), A.loc[np.setdiff1d(failures, groups[m])].to_numpy()
        trig = np.isin(A.columns, TRIGGER_FEATURES[m])
        share = lambda X: X[:, trig].sum() / X.sum()
        Rb, Db = np.empty(B), np.empty(B)
        for b in range(B):
            o, r = own[rng.integers(0, len(own), len(own))], rest[rng.integers(0, len(rest), len(rest))]
            Rb[b], Db[b] = share(o), share(o) - share(r)
        out[m] = {}
        for s, boot in (('R', Rb), ('D', Db)):
            null = np.asarray(raw['all']['null_draws'][f'{m}.{s}'])
            out[m][s] = {'ci95': [float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                         'z': float((obs[m][s] - null.mean()) / null.std(ddof=1))}
    cache.write_text(json.dumps({'B': B, 'mechanisms': out}, indent=1))
raw_eff = json.loads(cache.read_text())['mechanisms']

# ------------------------------------------------------------------------------- audit + raw block table
counts = bench['label_facts']['mode_counts']
L = [r'\begin{tabular}{@{}lrlrrrrrrrrrrrl@{}}', r'\toprule',
     r'& $n$ & Top feature & $R_m$ [95\% CI] & Null & $z$ & $p$ & $D_m$ [95\% CI] & Null & $z$ & $p$ '
     r'& $C_m$ & Null & $p$ & Verdict \\', r'\midrule']
for title, res, eff in (('Engineered inputs (primary)', audit['primary'], effects), ('Raw inputs only', audit['raw'], raw_eff)):
    if res is audit['raw']:
        L.append(r'\addlinespace[2pt]')
    L.append(rf'\multicolumn{{15}}{{@{{}}l}}{{\textit{{{title}}}}} \\')
    a = res['all']
    for m in PHYSICAL:
        o = res['observed']['all'][m]
        cells = [f"{f3(o[t])} [{f2(eff[m][t]['ci95'][0])}, {f2(eff[m][t]['ci95'][1])}] & "
                 f"{f3(a['null_mean'][f'{m}.{t}'])} & {eff[m][t]['z']:.1f} & {pv(a[f'p_holm_{t}'][m])}" for t in ('R', 'D')]
        cells.append(f"{f3(o['S'])} & {f3(a['null_mean'][f'{m}.S'])} & {pv(a['p_holm_S'][m], False)} & {verdict(res, m)}")
        L.append(f"{m} & {counts[m]} & {SHORT.get(o['top_feature'], o['top_feature'])} & " + ' & '.join(cells) + r' \\')
    if res is audit['primary']:
        o = res['observed']['all']['RNF']
        L.append(f"RNF & {counts['RNF']} & {SHORT.get(o['top_feature'], o['top_feature'])} & {f3(o['top_share'])} & "
                 f"{f3(a['null_mean']['RNF.top_share'])} & -- & {f3(a['p_RNF_more_concentrated'])} & "
                 r"-- & -- & -- & -- & -- & -- & -- & -- \\")
write('audit_review', L + [r'\bottomrule', r'\end{tabular}'])

# ------------------------------------------------------------------------------------ sensitivity table
grid = [('Eng., path', audit['primary']), ('Eng., normal', explainers['interventional']['100/0']),
        ('Raw, path', audit['raw']), ('Raw, normal', twf_diag['raw'])]
expl = [audit['primary'], explainers['interventional']['100/0'], twf_diag['non_failures'], twf_diag['oversampled'],
        twf_diag['failures'], final['pairwise'], dep['conditional'], final['lime'], explainers['random_forest'],
        other['mlp_kernel'], other['mlp_ig'], final['mlp_deep'], final['mlp_smooth']]
SYM = {'aligned': r'\checkmark', 'not aligned': r'$\times$', 'flagged': r'\textbf{!}'}
spec, sets = rev['primary']['specificity']['label_perm'], rev['primary']['specificity']['random_sets']
L = [r'\begin{tabular}{@{}lccccccccl@{}}', r'\toprule',
     r'& \multicolumn{4}{c}{Prespecified set} & & \multicolumn{2}{c}{Specificity $D_m$} & & \\',
     r'\cmidrule(lr){2-5}\cmidrule(lr){7-8}',
     r'& \multicolumn{2}{c}{Eng.} & \multicolumn{2}{c}{Raw} & Expl. & Label & Set & $C^{(3)}_m$ & \\',
     r'Mech. & path & norm. & path & norm. & agree & perm.\ $z$ & rank & $p$ & Reported \\', r'\midrule']
summary = {}
for m in PHYSICAL:
    vs = [verdict(r, m) for _, r in grid]
    agree = sum(verdict(r, m) == 'aligned' for r in expl)
    stable = len(set(vs)) == 1
    rep = 'stable' if stable else 'dependent'
    summary[m] = {'grid': vs, 'agree': agree, 'reported': rep}
    L.append(f"{m} & " + ' & '.join(SYM[v] for v in vs) + f" & {agree}/{len(expl)} & "
             f"{spec[m]['z_D']:.1f} & {sets[m]['rank_D']}/{sets[m]['n_sets']} & "
             f"{pv(rev['primary']['p_holm_S3'][m], False)} & {rep} \\\\")
write('sensitivity', L + [r'\bottomrule', r'\end{tabular}'])
print(json.dumps(summary, indent=1))
print('specificity (primary)', {m: (round(spec[m]['D'], 3), round(spec[m]['null_mean_D'], 3), round(spec[m]['z_D'], 1)) for m in PHYSICAL})
print('specificity R p', spec['p_holm_R'])
print('random sets', {m: (sets[m]['rank_R'], sets[m]['rank_D'], sets[m]['n_sets']) for m in PHYSICAL})
pm = rev['permuted_model']
print('permuted model: label-perm pD', pm['label_perm']['p_holm_D'], 'sets pD', pm['random_sets']['p_D'])
rs = rev['raw']['specificity']
print('raw: label-perm pD', rs['label_perm']['p_holm_D'], 'sets', {m: (rs['random_sets'][m]['rank_D'], rs['random_sets'][m]['n_sets']) for m in PHYSICAL})
print('primary S3', {m: (round(rev['primary']['observed'][m]['S3'], 3), round(rev['primary']['p_holm_S3'][m], 3)) for m in PHYSICAL})
print('raw S3', {m: (round(rev['raw']['observed'][m]['S3'], 3), round(rev['raw']['p_holm_S3'][m], 3)) for m in PHYSICAL})

# ------------------------------------------------------------------ shortcut statistics: detection + FPR
px = RESULTS_DIR / 'review_proxies.json'
cal = RESULTS_DIR / 'review_calib.json'
if px.exists():
    P = R('review_proxies.json')['rows']
    sig = sorted({r['sigma'] for r in P}, reverse=True)
    L = [r'\begin{tabular}{@{}l' + 'c' * len(sig) + '@{}}', r'\toprule',
         r'Sensors & ' + ' & '.join(f'{s:g}' for s in sig) + r' \\', r'\midrule']
    for k in sorted({r['n_proxies'] for r in P}):
        cells = []
        for s in sig:
            r = next((x for x in P if x['n_proxies'] == k and x['sigma'] == s), None)
            if r is None:
                cells.append('')
                continue
            pc, pg, prr = r['p_holm_S']['HDF'], r['p_holm_S3']['HDF'], r['p_holm_R']['HDF']
            mark = ('C' if pc < 0.05 else '') + ('G' if pg < 0.05 else '')
            cells.append(f"{r['proxy_share_HDF']:.2f}" + (rf"\,{{\scriptsize {mark}}}" if mark else
                                                         (r"\,{\scriptsize $\times$}" if prr >= 0.05 else '')))
        L.append(f'{k} & ' + ' & '.join(cells) + r' \\')
    write('shortcut_stats', L + [r'\bottomrule', r'\end{tabular}'])
    for r in P:
        print('proxy', r['n_proxies'], r['sigma'], 'share', round(r['proxy_share_HDF'], 3), 'max', round(r['max_proxy_share_HDF'], 3),
              'pC', round(r['p_holm_S']['HDF'], 3), 'pC3', round(r['p_holm_S3']['HDF'], 3), 'pR', round(r['p_holm_R']['HDF'], 3),
              'F1', round(r['test_f1'], 3), 'others', {m: r['verdict_grouped'][m] for m in PHYSICAL if m != 'HDF'})
if cal.exists():
    C = R('review_calib.json')['rows']
    nulls = [r for r in C if not r['n_proxies']]
    for n in sorted({r['n'] for r in nulls}):
        rs = [r for r in nulls if r['n'] == n]
        anyC = sum(any(r['p_holm_S'][m] < 0.05 for m in PHYSICAL) for r in rs)
        anyG = sum(any(r['p_holm_S3'][m] < 0.05 for m in PHYSICAL) for r in rs)
        exc = [r['observed'][m]['S'] for r in rs for m in PHYSICAL if r['p_holm_S'][m] < 0.05]
        flags = [(m, r['observed'][m]['S_feature'], round(r['observed'][m]['S'], 3)) for r in rs for m in PHYSICAL
                 if r['p_holm_S'][m] < 0.05]
        print(f'null n={n}: runs={len(rs)} any C flag={anyC} any C3 flag={anyG} flagged excesses={flags}')
        print('   verdicts', {m: {v: sum(r['verdict'][m] == v for r in rs) for v in ('aligned', 'not aligned', 'flagged')} for m in PHYSICAL})
    allexc = [r['observed'][m]['S'] for r in nulls for m in PHYSICAL if r['p_holm_S'][m] < 0.05]
    print('max flagged excess in no-shortcut runs:', max(allexc) if allexc else None,
          'largest C in any null run:', max(r['observed'][m]['S'] for r in nulls for m in PHYSICAL))
    det = [r for r in C if r['n_proxies']]
    for k in sorted({r['n_proxies'] for r in det}):
        for s in sorted({r['sigma'] for r in det}, reverse=True):
            rs = [r for r in det if r['n_proxies'] == k and r['sigma'] == s]
            if rs:
                print(f'detect proxies={k} sigma={s}: runs={len(rs)} C={sum(r["p_holm_S"]["HDF"] < 0.05 for r in rs)} '
                      f'C3={sum(r["p_holm_S3"]["HDF"] < 0.05 for r in rs)} either={sum(r["verdict_grouped"]["HDF"] == "flagged" for r in rs)} '
                      f'lost alignment={sum(r["verdict_grouped"]["HDF"] != "aligned" for r in rs)} '
                      f'share={np.mean([r["proxy_share_HDF"] for r in rs]):.3f} '
                      f'minC={min(r["observed"]["HDF"]["S"] for r in rs):.3f}')
tp = RESULTS_DIR / 'review_tep_placebo.json'
if tp.exists():
    T = R('review_tep_placebo.json')['reps']
    for kind in ('within', 'shift'):
        print(f'TEP placebo {kind}: reps={len(T)}',
              {v: sum(r[kind]['verdict'][m] == v for r in T for m in ('IDV4', 'IDV5')) for v in ('aligned', 'not aligned', 'flagged')},
              'R-test passes', sum(r[kind]['p_holm_R'][m] < 0.05 for r in T for m in ('IDV4', 'IDV5')),
              'C3 flags', sum(r[kind]['p_holm_S3'][m] < 0.05 for r in T for m in ('IDV4', 'IDV5')))
