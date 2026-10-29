"""
Stage 21: the workshop paper's data figures as pgfplots sources (Times fonts, as in the paper),
compiled with pdflatex into paper/figures/:

  twf_reference         the TWF relevant share under attribution formulations grouped by reference
                        distribution, with each variant's own null mean (stages 2, 11, 17, 18 and 6)
  shortcut_sweep_tikz   the planted-sensor sweep: sensor share and R_HDF, marking where the
                        relevant-share test fails and where the outside-set test flags (stage 3)
  attribution_heatmap   attribution shares by feature and mechanism with E_m outlined (stage 2)
  explainer_dependence  R_m against its null mean for XGBoost and four MLP explainers (stages 2, 12, 17)
  cwru_shares           CWRU attribution on defect bands, log-RMS and other features, with and without gain (stage 14)

Usage: python experiments/21_reference_figure.py
"""
import json
import subprocess
import tempfile
from pathlib import Path

import pdm_audit  # noqa: F401
import numpy as np

from pdm_audit.config import PAPER_FIGURES_DIR, PHYSICAL, RESULTS_DIR

ALPHA = 0.05
FIGURES = PAPER_FIGURES_DIR
PREAMBLE = r"""\documentclass[border=1pt]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{pgfplots}
\usetikzlibrary{calc}
\pgfplotsset{compat=1.18}
\definecolor{obs}{HTML}{D55E00}
\definecolor{sens}{HTML}{0072B2}
"""


def load(name):
    return json.loads((RESULTS_DIR / f'{name}.json').read_text())


def compile_tex(name, body):
    (FIGURES / f'{name}.tex').write_text(PREAMBLE + '\\begin{document}\n' + body + '\n\\end{document}\n')
    with tempfile.TemporaryDirectory() as tmp:
        run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', tmp,
                              str(FIGURES / f'{name}.tex')], capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(f'{name}: ' + '\n'.join(l for l in run.stdout.splitlines() if l.startswith('!'))[:2000])
        (FIGURES / f'{name}.pdf').write_bytes((Path(tmp) / f'{name}.pdf').read_bytes())
    print('saved', FIGURES / f'{name}.pdf')


def run_audit_row(res):
    a, o = res['all'], res['observed']['all']
    return o['TWF']['R'], a['null_mean']['TWF.R'], a['p_holm_R']['TWF'], a['p_holm_D']['TWF']


def group_row(res):
    return res['observed']['TWF']['R'], res['null_mean']['TWF.R'], res['p_holm_R']['TWF'], res['p_holm_D']['TWF']


audit = load('audit')['primary']
explainers, twf, final, dep = load('explainers'), load('twf_diagnostics'), load('final_checks'), load('dependence')

# ---- Fig. 3: TWF under attribution formulations, grouped by reference distribution
GROUPS = [
    ('oversampled\\\\training cover', [('primary\\\\path', run_audit_row(audit)),
                                      ('interv.\\\\overs.\\\\bg.', run_audit_row(twf['oversampled']))]),
    ('failures', [('interv.\\\\failure\\\\bg.', run_audit_row(twf['failures']))]),
    ('normal\\\\operation', [('interv.\\\\training\\\\bg.', run_audit_row(explainers['interventional']['100/0'])),
                            ('pair-\\\\wise', run_audit_row(final['pairwise'])),
                            ('condi-\\\\tional\\\\groups', group_row(dep['conditional']))]),
    ('other\\\\changes', [('focal\\\\loss', run_audit_row(final['focal'])),
                         ('positive\\\\part', run_audit_row(final['positive']))]),
]
ticks, heads, seps, body, x = [], [], [], [], 0.0
for g, (group, members) in enumerate(GROUPS):
    start = x
    for label, (r, null, p_r, p_d) in members:
        style = 'aligned' if (p_r < ALPHA and p_d < ALPHA) else 'notaligned'
        body += [f'\\draw[stick] (axis cs:{x},{null:.4f}) -- (axis cs:{x},{r:.4f});',
                 f'\\draw[nullmark] (axis cs:{x - 0.26},{null:.4f}) -- (axis cs:{x + 0.26},{null:.4f});',
                 f'\\node[{style}] at (axis cs:{x},{r:.4f}) {{}};']
        ticks.append((x, label))
        x += 1.15
    heads.append(((start + x - 1.15) / 2, group))
    if g < len(GROUPS) - 1:
        seps.append(x - 0.25)
    x += 0.5
body += [f'\\draw[sep] (axis cs:{s},0) -- (axis cs:{s},0.8);' for s in seps]
body += [f'\\node[head] at (axis cs:{h},0.8) {{{g}}};' for h, g in heads]
compile_tex('twf_reference', r"""\begin{tikzpicture}[
  stick/.style={line width=1.1pt, black!25}, nullmark/.style={line width=1.3pt, black!55},
  aligned/.style={circle, fill=obs, draw=obs, line width=0.6pt, inner sep=0pt, minimum size=5.4pt},
  notaligned/.style={circle, fill=white, draw=obs, line width=0.9pt, inner sep=0pt, minimum size=5.4pt},
  sep/.style={black!14, line width=0.5pt}, head/.style={font=\scriptsize\itshape, align=center, anchor=south, text=black!70}]
\begin{axis}[width=8.9cm, height=4.6cm, xmin=-0.55, xmax=""" + f'{x - 0.5 - 0.45:.2f}' + r""", ymin=0, ymax=0.8,
  axis x line*=bottom, axis y line*=left, clip=false,
  xtick={""" + ','.join(f'{t:.2f}' for t, _ in ticks) + r"""}, xticklabels={""" + ','.join('{' + l + '}' for _, l in ticks) + r"""},
  xticklabel style={align=center, font=\scriptsize}, x tick style={draw=none},
  ytick={0,0.2,0.4,0.6}, yticklabel style={font=\footnotesize, /pgf/number format/fixed},
  ylabel={$R_{\mathrm{TWF}}$}, ylabel style={font=\footnotesize}, ymajorgrids, grid style={black!8}]
""" + '\n'.join(body) + r"""
\end{axis}
\end{tikzpicture}""")

# ---- Fig. 4: planted-sensor sweep
rows = load('shortcut_sweep')['rows']


def coords(key, keep=lambda r: True):
    return ' '.join(f'({r["sigma"]},{r[key]:.4f})' for r in rows if keep(r))


compile_tex('shortcut_sweep_tikz', r"""\begin{tikzpicture}
\begin{semilogxaxis}[width=8.4cm, height=4.6cm, x dir=reverse, xmin=0.085, xmax=2.3, ymin=0, ymax=0.8,
  axis x line*=bottom, axis y line*=left, tick label style={font=\footnotesize},
  xtick={2,1,0.5,0.25,0.1}, xticklabels={2,1,0.5,0.25,0.1},
  xlabel={shortcut noise $\sigma$ (smaller: stronger shortcut)}, ylabel={attribution share},
  label style={font=\footnotesize}, ymajorgrids, grid style={black!8},
  legend style={draw=none, fill=none, font=\scriptsize, at={(0.0,1.02)}, anchor=south west, legend columns=3,
                /tikz/every even column/.append style={column sep=6pt}}, legend cell align=left]
\addplot[obs, line width=1pt, mark=square*, mark size=1.6pt] coordinates {""" + coords('R_HDF') + r"""};
\addlegendentry{$R_{\mathrm{HDF}}$}
\addplot[sens, line width=1pt, mark=*, mark size=1.6pt] coordinates {""" + coords('sensor_share_HDF') + r"""};
\addlegendentry{sensor share}
\addplot[black!50, dashed, line width=0.8pt] coordinates {""" + coords('null_mean_R_HDF') + r"""};
\addlegendentry{null mean of $R_{\mathrm{HDF}}$}
\addplot[only marks, mark=o, mark size=3.6pt, black, line width=0.8pt] coordinates {"""
    + coords('R_HDF', lambda r: r['p_holm_R']['HDF'] >= ALPHA) + r"""};
\addlegendentry{$R$ test fails}
\addplot[only marks, mark=triangle, mark size=4pt, black, line width=0.8pt] coordinates {"""
    + coords('sensor_share_HDF', lambda r: r['p_holm_S']['HDF'] < ALPHA) + r"""};
\addlegendentry{outside-set flag}
\end{semilogxaxis}
\end{tikzpicture}""")

# ---- attribution shares by feature and mechanism, expected sets outlined (primary audit)
from pdm_audit.config import TRIGGER_FEATURES  # noqa: E402

ROWS = [('tool wear', ['Tool wear [min]']), ('overstrain', ['Overstrain [minNm]']),
        ('overstrain ratio', ['Overstrain_ratio']), ('torque', ['Torque [Nm]']), ('power', ['Power [W]']),
        ('rotational speed', ['Rotational speed [rpm]']), ('temp.\\ difference', ['Temp_diff [K]']),
        ('air temp.', ['Air temperature [K]']), ('process temp.', ['Process temperature [K]']),
        ('product type', ['Type_H', 'Type_L', 'Type_M'])]
MECHS = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
mean_abs = audit['mean_abs_shap']
cells, outlines, labels = [], [], []
for c, m in enumerate(MECHS):
    total = sum(mean_abs[m].values())
    for r, (_, feats) in enumerate(ROWS):
        share = sum(mean_abs[m][f] for f in feats) / total
        stops = [(255, 255, 255), (158, 202, 225), (33, 113, 181), (8, 48, 107)]
        t = min(share / 0.6, 1) * 3
        lo = min(int(t), 2)
        rgb = [round(stops[lo][k] + (stops[lo + 1][k] - stops[lo][k]) * (t - lo)) for k in range(3)]
        cells.append(f'\\fill[fill={{rgb,255:red,{rgb[0]};green,{rgb[1]};blue,{rgb[2]}}}] '
                     f'(axis cs:{c - 0.5},{r - 0.5}) rectangle (axis cs:{c + 0.5},{r + 0.5});')
        if share >= 0.10:
            labels.append(f'\\node[font=\\tiny, text={"white" if share >= 0.3 else "black"}] at (axis cs:{c},{r}) {{{share:.2f}}};')
        if m in TRIGGER_FEATURES and all(f in TRIGGER_FEATURES[m] for f in feats):
            outlines.append(f'\\draw[trig, line width=1.1pt] (axis cs:{c - 0.5},{r - 0.5}) rectangle (axis cs:{c + 0.5},{r + 0.5});')
compile_tex('attribution_heatmap', r"""\definecolor{trig}{HTML}{009E73}
\begin{tikzpicture}
\begin{axis}[width=5.4cm, height=6.6cm, scale only axis=false, enlargelimits=false, axis on top,
  xmin=-0.5, xmax=4.5, ymin=-0.5, ymax=9.5, y dir=reverse,
  xtick={0,...,4}, xticklabels={""" + ','.join(MECHS) + r"""}, xticklabel pos=upper,
  ytick={0,...,9}, yticklabels={""" + ','.join('{' + n + '}' for n, _ in ROWS) + r"""},
  tick label style={font=\footnotesize}, tick style={draw=none},
  colormap={blues}{rgb255=(255,255,255) rgb255=(158,202,225) rgb255=(33,113,181) rgb255=(8,48,107)},
  point meta min=0, point meta max=0.6,
  colorbar, colorbar style={width=0.18cm, font=\scriptsize, ytick={0,0.2,0.4,0.6},
    yticklabel style={/pgf/number format/fixed}, ylabel={attribution share}, ylabel style={font=\scriptsize}}]
\addplot[draw=none] coordinates {(0,0)};
""" + '\n'.join(cells + outlines + labels) + r"""
\end{axis}
\end{tikzpicture}""")

# ---- explainer dependence: R_m against its null mean for XGBoost and four MLP explainers
other = load('other_explainers')
EXPLAINERS = [('XGBoost, path TreeSHAP', audit, 'obs', '*'), ('MLP, KernelSHAP', other['mlp_kernel'], 'sens', 'square*'),
              ('MLP, Integrated Gradients', other['mlp_ig'], 'trig', 'diamond*'),
              ('MLP, DeepSHAP', final['mlp_deep'], 'purp', 'triangle*'),
              ('MLP, SmoothGrad', final['mlp_smooth'], 'yel', 'pentagon*')]
body, x, ticks = [], 0.0, []
for m in ['TWF', 'HDF', 'PWF', 'OSF']:
    start = x
    for name, res, colour, mark in EXPLAINERS:
        a, o = res['all'], res['observed']['all']
        r, null = o[m]['R'], a['null_mean'][f'{m}.R']
        fill = colour if (a['p_holm_R'][m] < ALPHA and a['p_holm_D'][m] < ALPHA) else 'white'
        body += [f'\\draw[stick] (axis cs:{x},{null:.4f}) -- (axis cs:{x},{r:.4f});',
                 f'\\draw[nullmark] (axis cs:{x - 0.3},{null:.4f}) -- (axis cs:{x + 0.3},{null:.4f});',
                 f'\\addplot[only marks, mark={mark}, mark size=2.4pt, draw={colour}, fill={fill}, line width=0.8pt] '
                 f'coordinates {{({x},{r:.4f})}};']
        x += 1
    ticks.append(((start + x - 1) / 2, m))
    x += 1.2
legend = '\n'.join(f'\\addlegendimage{{only marks, mark={mark}, mark size=2.4pt, draw={colour}, fill={colour}}}'
                   f'\\addlegendentry{{{name}}}' for name, _, colour, mark in EXPLAINERS)
compile_tex('explainer_dependence', r"""\definecolor{trig}{HTML}{009E73}
\definecolor{purp}{HTML}{CC79A7}
\definecolor{yel}{HTML}{E69F00}
\begin{tikzpicture}[stick/.style={line width=1pt, black!25}, nullmark/.style={line width=1.2pt, black!55}]
\begin{axis}[width=8.6cm, height=4.6cm, xmin=-0.8, xmax=""" + f'{x - 1.2 - 0.2:.1f}' + r""", ymin=0, ymax=0.8,
  axis x line*=bottom, axis y line*=left, xtick={""" + ','.join(f'{t:.1f}' for t, _ in ticks) + r"""},
  xticklabels={""" + ','.join(m for _, m in ticks) + r"""}, x tick style={draw=none},
  tick label style={font=\footnotesize}, ylabel={$R_m$ (relevant share)}, ylabel style={font=\footnotesize},
  yticklabel style={/pgf/number format/fixed}, ymajorgrids, grid style={black!8},
  legend style={draw=none, fill=none, font=\scriptsize, at={(0.5,1.03)}, anchor=south, legend columns=3,
    /tikz/every even column/.append style={column sep=5pt}}, legend cell align=left]
""" + legend + '\n' + '\n'.join(body) + r"""
\end{axis}
\end{tikzpicture}""")

# ---- CWRU: where attribution goes, with and without the acquisition-gain shortcut
cwru = load('cwru_audit')
bars, x, ticks = [], 0.0, []
for fault, label in (('IR', 'inner race'), ('B', 'ball'), ('OR', 'outer race')):
    for condition, tag in (('original', 'orig.'), ('gain', 'gain')):
        o = cwru[condition]['observed'][fault]
        band, rms = o['R'], o['share_log_RMS']
        bars += [f'\\fill[obs] (axis cs:{x - 0.36},0) rectangle (axis cs:{x + 0.36},{band:.4f});',
                 f'\\fill[sens] (axis cs:{x - 0.36},{band:.4f}) rectangle (axis cs:{x + 0.36},{band + rms:.4f});',
                 f'\\fill[black!15] (axis cs:{x - 0.36},{band + rms:.4f}) rectangle (axis cs:{x + 0.36},1);',
                 f'\\node[font=\\scriptsize, anchor=north] at (axis cs:{x},-0.01) {{{tag}}};']
        if cwru[condition]['p_holm_S'][fault] < ALPHA:
            bars.append(f'\\node[font=\\footnotesize\\bfseries, anchor=south] at (axis cs:{x},1.0) {{!}};')
        x += 1
    ticks.append((x - 1.5, label))
    x += 0.6
compile_tex('cwru_shares', r"""\begin{tikzpicture}
\begin{axis}[width=8.2cm, height=4.2cm, xmin=-0.6, xmax=""" + f'{x - 0.6 - 0.4:.1f}' + r""", ymin=0, ymax=1.08,
  axis x line*=bottom, axis y line*=left, clip=false, xtick={""" + ','.join(f'{t:.1f}' for t, _ in ticks) + r"""},
  xticklabels={""" + ','.join(l for _, l in ticks) + r"""}, xticklabel style={font=\footnotesize, yshift=-9pt},
  x tick style={draw=none}, ytick={0,0.25,0.5,0.75,1}, yticklabel style={font=\footnotesize, /pgf/number format/fixed},
  ylabel={attribution share}, ylabel style={font=\footnotesize},
  legend style={draw=none, fill=none, font=\scriptsize, at={(0.5,1.06)}, anchor=south, legend columns=3,
    /tikz/every even column/.append style={column sep=6pt}}, legend cell align=left]
\addlegendimage{area legend, fill=obs, draw=none}\addlegendentry{defect-frequency bands ($E_m$)}
\addlegendimage{area legend, fill=sens, draw=none}\addlegendentry{log-RMS}
\addlegendimage{area legend, fill=black!15, draw=none}\addlegendentry{other features}
""" + '\n'.join(bars) + r"""
\end{axis}
\end{tikzpicture}""")
