"""
Stage 23: one TWF observation explained under two reference distributions.

Picks the single-mechanism TWF observation whose held-out path-dependent tool-wear share is closest to
the TWF mean, refits the cross-fitting fold that holds it out, and explains it with path-dependent
TreeSHAP (training cover of the oversampled data) and with interventional TreeSHAP against 100
original training observations (about 97% normal operation). Writes the signed attributions to
experiments/results/twf_example.json and a pgfplots figure to big_data_workshop_TUAI/figures/twf_example.pdf.

Usage: python experiments/23_twf_example.py
"""
import json
import subprocess
import tempfile
from pathlib import Path

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pdm_audit.audit import abs_shap
from pdm_audit.config import ENGINEERED, MODELS_DIR, RESULTS_DIR, SEED, WORKSHOP_TABLES_DIR
from pdm_audit.data import load, mechanism_groups
from pdm_audit.models import fit_pipeline

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
y = df['Machine failure']
path_all = pd.read_csv(RESULTS_DIR / 'attributions_primary.csv', index_col=0)  # signed held-out path SHAP (stage 13)
features = [c for c in path_all.columns if c in df.columns or c.startswith('Type_') or c in ENGINEERED]
twf_rows = np.intersect1d(mechanism_groups(df, exclusive=True)['TWF'], path_all.index)
A = path_all.loc[twf_rows, features].abs()
share = A['Tool wear [min]'] / A.sum(axis=1)
row = int((share - share.mean()).abs().idxmin())

for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(df, y):
    if row in df.index[te]:
        break
pre, est = fit_pipeline(df, df.index[tr], ENGINEERED, params, SEED)
kw = dict(numeric_cols=ENGINEERED, background_rows=df.index[tr], seed=SEED, background_y=y.loc[df.index[tr]],
          transform='signed')
path = abs_shap(est, pre, df, [row], explainer='path', **kw).iloc[0]
interv = abs_shap(est, pre, df, [row], explainer='interventional', n_background=100, **kw).iloc[0]
proba = float(est.predict_proba(pre.transform(df.loc[[row], ENGINEERED + ['Type']]))[0, 1])

SHORT = {'Tool wear [min]': 'tool wear', 'Overstrain [minNm]': 'overstrain', 'Overstrain_ratio': 'overstrain ratio',
         'Torque [Nm]': 'torque', 'Power [W]': 'power', 'Rotational speed [rpm]': 'rotational speed',
         'Temp_diff [K]': 'temp.\\ difference', 'Air temperature [K]': 'air temp.',
         'Process temperature [K]': 'process temp.'}
shown = sorted(SHORT, key=lambda f: -max(abs(path[f]), abs(interv[f])))[:7]
out = {'row': row, 'tool_wear_min': float(df.loc[row, 'Tool wear [min]']), 'failure_probability': proba,
       'tool_wear_share_path': float(abs(path).pipe(lambda v: v['Tool wear [min]'] / v.sum())),
       'tool_wear_share_interventional': float(abs(interv).pipe(lambda v: v['Tool wear [min]'] / v.sum())),
       'path': path.to_dict(), 'interventional': interv.to_dict()}
(RESULTS_DIR / 'twf_example.json').write_text(json.dumps(out, indent=1))
print({k: v for k, v in out.items() if not isinstance(v, dict)})

lim = max(max(abs(path[f]), abs(interv[f])) for f in shown) * 1.1
panels = []
for name, values, title in (('P', path, 'path-dependent (training cover)'),
                            ('I', interv, 'interventional (normal operation)')):
    bars = '\n'.join(
        f'\\fill[{"obs" if values[f] > 0 else "sens"}] (axis cs:0,{i - 0.32}) rectangle (axis cs:{values[f]:.4f},{i + 0.32});'
        for i, f in enumerate(shown))
    place = 'at={(P.east)}, anchor=west, xshift=0.25cm, yticklabels={},' if name == 'I' else \
        'yticklabels={' + ','.join('{' + SHORT[f] + '}' for f in shown) + '},'
    panels.append(rf"""\begin{{axis}}[name={name}, {place} width=3.55cm, height=4.3cm, scale only axis,
  xmin={-lim:.3f}, xmax={lim:.3f}, ymin=-0.6, ymax={len(shown) - 0.4}, y dir=reverse,
  ytick={{0,...,{len(shown) - 1}}}, tick label style={{font=\scriptsize}}, ytick style={{draw=none}},
  title={{{title}}}, title style={{font=\scriptsize, yshift=-4pt}}, xlabel={{SHAP value (log-odds)}},
  xlabel style={{font=\scriptsize}}, xmajorgrids, grid style={{black!8}}, axis y line*=left, axis x line*=bottom]
\addplot[draw=none] coordinates {{(0,0)}};
{bars}
\draw[black!60, line width=0.5pt] (axis cs:0,-0.6) -- (axis cs:0,{len(shown) - 0.4});
\end{{axis}}""")
tex = r"""\documentclass[border=1pt]{standalone}
\usepackage{newtxtext,newtxmath}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\definecolor{obs}{HTML}{D55E00}
\definecolor{sens}{HTML}{0072B2}
\begin{document}
\begin{tikzpicture}
""" + '\n'.join(panels) + r"""
\end{tikzpicture}
\end{document}
"""
figures = WORKSHOP_TABLES_DIR.parent / 'figures'
(figures / 'twf_example.tex').write_text(tex)
with tempfile.TemporaryDirectory() as tmp:
    run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-output-directory', tmp, str(figures / 'twf_example.tex')],
                         capture_output=True, text=True)
    if run.returncode:
        raise SystemExit('\n'.join(l for l in run.stdout.splitlines() if l.startswith('!')))
    (figures / 'twf_example.pdf').write_bytes((Path(tmp) / 'twf_example.pdf').read_bytes())
print('saved', figures / 'twf_example.pdf')
