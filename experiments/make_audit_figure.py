"""
Overview figure of the audit (paper Fig. 1) from experiments/results/audit_experiments.json.

Run after audit_experiments.py:  python experiments/make_audit_figure.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
res = json.load(open(os.path.join(ROOT, 'experiments', 'results', 'audit_experiments.json')))

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
})
pal = sns.color_palette('colorblind')
mechs = ['TWF', 'HDF', 'PWF', 'OSF']
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for i, m in enumerate(mechs):
    null = np.array(res['main']['null_draws'][m])
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(null)), null, s=5, color='0.65', linewidths=0,
               label='permuted-label models' if i == 0 else None, zorder=1)
    ax.scatter(i, res['main']['observed'][m], s=55, color=pal[3], edgecolor='k', linewidths=0.5,
               label='tuned model' if i == 0 else None, zorder=3)
    ax.scatter(i + 0.28, res['shortcut_strong']['observed'][m], s=40, marker='X', color=pal[0],
               edgecolor='k', linewidths=0.4, label='strong shortcut model' if i == 0 else None, zorder=3)
ax.set(xticks=range(len(mechs)), xticklabels=mechs, ylabel='Relevant share $R_m$', ylim=(0, 1))
ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.02), ncol=1, handletextpad=0.3)

out = os.path.join(ROOT, 'paper', 'figures')
fig.savefig(os.path.join(out, 'audit_overview.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(out, 'audit_overview.png'), dpi=300, bbox_inches='tight')
print('saved audit_overview')
