"""
Stage 5: exploratory figures (model comparison, SHAP summaries, embeddings) -> experiments/figures/
(PDF + 300 dpi PNG). The paper's own figures come from stages 21 and 23.

Reads experiments/results/{benchmark.json, test_predictions.npz, audit.json, shortcut_sweep.json} and
the committed model; only the feature-space projections are recomputed.
"""
import json

import pdm_audit  # noqa: F401
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import umap
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score, precision_recall_curve

from pdm_audit.config import ENGINEERED, FIGURES_DIR, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED
from pdm_audit.data import load, split

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False, 'axes.spines.right': False,
    # embed TrueType instead of Type 3 fonts; IEEE PDF eXpress flags Type 3
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
PAL = sns.color_palette('colorblind')
C_NEG, C_POS = PAL[0], PAL[3]
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    fig.savefig(FIGURES_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIGURES_DIR / f'{name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


bench = json.loads((RESULTS_DIR / 'benchmark.json').read_text())
preds = np.load(RESULTS_DIR / 'test_predictions.npz')
audit = json.loads((RESULTS_DIR / 'audit.json').read_text())
sweep = json.loads((RESULTS_DIR / 'shortcut_sweep.json').read_text())
prim = audit['primary']

# ---- Fig. 1: audit overview, R and D of the tuned model against its permuted-label null
fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.3))
rng = np.random.default_rng(0)
for ax, stat, label in [(axes[0], 'R', '$R_m$ (relevant share)'), (axes[1], 'D', '$D_m$ (specificity)')]:
    for i, m in enumerate(PHYSICAL):
        null = np.array(prim['all']['null_draws'][f'{m}.{stat}'])
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(null)), null, s=2, color='0.7', linewidths=0,
                   rasterized=True, label='permuted-label models' if i == 0 else None)
        ax.scatter(i, prim['observed']['all'][m][stat], s=45, color=PAL[3], edgecolor='k', linewidths=0.5,
                   zorder=3, label='tuned model' if i == 0 else None)
    ax.set(xticks=range(len(PHYSICAL)), xticklabels=PHYSICAL, ylabel=label)
    if stat == 'D':
        ax.axhline(0, color='0.4', linewidth=0.6, linestyle='--')
for ax in axes:
    ax.tick_params(axis='x', labelsize=7)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.06),
           handletextpad=0.2, markerscale=1.5)
fig.tight_layout()
save(fig, 'audit_overview')

# ---- shortcut sweep
rows = sweep['rows']
sig = [r['sigma'] for r in rows]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
ax.plot(sig, [r['sensor_share_HDF'] for r in rows], marker='o', color=PAL[0], label='shortcut share on HDF rows')
ax.plot(sig, [r['R_HDF'] for r in rows], marker='s', color=PAL[3], label='$R_{\\mathrm{HDF}}$')
ax.plot(sig, [r['null_mean_R_HDF'] for r in rows], color='0.6', linestyle='--', label='null mean of $R_{\\mathrm{HDF}}$')
for r in rows:
    if r['p_holm_R']['HDF'] >= 0.05:
        ax.scatter(r['sigma'], r['R_HDF'], s=90, facecolors='none', edgecolors='k', linewidths=1, zorder=4)
ax.set(xscale='log', xlabel='Shortcut noise $\\sigma$ (stronger shortcut $\\rightarrow$)', ylabel='Attribution share', ylim=(0, 1))
ax.invert_xaxis()
ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
save(fig, 'shortcut_sweep')

# ---- attribution heatmap from cross-fitted (out-of-sample) SHAP of the primary analysis
mat = pd.DataFrame(prim['mean_abs_shap']).rename(columns=lambda m: f'{m}\n(n={bench["label_facts"]["mode_counts"][m]})')
fig, ax = plt.subplots(figsize=(3.6, 3.4))
sns.heatmap(mat, annot=True, fmt='.2f', cmap='viridis', ax=ax, cbar_kws={'label': 'mean |SHAP|'},
            annot_kws={'fontsize': 7.5})
ax.set(xlabel='Failure mechanism', ylabel='')
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, 'shap_crosscheck_heatmap')

# ---- PR curves and confusion matrix (test split)
y_test = preds['y_test']
order = ['Logistic regression', 'SVM', 'k-NN', 'Decision tree', 'Random forest', 'MLP', 'XGBoost', 'XGBoost (tuned)']
fig, ax = plt.subplots(figsize=(3.4, 3.9))
for name, color in zip(order, sns.color_palette('colorblind', len(order))):
    prec, rec, _ = precision_recall_curve(y_test, preds[name])
    ax.plot(rec, prec, color=color, linewidth=2.2 if name == 'XGBoost (tuned)' else 1.1,
            label=f'{name} ({average_precision_score(y_test, preds[name]):.3f})')
ax.axhline(y_test.mean(), color='k', linestyle='--', linewidth=0.8)
ax.set(xlabel='Recall', ylabel='Precision')
ax.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.16), ncol=2,
          handlelength=1.2, columnspacing=0.8)
save(fig, 'pr_curves')

cm = np.array(bench['confusion_matrix_tuned'])
fig, ax = plt.subplots(figsize=(2.8, 2.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', cbar=False, ax=ax, annot_kws={'fontsize': 10},
            xticklabels=['No failure', 'Failure'], yticklabels=['No failure', 'Failure'])
ax.set(xlabel='Predicted', ylabel='Actual')
plt.setp(ax.get_yticklabels(), rotation=0, va='center')
save(fig, 'confusion_matrix_best')

# ---- global SHAP summary (tuned model, all rows)
df = load()
cols = ENGINEERED + ['Type']
pre = joblib.load(MODELS_DIR / 'preprocessor.joblib')
model = joblib.load(MODELS_DIR / 'xgboost_tuned.joblib')
X_all = pre.transform(df[cols])
names = ENGINEERED + list(pre.named_transformers_['cat'].get_feature_names_out(['Type']))
sv = shap.TreeExplainer(model).shap_values(X_all)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
plt.figure()
shap.summary_plot(sv, X_all, feature_names=names, show=False, plot_size=None)
fig = plt.gcf()
fig.set_size_inches(4.4, 3.6)
for t in fig.findobj(match=plt.Text):
    t.set_fontsize(7.5)
save(fig, 'shap_summary')

# ---- feature-space structure (training split)
idx_train, _, _ = split(df)
Xtr = pre.transform(df.loc[idx_train, cols])
ytr = df.loc[idx_train, 'Machine failure'].values
pca = PCA(n_components=2, random_state=SEED)
P = pca.fit_transform(Xtr)


def scatter(ax, Z):
    for cls, color, label, size, alpha in [(0, '0.75', 'No failure', 2, 0.4), (1, C_POS, 'Failure', 7, 0.9)]:
        m = ytr == cls
        ax.scatter(Z[m, 0], Z[m, 1], s=size, alpha=alpha, color=color, label=label, rasterized=True, linewidths=0)


fig, axes = plt.subplots(1, 4, figsize=(12, 3))
grid = np.meshgrid(np.linspace(P[:, 0].min() - 1, P[:, 0].max() + 1, 250),
                   np.linspace(P[:, 1].min() - 1, P[:, 1].max() + 1, 250))
for ax, (name, sampler) in zip(axes, [('Original', None), ('SMOTE', SMOTE(random_state=SEED)),
                                      ('Borderline-SMOTE', BorderlineSMOTE(random_state=SEED)),
                                      ('ADASYN', ADASYN(random_state=SEED))]):
    Xf, yf = (P, ytr) if sampler is None else sampler.fit_resample(P, ytr)
    Z = LogisticRegression().fit(Xf, yf).predict(np.c_[grid[0].ravel(), grid[1].ravel()]).reshape(grid[0].shape)
    ax.contourf(grid[0], grid[1], Z, alpha=0.25, colors=[C_NEG, C_POS], levels=[-0.5, 0.5, 1.5])
    scatter(ax, P)
    ax.set(title=name, xlabel='PC1')
axes[0].set_ylabel('PC2')
fig.tight_layout()
save(fig, 'decision_boundary_2d')

projections = {'PCA': P,
               't-SNE': TSNE(n_components=2, perplexity=30, init='pca', random_state=SEED).fit_transform(Xtr),
               'UMAP': umap.UMAP(n_components=2, random_state=SEED).fit_transform(Xtr)}
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for ax, (name, Z) in zip(axes, projections.items()):
    scatter(ax, Z)
    ax.set(title=name, xlabel='dim 1', ylabel='dim 2')
axes[0].legend(frameon=False, markerscale=3, loc='upper right')
fig.tight_layout()
save(fig, 'dimred_comparison')

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, perp in zip(axes, [5, 30, 50, 100]):
    scatter(ax, TSNE(n_components=2, perplexity=perp, init='pca', random_state=SEED).fit_transform(Xtr))
    ax.set(title=f'perplexity={perp}', xticks=[], yticks=[])
    sns.despine(ax=ax, left=True, bottom=True)
axes[0].legend(frameon=False, markerscale=3, loc='upper right', fontsize=6.5)
fig.tight_layout()
save(fig, 'tsne_perplexity')
print('ALL FIGURES DONE')
