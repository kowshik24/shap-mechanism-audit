"""
Publication figures for the paper -> paper/figures/ (PNG at 300 dpi + PDF).

Individual purpose-built panels, labeled axes, colorblind-safe categorical palette, viridis for
non-negative continuous data, despined axes. Uses the committed model artifacts in models/, so the
hyperparameter search is not re-run.  Run from the repo root:  python experiments/make_figures.py
"""
import os
for _v in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS',
           'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[_v] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import xgboost as xgb
import umap
import shap

RANDOM_STATE = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})
PALETTE = sns.color_palette('colorblind')
C_NEG, C_POS = PALETTE[0], PALETTE[3]
SEQ_CMAP = 'viridis'


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f'{name}.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(OUTDIR, f'{name}.pdf'), bbox_inches='tight')
    plt.close(fig)
    print('saved', name)


# ---------------------------------------------------------------- data + features (mirrors notebook)
df = pd.read_csv(os.path.join(ROOT, 'dataset', 'ai4i2020.csv'))
numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]',
                'Torque [Nm]', 'Tool wear [min]']
df['Power [W]'] = df['Torque [Nm]'] * df['Rotational speed [rpm]'] * (2 * np.pi / 60)
df['Temp_diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']
df['Overstrain [minNm]'] = df['Tool wear [min]'] * df['Torque [Nm]']
df['Overstrain_ratio'] = df['Overstrain [minNm]'] / df['Type'].map({'L': 11000, 'M': 12000, 'H': 13000})
engineered_cols = ['Power [W]', 'Temp_diff [K]', 'Overstrain [minNm]', 'Overstrain_ratio']
feature_cols = numeric_cols + engineered_cols + ['Type']
failure_modes = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

X = df[feature_cols].copy()
y = df['Machine failure'].copy()
fm_df = df[failure_modes].copy()

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, stratify=y_temp,
                                                random_state=RANDOM_STATE)

preprocessor = joblib.load(os.path.join(ROOT, 'models', 'preprocessor.joblib'))
xgb_tuned = joblib.load(os.path.join(ROOT, 'models', 'xgboost_tuned.joblib'))
threshold = joblib.load(os.path.join(ROOT, 'models', 'metadata.joblib'))['threshold']

X_train_proc = preprocessor.transform(X_train)
X_test_proc = preprocessor.transform(X_test)
X_full_proc = preprocessor.transform(X)
feature_names = (numeric_cols + engineered_cols +
                 list(preprocessor.named_transformers_['cat'].get_feature_names_out(['Type'])))

# ============================================================ class imbalance
fig, ax = plt.subplots(figsize=(3.2, 3.2))
counts = y.value_counts().sort_index()
bars = ax.bar(['No failure', 'Failure'], counts.values, color=[C_NEG, C_POS], width=0.6)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f'{v}\n({v/len(y)*100:.1f}%)', ha='center', va='bottom', fontsize=8)
ax.set(ylabel='Count', ylim=(0, counts.max() * 1.22))
save(fig, 'class_imbalance')

# ============================================================ PCA elbow
pca_full = PCA(random_state=RANDOM_STATE).fit(X_train_proc)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_for_95 = int(np.searchsorted(cumulative, 0.95) + 1)
comps = np.arange(1, len(explained) + 1)

fig, ax = plt.subplots(figsize=(3.6, 3.2))
ax.plot(comps, cumulative, marker='o', color=PALETTE[2], linewidth=1.5, markersize=4, label='cumulative')
ax.bar(comps, explained, color=PALETTE[0], alpha=0.5, width=0.6, label='per-component')
ax.axhline(0.95, color='grey', linestyle='--', linewidth=1)
ax.axvline(n_for_95, color='grey', linestyle=':', linewidth=1)
ax.annotate(f'{n_for_95} PCs\nfor 95%', xy=(n_for_95, 0.95), xytext=(n_for_95 + 1, 0.6),
            fontsize=7.5, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.set(xlabel='Principal component', ylabel='Explained variance ratio', xticks=comps)
ax.legend(loc='center right', frameon=False)
save(fig, 'pca_elbow')

# ============================================================ PCA 2D
pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca2 = pca_2d.fit_transform(X_train_proc)

fig, ax = plt.subplots(figsize=(3.4, 3.2))
for cls, color, label in [(0, C_NEG, 'No failure'), (1, C_POS, 'Failure')]:
    m = y_train.values == cls
    ax.scatter(X_train_pca2[m, 0], X_train_pca2[m, 1], s=6, alpha=0.5, color=color, label=label,
               rasterized=True, linewidths=0)
ax.set(xlabel=f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)',
       ylabel=f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
ax.legend(frameon=False, markerscale=2, loc='upper right')
save(fig, 'pca_2d')

# ============================================================ PCA vs t-SNE vs UMAP
X_train_tsne2 = TSNE(n_components=2, perplexity=30, init='pca', random_state=RANDOM_STATE).fit_transform(X_train_proc)
X_train_umap2 = umap.UMAP(n_components=2, random_state=RANDOM_STATE).fit_transform(X_train_proc)

fig, axes = plt.subplots(1, 3, figsize=(3 * 3.2, 3.2))
for ax, (name, proj) in zip(axes, {'PCA': X_train_pca2, 't-SNE': X_train_tsne2, 'UMAP': X_train_umap2}.items()):
    for cls, color, label in [(0, C_NEG, 'No failure'), (1, C_POS, 'Failure')]:
        m = y_train.values == cls
        ax.scatter(proj[m, 0], proj[m, 1], s=5, alpha=0.5, color=color, label=label, rasterized=True, linewidths=0)
    ax.set(title=name, xlabel='dim 1', ylabel='dim 2')
axes[0].legend(frameon=False, markerscale=2, loc='upper right')
fig.tight_layout()
save(fig, 'dimred_comparison')

# ============================================================ resampling decision boundaries
y_train_arr = y_train.values
viz_samplers = {'Original': None, 'SMOTE': SMOTE(random_state=RANDOM_STATE),
                'Borderline-SMOTE': BorderlineSMOTE(random_state=RANDOM_STATE),
                'ADASYN': ADASYN(random_state=RANDOM_STATE)}
step = (X_train_pca2[:, 0].max() - X_train_pca2[:, 0].min()) / 250
xx, yy = np.meshgrid(np.arange(X_train_pca2[:, 0].min() - 1, X_train_pca2[:, 0].max() + 1, step),
                     np.arange(X_train_pca2[:, 1].min() - 1, X_train_pca2[:, 1].max() + 1, step))

fig, axes = plt.subplots(1, 4, figsize=(4 * 3.0, 3.2))
for ax, (name, sampler) in zip(axes, viz_samplers.items()):
    Xf, yf = (X_train_pca2, y_train_arr) if sampler is None else sampler.fit_resample(X_train_pca2, y_train_arr)
    clf = LogisticRegression().fit(Xf, yf)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, colors=[C_NEG, C_POS], levels=[-0.5, 0.5, 1.5])
    for cls, color in [(0, C_NEG), (1, C_POS)]:
        m = y_train_arr == cls
        ax.scatter(X_train_pca2[m, 0], X_train_pca2[m, 1], s=4, alpha=0.6, color=color, rasterized=True, linewidths=0)
    ax.set(title=name, xlabel='PC1')
axes[0].set_ylabel('PC2')
fig.tight_layout()
save(fig, 'decision_boundary_2d')

# ============================================================ baselines for ROC / PR / F1 (not tuned)
Xtr_res, ytr_res = BorderlineSMOTE(random_state=RANDOM_STATE).fit_resample(X_train_proc, y_train)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE),
    'Random Forest': RandomForestClassifier(n_estimators=300, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                                 random_state=RANDOM_STATE, n_jobs=1),
}
test_probas = {name: m.fit(Xtr_res, ytr_res).predict_proba(X_test_proc)[:, 1] for name, m in models.items()}
test_probas['XGBoost (tuned)'] = xgb_tuned.predict_proba(X_test_proc)[:, 1]
model_order = list(test_probas)
line_colors = sns.color_palette('colorblind', n_colors=len(model_order))

f1s = {n: f1_score(y_test, (p >= (threshold if n == 'XGBoost (tuned)' else 0.5)).astype(int), zero_division=0)
       for n, p in test_probas.items()}
order = sorted(f1s, key=f1s.get, reverse=True)
fig, ax = plt.subplots(figsize=(3.6, 3.4))
bars = ax.barh(order, [f1s[n] for n in order],
               color=[PALETTE[3] if n == 'XGBoost (tuned)' else PALETTE[7] for n in order])
for b, n in zip(bars, order):
    ax.text(b.get_width() + 0.01, b.get_y() + b.get_height() / 2, f'{f1s[n]:.2f}', va='center', fontsize=7.5)
ax.set(xlabel='Positive-class F1 (test)', xlim=(0, 1.05))
ax.invert_yaxis()
save(fig, 'model_comparison_f1')

fig, ax = plt.subplots(figsize=(3.8, 4.4))
for name, color in zip(model_order, line_colors):
    fpr, tpr, _ = roc_curve(y_test, test_probas[name])
    ax.plot(fpr, tpr, label=f'{name} ({roc_auc_score(y_test, test_probas[name]):.3f})', color=color,
            linewidth=2.2 if name == 'XGBoost (tuned)' else 1.2)
ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
ax.set(xlabel='False positive rate', ylabel='True positive rate')
ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
save(fig, 'roc_curves')

fig, ax = plt.subplots(figsize=(3.8, 4.4))
for name, color in zip(model_order, line_colors):
    prec, rec, _ = precision_recall_curve(y_test, test_probas[name])
    ax.plot(rec, prec, label=f'{name} ({average_precision_score(y_test, test_probas[name]):.3f})', color=color,
            linewidth=2.2 if name == 'XGBoost (tuned)' else 1.2)
ax.axhline(y_test.mean(), color='k', linestyle='--', linewidth=0.8)
ax.set(xlabel='Recall', ylabel='Precision')
ax.legend(frameon=False, fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
save(fig, 'pr_curves')

cm = confusion_matrix(y_test, (test_probas['XGBoost (tuned)'] >= threshold).astype(int))
fig, ax = plt.subplots(figsize=(3.4, 3.2))
sns.heatmap(cm, annot=True, fmt='d', cmap=SEQ_CMAP, cbar=False, ax=ax,
            xticklabels=['No failure', 'Failure'], yticklabels=['No failure', 'Failure'], annot_kws={'fontsize': 11})
ax.set(xlabel='Predicted', ylabel='Actual')
plt.setp(ax.get_yticklabels(), rotation=0, va='center')
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, 'confusion_matrix_best')

# ============================================================ t-SNE perplexity sensitivity
fig, axes = plt.subplots(1, 4, figsize=(4 * 3.0, 3.2))
for ax, perp in zip(axes, [5, 30, 50, 100]):
    emb = TSNE(n_components=2, perplexity=perp, init='pca', random_state=RANDOM_STATE).fit_transform(X_train_proc)
    for cls, color, label in [(0, C_NEG, 'No failure'), (1, C_POS, 'Failure')]:
        m = y_train.values == cls
        ax.scatter(emb[m, 0], emb[m, 1], s=5, alpha=0.5, color=color, label=label, rasterized=True, linewidths=0)
    ax.set(title=f'perplexity={perp}', xlabel='dim 1')
    ax.set_xticks([])
    ax.set_yticks([])
    sns.despine(ax=ax, left=True, bottom=True)
axes[0].set_ylabel('dim 2')
axes[0].legend(frameon=False, markerscale=2, loc='upper right', fontsize=7)
fig.tight_layout()
save(fig, 'tsne_perplexity')

# ============================================================ SHAP
sv = shap.TreeExplainer(xgb_tuned).shap_values(X_full_proc)
shap_vals = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)

plt.figure(figsize=(4.2, 3.6))
shap.summary_plot(shap_vals, X_full_proc, feature_names=feature_names, show=False, plot_size=None)
fig = plt.gcf()
fig.set_size_inches(4.6, 3.8)
for txt in fig.findobj(match=plt.Text):
    txt.set_fontsize(8)
save(fig, 'shap_summary')

mode_importance = pd.DataFrame(index=feature_names)
for mode in failure_modes:
    mask = (fm_df[mode] == 1).values
    mode_importance[f'{mode}\n(n={mask.sum()})'] = np.abs(shap_vals[mask]).mean(axis=0)
fig, ax = plt.subplots(figsize=(4.6, 4.2))
sns.heatmap(mode_importance, annot=True, fmt='.2f', cmap=SEQ_CMAP, ax=ax,
            cbar_kws={'label': 'mean |SHAP value|'}, annot_kws={'fontsize': 7})
ax.set(xlabel='Failure mechanism', ylabel='')
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, 'shap_crosscheck_heatmap')

print('ALL FIGURES DONE')
