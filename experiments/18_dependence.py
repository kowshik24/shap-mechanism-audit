"""
Stage 18: dependence-aware attribution, power, and the choice of K.

  grouped       the twelve inputs form four groups of physically related, correlated features
                (wear: tool wear, overstrain, its ratio; thermal: both temperatures, their difference;
                load: torque, speed, power; type: product type). Grouped SHAP sums each group's signed
                path-dependent TreeSHAP values.
  conditional   exact conditional (observational) Shapley values of the four groups [Aas et al.]:
                v(S; x) = E[f(X) | X_S = x_S], estimated by Gaussian-kernel weights (bandwidth 0.3 in
                standardized units) over 2,000 training rows of the fold, f the XGBoost margin;
                v(all) = f(x). All 16 coalitions are evaluated, so the values are exact given v.
  Both use the cross-fitted primary pipeline and a K-model shuffled-label null with the same
  permutations as stage 2. E_m at group level: TWF {wear}, HDF {thermal, load}, PWF {load},
  OSF {wear, load}.

  power         normal-approximation power of each primary test: the Holm critical value
                null mean + z(1 - alpha/4) * null s.d., against the observed statistic with its
                bootstrap s.d. (stage 10); and the minimum detectable excess over the null mean.
  k_and_size    Monte Carlo error of p near alpha for several K, and the null spread of R against
                mechanism size across the generated-data runs of stage 7.

Usage: python experiments/18_dependence.py [--quick]
"""
import argparse
import itertools
import json
import math

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold

from pdm_audit.audit import audit_rows, holm, permutation_p
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RESULTS_DIR, SEED
from pdm_audit.data import load, mechanism_groups
from pdm_audit.models import feature_names, fit_pipeline

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
K = 4 if args.quick else 200

GROUPS = {'wear': ['Tool wear [min]', 'Overstrain [minNm]', 'Overstrain_ratio'],
          'thermal': ['Air temperature [K]', 'Process temperature [K]', 'Temp_diff [K]'],
          'load': ['Torque [Nm]', 'Rotational speed [rpm]', 'Power [W]'],
          'type': ['Type_H', 'Type_L', 'Type_M']}
G = list(GROUPS)
E_GROUP = {'TWF': ['wear'], 'HDF': ['thermal', 'load'], 'PWF': ['load'], 'OSF': ['wear', 'load']}
COALITIONS = [frozenset(c) for r in range(len(G) + 1) for c in itertools.combinations(G, r)]

df = load()
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())
y = df['Machine failure']
rows = audit_rows(df)
mech = mechanism_groups(df)
failures = df.index[y == 1]
cols = ENGINEERED + ['Type']


def group_attributions(labels, seed=SEED, n_bg=2000, bw=0.3):
    """(grouped |path SHAP|, |conditional group Shapley|) for the audit rows, cross-fitted."""
    grouped, conditional = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
        pre, est = fit_pipeline(df, df.index[tr], ENGINEERED, params, seed, y=labels)
        names = feature_names(pre, ENGINEERED)
        idx = {g: [names.index(f) for f in fs] for g, fs in GROUPS.items()}
        held = np.intersect1d(df.index[te], rows)
        X = pre.transform(df.loc[held, cols])
        sv = shap.TreeExplainer(est).shap_values(X)
        grouped.append(pd.DataFrame({g: np.abs(sv[:, idx[g]].sum(axis=1)) for g in G}, index=held))
        bg_rows = np.random.default_rng(seed).choice(df.index[tr], n_bg, replace=False)
        Xb = pre.transform(df.loc[bg_rows, cols])
        fb = est.predict(Xb, output_margin=True)
        v = {frozenset(G): est.predict(X, output_margin=True), frozenset(): np.full(len(X), fb.mean())}
        for S in COALITIONS:
            if S in v:
                continue
            c = sum((idx[g] for g in S), [])
            d2 = ((X[:, None, c] - Xb[None, :, c]) ** 2).mean(axis=2)
            w = np.exp(-d2 / (2 * bw ** 2))
            v[S] = (w @ fb) / np.maximum(w.sum(axis=1), 1e-300)
        phi = {}
        for g in G:
            others = [h for h in G if h != g]
            total = 0
            for r in range(len(others) + 1):
                weight = math.factorial(r) * math.factorial(len(G) - r - 1) / math.factorial(len(G))
                for S in itertools.combinations(others, r):
                    S = frozenset(S)
                    total = total + weight * (v[S | {g}] - v[S])
            phi[g] = np.abs(total)
        conditional.append(pd.DataFrame(phi, index=held))
    return pd.concat(grouped).loc[rows], pd.concat(conditional).loc[rows]


def stats(A):
    out = {}
    for m in PHYSICAL:
        own, rest = A.loc[mech[m]].mean(), A.loc[np.setdiff1d(failures, mech[m])].mean()
        a, b = own / own.sum(), rest / rest.sum()
        excess = (a - b).drop(E_GROUP[m])
        out[m] = {'R': float(a[E_GROUP[m]].sum()), 'D': float(a[E_GROUP[m]].sum() - b[E_GROUP[m]].sum()),
                  'S': float(excess.max()), 'S_feature': str(excess.idxmax()), 'shares': a.to_dict()}
    return out


def draw(k):
    rng = np.random.default_rng(SEED * 100_003 + k)  # same permutation as stage-2 draw k
    perm = pd.Series(rng.permutation(y.values), index=y.index)
    g, c = group_attributions(perm)
    return {'grouped': stats(g), 'conditional': stats(c)}


obs_g, obs_c = group_attributions(None)
observed = {'grouped': stats(obs_g), 'conditional': stats(obs_c)}
draws = Parallel(n_jobs=-1)(delayed(draw)(k) for k in range(K))
results = {'K': K, 'groups': GROUPS, 'E_group': E_GROUP}
for kind in ('grouped', 'conditional'):
    p = {f'{m}.{s}': permutation_p(observed[kind][m][s], [d[kind][m][s] for d in draws]) for m in PHYSICAL for s in 'RDS'}
    results[kind] = {'observed': observed[kind],
                     **{f'p_holm_{s}': holm({m: p[f'{m}.{s}'] for m in PHYSICAL}) for s in 'RDS'},
                     'null_mean': {f'{m}.{s}': float(np.mean([d[kind][m][s] for d in draws])) for m in PHYSICAL for s in 'RDS'}}
    r = results[kind]
    print(kind, {m: (round(observed[kind][m]['R'], 3), round(r['null_mean'][m + '.R'], 3), round(r['p_holm_R'][m], 3),
                     round(r['p_holm_D'][m], 3), round(r['p_holm_S'][m], 3)) for m in PHYSICAL}, flush=True)

# ---- power of the primary tests (normal approximation)
eff = json.loads((RESULTS_DIR / 'effect_sizes.json').read_text())['mechanisms']
z_crit = norm.ppf(1 - 0.05 / 4)  # the first Holm step, the most demanding
results['power'] = {}
for m in PHYSICAL:
    for s in ('R', 'D'):
        e = eff[m][s]
        boot_sd = (e['ci95'][1] - e['ci95'][0]) / (2 * 1.96)
        crit = e['null_mean'] + z_crit * e['null_sd']
        results['power'][f'{m}.{s}'] = {'critical': crit, 'power': float(norm.cdf((e['observed'] - crit) / boot_sd)),
                                        'min_detectable_excess': float(z_crit * e['null_sd'] + norm.ppf(0.8) * boot_sd)}
print('power', {k: (round(v['power'], 3), round(v['min_detectable_excess'], 3)) for k, v in results['power'].items()})

# ---- K and mechanism size
results['mc_error_at_alpha'] = {k: math.sqrt(0.05 * 0.95 / k) for k in (100, 200, 500, 1000)}
scale = json.loads((RESULTS_DIR / 'scale.json').read_text())['rows']
pairs = [(r['mode_counts'][m], (r['null_quantiles_50_95'][f'{m}.R'][1] - r['null_quantiles_50_95'][f'{m}.R'][0]) / 1.645)
         for r in scale if r['sigma'] is None for m in PHYSICAL if r['mode_counts'].get(m, 0) > 0]
size, sd = np.log([p[0] for p in pairs]), np.log([p[1] for p in pairs])
results['null_sd_vs_size'] = {'slope_loglog': float(np.polyfit(size, sd, 1)[0]), 'n_points': len(pairs)}
print('K/size', results['mc_error_at_alpha'], results['null_sd_vs_size'])

name = 'dependence_quick.json' if args.quick else 'dependence.json'
(RESULTS_DIR / name).write_text(json.dumps(results, indent=1))
print('saved', RESULTS_DIR / name)
