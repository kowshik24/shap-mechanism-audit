"""
Stage 25: checks requested by the TUAI 2026 reviewers for the camera-ready version.

  ai4i         primary (engineered) and raw-feature audits at K=1,000 with the grouped outside-set
               statistic C3_m (sum of the three largest positive excesses outside E_m), plus two
               mechanism-specificity references that keep the fitted model and its held-out SHAP values:
                 (a) mechanism labels permuted among failures (B=10,000 draws), and
                 (b) every expected set of the same size drawn from the model inputs (exact enumeration).
               Both are also applied to a model trained on permuted labels.
  proxies      the planted HDF shortcut split over 1, 3 or 5 sensors, noise grid, K=500: detection of
               the maximum statistic C_m versus the grouped statistic C3_m. 1 proxy reproduces stage 3,
               3 proxies reproduce stage 24.
  calib        independent generated datasets (seeds 100+, disjoint from stage 7's seeds 0-2) without any
               shortcut at n = 10^4, 3*10^4, 10^5: false-flag rates of C_m and C3_m and the excess
               distribution used to calibrate an effect-size threshold; plus replicated detection runs
               (n = 10^4, 5 seeds, 1 or 3 proxies, sigma 0.35/0.25).
  tep          placebo calibration of the two temporal nulls on Tennessee Eastman: only pre-fault
               (normal-operation) samples are used, a placebo onset splits each run into "inactive" and
               "active", so no mechanism exists; repeated over random run subsets and onsets.

Usage: python experiments/25_review_checks.py STAGE [--quick]
"""
import argparse
import itertools
import json
import time

import pdm_audit  # noqa: F401
import numpy as np
import pandas as pd

from pdm_audit.audit import holm, permutation_p, run_audit, shap_cross_fitted
from pdm_audit.config import ENGINEERED, MODELS_DIR, PHYSICAL, RAW_NUMERIC, RESULTS_DIR, ROOT, SEED, TRIGGER_FEATURES
from pdm_audit.data import load, mechanism_groups, split
from pdm_audit.metrics import point_metrics
from pdm_audit.models import f1_threshold, fit_pipeline
from pdm_audit.sim import simulate

ap = argparse.ArgumentParser()
ap.add_argument('stage', choices=['ai4i', 'proxies', 'calib', 'tep'])
ap.add_argument('--quick', action='store_true')
args = ap.parse_args()
Q = args.quick
params = json.loads((MODELS_DIR / 'xgboost_params.json').read_text())


def save(obj, name):
    name = name.replace('.json', '_quick.json') if Q else name
    (RESULTS_DIR / name).write_text(json.dumps(obj, indent=1))
    print('saved', RESULTS_DIR / name, flush=True)


def summary(res, extra=()):
    a, obs = res['all'], res['observed']['all']
    return {'observed': {m: {k: obs[m][k] for k in ('R', 'D', 'S', 'S3', 'S_feature', 'excess', 'shares', 'top_feature',
                                                     *[f'share[{f}]' for f in extra])} for m in PHYSICAL},
            **{f'p_holm_{s}': a[f'p_holm_{s}'] for s in ('R', 'D', 'S', 'S3')},
            'null_mean': {k: v for k, v in a['null_mean'].items() if k.split('.')[1] in ('R', 'D', 'S', 'S3')},
            'null_q95': {k: float(np.quantile(v, 0.95)) for k, v in a['null_draws'].items()
                         if k.split('.')[1] in ('R', 'D', 'S', 'S3')}}


def verdict(r, m, grouped=False):
    flag = r['p_holm_S'][m] < 0.05 or (grouped and r['p_holm_S3'][m] < 0.05)
    if flag:
        return 'flagged'
    return 'aligned' if r['p_holm_R'][m] < 0.05 and r['p_holm_D'][m] < 0.05 else 'not aligned'


# ----------------------------------------------------------------------------------------------- ai4i
def stats_from_sums(own_sum, own_n, rest_sum, rest_n, inside):
    own, rest = own_sum / own_n[:, None], rest_sum / rest_n[:, None]
    so, sr = own / own.sum(axis=1, keepdims=True), rest / rest.sum(axis=1, keepdims=True)
    R = (so * inside).sum(axis=1)
    return R, R - (sr * inside).sum(axis=1)


def specificity(A, df, B):
    """Mechanism-label permutation among failures and exact random expected sets, model fixed."""
    failures = df.index[df['Machine failure'] == 1].to_numpy()
    groups = mechanism_groups(df)
    assert all(np.isin(groups[m], failures).all() for m in PHYSICAL)
    AF = A.loc[failures].to_numpy().astype(np.float64)
    Z = df.loc[failures, PHYSICAL].to_numpy().astype(float)  # |F| x M
    cols = list(A.columns)
    inside = np.array([[c in TRIGGER_FEATURES[m] for c in cols] for m in PHYSICAL], dtype=float)
    total = AF.sum(axis=0)

    def RD(Zp):
        own_sum = Zp.T @ AF
        n = Zp.sum(axis=0)
        R, D = stats_from_sums(own_sum, n, total[None, :] - own_sum, len(failures) - n, inside)
        return R, D

    R_obs, D_obs = RD(Z)
    rng = np.random.default_rng(SEED)
    draws = np.array([np.r_[RD(Z[rng.permutation(len(failures))])] for _ in range(B)])  # B x 2M
    Rn, Dn = draws[:, :len(PHYSICAL)], draws[:, len(PHYSICAL):]
    out = {'B': B, 'label_perm': {}, 'random_sets': {}}
    pR = {m: permutation_p(R_obs[i], Rn[:, i]) for i, m in enumerate(PHYSICAL)}
    pD = {m: permutation_p(D_obs[i], Dn[:, i]) for i, m in enumerate(PHYSICAL)}
    for i, m in enumerate(PHYSICAL):
        out['label_perm'][m] = {'R': float(R_obs[i]), 'D': float(D_obs[i]),
                                'null_mean_R': float(Rn[:, i].mean()), 'null_mean_D': float(Dn[:, i].mean()),
                                'z_R': float((R_obs[i] - Rn[:, i].mean()) / Rn[:, i].std(ddof=1)),
                                'z_D': float((D_obs[i] - Dn[:, i].mean()) / Dn[:, i].std(ddof=1))}
    out['label_perm']['p_holm_R'], out['label_perm']['p_holm_D'] = holm(pR), holm(pD)
    # random expected sets of the same size, every subset of the model inputs
    own = np.stack([AF[Z[:, i] == 1].mean(axis=0) for i in range(len(PHYSICAL))])
    rest = np.stack([AF[Z[:, i] == 0].mean(axis=0) for i in range(len(PHYSICAL))])
    so, sr = own / own.sum(axis=1, keepdims=True), rest / rest.sum(axis=1, keepdims=True)
    pRs, pDs = {}, {}
    for i, m in enumerate(PHYSICAL):
        k = int(inside[i].sum())
        subs = np.array(list(itertools.combinations(range(len(cols)), k)))
        Rs, Ds = so[i][subs].sum(axis=1), so[i][subs].sum(axis=1) - sr[i][subs].sum(axis=1)
        pRs[m] = float(np.mean(Rs >= R_obs[i] - 1e-9))
        pDs[m] = float(np.mean(Ds >= D_obs[i] - 1e-9))
        out['random_sets'][m] = {'size': k, 'n_sets': len(subs), 'rank_R': int((Rs > R_obs[i] + 1e-9).sum()) + 1,
                                 'rank_D': int((Ds > D_obs[i] + 1e-9).sum()) + 1,
                                 'mean_R': float(Rs.mean()), 'mean_D': float(Ds.mean())}
    out['random_sets']['p_R'], out['random_sets']['p_D'] = pRs, pDs
    out['random_sets']['p_holm_R'], out['random_sets']['p_holm_D'] = holm(pRs), holm(pDs)
    return out


def stage_ai4i():
    df = load()
    K, B = (4, 200) if Q else (1000, 10_000)
    out = {'K': K}
    for name, cols in (('primary', ENGINEERED), ('raw', RAW_NUMERIC)):
        t0 = time.time()
        out[name] = summary(run_audit(df, numeric_cols=cols, params=params, seed=SEED, K=K))
        A = shap_cross_fitted(df, cols, params, SEED)
        out[name]['specificity'] = specificity(A, df, B)
        print(name, {m: verdict(out[name], m) for m in PHYSICAL}, 'pS3', out[name]['p_holm_S3'],
              'spec pD', out[name]['specificity']['label_perm']['p_holm_D'],
              'sets pD', out[name]['specificity']['random_sets']['p_D'], f'{time.time() - t0:.0f}s', flush=True)
    y = df['Machine failure']
    perm = pd.Series(np.random.default_rng(7).permutation(y.values), index=y.index)  # as stage 2
    out['permuted_model'] = specificity(shap_cross_fitted(df, ENGINEERED, params, SEED, labels=perm), df, B)
    print('permuted model spec', out['permuted_model']['label_perm']['p_holm_D'],
          out['permuted_model']['random_sets']['p_holm_D'], flush=True)
    save(out, 'review_ai4i.json')


# -------------------------------------------------------------------------------------------- proxies
def add_proxies(df, n_proxies, sigma, seed):
    df = df.copy()
    if n_proxies == 1:  # stage 3 / stage 7 construction
        names = ['Shortcut sensor']
        df[names[0]] = df['HDF'] + np.random.default_rng(seed).normal(0, sigma, len(df))
    else:  # stage 24 construction
        names = [f'Shortcut sensor {k + 1}' for k in range(n_proxies)]
        rng = np.random.default_rng(seed)
        for nm in names:
            df[nm] = df['HDF'] + rng.normal(0, sigma, len(df))
    return df, names


def stage_proxies():
    base = load()
    K = 4 if Q else 500
    grid = [(1, [0.35, 0.25])] if Q else [(n, [1.0, 0.67, 0.5, 0.4, 0.35, 0.3, 0.25]) for n in (1, 3, 5)]
    idx_train, idx_val, idx_test = split(base)
    y = base['Machine failure']
    rows = []
    for n_prox, sigmas in grid:
        for sigma in sigmas:
            t0 = time.time()
            df, names = add_proxies(base, n_prox, sigma, SEED)
            cols = ENGINEERED + names
            pre, model = fit_pipeline(df, idx_train, cols, params)
            Xv, Xt = (pre.transform(df.loc[i, cols + ['Type']]) for i in (idx_val, idx_test))
            thr = f1_threshold(y[idx_val], model.predict_proba(Xv)[:, 1])
            pt = model.predict_proba(Xt)[:, 1]
            f1 = point_metrics(y[idx_test], (pt >= thr).astype(int), pt)['f1']
            r = summary(run_audit(df, numeric_cols=cols, params=params, seed=SEED, K=K, extra_features=tuple(names)),
                        names)
            o = r['observed']['HDF']
            row = {'n_proxies': n_prox, 'sigma': sigma, 'test_f1': f1,
                   'proxy_share_HDF': sum(o[f'share[{p}]'] for p in names),
                   'max_proxy_share_HDF': max(o[f'share[{p}]'] for p in names), **r,
                   'verdict': {m: verdict(r, m) for m in PHYSICAL},
                   'verdict_grouped': {m: verdict(r, m, grouped=True) for m in PHYSICAL}}
            rows.append(row)
            print(f'proxies={n_prox} sigma={sigma:<5} F1={f1:.3f} share={row["proxy_share_HDF"]:.3f} '
                  f'max={row["max_proxy_share_HDF"]:.3f} C={o["S"]:.3f} pC={r["p_holm_S"]["HDF"]:.3f} '
                  f'C3={o["S3"]:.3f} pC3={r["p_holm_S3"]["HDF"]:.3f} pR={r["p_holm_R"]["HDF"]:.3f} '
                  f'others={ {m: row["verdict_grouped"][m] for m in PHYSICAL if m != "HDF"} } '
                  f'({time.time() - t0:.0f}s)', flush=True)
    save({'K': K, 'rows': rows}, 'review_proxies.json')


# ---------------------------------------------------------------------------------------------- calib
def stage_calib():
    K = 4 if Q else 200
    if Q:
        null_plan, det_plan, late_null_plan = [(2_000, [100])], [(2_000, [100], [1], [0.25])], []
    else:
        # Trimmed for camera-ready compute: null runs first (they calibrate the threshold), then detection runs,
        # then the largest size. Each row is saved as it finishes.
        null_plan = [(10_000, range(100, 110)), (30_000, range(100, 106))]
        det_plan = [(10_000, range(100, 103), [1, 3], [0.35, 0.25])]
        late_null_plan = [(100_000, range(100, 103))]
    rows = []

    def run(n, seed, n_prox, sigma):
        t0 = time.time()
        base = simulate(n, seed=seed)
        df, names, cols = base, [], ENGINEERED
        if n_prox:
            df, names = add_proxies(base, n_prox, sigma, seed)
            cols = ENGINEERED + names
        r = summary(run_audit(df, numeric_cols=cols, params=params, seed=seed, K=K, extra_features=tuple(names)),
                    names)
        row = {'n': n, 'seed': seed, 'n_proxies': n_prox, 'sigma': sigma,
               'mode_counts': {m: int(df[m].sum()) for m in PHYSICAL}, **r,
               'verdict': {m: verdict(r, m) for m in PHYSICAL},
               'verdict_grouped': {m: verdict(r, m, grouped=True) for m in PHYSICAL}, 'seconds': time.time() - t0}
        if names:
            row['proxy_share_HDF'] = sum(r['observed']['HDF'][f'share[{p}]'] for p in names)
        rows.append(row)
        print(f'n={n:<7} seed={seed} proxies={n_prox} sigma={sigma} verdict={row["verdict"]} '
              f'pC={ {m: round(v, 3) for m, v in r["p_holm_S"].items()} } '
              f'pC3={ {m: round(v, 3) for m, v in r["p_holm_S3"].items()} } '
              f'C={ {m: round(r["observed"][m]["S"], 3) for m in PHYSICAL} } ({row["seconds"]:.0f}s)', flush=True)
        save({'K': K, 'rows': rows}, 'review_calib.json')

    for n, seeds in null_plan:
        for seed in seeds:
            run(n, seed, 0, None)
    for n, seeds, prox, sigmas in det_plan:
        for seed in seeds:
            for n_prox in prox:
                for sigma in sigmas:
                    run(n, seed, n_prox, sigma)
    for n, seeds in late_null_plan:
        for seed in seeds:
            run(n, seed, 0, None)


# ------------------------------------------------------------------------------------------------ tep
def stage_tep():
    import shap
    import xgboost as xgb
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedGroupKFold

    K, REPS = (4, 2) if Q else (200, 10)
    labels_txt = (ROOT / 'experiments' / '15_tep_audit.py').read_text()
    ns = {}
    exec(labels_txt[labels_txt.index('LABELS = '):labels_txt.index('MECH = ')], ns)  # LABELS, EXPECTED from stage 15
    LABELS, EXPECTED = ns['LABELS'], ns['EXPECTED']
    MECH = list(EXPECTED)
    full = {(f, r): np.fromfile(ROOT / 'data' / 'dtu_tep' / 'prospective_runs' / f'{f}_{r:03d}.bin',
                                dtype='<f8').reshape(2001, 54) for f in (4, 5) for r in range(1, 201)}
    times = np.linspace(1, 29, 32)  # all before the 30 h fault injection: normal operation only
    reps = []
    for rep in range(REPS):
        rng = np.random.default_rng(1000 + rep)
        runs = [(f, int(r)) for f in (4, 5) for r in rng.choice(np.arange(1, 201), 100, replace=False)]
        onset = int(rng.integers(8, 25))  # placebo onset index among the 32 samples
        Xl, g, fault = [], [], []
        for f, r in runs:
            a = full[(f, r)]
            idx = np.abs(a[:, 0, None] - times).argmin(axis=0)
            Xl.append(a[idx, 1:]); g += [f * 1000 + r] * 32; fault += [f] * 32
        X, groups, fault = np.vstack(Xl), np.array(g), np.array(fault)
        y = np.tile((np.arange(32) >= onset).astype(int), len(runs))
        mech = {f'IDV{f}': (fault == f) & (y == 1) for f in (4, 5)}

        def cross_fit(lab):
            A = np.zeros((len(X), len(LABELS)))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, fault, groups):
                model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric='logloss',
                                          random_state=SEED, n_jobs=1).fit(X[tr], lab[tr])
                te_f = te[y[te] == 1]
                A[te_f] = np.abs(shap.TreeExplainer(model).shap_values(X[te_f]))
            return A

        def stats(A):
            out = {}
            for m in MECH:
                own, rest = A[mech[m]].mean(axis=0), A[(y == 1) & ~mech[m]].mean(axis=0)
                a_, b_ = pd.Series(own / own.sum(), LABELS), pd.Series(rest / rest.sum(), LABELS)
                ex = (a_ - b_).drop(EXPECTED[m])
                out[m] = {'R': float(a_[EXPECTED[m]].sum()),
                          'D': float(a_[EXPECTED[m]].sum() - b_[EXPECTED[m]].sum()),
                          'S': float(ex.max()), 'S3': float(ex.clip(lower=0).nlargest(3).sum()),
                          'S_feature': str(ex.idxmax())}
            return out

        def draw(k, kind):
            r2 = np.random.default_rng(SEED * 100_003 + 7919 * rep + k)
            perm = y.copy()
            for gg in np.unique(groups):
                ii = np.flatnonzero(groups == gg)
                perm[ii] = np.roll(y[ii], r2.integers(len(ii))) if kind == 'shift' else r2.permutation(y[ii])
            return stats(cross_fit(perm))

        t0 = time.time()
        obs = stats(cross_fit(y))
        res = {'rep': rep, 'onset_index': onset, 'onset_hour': float(times[onset]), 'observed': obs}
        for kind in ('within', 'shift'):
            d = Parallel(n_jobs=-1)(delayed(draw)(k, kind) for k in range(K))
            p = {f'{m}.{s}': permutation_p(obs[m][s], [x[m][s] for x in d]) for m in MECH for s in ('R', 'D', 'S', 'S3')}
            res[kind] = {f'p_holm_{s}': holm({m: p[f'{m}.{s}'] for m in MECH}) for s in ('R', 'D', 'S', 'S3')}
            res[kind]['verdict'] = {m: verdict(res[kind], m) for m in MECH}
            res[kind]['verdict_grouped'] = {m: verdict(res[kind], m, grouped=True) for m in MECH}
        res['seconds'] = time.time() - t0
        reps.append(res)
        print(f'rep {rep} onset {times[onset]:.1f}h within={res["within"]["verdict"]} '
              f'shift={res["shift"]["verdict"]} pR(within)={res["within"]["p_holm_R"]} '
              f'pC(within)={res["within"]["p_holm_S"]} ({res["seconds"]:.0f}s)', flush=True)
        save({'K': K, 'reps': reps}, 'review_tep_placebo.json')


{'ai4i': stage_ai4i, 'proxies': stage_proxies, 'calib': stage_calib, 'tep': stage_tep}[args.stage]()
