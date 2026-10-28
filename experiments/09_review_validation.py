"""Bounded review validation; see REVIEW_VALIDATION_PROTOCOL.md before interpreting results.

Writes a separate JSON result and checkpoint directory; historical artifacts are untouched.
Run with --quick only for a smoke test. Full mode: 100 conditional label controls and
20 generated 10,000-row audits, each with 200 independent shuffled-label draws.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import pdm_audit  # thread configuration before numerical imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold

from pdm_audit.audit import abs_shap, audit_rows, holm, permutation_p, shap_cross_fitted, statistics
from pdm_audit.config import ENGINEERED, MODES, PHYSICAL, RESULTS_DIR, TRIGGER_FEATURES
from pdm_audit.data import load, mechanism_groups, split
from pdm_audit.metrics import clopper_pearson, point_metrics
from pdm_audit.models import fit_pipeline, f1_threshold
from pdm_audit.sim import simulate
from pdm_audit.validation import disjoint_feature_decoys, group_statistics, monte_carlo_family


def read(name):
    return json.loads((RESULTS_DIR / name).read_text())


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False))
    tmp.replace(path)


def crossfit_both(df, cols, params, seed=42):
    parts, probabilities = [], pd.Series(index=df.index, dtype=float)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, df['Machine failure']):
        pre, model = fit_pipeline(df, df.index[tr], cols, params, seed)
        held = df.index[te]
        probabilities.loc[held] = model.predict_proba(pre.transform(df.loc[held, cols+['Type']]))[:, 1]
        rows = np.intersect1d(held, audit_rows(df))
        parts.append(abs_shap(model, pre, df, rows, cols))
    return pd.concat(parts).loc[audit_rows(df)], probabilities


def families(observed, draws):
    return {s: monte_carlo_family({m: observed[m][s] for m in PHYSICAL},
                                  {m: draws[f'{m}.{s}'] for m in PHYSICAL}) for s in 'RDS'}


def support_summary(records, family_key='families'):
    out = {}
    for m in PHYSICAL:
        n = len(records)
        counts = {'R_support': sum(r[family_key]['R'][m]['p_holm'] < .05 for r in records),
                  'D_support': sum(r[family_key]['D'][m]['p_holm'] < .05 for r in records),
                  'RD_support': sum(max(r[family_key][s][m]['p_holm'] for s in 'RD') < .05 for r in records),
                  'S_warning': sum(r[family_key]['S'][m]['p_holm'] < .05 for r in records)}
        out[m] = {k: {'count': int(v), 'n': n, 'rate': v/n,
                       'exact_95_interval': list(clopper_pearson(v, n))} for k, v in counts.items()}
        out[m]['S_floor_counts'] = {str(f): int(sum(r[family_key]['S'][m]['p_holm'] < .05 and
                                      r['observed'][m]['S'] >= f for r in records)) for f in [0, .025, .05, .10]}
    return out


def negative_control(i, df, params, archived, checkpoint):
    path = checkpoint / f'negative_{i}.json'
    if path.exists():
        return json.loads(path.read_text())
    labels = pd.Series(np.random.default_rng(900_000+i).permutation(df['Machine failure']), index=df.index)
    a = shap_cross_fitted(df, ENGINEERED, params, 42, labels=labels)
    obs = statistics(a, df, mechanism_groups(df))
    out = {'control_seed': 900_000+i, 'observed': obs,
           'families': families(obs, archived['all']['null_draws'])}
    write(path, out)
    return out


def generated_condition(seed, sigma, params, K, n, checkpoint):
    name = f'generated_{n}_{seed}_{sigma}'
    path = checkpoint / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text())
    started = time.time(); df = simulate(n, seed=seed); cols = list(ENGINEERED)
    if sigma is not None:
        df['Shortcut sensor'] = df['HDF'] + np.random.default_rng(500_000+seed).normal(0, sigma, n)
        cols += ['Shortcut sensor']
    groups = mechanism_groups(df)
    a = shap_cross_fitted(df, cols, params, 42)
    obs = statistics(a, df, groups, ('Shortcut sensor',) if sigma else ())
    draws = {f'{m}.{s}': [] for m in PHYSICAL for s in 'RDS'}
    for k in range(K):
        labels = pd.Series(np.random.default_rng(np.random.SeedSequence([20261022, seed, int(sigma is not None), k]))
                           .permutation(df['Machine failure']), index=df.index)
        null_a = shap_cross_fitted(df, cols, params, 42, labels=labels)
        st = statistics(null_a, df, groups)
        for m in PHYSICAL:
            for s in 'RDS':
                draws[f'{m}.{s}'].append(st[m][s])
    out = {'generation_seed': seed, 'fold_model_seed': 42, 'n': n, 'sigma': sigma, 'K': K,
           'observed': obs, 'families': families(obs, draws), 'null_draws': draws,
           'seconds': time.time()-started}
    write(path, out)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); ap.add_argument('--jobs', type=int, default=4)
    args = ap.parse_args()
    K, seeds, controls, n = (4, [10], 2, 2_000) if args.quick else (200, list(range(10, 20)), 100, 10_000)
    suffix = '_quick' if args.quick else ''
    checkpoint = RESULTS_DIR / f'review_validation{suffix}_checkpoints'; checkpoint.mkdir(exist_ok=True)
    manifest = {'K': K, 'seeds': seeds, 'controls': controls, 'n': n,
                'dataset_sha256': hashlib.sha256(Path('dataset/ai4i2020.csv').read_bytes()).hexdigest(),
                'params_sha256': hashlib.sha256(Path('models/xgboost_params.json').read_bytes()).hexdigest(),
                'protocol_sha256': hashlib.sha256(Path('experiments/REVIEW_VALIDATION_PROTOCOL.md').read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    manifest_path = checkpoint / 'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'Checkpoint settings or sources changed; use a fresh checkpoint directory.'
    write(manifest_path, manifest)
    params = json.loads(Path('models/xgboost_params.json').read_text()); df = load()
    archived = read('audit.json')['primary']; a, p = crossfit_both(df, list(ENGINEERED), params)
    np.testing.assert_allclose([statistics(a, df, mechanism_groups(df))[m]['R'] for m in PHYSICAL],
                               [archived['observed']['all'][m]['R'] for m in PHYSICAL], rtol=1e-5)
    out = {'protocol': 'experiments/REVIEW_VALIDATION_PROTOCOL.md', 'quick': args.quick,
           'settings': {'K': K, 'generation_seeds': seeds, 'negative_controls': controls, 'n': n, 'jobs': args.jobs}}
    out['archive_monte_carlo'] = {name: families(v['observed']['all'], v['all']['null_draws'])
                                  for name, v in [('primary', archived), ('raw', read('audit.json')['raw'])]}
    out['reference_effects'] = {m: {'R': archived['observed']['all'][m]['R'],
          'reference_R': archived['all']['null_mean'][f'{m}.R'],
          'headroom_lift': (archived['observed']['all'][m]['R']-archived['all']['null_mean'][f'{m}.R'])/
                           (1-archived['all']['null_mean'][f'{m}.R'])} for m in PHYSICAL}
    # Holding predictions fixed, enumerate feature-set decoys and shuffle joint flags within failures.
    groups = mechanism_groups(df); failures = df.index[df['Machine failure'] == 1].to_numpy()
    out['feature_decoys'] = {m: disjoint_feature_decoys(a, groups[m], TRIGGER_FEATURES[m]) for m in PHYSICAL}
    out['osf_including_type'] = disjoint_feature_decoys(a, groups['OSF'], TRIGGER_FEATURES['OSF']+['Type_H','Type_L','Type_M']) \
        if len(a.columns)-7 >= 7 else None
    # Seven expected features leave only five disjoint alternatives; compare the actual statistic directly.
    out['osf_type_statistics'] = group_statistics(a, groups['OSF'], np.setdiff1d(failures, groups['OSF']),
                                                  TRIGGER_FEATURES['OSF']+['Type_H','Type_L','Type_M'])
    rng = np.random.default_rng(20261022); subgroup_draws = {f'{m}.{s}': [] for m in PHYSICAL for s in 'RDS'}
    flags = df.loc[failures, PHYSICAL].values
    for _ in range(20 if args.quick else 2000):
        shuffled = rng.permutation(flags)
        for j, m in enumerate(PHYSICAL):
            own = failures[shuffled[:,j] == 1]
            st = group_statistics(a, own, np.setdiff1d(failures, own), TRIGGER_FEATURES[m])
            for s in 'RDS': subgroup_draws[f'{m}.{s}'].append(st[s])
    out['within_failure_flag_reference'] = families(archived['observed']['all'], subgroup_draws)
    # Per-mechanism recall and TWF cohort/representation sensitivity.
    out['oof_recall'] = {}
    for m in PHYSICAL:
        rows = groups[m]; hits = int((p.loc[rows] >= .5).sum())
        out['oof_recall'][m] = {'hits': hits, 'n': len(rows), 'recall': hits/len(rows),
                               'exact_95_interval': list(clopper_pearson(hits,len(rows)))}
    single = df[MODES].sum(axis=1) == 1
    own = groups['TWF']; rest = np.setdiff1d(failures, own)
    single_own = df.index[(df['TWF']==1)&single].to_numpy()
    single_other = df.index[(df['Machine failure']==1)&single&(df['TWF']==0)].to_numpy()
    rest_no_osf = np.array([i for i in rest if df.loc[i,'OSF']==0])
    wear_group = ['Tool wear [min]', 'Overstrain [minNm]', 'Overstrain_ratio']
    out['twf_sensitivity'] = {name: group_statistics(a, own_rows, other_rows, features)
        for name, own_rows, other_rows, features in [
            ('historical',own,rest,TRIGGER_FEATURES['TWF']),
            ('exclude_OSF_comparison',own,rest_no_osf,TRIGGER_FEATURES['TWF']),
            ('strict_single_both_cohorts',single_own,single_other,TRIGGER_FEATURES['TWF']),
            ('wear_representations',own,rest,wear_group),
            ('wear_representations_no_OSF',own,rest_no_osf,wear_group) ]}
    # Fit a genuinely predictive leakage control and break only its test shortcut association.
    leaked = df.copy(); leaked['Shortcut sensor'] = leaked['HDF']+np.random.default_rng(42).normal(0,.25,len(df))
    cols = list(ENGINEERED)+['Shortcut sensor']; tr, va, te = split(leaked)
    pre, model = fit_pipeline(leaked,tr,cols,params,42)
    vp = model.predict_proba(pre.transform(leaked.loc[va,cols+['Type']]))[:,1]
    threshold = f1_threshold(leaked.loc[va,'Machine failure'],vp)
    shifted = leaked.loc[te].copy(); shifted['Shortcut sensor'] = np.random.default_rng(20261022).permutation(shifted['Shortcut sensor'])
    out['predictive_shortcut_control'] = {'sigma': .25, 'threshold': threshold}
    for name, data in [('original',leaked.loc[te]),('shortcut_permuted_test',shifted)]:
        prob = model.predict_proba(pre.transform(data[cols+['Type']]))[:,1]
        out['predictive_shortcut_control'][name] = point_metrics(data['Machine failure'],(prob>=threshold).astype(int),prob)
        mask = data['HDF']==1
        out['predictive_shortcut_control'][name]['HDF_recall'] = float(((prob>=threshold)&mask).sum()/mask.sum())
    print('Saved-evidence diagnostics and predictive controls complete', flush=True)
    out['negative_controls'] = Parallel(n_jobs=args.jobs)(delayed(negative_control)(i,df,params,archived,checkpoint) for i in range(controls))
    out['negative_control_summary'] = support_summary(out['negative_controls'])
    result_path = RESULTS_DIR / f'review_validation{suffix}.json'; write(result_path,out)
    print('Conditional negative controls complete:', controls, flush=True)
    tasks = list(itertools.product(seeds,[None,.25])); records=[]
    stream=Parallel(n_jobs=args.jobs,return_as='generator_unordered')(
        delayed(generated_condition)(seed,sigma,params,K,n,checkpoint) for seed,sigma in tasks)
    for record in stream:
        records.append(record)
        out['generated_rows']=sorted(records,key=lambda r:(r['generation_seed'], r['sigma'] is not None))
        out['complete']=len(records)==len(tasks)
        write(result_path,out)
        print('Generated audit complete:',record['generation_seed'],record['sigma'],round(record['seconds']),
              f'({len(records)}/{len(tasks)})',flush=True)
    out['generated_summary']={str(sigma):support_summary([r for r in records if r['sigma']==sigma]) for sigma in [None,.25]}
    write(result_path,out)
    assert out['complete'] and len(out['negative_controls'])==controls
    print('COMPLETE',result_path,flush=True)


if __name__ == '__main__':
    main()
