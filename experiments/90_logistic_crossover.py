"""Paired design contrasts and affine-logit identities on frozen logistic fits."""
import pdm_audit
import hashlib
import json
from pathlib import Path

import numpy as np
from pdm_audit.exact_oracle import STATES, exact_shap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'experiments/results'


def main():
    design = json.loads((OUT/'score_design.json').read_text())
    reference = json.loads((OUT/'reference_transfer.json').read_text())
    saved = np.load(OUT/'score_design_draws.npz')
    matched = np.load(OUT/'matched_population_draws.npz')
    variants = ['batch_share_failure','batch_raw_failure','frozen_share_failure','frozen_raw_failure','flip_failure']
    rows, points, boots = [], {}, {}
    for key in variants:
        row = next(r for r in design['summary'] if (r['family'],r['score'],r['partition']) == ('logistic',key,'global'))
        points[key] = np.asarray(row['matched_localization'])
        boots[key] = saved['logistic_10000_'+key+'_global_bootstrap']
        rows.append(dict(score=key,rates=row['matched_localization'],ci95=row['matched_ci95']))
    contrasts = []
    for key in variants[:-1]:
        delta = points['flip_failure']-points[key]
        bootstrap = boots['flip_failure']-boots[key]
        contrasts.append(dict(contrast='flip_minus_'+key,delta=delta.tolist(),
            ci95=np.moveaxis(np.quantile(bootstrap,[.025,.975],axis=0),0,-1).tolist()))
        if key != 'batch_share_failure':
            delta = points[key]-points['batch_share_failure']
            bootstrap = boots[key]-boots['batch_share_failure']
            contrasts.append(dict(contrast=key+'_minus_batch_share',delta=delta.tolist(),
                ci95=np.moveaxis(np.quantile(bootstrap,[.025,.975],axis=0),0,-1).tolist()))
    for k,name in enumerate(['SHAP','flip']):
        point = matched['logistic_10000_point'][k]
        bootstrap = matched['logistic_10000_bootstrap'][:,k]
        contrasts.append(dict(contrast=name+'_fixed_bins_minus_global',
            delta=(point[1]-point[0]).tolist(),
            ci95=np.moveaxis(np.quantile(bootstrap[:,1]-bootstrap[:,0],[.025,.975],axis=0),0,-1).tolist()))
    x = np.column_stack([np.ones(16),STATES])
    models, max_error, max_shap_error, max_flip_range = [], 0., 0., 0.
    rng = np.random.default_rng(2026102590)
    for row in reference['records']:
        if row['family'] != 'logistic':
            continue
        probability = np.asarray(row['candidate_table'])
        assert np.all((probability > 0)&(probability < 1))
        logit = np.log(probability/(1-probability))
        beta = np.linalg.lstsq(x,logit,rcond=None)[0]
        error = float(np.max(abs(logit-x@beta)))
        assert error < 1e-10
        background = rng.dirichlet(np.ones(16))
        analytic = beta[1:][None,:]*(STATES-(background@STATES)[None,:])
        difference = float(np.max(abs(exact_shap(logit,background)-analytic)))
        assert difference < 1e-10
        response = np.stack([abs(logit-logit[np.arange(16)^(1<<(3-j))]) for j in range(4)],axis=1)
        flip_range = float(np.ptp(response,axis=0).max())
        assert flip_range < 1e-10
        max_error, max_shap_error, max_flip_range = max(max_error,error),max(max_shap_error,difference),max(max_flip_range,flip_range)
        tag = f"logistic_{row['n']}"
        ids = matched[tag+'_case_indices']
        # Reference records retain their global array index; locate the paired harm label.
        own_index = reference['records'].index(row)
        index = int(np.flatnonzero(ids == own_index)[0])
        harm = bool(matched[tag+'_harm'][index])
        models.append(dict(n=row['n'],case_seed=row['case_seed'],intercept=float(beta[0]),
            coefficients=beta[1:].tolist(),healthy_C1_logit=float(beta[0]+beta[4]),harm=harm,
            healthy_C1_fault=bool(probability[1]>=.5),affine_error=error))
    assert len(models) == 400
    files = [Path(__file__),OUT/'score_design.json',OUT/'score_design_draws.npz',
        OUT/'reference_transfer.json',OUT/'matched_population_draws.npz',
        ROOT/'src/pdm_audit/exact_oracle.py']
    result = dict(complete=True,n=10000,matched_rate=.05,partition='global',replicates=400,
        rates=rows,contrasts=contrasts,models=models,
        affine_logit_max_error=max_error,analytic_SHAP_max_error=max_shap_error,
        logit_flip_max_state_range=max_flip_range,
        healthy_context_threshold_harm_agreement=sum(r['healthy_C1_fault']==r['harm'] for r in models),
        scope='Retrospective paired contrasts from frozen 10k global design ablations. Identities checked on all 400 logistic candidates. Logit flip excess is analytically zero because per-feature responses are constant; floating residuals are not calibrated as signal. These checks do not identify a unique cause for every learner/rate crossover.',
        sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (OUT/'logistic_crossover.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in contrasts:
        if 'fixed_bins' in row['contrast']:
            print(row['contrast'],np.round(100*np.asarray(row['delta'])[0],2),flush=True)
        else:
            print(row['contrast'],np.round(100*np.asarray(row['delta'])[0,0],2),np.round(100*np.asarray(row['ci95'])[0,0],2),flush=True)
    print('Affine-logit/SHAP errors',max_error,max_shap_error,'healthy context gate agreement',result['healthy_context_threshold_harm_agreement'],'/400')


if __name__ == '__main__':
    main()
