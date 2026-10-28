"""Paired fitted-proxy strength curves at independently calibrated target size."""
import pdm_audit
import hashlib
import importlib.util
import json
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
from pdm_audit.exact_oracle import STATES
from pdm_audit.learned_reference import FAMILIES, estimator, functional_gate
from pdm_audit.conditional_audit import (METHODS, batch_scores, thresholds_on, residual,
                                        warning, permute_context_counts, table_f1)

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'
spec=importlib.util.spec_from_file_location('conditional_study',ROOT/'experiments/36_observed_overlap.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
ERRORS=[0.,.01,.05,.10,.20,.30,.50]


def case(old, row, counts, critical):
    n,seed,family=old['n'],old['case_seed'],old['family']
    rng=np.random.default_rng([20261023,20,n,seed,1])
    z=rng.integers(0,2,size=(512,3));uniform=rng.random(512)
    y=z.max(axis=1) ^ (rng.random(512)<.05)
    permutation_rng=np.random.default_rng([20261024,37,n,seed,FAMILIES.index(family),1])
    references=[]
    for ref in row['references']:
        references.append({'global':float(ref['global']),'observed':[float(q) for q in ref['observed']]})
    tables=[]; gates=[]; f1=[]; shifted=[]; localizations=[];permutations=[]
    for e_i,error in enumerate(ERRORS):
        x=np.column_stack([z,z[:,0] ^ (uniform<error)])
        fit=estimator(family,4,seed).fit(x,y)
        values=fit.predict_proba(STATES)[:,1]
        if error==.01:
            assert np.array_equal(values,old['views']['unrestricted']['probability_table'])
        _,local,valid=batch_scores(values,counts[e_i])
        flags=[]
        for m_i,method in enumerate(METHODS):
            if m_i<2:
                flags.append(warning(local,critical[m_i],valid))
            else:
                q=thresholds_on(references[0 if method.endswith('source') else 1],counts[e_i],True)
                delta=float(row['matched_offsets'][m_i][2])
                flags.append(np.isfinite(q)&valid&(residual(local,q,valid)>delta))
        permuted=permute_context_counts(counts[e_i],permutation_rng)
        tables.append(values.tolist());gates.append(functional_gate(values))
        f1.append(float(table_f1(values,counts[e_i]).mean()))
        shifted.append(float(table_f1(values,permuted).mean()))
        localizations.append(flags);permutations.append(permuted)
    return {'n':n,'case_seed':seed,'family':family,'tables':tables,'gates':gates,
            'mean_F1':f1,'mean_context_permuted_F1':shifted},np.asarray(localizations),np.asarray(permutations,dtype=np.int16)


def main():
    old=json.loads((OUT/'learned_calibration.json').read_text())
    lookup={(r['n'],r['case_seed'],r['family']):r for r in old['records'] if r['role']=='evaluation'}
    conditional=json.loads((OUT/'observed_overlap.json').read_text());assert conditional['complete']
    draws=np.load(OUT/'observed_overlap_draws.npz')
    counts=draws['strength_counts'];critical=draws['matched_thresholds'][:,:,2,:]
    results=Parallel(n_jobs=4,verbose=10)(delayed(case)(lookup[r['n'],r['case_seed'],r['family']],r,counts[i],critical[i])
                                        for i,r in enumerate(conditional['records']))
    records=[r[0] for r in results];flags=np.array([r[1] for r in results]);permuted=np.array([r[2] for r in results])
    assert np.array_equal(flags[:,1],draws['matched_localization'][:,:,2,:])
    summaries=[];rng=np.random.default_rng(2026102437)
    for n in [1000,10000]:
        for family in FAMILIES:
            indices=[i for i,r in enumerate(records) if r['n']==n and r['family']==family]
            for e_i,error in enumerate(ERRORS):
                material=[i for i in indices if records[i]['gates'][e_i]['material_context_dependence']]
                for m_i,method in enumerate(METHODS):
                    summaries.append({'n':n,'family':family,'error_probability':error,'method':method,
                        'all_cases':base.interval(flags[indices,e_i,m_i].mean(axis=1),rng),
                        'material_cases':base.interval(flags[material,e_i,m_i].mean(axis=1),rng) if material else None,
                        'material_case_n':len(material),'all_case_n':len(indices),
                        'mean_F1':float(np.mean([records[i]['mean_F1'][e_i] for i in indices])),
                        'mean_context_permuted_F1':float(np.mean([records[i]['mean_context_permuted_F1'][e_i] for i in indices]))})
    path=OUT/'proxy_power_draws.npz';np.savez_compressed(path,localization_flags=flags,permuted_counts=permuted)
    paths=[Path(__file__),ROOT/'experiments/REVIEW2_PROTOCOL.md',OUT/'observed_overlap.json',
           OUT/'observed_overlap_draws.npz',OUT/'learned_calibration.json',path]
    out={'complete':len(records)==1200,'new_unrestricted_fits':8400,'error_probabilities':ERRORS,
         'alpha':.05,'methods':METHODS,'records':records,'summary':summaries,
         'matched_base_case_reproduced_exactly':True,
         'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (OUT/'proxy_power.json').write_text(json.dumps(out,allow_nan=False))
    print('COMPLETE: all strengths and paired methods retained; no minimum-detectable-strength claim',flush=True)


if __name__=='__main__':main()
