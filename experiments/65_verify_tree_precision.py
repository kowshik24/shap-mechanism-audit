"""Bind the numerical replay to unchanged historical manuscript attributions."""
import pdm_audit
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/results'


def main():
    r=json.loads((OUT/'tree_precision_audit.json').read_text());assert r['complete']
    for p,h in r['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    current=np.load(OUT/'tree_precision_draws.npz');cw=np.load(OUT/'cwru_attribution_draws.npz');sca=np.load(OUT/'sca_field_arrays.npz')
    assert np.array_equal(current['cwru_shap'],cw['signed_shap'],equal_nan=True)
    assert np.max(abs(current['cwru_shap'].sum(axis=3)+current['cwru_expected']-cw['probabilities']))<1e-6
    for view in ['anchors','all']:
        assert np.array_equal(current['sca_'+view+'_shap'],sca[view+'_shap'],equal_nan=True)
        delta=current['sca_'+view+'_shap'].sum(axis=1)+current['sca_'+view+'_expected']-sca[view+'_probabilities']
        assert np.nanmax(abs(delta))<1e-6
    for parent,derived in [('cwru_attribution_audit.json','cwru_attribution_precision.json'),('sca_field.json','sca_field_precision.json')]:
        a=json.loads((OUT/parent).read_text());b=json.loads((OUT/derived).read_text())
        assert b['parent_sha256']==hashlib.sha256((OUT/parent).read_bytes()).hexdigest()
        if 'records' in a and isinstance(a['records'],list):assert a['records']==b['records']
        else:
            for x,y in zip(a['summary'],b['summary']):
                assert x['audit']==y['audit'] and x['within_case_audit']==y['within_case_audit']
    assert len(r['checks'])==32 and r['max_attribution_change']==0
    assert all(x['max_attribution_change']==0 and x['additivity_error']<1e-6 for x in r['checks'])
    files=[OUT/'tree_precision_audit.json',Path(__file__)]
    (OUT/'tree_precision_verification.json').write_text(json.dumps({'verified':True,'all_earlier_attributions_exactly_unchanged':True,
        'sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}},indent=2)+'\n')
    print('PASS every CWRU/SCA attribution and descriptive audit unchanged under equivalent split encoding')


if __name__=='__main__':main()
