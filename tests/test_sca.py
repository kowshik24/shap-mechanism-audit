import numpy as np
import pandas as pd
import pytest
from pdm_audit.sca import features,correspondence,ANCHORS,FEATURES


def test_spectral_anchors_have_the_documented_scaling_and_frequency_response():
    t=np.arange(4096)/256
    x=np.sin(2*np.pi*10*t)+.4*np.sin(2*np.pi*20*t)
    a=features(x,256,600,{'IR':1.,'B':.5,'OR':1.5})
    b=features(3*x,256,600,{'IR':1.,'B':.5,'OR':1.5})
    assert set(a)==set(FEATURES)
    assert a['raw_IR_1']>.8 and a['raw_OR_1']<.001
    assert np.isclose(b['log_RMS']-a['log_RMS'],np.log(3))
    for key in ANCHORS:assert np.isclose(a[key],b[key],atol=1e-12)
    with pytest.raises(ValueError,match='unavailable_anchor'):
        features(x,256,600,{'IR':20.,'B':.5,'OR':1.5})


def test_relevance_survives_a_missing_comparison_and_case_weights_ignore_replication():
    metadata=pd.DataFrame({'case':[1,2,3],'label':[1,2,3]})
    values=np.arange(1,37,dtype=float).reshape(3,12)
    pooled=correspondence(values,metadata,ANCHORS)
    repeated=correspondence(np.r_[values,values[:1].repeat(20,axis=0)],
                           pd.concat([metadata,metadata.iloc[:1].loc[metadata.index[:1].repeat(20)]],ignore_index=True),ANCHORS)
    assert pooled==repeated
    own=correspondence(values[:1],metadata.iloc[:1],ANCHORS)
    assert not own[0]['available'] and own[0]['reason']=='missing_other_failure_cohort'
    assert np.isclose(own[0]['R'],values[0,:4].sum()/values[0].sum())
