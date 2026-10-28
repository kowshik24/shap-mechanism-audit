import numpy as np
from pdm_audit.process_audit import score_packet,prediction_response,predictive_metrics


def test_continuous_response_matches_a_known_additive_probability_model():
    class Model:
        def predict_proba(self,x):
            p=.1+x@np.array([.2,.3,.2,0.]);return np.c_[1-p,p]
    x=np.array([[1.,0.,1.,0.],[0.,1.,0.,1.]])
    observed=prediction_response(Model(),x,np.zeros(4))
    assert np.allclose(observed,x*np.array([.2,.3,.2,0.]))


def test_packet_localization_and_missing_cohorts_are_distinct():
    values=np.array([[1.,0.,3.,0.],[1.,0.,3.,0.],[0.,3.,1.,0.],[0.,3.,1.,0.]])
    flags=np.array([4,4,5,5]);expected={4:[0],5:[1]}
    normalized=score_packet(values,flags,expected,2,True)
    response=score_packet(values,flags,expected,2,False)
    assert normalized['valid'] and np.isclose(normalized['score'],.5) and np.isclose(normalized['localization'],.5)
    assert response['score']==2 and response['localization']==2
    missing=score_packet(values,np.full(4,4),expected,2,True)
    assert not missing['valid'] and missing['score']==np.inf and missing['localization']==-np.inf
    metrics=predictive_metrics(np.r_[np.zeros(32),np.ones(96)],np.ones(128))
    assert np.isclose(metrics['F1'],6/7) and metrics['balanced_accuracy']==.5 and metrics['healthy_FPR']==1
