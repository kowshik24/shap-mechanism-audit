import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pdm_audit.tree_precision import float32_threshold_floor,probability_explainer


def test_threshold_floor_preserves_every_float32_side_of_a_midpoint():
    rng=np.random.default_rng(42)
    lo=rng.uniform(-1000,1000,2000).astype(np.float32)
    hi=np.nextafter(lo,np.float32(np.inf))
    thresholds=(lo.astype(float)+hi.astype(float))/2
    floor=float32_threshold_floor(thresholds)
    assert np.array_equal(floor,lo.astype(float))
    for values in [np.nextafter(lo,np.float32(-np.inf)),lo,hi,np.nextafter(hi,np.float32(np.inf))]:
        assert np.array_equal(values<=thresholds,values<=floor)


def test_canonical_explainer_does_not_mutate_the_fitted_predictor():
    a=np.float32(37.258968353271484);b=np.nextafter(a,np.float32(np.inf))
    x=np.r_[np.full(30,a),np.full(30,b)].reshape(-1,1);y=np.r_[np.zeros(30),np.ones(30)]
    model=RandomForestClassifier(n_estimators=1,max_depth=1,bootstrap=False,random_state=0).fit(x,y)
    original=model.estimators_[0].tree_.threshold.copy()
    explainer=probability_explainer(model,x[[0,-1]])
    phi=explainer.shap_values(x,check_additivity=False)
    assert np.array_equal(model.estimators_[0].tree_.threshold,original)
    assert np.max(abs(phi.sum(axis=1)+explainer.expected_value-model.predict_proba(x)))<1e-7


def test_boundary_coalition_values_match_exact_two_feature_shap():
    a=np.float32(37.258968353271484);b=np.nextafter(a,np.float32(np.inf))
    states=np.array([[a,0],[a,1],[b,0],[b,1]],dtype=np.float32)
    x=np.repeat(states,30,axis=0);y=np.repeat([0,0,0,1],30)
    model=RandomForestClassifier(n_estimators=3,max_depth=2,bootstrap=False,
                                 max_features=None,random_state=0).fit(x,y)
    actual=np.asarray(probability_explainer(model,states).shap_values(states,check_additivity=False))[:,:,1]
    base=model.predict_proba(states)[:,1].mean();expected=[]
    for row in states:
        singleton=[]
        for j in range(2):
            hybrid=states.copy();hybrid[:,j]=row[j]
            singleton.append(model.predict_proba(hybrid)[:,1].mean())
        full=model.predict_proba(row[None])[:,1][0]
        expected.append([.5*(singleton[0]-base+full-singleton[1]),
                         .5*(singleton[1]-base+full-singleton[0])])
    assert np.allclose(actual,expected,atol=1e-7)
