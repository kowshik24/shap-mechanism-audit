"""Prediction-equivalent threshold representation for interventional TreeSHAP.

SHAP 0.52's independent-tree routine stores Node.thres as float, while sklearn
stores split thresholds as double and casts prediction inputs to float32.
Rounding a midpoint upward can therefore route a coalition differently. Flooring
each threshold to the float32 lattice preserves every sklearn split decision
on that input lattice and is exactly representable in the SHAP routine.
"""
import copy
import numpy as np
import shap


def float32_threshold_floor(thresholds):
    original=np.asarray(thresholds,dtype=np.float64)
    rounded=original.astype(np.float32)
    return np.where(rounded.astype(np.float64)>original,
                    np.nextafter(rounded,np.float32(-np.inf)),rounded).astype(np.float64)


def probability_explainer(model,background):
    canonical=copy.deepcopy(model)
    for original,tree in zip(model.estimators_,canonical.estimators_):
        mask=tree.tree_.children_left>=0
        thresholds=original.tree_.threshold[mask]
        floor=float32_threshold_floor(thresholds)
        upper=np.nextafter(floor.astype(np.float32),np.float32(np.inf)).astype(np.float64)
        assert np.all(floor<=thresholds) and np.all(upper>thresholds)
        tree.tree_.threshold[mask]=floor
    return shap.TreeExplainer(canonical,data=np.asarray(background,dtype=np.float32),
        feature_perturbation='interventional',model_output='probability')
