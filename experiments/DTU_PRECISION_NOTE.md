# Numerical representation correction before prospective outcomes

October 25, 2026. The first DTU calibration run stopped at the required SHAP
additivity check, before producing a complete calibration bank or prospective
results. Calibration model 54 had a probability residual .0053686177 on one row;
the SHAP internal predictor and sklearn predictor agreed exactly. Casting inputs
and backgrounds to float32 did not fix it. No tolerance was relaxed.

The pinned SHAP 0.52 implementation in `shap/cext/tree_shap.h` stores the
interventional independent routine's `Node.thres` and local threshold as float.
Sklearn stores thresholds as double but predicts on float32 inputs. In the failing
path, a threshold halfway between adjacent float32 values rounded upward inside
the independent routine. The source is:
https://github.com/shap/shap/blob/v0.52.0/shap/cext/tree_shap.h

For a double threshold t, let q be the largest float32 value no greater than t.
Every finite float32 x satisfies x<=t iff x<=q, since no float32 lies strictly
between q and t. Applying that replacement to every split therefore preserves
the sklearn prediction function on its input lattice, including all hybrids
formed from explained and background inputs. The helper deep-copies the model;
the fitted model and its saved file are unchanged. Backgrounds are represented
at the same float32 input precision. The failing residual falls below 1e-8.

This is a numerical implementation correction, not a different explanation game,
new learned model, tolerance relaxation, or selection on predictive performance.
The experiment retains its frozen data roles, settings, partial feature map and
pressure test. Earlier continuous-model attributions are separately audited for
the same representation issue; Boolean exact-enumeration results are unaffected.
