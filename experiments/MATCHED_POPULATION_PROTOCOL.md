# Matched acceptable-population false-warning follow-up

October 25, 2026. Exploratory addition after the third-review coverage audit,
before executing this experiment. Existing studies and protocols are unchanged.

Use all six learner/size strata and the disjoint eligible calibration and
evaluation fits already recorded in experiments 43 and 47. Calibration pairs
are the 200 mechanism-only calibration fits, each with its first target-base
batch. Evaluate the 200 other mechanism-only fits on their 200 untouched test
batches. Candidate localization uses the paired unrestricted models on intact
and independent-context batches. Retain the existing realized-harm definition.

Compare SHAP and flip-excess, globally and with the original fixed overlap bins.
For each calibration bank, compute upper-tail conformal p-values globally or
within its overlap bin. Retain strict SHAP ranks and lexicographic randomized
flip-score ties. Invalid inputs and empty/undersupported strata abstain. At
least 19 calibration pairs are required for a bin in this five-percent study.

Report two distinct comparisons:

1. Apply nominal p<=.05 without looking at evaluated null outcomes. Report the
   independently fitted null warning frequency and harmful-candidate localization.
2. Construct retrospective localization-versus-false-warning curves from the
   evaluated acceptable fits at rates .01, .025, .05, .10 and .20. At a null-score
   atom, randomize between boundary decisions to match the specified empirical
   null frequency in expectation. Apply that same boundary mixture to candidate
   localization. The null score is maximum excess A; the candidate localization
   score is L. This is an empirical ROC comparison, not a threshold selected
   independently of evaluated null outcomes or a deployment guarantee.

Use 400 paired bootstrap replicates. Independently resample the 200 calibration
fit/batch pairs and the 200 evaluated fitted-case clusters with replacement.
Recompute ranks and empirical matched-rate boundaries in every replicate; use
the same resampled indices for both scores, partitions, and context laws.
Within each resampled evaluated case retain all its batches, both its acceptable
model and candidate, and its fixed realized-harm label. Percentile intervals
approximate uncertainty over the observed calibration-pair and case populations;
these are not 400 independently fitted calibration banks or field intervals.
Save bootstrap indices, scores, validity masks, ranks, boundaries and rates so
the comparison can be reconstructed. Report paired flip-minus-SHAP differences.

Do not tune a score, bins, harm floor, or feature map from these outcomes. A
negative result is reported. This study addresses controlled-population size
comparability, not physical alignment, cross-family or field validity.
