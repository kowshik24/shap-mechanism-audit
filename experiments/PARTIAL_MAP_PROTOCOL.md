# Partial-map identification and conservative rank audit

October 25, 2026. Post hoc mathematical/method extension after the existing
Boolean, CWRU, SCA, DTU and physical-chamber outcomes are known. It is not a
preregistered experiment, new physical acquisition, or independently selected
validation set. Preserve all existing model fits, scores and source studies.

## Purpose

An exhaustive physical feature map is not available. Instead of treating unknown
features as excluded, characterize every audit score compatible with a declared
family of maps. For each mechanism let P be required/supported features, N be
independently excluded features, and U the remaining features. Admissible expected
sets satisfy P subset E subset P union U. These constraints are assumptions with
provenance; the interval calculation does not establish their physical truth.

Given normalized own/comparison shares a,b and delta=a-b, calculate sharp ranges:

- R: [sum_P a, sum_(P union U) a].
- D: [sum_P delta + sum_U min(delta,0),
      sum_P delta + sum_U max(delta,0)].
- S: [max_N delta, max_(N union U) delta].

The maximum over an empty outside set is -Infinity, meaning no eligible outside
feature warning, rather than a numerical mechanism score. With N empty, no
outside-excess warning holds for every admissible map. Missing comparison cohorts
still make D/S unavailable; partial-map uncertainty does not create those cohorts.
These are identification ranges conditional on the measured shares, not confidence
intervals for population attribution or inferred causal effects.

For a global rank audit, replace each calibration score by its upper bound and
the candidate score by its lower bound. Use the same strict rank k=ceil((M+1)(1-alpha)),
with +Infinity if k>M. For any admissible actual map, the conservative-warning
event is a subset of the usual exact-score rank-warning event. Therefore the
usual marginal bound is retained under the stated exchangeable reference null.
The rule does not create exchangeability or validate the admissible-map family;
it is not a new conformal theorem. No field error guarantee is claimed here.

## Reanalysis

Reconstruct CWRU equal-recording shares (all three conditions/heads), SCA pooled
case/type shares and the available within-case relevance shares, and every DTU
calibration/evaluation packet's SHAP shares. Use the original partial expected
sets as P and no invented exclusions (N empty). Save a,b, intervals and scope.
Check that R's lower endpoint and S's upper endpoint recover the original
supplied-map results, and that the old D belongs to the new range. For DTU,
reconstruct the original upper-score rank cut and all conservative decisions.
Closed-map recovery uses N=complement(P) only as a labeled mathematical control,
not as physical evidence.

Independently enumerate all completions for the finite measured-bearing feature
sets and test exact endpoint attainment. For the 53-input process setting,
construct endpoint witness maps; enumerating 2^50 completions is unnecessary.
Retain source/model/array hashes and verify all original input records unchanged.

## Acquisition gap

No new physical identity, session log, reset certificate or source protocol is
invented. Preserve the unresolved 25 wind replay mismatches. Prepare a concrete
unsent provenance request identifying the needed original protocol versions,
generator/runtime details, device/session identifiers and reset/clock evidence.
The authors' published non-edge caveat remains controlling evidence against
claiming an exhaustive causal graph from the released adjacency matrices.
