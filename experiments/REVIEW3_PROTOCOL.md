# Reference transfer, harmful context shift, and direct-baseline follow-up

October 25, 2026. Written after reading the supplied third review, before running
these follow-up outcomes. This is an exploratory, review-driven extension, not
preregistration. Earlier protocols and result files remain immutable.

## Fixed cases and calibration

Use all 1,200 saved evaluation references/candidates, without eligibility-based
replacement. Retain the existing 1,000 source-base, 1,000 target-base, 400 target
calibration, and 200 untouched test count vectors per case. The null guarantee
is only for the fixed reference f under the stated sampling law. The following
cross-function evaluations explicitly do not inherit that guarantee.

For every reference evaluate mechanism-only fits from the disjoint 310000-series
calibration training seeds: the matching seed offset (evaluation seed minus
100000), both the same learner and each of the other two learners. Verify their
full-state decision equivalence and zero context dependence. Evaluate the saved
proxy-only model separately as a stable-surrogate candidate, not as a member of
the supplied mechanism-only null. Retain its actual intact and shifted F1.

Use alpha=0.05. Compare global target calibration and the original three observed
overlap bins. Additional target-base references use edges estimated exclusively
from source-base overlap quantiles: tertiles and quintiles. Freeze those edges,
then rank-calibrate residuals on the separate 400 target batches. Reuse all test
units across references, candidates and methods. Quantile support and abstention
are reported, not silently repaired.

## Harm and direct baselines

Keep every unrestricted predictor fixed at its original training association
C = Z0 XOR Bernoulli(.01). Target bit-flip probabilities are .01, .10, .25, .50,
and 1.00, with .50 independent context and 1.00 reversal. The .01 counts are the
original test counts. Other laws retain exactly the same mechanism-state counts;
new nested context-flip counts share a separate random stream
[20261025,43,n,case_seed,family_index]. They are not claimed to share the original
.01 uniforms. Labels and all mechanism frequencies remain fixed.

Realized transfer harm is the paired F1 loss from the intact law. The primary
harm-positive denominator is models whose mean F1 loss across their 200 test
batches under independent context (.50) is at least .05. Report all models,
this denominator, pre-shift localization and post-shift localization. Repeat
for each target law descriptively, with no threshold selected from outcomes.
The old maximum-probability-change gate is a reliance criterion, not harm truth.

Non-SHAP baseline 1: for every input j compute h_j(x)=|g(x)-g(x with j flipped)|.
Use the difference in mean h_j between mechanism-m and other failure groups,
maximized over j outside E_m. Calibrate its max-mechanism score using the same
base/target splits and global or observed-bin residual ranks. Localization uses
the same maximizing-context definition and tie order as the SHAP score.

Non-SHAP baseline 2: the targeted context test averages h_C on all audit rows.
All mechanism-only references have exactly zero response. Use a randomized tie
rank: a positive response warns; at zero response compare an independent uniform
test tie-breaker with the rank cut from 400 uniform calibration tie-breakers.
This yields a comparable nominal size for an input known to be context. Its
extra targeting information is explicit; it tests reliance, not transfer harm.
Use one shared calibration/test tie stream per case and reuse across target laws.

## Diagnostics, intervals, and comparisons

For each reference, compute the fraction of target null-score variance explained
by the fixed overlap bins and source-quantile bins. Report direct O/score squared
correlation and localization saturation across cases. The n=1000 and n=10000
models were fitted on different generated training samples; differences in their
conditional material subsets are not a controlled sample-size effect.

Use 1,000 bootstrap draws over the 200 independent case clusters per learner/size,
with shared indices across methods and contrasts. Primary paired sign tests use
all per-case localization frequencies at the intact target law, comparing observed
versus global calibration for SHAP and intervention scores. Holm adjustment covers
12 comparisons (two scores by six learner/size strata). Harm-conditioned summaries,
quantile-bin sensitivity, and individual input diagnostics remain descriptive.

## CWRU attribution audit and exact finite-inventory uncertainty

Replay the saved diameter-fold depths for original and spectral-confounded forests;
check exact recording predictions against the saved results. Use interventional
TreeSHAP probabilities with 96 equally spaced training-window background indices,
shared by original and confounded conditions. Retain signed attribution arrays,
background indices, expected values and additivity residuals. Audit original,
confounded, and confounded-model/filter-removed predictions. Report R,D,S and the
maximizing outside feature per fault head with equal recording weight. Report
whether the maximizer is one of the four broad-band features, without requiring
that outcome. Expected harmonic sets remain partial; these are descriptive audits.

Use all 2^9 paired swaps of matched versus filter-removed predictions at the nine
fault-location/diameter groups for an exact finite-inventory label-swap p-value.
This requires exchangeability of paired conditions under its null and is not a
randomized acquisition experiment or a field-population confidence interval.
Also enumerate the 3^9 ordered class-stratified group bootstrap configurations
with replacement; report exact percentile bounds of that empirical bootstrap law,
with an explicit nine-cluster reliability limitation. No asymptotic precision claim.

## Algebra, complexity and interpretation

Verify corrected intermediate OR-example magnitudes against independent coalition
enumeration at several p,q values. Match the manuscript's invalid-score semantics
to the saved implementation (+infinity before rank cuts; invalid test units abstain;
unsupported-bin residuals use -infinity). Do not change existing result arrays.

Provide a data-unit budget by alpha and number of strata, state why exact Boolean
compression does not scale to arbitrary d, and measure continuous-feature TreeSHAP
runtime in the actual CWRU replay. Distinguish compute from the independent-
acquisition requirement. The constant-offset identity is a remark. Do not claim
that stratification, conformal ranks, or a SHAP dependence test is novel. If direct
baselines dominate or aligned-reference transfer fails, make that a central result.

Before outcome execution, rank handling is specified further: SHAP retains the
existing strict cuts for direct comparability. The finite-difference baseline
uses independent uniform tie breakers at its residual rank cut, because its null
score can have an atom at zero. Calibration/test tie streams are shared across
its threshold variants; invalid test units still abstain. This avoids presenting
an atom-induced zero-size baseline as a matched-size comparison.
