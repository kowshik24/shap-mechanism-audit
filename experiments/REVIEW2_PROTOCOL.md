# Second-review scientific revision

Written October 24, 2026 after reading review 2 and the existing results, before
the new calibration, proxy-strength, and spectral-confound outcomes. Existing
studies remain archived. This is a review-driven follow-up, not preregistration.

## Primary corrections and estimands

Equal reference counts do not imply equal detector size. For a fixed fitted
function, uniform and latent-region references threshold the same scalar S
score. Their population ROC is identical under strictly monotone calibration;
different thresholds alone cannot establish a special localization penalty.
Treat the previous unequal-size comparison as operating-point sensitivity.

The latent overlap parameter u is not the observed conditional frequency:
the latter also includes baseline Z1 occurrences and finite-sample variation.
Compute observed overlap O = count(Z0=1,Z1=1)/count(Z0=1) from recorded flags.
Fixed bins are [0,0.8), [0.8,0.9), [0.9,1]. No test-driven bin selection occurs.
Conditioning on these bins guarantees a rank bound only when calibration and
future scores are exchangeable within the same bin. It does not automatically
repair a latent-population shift.

## Fresh fixed-function calibration and size/power analysis

Reuse the 1,200 eligible evaluation functions from the completed learned study
(200 cases in each learner/size stratum); do not replace any function. For each,
sample 1,000 source-population batches (u uniform on [0,1]), 1,000 target-population
batches (u uniform on [0.8,1]), 400 additional independent target size-tuning
batches, and 200 untouched target test batches. Retain all 16-state count vectors.
Use seed stream [20261024,36,n,case_seed,family_index]. All tests are paired.

Compare four reference procedures: global/source, global/target, observed-bin/
source, observed-bin/target. Algorithms receive count vectors/flags, never latent
u. Keep latent parameters only as generator provenance. Empty groups or
undefined statistics cause abstention; unsupported bins use an infinite threshold.
The base conditional thresholds use rank ceil((B_bin+1)*0.95).

For size-matched analysis, freeze the base reference and define residual score
T=A-Q_base(O). Independently size-tune T with the 400 target batches, using
rank ceil(401*(1-alpha)), for alpha in {0.01,0.025,0.05,0.10,0.20}. Unsupported
bins have T=-infinity and abstain. Under the same target law, the standard rank
bound is marginal over tuning/test data conditional on the fitted function and
base bank. It is not conditional on a realized tuning bank and does not validate
all physically aligned models. Global procedures use the identical canonical
raw-score tuning threshold, since their constant offsets cancel exactly.

For each method/alpha report null warning rate, abstention rate, and context
localization frequency on the fixed unrestricted alternative. Localization means
some warned mechanism has context as its maximizing outside feature. Also
report the max probability-difference >=0.05 material-dependence denominator.
Summaries average case-level test frequencies; confidence intervals resample
the 200 case clusters, not individual batches. Do not call inclusion of 5% in
an interval proof of calibration.

At primary alpha=0.05, compare global vs each observed-bin method with exact
paired McNemar tests using the first untouched test batch per case, restricted
to the same material alternatives. Holm adjustment covers all 12 comparisons
(two contrasts across six learner/size strata). Other curve points are descriptive.
Global-vs-global identity is a design fact, not an empirical discovery.

## Proxy-strength curve

Fit unrestricted alternatives at context bit-flip probabilities
{0,0.01,0.05,0.10,0.20,0.30,0.50}, holding each case's training Z, label noise,
random uniforms, and estimator settings fixed. The 0.01 case must reproduce the
saved unrestricted probability table exactly. Use common test Z counts and
nested context-error counts, seed [20261024,37,n,case_seed,family_index].
Permute context using conditional hypergeometric allocations, preserving the
observed context total. Save every fitted 16-state table, test count vector, and
functional-dependence gate. Report localization both across all alternatives and
conditional on material context dependence, plus F1 before/after permutation.
The primary threshold is the size-matched global rule at 5%; compare observed-
bin variants without searching the best strength/threshold. This grid does not
identify a universal minimum detectable shortcut or a Gaussian-noise equivalent.

## Measured spectral confound and partial perturbation

Use the same 36 official CWRU fault recordings and first 24 windows. Primary
outer holdout is fault diameter; load holdout is a secondary, within-bearing
operating-condition comparison, not independent-bearing generalization.

Assign a smooth acquisition resonance by location: centers 2.5/3.5/4.5 kHz for
ball/inner/outer, width 250 Hz, frequency response
H(f)=1+3 exp(-(f-center)^2/(2*250^2)). Apply to centered raw windows with rFFT,
then normalize back to the original RMS. This removes scalar amplitude as the
shortcut and changes spectral shape. Parameters are fixed before outcomes.
Fit original and resonance-confounded 19-feature forests with existing nested
diameter depth selection. Remove the acquisition filter at held-out evaluation
without refitting. Retain all predictions, feature changes, and a paired,
class-stratified recording-bootstrap interval for the F1 difference (10,000 draws).
This is a specified transfer-function artifact, not a discovered natural shortcut.

Replay the existing restricted binary heads with one anchor at a time replaced
by its training median, retaining the other three. Report every anchor and
whether predictions become constant; do not use full masking as evidence.
Single-anchor masking is a model sensitivity check, not on-manifold or physical
repair evidence. Gain invariance is stated as a feature-definition property.

## Writing and stopping

The main paper centers on the audit decision procedure, its rank assumptions,
matched-size/observed-overlap analysis, proxy-strength curve, and one measured
diagnosis case. Chronological, Paderborn, and scaling studies remain in the
repository archive, not an appended submission: the conference prohibits an
appendix. Do not add artificial Big Data/time-series claims or invent fleet
inspection outcomes. Align local title, PDF metadata, and prepared submission
metadata; portal state is unverified without access. Report all new outcomes,
including disappearance of the claimed localization effect.
