# Known-function calibration protocol

Written October 23, 2026 before running this experiment. Earlier AI4I and CWRU
results are already known. This is an internal prospective extension, not an
external preregistration. It addresses a deliberately narrower target than
causal correctness on physical machinery.

## Target and model classes

Three binary mechanism inputs Z0, Z1, Z2 determine an aggregate outcome by OR.
Each mechanism's exhaustive functional input set is its own Z coordinate.
A latent operating regime creates dependence between mechanisms: its frequency
is sampled uniformly from [0.1, 0.4] for each dataset; conditional mechanism
probabilities are sampled independently from [0.01, 0.04] in regime 0 and
[0.10, 0.30] in regime 1. This represents a distribution over synthetic operating
conditions, not an estimated industrial population. A fourth context coordinate
copies Z0 with independent 0.01 bit-flip probability.

The known-aligned function is f(Z,C)=OR(Z0,Z1,Z2); it ignores context exactly.
The deliberately misaligned function is g(Z,C)=OR(C,Z1,Z2), replacing one actual
mechanism by context. Both are explicit functions, not trained estimators; we
therefore know functional dependence without inferring it from feature rankings.
These are favorable oracle controls and do not validate a learned model.

Compute exact model-agnostic interventional Shapley values by enumerating all
16 input states, all coalitions, and the empirical joint distribution of absent
coordinates. Missing coordinates retain their joint distribution; interventions
break dependence with present coordinates. Compute R/D/S on positive aggregate
outcomes, using the same single-feature mechanism sets for both functions.
Retain overlapping mechanism rows. Outside-set causal variables are legitimate
inputs to the aggregate OR function; S is not itself a causal-error label.

## Calibration and held-out evaluation

Evaluate n=1,000, 10,000, and 30,000 separately. For each n, use 200 independent
known-aligned calibration datasets, seeds 100000--100199, and 200 fresh paired
evaluation datasets, seeds 200000--200199. Separate sizes use independent streams.
Dataset generation, not individual rows, is the exchangeable calibration unit.
The score is max_m S_m across all three mechanisms. Set the threshold to order
statistic ceil((200+1)*(1-0.05)) = 191 of the 200 calibration scores. Warn only
when the evaluation score is strictly larger than this threshold. Ties therefore
produce conservative decisions. This is a split-conformal rank threshold; its
finite-sample marginal bound applies only to exchangeable future datasets under
this same known-aligned function and generator distribution. It neither
calibrates the historical shuffled-label p-values nor transfers to AI4I/CWRU.

Report threshold, known-aligned warning rate, paired misaligned warning rate,
and exact binomial intervals across the 200 independent evaluation datasets.
Also report the predeclared uncalibrated magnitude rule max S > 0.05 and
predictive F1 before/after independently permuting context within the same
evaluation dataset, without changing g. Do not select a threshold or regime
distribution after observing detection results. Retain raw seed-level values
and scores, including unsuccessful detection. Do not call these detection rates
for learned classifiers or an industrial operational guarantee.

## Prospective follow-up: overlapping mechanisms

The first weak-dependence run has degenerate aligned scores: max S is zero at
all three sizes, so a zero threshold separates the simple oracles. This is a
sanity control, not convincing operational calibration. After inspecting that
result, we add a separately labelled overlap stress test before running it.
Do not replace or hide the initial result.

Keep all counts, sizes, functions, feature maps, and calibration/evaluation seed
roles fixed. Draw an overlap probability uniformly from [0,1] per dataset.
After drawing the original mechanisms, set Z1 to one on each Z0-positive row
with that probability. Draw context from the final Z0 as before. Thus some
datasets approach a nested-mechanism setting such as TWF/OSF, but every outcome
still equals the known OR function. Save results to `known_truth_overlap.json`.
Use a separate stream tag to keep this follow-up independent of the first run.
Do not optimize the overlap range or thresholds using its results. Interpret
positive S on this aligned oracle as attribution to another genuine cause,
not evidence of a nonphysical shortcut. Its calibration is specific to this
explicit overlap distribution and function class.
