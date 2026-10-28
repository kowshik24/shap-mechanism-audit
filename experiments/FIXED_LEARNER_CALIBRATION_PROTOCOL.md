# Per-fitted-function calibration follow-up

Written October 23, 2026 after the first learned-study runs, before computing
this follow-up. The pooled learner/dataset reference has a marginal target over
training randomness. This follow-up asks whether holding an eligible fitted
function fixed changes warnings and reliance detection. It is a separate
prospective follow-up, not a retroactive change to the pooled experiment.

Use every eligible mechanism-only evaluation function from the completed full
`20_learned_calibration.py` output, with its paired proxy-only and unrestricted
functions. Do not select functions by score or detection outcome. At each fixed
function, independently draw 200 new audit datasets from the original uniform-
overlap population and calibrate maximum S at order statistic 191. Reuse this
threshold on the original evaluation audit, its paired increased-overlap audit,
and the paired alternative functions. The reference function is the eligible
mechanism-only function throughout; do not calibrate a proxy-using function
against itself as if it were aligned.

The rank guarantee now conditions on this particular fitted eligible function
and is marginal over its independent calibration and future audit datasets.
It still does not condition on a realized calibration set, transfer across
overlap populations, or apply to industrial models without a valid alignment
reference. The alternative functions are power controls, not guaranteed-coverage
targets. Report any warning separately from correct context localization and
from material context dependence (the existing >=0.05 probability-difference
criterion). Preserve all nondetections.

Calibration audit counts are sampled directly from the 16-state distribution.
Given regime-specific base probabilities p0,p1,p2 and overlap u, the effective
Z1 probability is p1+(1-p1)*u*Z0. Context probability is 0.01+0.98*Z0. Mix the
two independent-regime probability products using the randomly drawn regime
frequency, then sample n counts multinomially. This is the same count law as
the original iid row generator, not a new generator family. Unit checks verify
normalization, marginals, and equivalence to scalar exact attribution statistics.

For speed, exploit linearity of signed interventional Shapley values in the
background distribution: enumerate the 16 one-state backgrounds once per
function, combine the signed values for each sampled count distribution, and
only then take absolute values and normalize group means. Do not average
absolute values across backgrounds. Save all 200 count vectors and resulting
scores per eligible function in a compressed NPZ, plus seeds, hashes, thresholds,
warning indicators, and aggregate exact binomial intervals in a separate JSON.

Calibration RNG stream is [20261023,21,n,case_seed,family_index]; the original
evaluation counts and functions remain unchanged. The output is complete only
when every eligible evaluation function has all 200 calibration datasets and
the alternative-function evaluation indicators. No partial result enters the
paper. This study does not require refitting the 7,200 models.
