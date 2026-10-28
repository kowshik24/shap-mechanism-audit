# Learned-function calibration and transfer protocol

Written October 23, 2026 before fitting this extension. The earlier OR-oracle,
CWRU, chronological, and scale results are already known. This is an internal
prospective extension, not an external preregistration. It changes the calibrated
unit from a fixed function/dataset to an independently trained learner/dataset
pair. It does not claim industrial ground truth or recover hidden mechanisms.

## Finite-world alignment contract

Use three Boolean mechanism variables Z0,Z1,Z2, context C, and true outcome
Y=OR(Z0,Z1,Z2). A mechanism-only learner receives only Z0,Z1,Z2. It is eligible
as a calibration control only if its fixed-threshold (probability >=0.5) decision
equals Y on every one of the 16 possible (Z,C) states and its probability output
is exactly context-invariant (maximum difference <=1e-12). Feature restriction
establishes the latter structurally; full enumeration checks both properties.
This certifies decision correctness and absence of context dependence in this
finite world, not exact Bayes probabilities or general causal faithfulness.

Train each model on 512 independent examples with uniformly distributed Boolean
Z coordinates. Context copies Z0 with 0.01 independent bit-flip probability.
Training labels independently flip the true outcome with probability 0.05.
Audit labels are the uncorrupted outcome. Label noise makes learned probability
functions vary across fits; no model is selected for having a favorable score.
Save every gate failure; do not replace failed seeds or retry hyperparameters.

Three fixed learning procedures are decision tree (maximum depth equal to input
count, minimum leaf 5), random forest (64 trees, same depth/leaf rule, sqrt feature
sampling, bootstrap, one thread), and logistic regression (C=1, lbfgs, maximum
1,000 iterations). Model random state is the case seed. No parameter search or
test-selected threshold is used.

For every learner fit three views on the same training data and labels:
mechanism-only (Z0,Z1,Z2); proxy-only replacement (C,Z1,Z2); and unrestricted
(Z0,Z1,Z2,C). Save all 16 predicted probabilities for each fitted function.
For the latter two, measure maximum context probability difference over the
eight Z states, the count of context-induced decision changes, and full-state
decision agreement. A material context effect means maximum probability
difference >=0.05, selected now rather than from observed warning outcomes.
Neither a restricted proxy-only failure nor its detection is attributed to the
unrestricted model.

## Independent calibration and evaluation

At each audit size n=1,000 and n=10,000, use 200 calibration cases (seeds
310000--310199) and 200 evaluation cases (410000--410199). Each case independently
draws training data, label noise, model fitting randomness, and an audit dataset.
Stream tags distinguish training and audit randomness and the two sizes. The
same case data are shared across algorithms/views to permit paired descriptions;
algorithm-specific frequencies are not independent replications of each other.

Audit generation follows the previously specified overlap family: draw regime
frequency uniformly on [0.1,0.4], independent conditional mechanism probabilities
uniformly on [0.01,0.04] and [0.10,0.30], and overlap u uniformly on [0,1]. Set Z1
on each Z0-positive row with probability u. Context copies Z0 with 0.01 flips.
Evaluate exact model-agnostic interventional SHAP from the complete fitted
probability table and empirical joint audit distribution, retaining the original
single-coordinate expected sets and R/D/S definitions.

Within each algorithm/size stratum, calibrate max_m S_m on the eligible
mechanism-only calibration pairs. With B eligible pairs, use rank
ceil((B+1)*0.95) and strict exceedance. Report the gate denominators. Conditional
on eligibility, independently drawn future pairs from this same procedure form
the exchangeability target; the rank guarantee is marginal over training,
calibration, and audit draws. It is not a conditional guarantee for an arbitrary
fixed fitted model, an exact-probability alignment guarantee, or a transfer
guarantee to continuous industrial inputs. If fewer than 19 calibration pairs
pass, use an infinite threshold, disclose it, and do not claim useful detection.

Report exact binomial intervals for eligible mechanism-only evaluation warnings,
and all proxy-only/unrestricted warning frequencies. Separately report correct
context-localizing warnings (maximizing feature is C and score exceeds threshold)
and frequencies among prespecified materially context-dependent fits. For those
fits report prediction F1 before/after within-audit context permutation, without
refitting. Retain small effects and nondetections.

## Transfer stress tests

1. Apply the historical fixed-OR overlap threshold for the corresponding n to
   eligible learned evaluation pairs, without recalibrating it.
2. Apply the decision-tree stratum threshold to eligible forest and logistic
   evaluation pairs. This is out-of-family use, without a claimed guarantee.
3. On the same evaluation functions, increase overlap to u'=0.8+0.2u using the
   same base mechanism draws, nested Bernoulli uniforms, and context noise.
   This changes only additional Z1 flags on already Z0-positive rows, so true
   aggregate outcomes and context are unchanged. Reuse the original calibration
   threshold. This isolates overlap shift; no validity guarantee is asserted.

Save seed-level audit counts, training hashes, fitted probability tables, gates,
statistics, scores, and all summary denominators. The extension is complete only
when all 2,400 algorithm/size/case records are present (7,200 fitted models across
three views). No partial checkpoint supplies a manuscript result. Every old
result file remains unchanged. Follow-up thresholds or generators require a
separate plan and result file; do not tune this study to improve detection.
