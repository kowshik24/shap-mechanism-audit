# Oracle operating-region reference follow-up

Written October 23, 2026 after observing the learned and fixed-function transfer
results, before drawing this follow-up's calibration or test batches. The
earlier increase to 22--25.5% warnings motivates this separate prospective test.
No existing evaluation sample or fitted function is altered.

Use all 1,200 eligible evaluation functions and their paired alternatives from
the completed learned-control study. For each fixed function, draw 200 new
calibration batches with overlap u uniform on [0,1] and one fresh high-overlap
test batch with u uniform on [0.8,1]. All other population parameters retain the
previous law. The threshold 0.8 comes from the already fixed transfer stress;
do not tune it or any score floor to this follow-up.

Compare two references using the same calibration collection and fresh test:
(1) all 200 calibration scores; (2) scores only from batches labelled high region,
u>=0.8. The region label is oracle metadata supplied by the simulator, not an
estimated statistic inferred from the evaluation outcomes. The high conditional
calibration law exactly matches the high test law; a stratified rank argument
therefore applies conditional on the fitted eligible function and region, while
remaining marginal over reference/test draws. This is a standard conditional
calibration construction, not a new theorem. It cannot be transferred to real
equipment without defensible region metadata and conditional exchangeability.

The number B of high-region calibration batches is random (mean 40). Use rank
ceil((B+1)*0.95) with strict exceedance, retaining the finite-sample conservatism.
If B<19, use an infinite threshold, report that case as insufficient reference,
and count no warning; never replace it with a new seed. Report B ranges and
counts of infinite thresholds, not just nominal alpha. Alternative functions
reuse their paired mechanism-only reference's threshold.

Retain maximum-S warnings and correct context localization separately, for both
references. Also retain material-context denominators (existing >=0.05 maximum
probability effect), and prediction F1 before/after independently permuting C
within the fresh test batch. Reduced localization need not imply equal population
harm: higher overlap supplies another genuine cause and can change reliance.
Do not describe this oracle reference as an industrial solution.

RNG stream is [20261023,24,n,case_seed,family_index]. Generate calibration counts
with the already verified multinomial population law while saving each u and
all count vectors. Save fresh test/permuted counts and every seed-level score,
threshold, warning, localization, and reference denominator. Use exact binomial
intervals across the 200 independently trained evaluation functions per stratum.
All 1,200 functions must complete. This uses fresh audit batches and existing
fixed functions, and requires no new classifier fitting.
