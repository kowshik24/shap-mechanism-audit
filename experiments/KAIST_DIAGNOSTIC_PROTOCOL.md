# KAIST load-transfer diagnostics: exploratory follow-up

Written October 26, 2026 after the predictive pilot and its six directed source
controls. The controls revealed healthy false positives for a depth-four forest
trained at 0 Nm and evaluated at 4 Nm, whereas training on both 0 and 2 Nm
removed those errors. This diagnostic follow-up does not redefine the original
endpoints or select a primary contrast after its outcome is known.

Reconstruct those two frozen forest fits with seed 20261026 and the original
features, class/record weights, and source-fitted standardization. Reproduce all
saved out-of-record predictions before interpreting explanations. Report all
three loads, marking training loads as in-sample rather than held-out tests.

Explain the summed inner/outer fault probability with interventional TreeSHAP
and a fixed, source-only background of 30 rows: ten per diagnosis class,
selected by equally spaced indices in deterministic feature-table order.
Use no target rows in the background. Check probability reconstruction to 1e-6.
Summarize every original feature's signed and absolute attribution separately
for normal, inner-race, and outer-race rows at every load. Inspect normal 4-Nm
false positives using the five features with the largest mean positive signed
fault-probability contribution; show all 40 features in the saved arrays.

Report the original three-class argmax collapsed to normal/fault and the
summed fault-probability threshold at 0.5 separately. They need not coincide.
Do not call generic spectral features an exhaustive physical map; do not
calibrate an outside-map warning or infer a five-percent false-alarm rate.
Background choice, reference cohorts, and the observed diagnosis/date coupling
prevent a causal shortcut conclusion. No input values or labels are edited.

The report must retain every directed load pair and both learners from the
original pilot, including zero-error controls. The follow-up describes observed
conditional failure in one rig. It does not estimate independent-asset harm
sensitivity, establish a naturally occurring label/condition confound, or prove
that load coverage alone caused the performance change.

Before fitting additional seeds, extend the diagnostic to the fixed twenty-seed
grid 20261026 through 20261045 for each of the same two training-coverage sets.
Keep all other forest/scaling/weighting parameters fixed; evaluate all loads.
Report ranges and all seed outcomes for both binary decision rules, without
asset-population confidence intervals. Seed variation is algorithmic variation
on the same recordings, not twenty independent assets. The original seed is
included, and no seed is selected by its target performance.
