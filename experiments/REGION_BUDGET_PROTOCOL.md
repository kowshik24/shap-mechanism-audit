# Equal-budget oracle-region confirmation

Written October 23, 2026 after the bank-filtered region experiment, before the
new calibration or test draws below. Filtering a 200-batch bank leaves only
23--61 high-region references in that completed run. This follow-up separates
reference-population choice from effective reference count. Neither the high
region cut (u>=0.8) nor any effect floor is changed.

For every previously eligible fixed evaluation function, draw a fresh uniform-
overlap calibration bank of 200 batches, a separate targeted high-overlap bank
of 200 batches, and one fresh high-overlap evaluation batch. Compare:

1. all 200 uniform-bank scores, an unmatched population reference;
2. the high-region subset of that bank, retaining its random denominator;
3. all 200 targeted high-bank scores, an equal-count matched-population reference.

All references use the same known mechanism-only fitted function. The new test
batch and paired alternative functions are identical across reference choices.
Model parameters remain fixed. Region metadata remain oracle knowledge supplied
by the simulator. Conditional rank validity applies to the two matched references
under their stated sampling laws, not to real data or an estimated region tag.
Report every reference, all warning/localization denominators, and F1 before/
after context permutation. A reduced localization rate is not labelled an
intrinsic impossibility theorem; it is an empirical finding for these functions,
scores, populations, and reference budgets.

Use RNG stream [20261023,26,n,case_seed,family_index]. Retain both 200-batch banks,
overlaps, scores, all rank thresholds, fresh test/permuted counts, and all output
statistics. Complete all 1,200 functions with no outcome-based replacement.
This gives 480,000 additional count-represented calibration batches and 1,200
new high-region tests, with no new model fitting. The earlier bank-filtered
study remains unchanged and available as a separately labelled sensitivity.
