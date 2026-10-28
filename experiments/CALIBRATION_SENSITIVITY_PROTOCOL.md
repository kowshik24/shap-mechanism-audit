# Retrospective sensitivity checks

This extension is designed after inspecting the existing study and review. It
is exploratory, not a preregistration or a new confirmatory family. No model,
calibration bank, partition, warning boundary, or primary result is retuned.

For every frozen candidate, generate 200 fresh target-law batches using the
original batch generator and an independent random stream keyed by
`[20261025, 88, n, case_seed, family_index, 0]`. Pair each with independent-context
counts retaining its exact mechanism counts. Recompute mean F1 loss and the
0.05 harm label. Evaluate the existing global five-percent screening boundaries
on the original audit batches using these independently obtained labels. Report
label changes and conditional detection rates; do not replace original labels
or reinterpret retrospective results as a held-out confirmatory study.

On the original audit batches, repeat the single-permutation F1 screen with ten
additional independent permutation streams keyed by
`[20261025, 88, n, case_index, law_index, replicate, 1]`. Keep the original
calibration ties, boundaries, and harm labels. Report the range of the ten
empirical screening frequencies separately from bootstrap intervals. This
measures randomization sensitivity, not between-model sampling uncertainty.

The manuscript length pass removes duplicated displays and moves supporting
derivations to external technical materials. IEEE type size and margins remain
unchanged. Original result files and the standalone native-vector figures are
preserved.
