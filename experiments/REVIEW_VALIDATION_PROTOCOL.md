# Review validation protocol

Frozen before running `09_review_validation.py` on October 22, 2026. This is an internal analysis plan, not an externally registered preregistration. Earlier results and the review were already observed. It does not retrofit confirmatory thresholds to those results.

## Questions and fixed design

1. **Predictive correspondence controls:** on the released data, keep the fitted predictions and cross-fitted attribution fixed. Compare the original mechanism feature set with all equally sized disjoint feature-set decoys and 2,000 within-failure permutations of the joint physical-flag vector. These controls test feature-set/subgroup correspondence while preserving prediction, not causal correctness. In addition, fit the established strong HDF flag-copy shortcut (sigma 0.25) and measure held-out prediction before and after independently permuting only the shortcut input. This is a synthetic leakage stress test, not a realistic industrial shortcut.
2. **Conditional negative-control calibration:** independently permute aggregate training labels 100 times on the released data, keeping folds/model seed 42 and settings fixed. Compare each with the archived 1,000-draw shuffled-label reference. Report per-statistic support and warning counts with exact binomial intervals, conditional on this dataset and shared reference. The reference is reused and is not a new independent dataset per control.
3. **Generated paired replication:** generate 10,000 rows with generation seeds 10--19, with no injected sensor and with the established sigma-0.25 HDF sensor. For each of 20 conditions, fit a fresh 200-draw shuffled-label reference, keeping training/fold seed 42. Report counts and effect distributions, not an asserted calibrated 5% false-warning guarantee. Uninjected warnings need not be false because a learned classifier can itself be misaligned.
4. **TWF sensitivity:** on the released data, compare original and strictly single-flag cohorts, exclude OSF rows from the comparison, and group the three wear representations (wear, overstrain, overstrain ratio). Report descriptive effects; do not transfer the original permutation p-values to changed statistics. Report per-mechanism cross-fitted recall at threshold 0.5, with exact intervals.
5. **OSF map sensitivity:** include the three type indicators in the OSF expected set, then recompute fixed-model effects and feature-set decoys. Do not overwrite historical feature maps or results.
6. **Monte Carlo uncertainty:** compute Clopper--Pearson intervals for reference exceedance probability. Within each four-mechanism family, use alpha/4 per interval and propagate lower/upper bounds through Holm's monotone adjustment. These simultaneous Monte Carlo bounds quantify finite-reference uncertainty, not uncertainty about causal validity or data generation. Store raw hit counts in all newly generated results.

## Effects and interpretation

Report R minus reference mean and available-headroom normalized lift `(R - mean_ref_R)/(1 - mean_ref_R)`, with raw D/S differences. These descriptive normalizations do not remove feature dependence or make mechanisms exchangeable.

Report S-warning counts at magnitude floors 0, 0.025, 0.05, and 0.10 as an explicitly retrospective sensitivity grid motivated by the existing review. No floor is adopted as a validated operational threshold. Retain unfavorable and favorable outcomes. Do not tune the threshold to eliminate earlier warnings.

## Scope and stopping rule

The script writes a new results file, checkpoints completed tasks, and never overwrites the original six result files. Complete all 100 controls and all 20 paired audits; report failure if any task is missing. A quick option exists only for smoke tests and writes a different filename. No second dataset, temporal validation, fully nested selection, new realistic shortcut, 300,000-row seed replication, or formal error-control theorem is claimed. Those remain open. Timings are recorded; the original model configuration is fixed at every sample size.
