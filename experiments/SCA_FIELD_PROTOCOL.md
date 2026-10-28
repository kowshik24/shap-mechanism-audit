# SCA descriptive field audit

October 25, 2026. Written after checking public documentation and download metadata,
before opening SCA raw signals, extracting features, fitting models, or inspecting
predictive/attribution outcomes. This is an exploratory extension, not preregistration.

## Source and statistical target

Use all 22 original train/test MAT files from SCA release 2, authenticated against
SHA-256 values supplied by the official Mendeley API. Preserve original class labels
and metadata. These are real mill records, but case independence, exact physical
bearing identities across replacements, exhaustive mechanism truth, and independent
acquisition sessions are not established by file names. This is a case-held-out
descriptive study, not a five-percent calibrated industrial-warning experiment.

The primary prediction is any recorded bearing fault (labels 1/2/3) versus label 0.
Exclude label -1, invalid/nonfinite signals, missing or nonpositive RPM/fs, and
unavailable frequency anchors; report exclusions. Do not use dates, case IDs,
published train/test indicators, or fault labels as prediction inputs. Native train
files describe healthy periods; both native files follow their case in our split.

## Frozen design

Leave each of cases 1–10 out in turn; train on the other nine cases. Keep every
measurement and sensor position from a case together. Case 11 never enters fitting
or background selection. Evaluate its native train/test periods separately as an
external non-bearing challenge; its published bearing-fault label remains zero.
Do not assume that every measurement in that test period is a verified misalignment.

Use two feature views: twelve frequency anchors and all 21 features. Anchors use
the release's BPFIMultiple, BPFOMultiple and BPFMultiple literally, each multiplied
by RPM/60; do not replace their shaft-reference conventions with guessed geometry.
For each provided multiple use raw and Hilbert-envelope power at harmonics 1 and 2.
Center the raw signal, apply a Hann window for spectral estimation, remove the
envelope mean, and use squared FFT magnitudes. Band half-width is max(3% of its
center, one FFT bin); divide by total non-DC power. An out-of-Nyquist or empty anchor
makes that measurement unavailable rather than assigning artificial zero relevance.
The remaining features are log RMS, kurtosis, crest factor, four power bands with
edges [0,.05,.15,.30,.50] times fs, RPM, and fs. These acquisition variables are not
automatically illegitimate features.

Fit 64-tree random forests, depth 6, leaf minimum 5, sqrt feature sampling,
balanced class weights, fixed random state 2026. Use inverse case record counts as
sample weights. There is no outcome-based feature/model/hyperparameter selection.
Threshold is .5. The two views share folds and background row indices.

Interventional TreeSHAP explains positive-class probability using 96 training-only
backgrounds, allocated as equally as possible across training cases in case-ID order,
then evenly spaced within each case. Save fitted models, row IDs, signed attributions,
expected values, predictions, backgrounds, and additivity errors. No held-out case
or case-11 row enters any background.

## Audit and reporting

For a recorded fault type, its four own-frequency anchors form the partial expected
set. Compute R/D/S from held-out attributions for each type, first averaging absolute
attribution within each case/type, then weighting contributing cases equally.
Comparison groups contain other recorded bearing-fault types. Report the maximizing
outside feature and acquisition-variable share. Empty comparisons abstain. This map
is partial; source labels were manually assigned from related spectra, with inspection
when available. Do not claim independent causal alignment from correspondence.

Clarification before feature extraction/fitting, after a structural read of native
case-1 training metadata: pooled held-out attribution contrasts combine different
fold models and backgrounds. Label them as cross-fitted procedure summaries, not
an audit of one fixed function. Also compute within-held-out-case contrasts and
report their unavailable mechanism/comparison cohorts; do not silently substitute
the pooled contrast for a missing within-model comparison.

Report pooled case-weighted F1, per-case metrics, healthy false-positive rates, and
case-11 native-period false-positive rates for the frozen fitted-model ensemble.
Save labels and all counts; no i.i.d.-window confidence intervals are permitted.
Report the number of cases supplying each mechanism. Treat rates as descriptive
of the observed inventory, not industrial population probabilities.

Fingerprint raw signals across files before fitting. If exact duplicate payloads
cross different cases, stop and revise the grouping contract explicitly; do not
silently train/test across duplicate signals. Repeated measurements within a case
remain in that case and are not counted as independent calibration units.
