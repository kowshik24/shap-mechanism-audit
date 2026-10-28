# KAIST measured-load pilot: predictive analysis protocol

Frozen before reading vibration values or fitting models, October 26, 2026.
This extends the metadata contract in `KAIST_LOAD_PILOT_CONTRACT.json`.

## Question and scope

Can a classifier trained on two measured torque loads diagnose normal,
inner-race, and outer-race states at the third load? This is a within-rig,
recording-separated transfer illustration. It does not establish field
performance, new-bearing independence, a natural acquisition confound, or a
calibrated mechanism-level warning. All three target loads will be reported.

## Authentication and signal checks

Authenticate the complete vibration ZIP against the publisher SHA-256. Read the
21 normal/BPFI/BPFO MAT records. Report variable structure, timestamp-derived
sample rate and regularity, signal range and finite values, sample count,
record/payload hashes, and exact duplicate-window checks. Use the four vibration
channels in their publisher order, in g. Time and file identifiers are never
predictors. Stop or explicitly amend this protocol if the expected schema or
sampling integrity fails. Do not infer independent asset identities from files.

## Features fixed before target inspection

Extract 30 disjoint one-second windows per native recording. Locate starts
uniformly from zero to the last complete one-second window; require no overlap.
Use all four vibration channels. Per channel, subtract the window mean and
compute log RMS, log standard deviation, Pearson kurtosis, skewness, crest
factor, and five Hann-periodogram relative powers in [0,500), [500,1000),
[1000,2000), [2000,4000), [4000,Nyquist] Hz. Thus there are 40 predictors.
These generic signal features do not imply a certified physical feature map.
Report and fix deterministic numerical faults without choosing features by
target performance. The RMS/standard-deviation redundancy is retained as a
prespecified feature set, not selected after results.

## Folds, training and selection

Use all three leave-one-load-out folds. Within each outer fold, train once on
each of its two source loads and validate on the other source load; select
hyperparameters by mean class-balanced, recording-weighted macro F1 of these
two directed validations. No window from a validation/target recording enters
training. No target-load normalization, background, hyperparameter, threshold,
or feature selection is allowed. Refit the selected model on both source loads.

Compare standardized logistic regression (C in 0.1, 1, 10; max_iter 2000) with
random forest (200 trees, depth 4, 8, or unlimited; min_samples_leaf 2;
max_features sqrt; n_jobs 2). Use seed 20261026, fixed parameter order for ties.
Standardization is fitted to training data only. Weight each diagnosis class
equally, and native recordings equally within class, so one normal recording
has the same total class weight as three fault-severity recordings. Use those
weights for fitting and evaluation. Models have the same available features.

## Endpoints and reporting

Report weighted three-class macro F1, three-class balanced accuracy, binary
normal/fault balanced accuracy, healthy false-positive rate, per-class recall,
and class-balanced confusion matrices. Binary labels collapse both bearing
fault classes; do not tune a binary threshold using target outcomes.
Save every window prediction and native recording identity, including source
validation predictions and selected hyperparameters. Report record-level
majority-vote metrics as a sensitivity analysis (ties use lowest class index).
Describe the six source-to-source directed results as descriptive controls;
their one-source training size differs from the final two-source fit. They are
not a matched causal estimate of deployment harm.

The 21 native records do not supply independent new-bearing replication.
Do not put window-binomial confidence intervals, significance tests, or a 5%
conformal guarantee on these results. Retain negative and zero-loss outcomes.
No sensor offsets or label-dependent signal edits are introduced. Attribution
and designed acquisition-selection experiments require a separate documented
analysis; they cannot retroactively redefine these predictive endpoints.
