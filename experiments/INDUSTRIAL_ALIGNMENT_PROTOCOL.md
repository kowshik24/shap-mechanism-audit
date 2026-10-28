# Measured-system alignment and operating-region validation

Written October 23, 2026 before the new fitted-model results. This follow-up
uses the existing 36 official CWRU recordings; it does not add independent field
machines. Historical results and the finite-world studies remain unchanged.

## Metadata validation

Parse the saved official table independently of the manifest. Cross-check each
recording's table row, download URL, load, fault location, diameter, fixed
outer-race position, and MAT-file RPM. Compare measured RPM to the table's
approximate RPM, reporting every residual and flagging absolute residuals above
50 RPM; this tolerance is a consistency check, not calibration of a speed sensor.
Retain the complete load/location/diameter contingency and source hashes.
Load is the prospective operating-region label. Fault and diameter are evaluation
annotations, not deployment-available inputs. RPM normalizes spectral orders;
none of these annotations enters a classifier directly.

## Learned alignment controls

For each fault location, train a binary one-versus-rest random forest using either
all 19 features or only its four documented raw/envelope harmonic anchors.
Use the existing 100-tree, minimum-leaf-3, class-balanced estimator and seed 2026.
Outer groups are four loads or three diameters. Select depth 3/6/unrestricted
using recording-level F1 averaged over inner held-out groups, with shallower ties.
All windows from a recording stay together; calibration and evaluation labels
never tune models. Report all 42 final fits, 0.5-threshold probabilities,
recording F1, sensitivity, specificity, and complete predictions.

Exhaustively inspect every fitted tree's splitting features. Restricting a head
to four anchors certifies invariance to excluded coordinates for that feature
representation on all inputs; it does not certify correct decisions, complete
physical mechanisms, Bayes probabilities, or independent physical bearings.
Do not select eligible controls using outer performance or attribution scores.

Test two fixed-model interventions on each held-out recording: multiply signal
gain by 0.25 or 4, implemented by the previously verified log-RMS transformation,
and replace the four own-anchor coordinates by their training medians. Report
all probability changes and F1 values, including improvements. The latter is
off-manifold feature masking, evidence of functional reliance only. It is not a
physical repair experiment. Save fitted estimators and training medians so an
independent verifier can replay predictions, masks, and split-support checks.

## Honest calibration feasibility

The paper's S statistic is a mechanism-versus-other-mechanisms batch statistic.
A non-overlapping batch in this inventory comprises one recording for each of
the three fault locations at a fixed load and diameter. This gives three batches
per recorded load, not 24 independent units per recording. Before training and
withholding a new test batch, at most two same-load calibration batches remain;
pooling every other batch gives at most eleven. These are optimistic inventory
counts: model training consumes records, and physical bearing identity is unknown.

Compute the deterministic split-conformal order rank ceil((B+1)(1-alpha)) at
alpha=0.05. If rank exceeds B, the honest threshold is infinity (abstain), not a
bootstrap-generated finite threshold. Report matched and pooled limits, their
minimum attainable strict-rank tail resolution, and the required minimum B=19.
No empirical 5% false-warning guarantee follows from these sparse measured data.
Window bootstrap or repeated fits cannot create independent calibration units.

## Claim and stop rule

Complete all metadata checks and 42 fits; report all failures and negative
outcomes. This is a measured-system feasibility and structural-alignment study,
not industrial mechanism-ground-truth validation. If adequate independent units
or complete physical alignment labels are missing, record that gate as unmet;
do not manufacture a calibrated reference by simulation or repetition.
