# Prospective training-recipe audit on continuous process trajectories

October 25, 2026. Exploratory follow-up after the DTU schema/prefix preflight,
before extracting the evaluation trajectories or fitting/inspecting any model
outcome below. No statement of preregistration or physical field validity is made.

## Source and units

Use Mode 1, completed IDV4 and IDV5 at 100% magnitude from the author-hosted
DTU release, DOI 10.11583/DTU.13385936.v1. Require Run1–Run200 in both branches.
Each packet combines the two scenarios with the same run number, conservatively
grouping potential shared noise across fault scenarios. A packet, not a timestamp,
is the calibration/evaluation unit. The publication describes independent seeds;
exchangeability of the specified simulation-replicate law is an assumption, not
proof from unequal hashes or evidence of independent physical acquisitions.

Download the complete contiguous processdata array for each selected run through
bounded byte ranges. Preserve source offsets, shapes, raw-payload hashes and exact
extracted values. The full 23.9-GB HDF5 object's MD5 remains unverified. Exclude
Time from model inputs, retaining all 53 process variables. Check that source
injection metadata and the documented 30-hour fault activation agree. For every
run select 16 evenly spaced observations from 10–25 hours and 48 from 40–95 hours,
using nearest distinct native times. These define healthy and active-fault labels
from simulation settings, not a rule computed from prediction inputs. Keep the
entire source run/packet in one role. Do not join unrelated economic arrays by row.

Packets 1–40 are frozen training data; 41–140 are calibration; 141–200 are untouched
prospective evaluation. There is no tuning/validation phase because all settings
are fixed here. Stop on missing required runs, duplicate numerical trajectories
across roles, changed settings, invalid times/values, or zero injection scale.

## Models and the reference population

Train a 32-tree random forest, depth 5, minimum leaf 5, sqrt feature sampling,
balanced classes on the same fixed training corpus for each independently seeded
draw. The reference population is this unaltered training recipe, conditional on
the observed training data, paired with a new trajectory packet. It is not a
population of causally certified models or newly acquired plants.

Calibration draws use seeds 1000+packet ID, one fitted model and one packet per
score. Evaluation draws use seeds 2000+packet ID. The seeds are disjoint. Fit
separate models per unit; do not calibrate on repeated batches of one model.
Freeze 32 equally spaced training background rows for interventional TreeSHAP,
explaining positive fault probability. Save signed attributions, expected values,
model files, training/background row identities, and probabilities.

Partial expected sets are fixed from the named process components:

- IDV4: Reactor Temperature, Reactor Coolant Temp, Reactor Coolant.
- IDV5: Product Sep Temp, Separator Coolant Temp, Condenser Coolant.

These are component-based correspondences, not exhaustive causal truth; propagated
process variables remain potentially legitimate. The two mechanisms are mutually
exclusive, so overlap stratification is not applicable in this study.

## Known acquisition pressure test and direct baseline

For each evaluation seed, also fit a paired forest to a pressure-test training
corpus where Compressor Work has an added 2*IQR offset on active-fault observations.
IQR is computed from unmodified training data only. This deliberately label-assigned
instrument bias is not claimed to be an observed fault in the released simulator.
Use the same model seed/background indices; background values follow that model's
training condition. Evaluate the fixed biased model with matched bias, no bias,
and reversed bias. Never retrain when removing/reversing the test response.

Compute R/D/S and outside-feature maximizers on each packet. A localization event
points to Compressor Work. Also compute a non-SHAP response score by replacing
each feature with its frozen training median and taking absolute prediction change;
use the same own/other-failure excess and maximization. Medians follow each model's
training condition. This is a direct model-response intervention, not a conditional
randomization or physical-causal test.

For each score, use the 100 calibration fit/packet draws to construct a global
strict conformal rank cut at alpha=.05. Warn/localize only above that cut. Under
the specified conditional training-recipe and simulation-packet exchangeability,
the rank target averages over model fitting and a new packet. Report prospective
unaltered-fit warning frequencies, not empirical-size-matched or outcome-selected
thresholds. Do not condition acceptance of reference models on test performance.

## Reporting

Report reference false warnings out of 60 prospective independent fit/packet draws,
with exact binomial intervals conditional on the realized calibration bank and
the stipulated replicate assumptions. Separately give the order-statistic
calibration-bank tolerance bound. Report paired mean F1 under matched/removed/
reversed response, warning/localization counts, cohort availability, and the
distribution of partial-map R/D/S. Include all evaluation units without selecting
harm-positive cases for the primary summary. Distinguish predictive false alarms
from explanation warnings. This validates a specified reference-population
experiment; it does not resolve industrial independence or causal alignment.
