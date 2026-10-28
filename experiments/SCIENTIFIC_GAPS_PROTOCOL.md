# Scientific extension protocol

Written on October 22, 2026 before fitting the extension experiments. This is an
internal, prospectively written extension plan, not an external preregistration.
The authors already know the original study and review-validation outcomes.
The aim is a falsifiable study of reference design, not a guaranteed acceptance
decision or a causal certificate for real machinery.

## 1. Independent measurements and context shortcut

Use the official CWRU 12 kHz drive-end fault table: inner-race, rolling-element,
and outer-race faults at the 6 o'clock position, diameters 0.007, 0.014, and
0.021 inches, loads 0--3 HP. This gives 36 recordings. Exclude 0.028-inch faults,
other outer-race positions, and normal recordings; do not guess a normal-file
sampling rate. The prediction task is fault-location diagnosis among known
faults, not binary failure detection or prospective maintenance.

Download from the university, record checksums and exact labels, and inspect
MATLAB keys and RPM. Stop on missing channel/RPM or inconsistent inventory.
Extract at most the first 24 nonoverlapping 4096-sample drive-end windows per
recording. A window is 0.3413 s. Never split windows from one recording across
training, validation, and test. Physical bearing identities are not supplied;
do not claim independent held-out bearings.

Features are defined before attribution inspection. For each documented drive-end
frequency multiplier (inner 5.4152, outer 3.5848, ball 4.7135), measure the first
two harmonics in raw acceleration and in the Hilbert envelope of a 2--5 kHz
fourth-order Butterworth bandpass. Each band has half-width 0.15 shaft orders;
each power is divided by the corresponding total spectral power. Add log RMS,
kurtosis, crest factor, and relative raw powers in 0--1, 1--2, 2--4, and 4--6 kHz.
Use recorded RPM, not fault labels, to convert orders to Hz. RPM itself is not
a model input. These 19 features provide four mechanism-anchored features per
fault location plus seven broad features. They are hypotheses about signatures,
not exhaustive causal truth: harmonics, sidebands and resonance propagation may
legitimately carry fault information elsewhere.

Outer evaluation holds out each load in turn (four folds). Within the remaining
loads, select among random-forest depths 3, 6, and unrestricted by leave-one-load-
out macro F1; tie-break toward smaller depth. Use 100 trees, minimum leaf 3,
balanced class weights, model seed 2026; fit on recording-balanced windows.
This is nested grouped selection, not random window cross-validation. Report
out-of-load recording majority-vote predictions and window metrics separately.
No exact binomial interval will treat the 864 windows as independent replicates.

Evaluate three conditions, using exactly the same splits: original amplitude;
an explicit acquisition-gain confound (multiply each recording's acceleration by
0.25, 1, or 4 for inner, ball, or outer faults respectively) in training and test;
and removal of that gain in held-out test only, without refitting or selecting
parameters again. Gain changes log RMS while normalized spectral powers,
kurtosis and crest factor are invariant up to numerical precision. This tests
a realistic type of acquisition confound, deliberately constructed using fault
labels; it does not claim discovery of a naturally occurring CWRU artifact.
Report paired recording-level outcome changes and fixed-model attributions.

Use class-specific path-dependent TreeSHAP for multiclass random forests. For
each fault location, explain its probability output on all held-out recordings;
R/D/S use this output consistently on own and other fault recordings. Aggregate
absolute attribution within each recording first, then equally across recordings.
Report leave-one-recording-out descriptive ranges and recording bootstrap
intervals; these describe this inventory, not independent physical bearings.
Include all equal-size disjoint feature-set decoys. A predictive-but-misaligned
gain model is the positive stress test. No outside-set warning is called a
verified false positive on these physical measurements.

## 2. Monte Carlo resolution and scale replication

Replicate 300,000-row generated conditions for seeds 1 and 2, both without an
injection and with the established sigma-0.25 HDF shortcut, alongside the
historical seed 0. Keep historical hyperparameters fixed and report this limit.
For the new runs retain all raw shuffled-label draws and use 1,000 draws when
practical; a fixed 200-draw first block is explicitly preliminary. Checkpoint
blocks so increasing Monte Carlo resolution never discards earlier draws.
Report simultaneous binomial Monte Carlo bounds. Do not call a decision stable
when its interval crosses 0.05. Replication seeds are data-generation seeds;
model/fold seeds are separately recorded. A missing or interrupted block is not
a completed experiment. Do not claim that generated scale establishes field
validity.

### Separate effect replication (added before fitting these runs)

Because the 1,000-draw reference extension is computationally expensive, run a
separate descriptive replication over sizes 10,000, 30,000, 100,000, and 300,000,
generation seeds 0--4, no injection and sigma 0.25. Keep model/fold seed 42 at
all sizes. Save the 40 observed audits without reference-relative p-values.
For each injected fit, also permute the shortcut input within each held-out fold
using seed [20261023, generation_seed, n, fold], then predict without refitting.
Report F1 and HDF recall at fixed threshold 0.5 before/after this test shift.
This provides five-seed replication of effect sizes and direct model reliance,
not a substitute for independent reference draws or a precision claim about
the historical p-values. It must be reported separately from an incomplete
reference extension. The five seeds include the historical generation seeds,
so these are controlled reruns with one fixed model seed, not five entirely
unseen generating processes or independently sampled industrial datasets.

## 3. Calibration and temporal limits

Keep conditional shuffled-label control rates, known-aligned predictor controls,
and predictive-but-misaligned controls separate. Learned classifiers on clean
generated data are not certified aligned. No effect threshold selected on the
existing outcomes becomes confirmatory. Operational calibration must reserve
independent calibration/evaluation seeds and define an actual alignment target;
until that design is implemented and completed, do not claim a 5% false-warning
guarantee. CWRU load-held-out splits address recording leakage, not wear-process
dependence. A separate wear-cycle or chronological study must identify its
independent units and uncertainty before a temporal generalization claim.

## Reporting and stopping

Each extension writes new results and leaves the six historical JSONs unchanged.
Report unfavorable outcomes. Integrate completed evidence into the ten-page
paper by replacing redundant benchmark prose/tables, rather than shrinking
fonts or exceeding the page limit. Remaining gates stay explicit.
