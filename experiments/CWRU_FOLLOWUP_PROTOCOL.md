# CWRU feature and shift follow-up

Written October 23, 2026 after the first CWRU load-held-out results, before this
follow-up is fitted. The original full-feature gain effect was weak (two changed
recordings, uncertainty including no degradation). This follow-up investigates
that outcome; it does not tune gain to make the effect larger.

Keep the exact 36-file inventory, 24 windows per recording, 19 features, gains,
random-forest settings, and nested selection criterion. Compare four prespecified
views: all 19 features; the 12 raw/envelope harmonic anchors; the seven broad
features; and log RMS alone. Gain is unchanged at 0.25/1/4 by fault location.
For each view fit original and gain-correlated training conditions, and remove
gain at test time without refitting or threshold selection. Harmonic-only
predictions must be identical before/after gain removal because these features
are gain invariant; this is an intervention-consistency check.

Evaluate two separate axes: the existing four leave-one-load-out folds, and
three leave-one-diameter-out folds. Inner selection leaves out each remaining
load or diameter, respectively, using only outer training groups. With a held-
out diameter this means training inner models on one diameter and validating on
the other. Never split windows from a recording. The two axes use the same
inventory and are not independent replications. Holding out diameter still does
not establish held-out physical bearings because identity is unknown.

Report all 16 axis/view/training-condition results, recording macro F1, exact
correct-recording counts, and fixed-model gain-removed performance. Retain every
selected depth and prediction. The full-feature load result must reproduce the
first study exactly. Do not select the best feature view using outer results or
replace the initial finding with an ablation chosen after observing its outcome.
The table is a diagnostic family, not a new corrected significance test. A model
restricted to log RMS is deliberately constrained; its failure does not imply
the unrestricted model has equal reliance.
