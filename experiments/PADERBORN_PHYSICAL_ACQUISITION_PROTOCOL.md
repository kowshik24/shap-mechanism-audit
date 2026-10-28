# Paderborn physical-signature and acquisition audit

Written October 23, 2026 before downloading or inspecting raw signals for this
follow-up. The earlier feasibility contract and 5% new-bearing calibration
failure remain unchanged. This audit fits no predictive model.

## Provenance and protected roles

The university archive host is unreachable from this environment. Inspect a
Zenodo rehosting (record 15845309), retaining its metadata, checksums, creator,
and archive inventory. This is not a publisher-authenticated replacement: MD5
agreement authenticates the downloaded mirror object only, not byte identity
with the inaccessible university original. Continue to cite the university
experiment and respect its stated academic dataset terms.

Select KA04 (outer-ring damage) and KI04 (inner-ring damage), both training
bearings in the frozen split. Inspect all available captures and conditions from
these two bearings without selecting favorable records. Do not inspect raw
reserved-calibration or test signals, and do not use findings to retune a frozen
calibration study. No such model or calibration experiment is executed.

## Acquisition and identity checks

List archive members before extraction, reject unsafe paths, verify mirror MD5,
and retain local SHA-256. Parse MAT headers, native channel names/units, relative
time axes, metadata fields, and any bundled measurement/fact-sheet documents.
Compare condition codes and measured speed/force/torque to documented settings;
report all discrepancies. Search metadata for acquisition/session time, physical
serial number, source recording, and reset/reinstallation evidence. Missing
evidence is unverified, not proof that no session occurred.

Fingerprint complete numerical payloads and each channel separately across all
inspected captures. Report exact duplicate measurements rather than treating
distinct filenames as independent acquisitions. Temporal non-overlap, different
timestamps, weak correlation, and passing an independence test do not by
themselves establish acquisition independence or exchangeability.

## Physical-signature checks

Use the vibration channel and native sampling rate, checking that each signal
contains the stated recording duration. Use native measured shaft speed if its
units are documented; otherwise retain nominal-condition speed and flag that
limitation. Inspect metadata before interpreting a channel's units.

For a localized-fault candidate, use the standard no-slip ball-pass orders
z/2*(1 +/- d/D*cos(theta)). Prefer bearing-specific geometry from an inspected
fact sheet. If unavailable, use only the original paper's example geometry
(z=8, d=6.75 mm, D=28.55 mm, theta=0) as an explicitly approximate anchor.
Geometry/manufacturer uncertainty and slip prevent an exact-mechanism claim.

The fixed signal transform is a fourth-order 2--10 kHz bandpass, Hilbert envelope,
mean removal, and Hann-window periodogram of the complete capture. For each of
the first three own and alternative-location harmonics, search within a half-
width of 0.15*h shaft orders. Compare the peak to the median power in flanks
0.30*h--0.60*h shaft orders from its center. Report peak frequency, order error,
band mass, and peak/flank dB. A descriptive screen requires at least two own
harmonics above 3 dB; this is not a calibrated hypothesis test or a mechanism-
correctness certificate. Retain every result and all alternative-location bands.

Localized spectral agreement supports a partial physical anchor only. It cannot
prove that all legitimate attribution lies in that anchor set, validate mixed or
distributed damage, authenticate mirror provenance, or validate an aligned
industrial reference function. Do not treat 160 captures as 160 bearings.

## Outcomes

Report separately: source-supported damage location, native metadata consistency,
physical-signature support, exact duplicate evidence, acquisition independence,
and exhaustive attribution-map alignment. Mark unsupported targets unverified.
Do not run new-bearing calibration or change its failed capacity gate.

## Metadata clarification before spectral computation

The downloaded KI04 fact sheet records both IR and OR damage, whereas the
summary table lists IR. Retain the selected bearing and both physical candidates;
do not replace it with a favorable pure-IR control. The frozen summary-table
partition is not silently relabelled; a separate physical-label overlay records
that its pure-IR interpretation is invalidated.

Native MAT channel units are empty and HostService timestamps are nonuniform.
Use the log's documented 64 kHz for the primary spectrum and nominal condition
RPM, as specified by the original fallback. Additionally retain a prespecified
sensitivity that interpolates vibration onto a uniform 64-kHz relative-time
grid before the identical transform. Report every paired outcome; do not select
between variants. Numeric speed/force/torque fields are reported without claiming
independently verified units, calibration, or equivalence to nominal setpoints.
MAT export dates, native DateTime fields, and log dates are distinguished.
