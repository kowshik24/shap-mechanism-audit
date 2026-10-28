# Physical intervention and feature-map verification

October 25, 2026. Exploratory source-validation study, fixed after reading the
publisher documentation/configuration tables and before opening numerical
measurement CSVs. No predictive outcomes or calibration thresholds are selected.

## Source and scope

Use the authors' `lt_validate_v1` and `wt_validate_v1` physical acquisitions,
dataset repository commit `0c8a77ea14bd4437b1fdd79adc1aa91ac2342569` and graph
package commit `9fb5d82e391bb91a89a64f2e1c9b6ae8e7aed6d6`. Download complete ZIPs
from the author-linked S3 URLs; require the MD5 values published in the pinned
README. Preserve each member, its ZIP CRC and SHA-256, configurations, generator,
variable dictionary and graph. Cite Gamella, Peters and Bühlmann (2025),
doi:10.1038/s42256-024-00964-x.

These are physical light/wind tunnel acquisitions, not maintenance failures or
independent new machines. A file/protocol is a distinct acquisition sequence.
Rows, target changes and randomized settings do not become independent machines.
Neither a unique timestamp nor a distinct hash proves statistical independence.

## Acquisition checks

For every source configuration, identify the matching measurement file. Replay
the published generator RNG (including discarded waiting-time draws) and verify
every treatment flag and intervention setting. Check row counts, counters,
timestamps, other fixed controls, finite measurements and duplicate numerical
payloads. Record timestamp overlap and experiment setup/waiting times. Do not
silently drop a failed acquisition; report its specific failed check. The
generator's fixed seed documents an intended randomization schedule; replay does
not certify physical reset, absence of carryover, random-stream independence or
exchangeability between different configurations. No new-machine calibration
claim is made.

## Exhaustive assessment over a finite feature universe

The light measurement universe is current, two measured angles, six visible/IR
readings, and the two recorded board/regulator voltage diagnostics (11 channels).
The wind universe is two currents, two RPMs, four pressures, microphone, and two
electrical signal readings (11 channels). Controls and administrative fields are
not diagnostic features. Assess every configured intervention against every one
of these channels; no correlation-based feature selection or new fitted graph.

For each intervention/channel, retain the authors' direct/path/no-listed-path
status. A missing path is **unknown**, not a physically excluded effect. The
authors explicitly state that graph non-edges do not preclude causal effects.
Keep diagnostic channels absent from the source graph as unmodeled rather than
deleting them to obtain a complete-looking map.

Independently screen randomized responses with two two-sided Fisher exact tests:
the treatment versus above-pooled-median indicator and versus above-median
absolute deviation indicator. Thresholds are fixed from the unlabelled pooled
outcome. These tests condition on treatment count and are randomization tests of
the global sharp no-effect null under the stipulated independent fair assignment
design and a stable recorded protocol. Serially dependent outcomes alone do not
require independent sensor readings under that null; carryover under alternatives
prevents an instantaneous/direct-effect interpretation. Adjust the entire
two-dataset/intervention/channel/endpoint family with Holm at 0.05. Save the raw
2x2 counts, risk differences and descriptive mean/median contrasts.

Rejection is evidence of a protocol-specific distributional response, not a
complete structural graph. Failure to reject is unknown, not equivalence or no
effect. A pair is supported if a published path or adjusted response test supports
it; otherwise it remains unresolved. The result is an exhaustive *assessment*
of this finite intervention/channel grid, not an exhaustive causal truth map.
No negatively excluded feature is invented. No conformal mechanism warning is
calibrated unless an eligible physical acquisition population and complete
comparison cohorts are independently justified.

## Deliverables

Source/member manifest, acquisition ledger, every-pair feature-map CSV, all
randomization tables and multiplicity adjustments, source-graph coverage,
independent offline verifier, and a concise report distinguishing the positive
physical evidence from unmet independence/causal-completeness requirements.
