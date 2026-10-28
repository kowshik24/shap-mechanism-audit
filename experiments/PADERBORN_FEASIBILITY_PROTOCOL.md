# Paderborn feasibility gate

Written October 23, 2026 after reading the official dataset documentation, before
constructing the bearing registry, split contract, or numerical capacity audit.
This is a retrospective source audit with a prospective training gate, not an
external preregistration. No signal-level result or fitted model is authorized
as evidence until its stated data requirements pass.

## Evidence and hierarchy

Use the university dataset, operating-condition, and damage-description pages
and the original Lessmeier et al. (2016) benchmark paper. Cache sources with
SHA-256 hashes. Transcribe bearing IDs and components from Tables 4, 5, and 7;
preserve IR (+OR) and OR + IR rather than silently collapsing mixed damage.
Record the hierarchy bearing / operating condition / capture / window. A capture
index is not evidence of a different bearing or an independent acquisition
session. Check actual filenames, headers, and logs only if original archives
are available; distinguish advertised inventories from verified signal files.

## Scientific target

The target is calibration of the paper's mechanism-versus-other-mechanisms S
batch score for an independently aligned fixed model on previously unseen
bearings within a prespecified operating region. This is not ordinary prediction
conformalization or calibration conditional on a previously seen bearing pair.
Matching speed/torque/radial-force labels does not itself imply exchangeability.

Compute both a pure IR/OR pairing capacity and an optimistic upper bound that
allows every damaged bearing. Each nondegenerate S batch requires at least two
different fault-bearing clusters; no cluster may cross training, tuning,
calibration, test, or another independent batch. A mixed-only single-bearing batch
does not supply the required mechanism-versus-other-mechanisms comparison.
Exclude healthy bearings from the original failure-only score. Pooling regions,
repeated captures, windows, or repeated fits does not increase bearing count.

At alpha=0.05 use rank ceil((B+1)*0.95). If rank exceeds B, the finite-threshold
capacity gate fails and the honest threshold is infinity. B>=19 is numerically
necessary, not sufficient: alignment and exchangeability are separate gates.
Evaluate capacity before training/test reservation as an optimistic bound, and
again after reserving at least one test batch.

## Frozen future split

Create a deterministic ID-based development/validation/reserved-calibration/test
partition for pure healthy/IR/OR diagnosis, keeping every condition and capture
from a bearing together. Keep the three mixed/secondary-damage bearings in a
separate challenge partition. Define explicit training-only model selection and
cross-condition evaluation; do not tune on reserved calibration or test labels.
The proposed split is a reviewable future diagnosis contract, not a claim of an
executed or statistically valid mechanism-calibration study.

## Stop rule

Do not train the proposed 5% new-bearing S-calibration study unless source-verified
independent units, adequate disjoint calibration batches, and an independently
justified physical-alignment reference all pass. Localized-fault spectral anchors
alone do not certify exhaustive relevance for distributed or mixed damage.
If a gate fails, retain the source audit, split, counts, and failure reason.
Do not manufacture independent units or change the target after seeing a failure.
