# DTU run-identity follow-up

October 25, 2026. Written after the metadata preflight exposed magnitude-specific
run counts, before reading the signal prefixes below. This is a source audit,
not a preregistered model evaluation. No model is fitted and no fault-detection
or explanation outcome selects data.

For IDV1, IDV2, and IDV4 in the Mode 1 completed branch, inspect Run1 under all
available magnitudes and Run2 under the 100% magnitude. Read only the first
64 process-data rows and the small time_info/idv_init arrays. Record schemas,
finite values, times, exact signal-prefix hashes, and duplicate-prefix pairs.
Also compare prefixes after an exact intersection of their Time column, since
the metadata preflight showed 2000 versus 2001 process rows in different branches.
Require at least 32 common times for this supplementary prefix comparison; do
not interpret shifted starting rows as evidence of distinct random streams.
Do not regard Run1 labels under different faults/magnitudes as independent seeds
without checking the documented generation procedure. Equal prefixes establish
shared observed numerical data; unequal prefixes do not prove independent RNG
streams or independent physical acquisitions. Preserve fault settings separately
from model inputs. Do not promote a partial expected-sensor map to causal truth.

Fetch bounded byte ranges directly from the author's repository, cache and hash
them, and state that the full 23.9-GB object's MD5 was not checked. Retain the
author's published file metadata. Stop if range or byte-budget checks fail.
