# Exploratory score-design and prospective-process follow-up

Recorded October 25, 2026, after receiving the fourth review and before running
these follow-ups. This is an exploratory amendment, not a preregistration of
the existing paper's headline claims. Existing artifacts remain unchanged.

## Boolean score design

Reuse all 600 evaluated pairs at n=10,000 and their disjoint 200-fit-per-family
calibration banks. Reuse the exact saved intact and independent-context counts,
harm labels, tie uniforms, and 400 paired calibration-bank/case bootstrap draws
from the matched-population experiment. Do not refit or select cases.

Compare a factorial combination of batch versus each model's frozen empirical
512-row training background, normalized versus unnormalized absolute SHAP,
and own-fault versus other-fault or healthy comparison cohorts. Add mean
absolute context SHAP across the entire batch and the existing mean context
flip response. Targeted scores know C; they measure use rather than harm.
Keep original SHAP/flip comparisons under both global and fixed-bin population
ranks. Use global ranks for the factorial and targeted comparisons to isolate
score design. All new scores use strict ranks, except the response comparator
which retains its saved lexicographic randomized ties. Report prospective
nominal size and retrospective matched 5% size separately. Report harmful and
no-harm localization for both laws, with paired bank/case percentile intervals.
Every bootstrap recomputes calibration ranks and the common null boundary;
retain whole model/batch clusters and frozen realized-harm labels.

Recover each empirical training background using training_data(n, case_seed),
including the calibration case seed (evaluation seed minus 100000). Verify
the regenerated training hash against learned_calibration.json. For incomplete
maps, declare P_m={Z_m}, N_m={C}, with other mechanism inputs unresolved.
These exclusions are known only by construction of the Boolean target law.
Calibrate upper bounds on the same disjoint fit/batch bank and evaluate lower
bounds, both globally, at nominal 5%. Report finite cuts, unaltered warnings,
and candidate warnings, without claiming physical exclusions.

## DTU follow-up

Use only packets 1--40 for training and hyperparameter selection, 41--140 for
calibration, and 141--200 for evaluation. Select a forest configuration by
packet-grouped training-only validation balanced accuracy; report F1 alongside
all-positive F1=0.857. Candidate configurations are (trees, depth, leaf,
class_weight): (64, 8, 2, balanced), (64, None, 2, balanced),
(64, None, 1, None). Validation uses packets 31--40, fitting packets 1--30.
No calibration/evaluation outcome chooses the model or bias magnitude.

Each fitted unit samples 40 whole training packets with replacement from
packets 1--40, plus an independent forest seed. Its 32 SHAP background rows
come from its own sampled training corpus. Generate 100 unaltered calibration
fits and 60 unaltered evaluated fits. For each evaluated fit, refit paired
common-fault and IDV4-only label-assigned Compressor Work offsets at
0.1, 0.25, 0.5, 1, 2 times that training sample's IQR. Evaluate matched and
removed offsets with the same forest seed, training packets, and background
indices. Reuse each score's unaltered rank-96 cut across all doses.
Record both scores' localization by mechanism, null warnings, F1 and balanced
accuracy, additivity errors, sampled packet indices, and input hashes. This
extends the training recipe to empirical packet resampling; it does not
establish independent plants or a physically authentic acquisition artifact.
