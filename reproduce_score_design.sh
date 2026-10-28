#!/usr/bin/env bash
# Focused exploratory follow-up; existing reference and DTU source artifacts required.
set -euo pipefail
cd "$(dirname "$0")"
followup_python="${PYTHON:-.venv/bin/python}"
"$followup_python" experiments/75_score_design.py
"$followup_python" experiments/76_dtu_dose_response.py
"$followup_python" experiments/79_dtu_predictive_baselines.py
"$followup_python" experiments/83_review_audit.py
"$followup_python" experiments/84_replay_dtu_predictions.py
"$followup_python" experiments/85_harm_screening.py
"$followup_python" experiments/86_calibration_assets.py
"$followup_python" experiments/88_calibration_sensitivity.py
"$followup_python" experiments/90_logistic_crossover.py
"$followup_python" experiments/77_review4_assets.py
"$followup_python" experiments/95_verify_kaist.py
"$followup_python" experiments/97_verify_kaist_diagnostics.py
"$followup_python" experiments/81_publication_figures.py
make -C IEEE-conference-template-062824 all figures
"$followup_python" experiments/78_verify_review4.py --enforce-page-limit
"$followup_python" experiments/82_verify_publication_figures.py
"$followup_python" experiments/87_verify_calibration_scope.py
