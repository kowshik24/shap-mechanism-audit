#!/usr/bin/env bash
# Regenerate the calibration-scope revision from the frozen reference/process assets.
set -euo pipefail
cd "$(dirname "$0")"
calibration_python="${PYTHON:-.venv/bin/python}"
"$calibration_python" experiments/83_review_audit.py
"$calibration_python" experiments/84_replay_dtu_predictions.py
"$calibration_python" experiments/85_harm_screening.py
"$calibration_python" experiments/86_calibration_assets.py
"$calibration_python" experiments/88_calibration_sensitivity.py
"$calibration_python" experiments/90_logistic_crossover.py
"$calibration_python" experiments/95_verify_kaist.py
"$calibration_python" experiments/97_verify_kaist_diagnostics.py
"$calibration_python" experiments/81_publication_figures.py
make -C IEEE-conference-template-062824 all figures
"$calibration_python" experiments/78_verify_review4.py --enforce-page-limit
"$calibration_python" experiments/82_verify_publication_figures.py
"$calibration_python" experiments/87_verify_calibration_scope.py
