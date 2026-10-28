#!/usr/bin/env bash
# Reproduce the bounded measured-load pilot; first run downloads 2.66 GB.
set -euo pipefail
cd "$(dirname "$0")"
kaist_python="${PYTHON:-.venv/bin/python}"
"$kaist_python" experiments/92_kaist_download.py
"$kaist_python" experiments/93_kaist_features.py
"$kaist_python" experiments/94_kaist_predictive.py
"$kaist_python" experiments/95_verify_kaist.py
"$kaist_python" experiments/96_kaist_diagnostics.py
"$kaist_python" experiments/97_verify_kaist_diagnostics.py
"$kaist_python" experiments/98_kaist_assets.py
cd IEEE-conference-template-062824/figures
latexmk -pdf -interaction=nonstopmode -halt-on-error kaist_load_pilot_standalone.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error kaist_load_controls_standalone.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error measured_load_plot_standalone.tex
