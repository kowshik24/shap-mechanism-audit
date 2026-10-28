#!/usr/bin/env bash
# Reproduce the completed scientific extensions; use the pinned project environment.
# The optional multi-hour 14_scale_replication.py is not needed for reported effects.
set -euo pipefail
cd "$(dirname "$0")"
scientific_python="${PYTHON:-python}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/pdm-matplotlib}"

if [[ ! -f data/cwru/manifest.json ]]; then
  "$scientific_python" experiments/11_cwru_download.py
fi
"$scientific_python" experiments/12_cwru.py
"$scientific_python" experiments/13_known_truth.py
"$scientific_python" experiments/13_known_truth.py --overlap
"$scientific_python" experiments/15_temporal.py
"$scientific_python" experiments/16_effect_replication.py
"$scientific_python" experiments/18_cwru_followup.py
"$scientific_python" experiments/17_scientific_assets.py
"$scientific_python" experiments/20_learned_calibration.py
"$scientific_python" experiments/21_fixed_learner.py
"$scientific_python" experiments/24_operating_region.py
"$scientific_python" experiments/26_region_budget.py
"$scientific_python" experiments/22_verify_learned.py
"$scientific_python" experiments/25_verify_region.py
"$scientific_python" experiments/27_verify_region_budget.py
"$scientific_python" experiments/23_learned_assets.py
"$scientific_python" experiments/28_industrial_alignment.py
"$scientific_python" experiments/29_verify_industrial.py
"$scientific_python" experiments/30_industrial_assets.py
"$scientific_python" experiments/31_paderborn_feasibility.py --download-documents
"$scientific_python" experiments/32_verify_paderborn.py
"$scientific_python" experiments/33_paderborn_physical_acquisition.py --download
"$scientific_python" experiments/35_physical_acquisition_assets.py
"$scientific_python" experiments/34_verify_physical_acquisition.py
PYTHON="$scientific_python" bash reproduce_review2.sh
"$scientific_python" experiments/19_verify_science.py
