#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper from the raw dataset.
#   ./reproduce.sh          full run (about a day on 8 cores; per-stage times in README.md)
#   ./reproduce.sh --quick  smoke test of the audit stages with tiny K (a few minutes)
set -euo pipefail
cd "$(dirname "$0")"

python -m pip install -q -r requirements.txt -e .
python -m pytest -q

python experiments/01_benchmark.py
python experiments/02_audit.py "$@"
python experiments/03_shortcut_sweep.py "$@"
python experiments/06_explainers.py "$@"
python experiments/07_scale.py "$@"
python experiments/08_compare.py "$@"
python experiments/09_retuned_null.py "$@"
[[ "${1:-}" == "--quick" ]] || python experiments/09_retuned_null.py --extend-to 500
python experiments/10_effect_sizes.py
python experiments/11_twf_diagnostics.py "$@"
python experiments/12_other_explainers.py "$@"
python experiments/13_review_checks.py "$@"
python experiments/13_review_checks.py --kernel "$@"
python experiments/14_cwru_audit.py "$@"
for opt in "--band 0.05" "--band 0.3" "--harmonics 1,2,3"; do
  python experiments/14_cwru_audit.py $opt --conditions original "$@"
done
python experiments/15_tep_audit.py "$@"
python experiments/15_tep_audit.py --window 10 "$@"
python experiments/15_tep_audit.py --null shift "$@"
python experiments/15_tep_audit.py --window 10 --null shift "$@"
python experiments/16_more_checks.py "$@"
python experiments/17_final_checks.py "$@"
python experiments/18_dependence.py "$@"
python experiments/18_dependence.py --bw 0.15 "$@"
python experiments/18_dependence.py --bw 0.6 "$@"
python experiments/18_dependence.py --split-wear "$@"
python experiments/19_flag_threshold.py
python experiments/20_rank_checks.py "$@"
python experiments/20_rank_checks.py --transform positive "$@"
python experiments/22_tep_lstm.py "$@"
python experiments/23_twf_example.py
python experiments/24_multi_proxy.py "$@"

if [[ "${1:-}" != "--quick" ]]; then
  python experiments/04_tables.py
  python experiments/05_figures.py
  python experiments/21_reference_figure.py
  (cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
     && latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex >/dev/null)
  echo "paper: paper/main.pdf, paper/supplement.pdf"
fi
