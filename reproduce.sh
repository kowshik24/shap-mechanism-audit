#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper from the raw dataset.
#   ./reproduce.sh          full run (about 45 minutes on 8 cores)
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
python experiments/10_effect_sizes.py
python experiments/11_twf_diagnostics.py "$@"
python experiments/12_other_explainers.py "$@"

if [[ "${1:-}" != "--quick" ]]; then
  python experiments/04_tables.py
  python experiments/05_figures.py
  (cd paper/ieee_bigdata && pdflatex -interaction=nonstopmode main.tex >/dev/null \
     && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null)
  python paper/sync_arxiv.py
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null \
     && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null)
  echo "paper: paper/ieee_bigdata/main.pdf (IEEE), paper/main.pdf (single-column preprint)"
fi
