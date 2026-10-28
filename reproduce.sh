#!/usr/bin/env bash
# Recompute scientific results, tables, and figures; measured runtimes vary by run.
#   ./reproduce.sh          full run (several hours; scaling alone can take about 7 h)
#   ./reproduce.sh --quick  reduced audit counts; benchmark still runs in full
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
python experiments/09_review_validation.py "$@"

if [[ "${1:-}" != "--quick" ]]; then
  python experiments/04_tables.py
  python experiments/05_figures.py
  python - <<'PY'
from pathlib import Path
import shutil
target = Path('IEEE-conference-template-062824')
for source in Path('paper/ieee_bigdata/tables').glob('*.tex'):
    shutil.copy2(source, target / 'tables' / source.name)
for name in ['audit_overview.pdf', 'shap_crosscheck_heatmap.pdf']:
    shutil.copy2(Path('paper/figures') / name, target / 'figures' / name)
PY
  python experiments/10_review_assets.py
  bash reproduce_scientific.sh
  echo "Revised paper: IEEE-conference-template-062824/IEEE-conference-template-062824.pdf"
fi
