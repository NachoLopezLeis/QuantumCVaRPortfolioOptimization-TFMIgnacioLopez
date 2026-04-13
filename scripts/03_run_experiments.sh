#!/usr/bin/env bash
# =============================================================================
# STEP 3 — Run all quantum CVaR portfolio experiments
# =============================================================================
# Executes run_all_experiments.py which runs main.ipynb via papermill for each
# of the 34 experiment configs (30 original + 4 new P1-002 configs).
#
# Usage:
#   bash scripts/03_run_experiments.sh                    # all 34 experiments
#   bash scripts/03_run_experiments.sh A1_PA A1_PB A1_PC  # selected subset
#   bash scripts/03_run_experiments.sh --dry-run          # show plan, no exec
#   bash scripts/03_run_experiments.sh --resume-from D1   # skip completed ones
#
# Estimated runtime:
#   Standard noiseless (A1, A3): ~30-45 min per experiment
#   Noisy (B2, C2, D1, D2):      ~45-90 min per experiment
#   Scalability K=100 (S1):      ~2-4 h per experiment
#
# Requirements: pip install papermill
#
# All results go to results/exp_<ID>/ and logs to results/batch_runs/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo " STEP 3: RUN ALL EXPERIMENTS (34 total)"
echo " Project root : $PROJECT_ROOT"
echo "============================================================"

cd "$PROJECT_ROOT"

# Check papermill
if ! python -c "import papermill" 2>/dev/null; then
    echo "ERROR: papermill not installed."
    echo "  Run: pip install papermill"
    exit 1
fi

# Check that universe selection outputs exist
if [ ! -f "results/network_selection/selection_PA.json" ]; then
    echo "WARNING: results/network_selection/selection_PA.json not found."
    echo "  Run step 2 first: bash scripts/02_run_universe_selection.sh"
    echo "  Continuing anyway (experiments that need PA will fail)..."
fi

# Pass any extra CLI args through (e.g., specific experiment IDs)
EXTRA_ARGS="$@"

echo ""
echo "Starting batch run..."
echo "Progress visible in: results/batch_runs/batch_runner_<timestamp>.log"
echo ""

python scripts/run_all_experiments.py $EXTRA_ARGS

if [ $? -ne 0 ]; then
    echo "ERROR: Batch runner exited with errors"
    echo "Check results/batch_runs/batch_summary_*.json for status"
    exit 1
fi

echo ""
echo "============================================================"
echo " STEP 3 COMPLETE"
echo " Results in: results/exp_*/"
echo " Summary:    results/batch_runs/batch_summary_*.json"
echo "============================================================"
echo ""
echo "Next step: bash scripts/04_run_bootstrap.sh"
