#!/usr/bin/env bash
# =============================================================================
# STEP 2 — Universe selection via QUBO (D-Wave or SimAnneal)
# =============================================================================
# Runs run_universe_selection.py to generate the four portfolio universes:
#   PA (K=10 peripheral), PB (K=10 central),
#   PA_100 (K=100 peripheral), PB_100 (K=100 central)
#
# Usage:
#   bash scripts/02_run_universe_selection.sh              # SimAnneal (local)
#   bash scripts/02_run_universe_selection.sh --dwave      # D-Wave Leap hybrid
#
# D-Wave acknowledgement [T-004]:
#   D-Wave provides access to LeapHybridBQMSampler under the Leap academic
#   research programme. Set DWAVE_API_TOKEN in your environment or .env file.
#
# Requires .env file with:
#   DWAVE_API_TOKEN=your_token_here
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USE_DWAVE=false

for arg in "$@"; do
    [[ "$arg" == "--dwave" ]] && USE_DWAVE=true
done

echo "============================================================"
echo " STEP 2: UNIVERSE SELECTION"
echo " Project root : $PROJECT_ROOT"
echo " Backend      : $([ "$USE_DWAVE" = true ] && echo 'D-Wave LeapHybrid' || echo 'SimulatedAnnealing (local)')"
echo "============================================================"

cd "$PROJECT_ROOT"

# Load .env if present
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo " .env loaded"
fi

if [ "$USE_DWAVE" = true ]; then
    if [ -z "${DWAVE_API_TOKEN:-}" ]; then
        echo "ERROR: DWAVE_API_TOKEN not set."
        echo "  Option 1: add DWAVE_API_TOKEN=<token> to .env"
        echo "  Option 2: export DWAVE_API_TOKEN=<token> before running"
        exit 1
    fi
    echo " D-Wave token found (${#DWAVE_API_TOKEN} chars)"
    BACKEND_FLAG="--backend hybrid"
else
    BACKEND_FLAG="--backend simulated"
fi

echo ""
echo "Running universe selection..."
python scripts/run_universe_selection.py $BACKEND_FLAG

if [ $? -ne 0 ]; then
    echo "ERROR: Universe selection failed"
    exit 1
fi

echo ""
echo "============================================================"
echo " STEP 2 COMPLETE"
echo " results/network_selection/selection_PA.json"
echo " results/network_selection/selection_PB.json"
echo " results/network_selection/selection_PA_100.json"
echo " results/network_selection/selection_PB_100.json"
echo "============================================================"
echo ""
echo "Next step: bash scripts/03_run_experiments.sh"
