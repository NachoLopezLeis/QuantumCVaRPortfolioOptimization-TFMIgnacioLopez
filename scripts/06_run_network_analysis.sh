#!/usr/bin/env bash
# =============================================================================
# STEP 6 — Extended network analysis report [T-005, T-006]
# =============================================================================
# Generates the topological analysis of the TMFG correlation graph
# and fat-tail statistics for portfolio returns.
#
# Usage:
#   bash scripts/06_run_network_analysis.sh
#
# Outputs:
#   results/network_analysis/network_analysis_report.json
#   results/network_analysis/network_analysis_report.html
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo " STEP 6: NETWORK ANALYSIS REPORT [T-005, T-006]"
echo "============================================================"

cd "$PROJECT_ROOT"

python src/phase_0/network_analysis_report.py \
    --config config/config.yaml \
    --output results/network_analysis

echo ""
echo "============================================================"
echo " STEP 6 COMPLETE"
echo " results/network_analysis/network_analysis_report.html"
echo " results/network_analysis/network_analysis_report.json"
echo "============================================================"
