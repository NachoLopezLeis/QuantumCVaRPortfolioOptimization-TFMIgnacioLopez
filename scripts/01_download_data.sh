#!/usr/bin/env bash
# =============================================================================
# STEP 1 — Download S&P 500 returns data (2018-2025)
# =============================================================================
# Fix [T-007]: extends training window to 2018-01-01 for ~6 years of data.
#
# Usage:
#   cd PROJECT_ROOT
#   bash scripts/01_download_data.sh
#
# Output:
#   data/returns_sp500_full.csv   (~374 tickers, 2018-2025, log returns)
#   data/returns_sp500_100.csv    (top-100 by liquidity, same period)
#
# Requirements: pip install yfinance pandas numpy
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo " STEP 1: DATA DOWNLOAD"
echo " Project root : $PROJECT_ROOT"
echo " Date range   : 2018-01-01 to 2025-12-31 (T-007)"
echo "============================================================"

cd "$PROJECT_ROOT"

# Full S&P 500 (~374 tickers after quality filter)
echo ""
echo "[1/2] Downloading full S&P 500 universe..."
python scripts/download_sp500_full.py \
    --start 2018-01-01 \
    --end   2025-12-31 \
    --min-obs 1500

if [ $? -ne 0 ]; then
    echo "ERROR: Full S&P 500 download failed"
    exit 1
fi

# Top-100 subset for scalability experiments (S1 group)
echo ""
echo "[2/2] Generating SP100 subset (data/returns_sp500_100.csv)..."
python - << 'PYEOF'
import pandas as pd
import numpy as np
from pathlib import Path

full = pd.read_csv("data/returns_sp500_full.csv", index_col=0, parse_dates=True)

# Select top-100 by Sharpe (proxy for liquidity / data quality)
sharpes = full.mean() / full.std() * (252 ** 0.5)
top100 = sharpes.nlargest(100).index.tolist()
subset = full[top100]
out = Path("data/returns_sp500_100.csv")
subset.to_csv(out)
print(f"SP100 subset saved: {out} — {subset.shape[0]} periods x {subset.shape[1]} tickers")
PYEOF

echo ""
echo "============================================================"
echo " STEP 1 COMPLETE"
echo " data/returns_sp500_full.csv  — full universe"
echo " data/returns_sp500_100.csv   — SP100 subset"
echo "============================================================"
echo ""
echo "Next step: bash scripts/02_run_universe_selection.sh"
