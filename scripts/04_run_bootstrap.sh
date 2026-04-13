#!/usr/bin/env bash
# =============================================================================
# STEP 4 — Bootstrap confidence intervals for OOS comparison [T-009]
# =============================================================================
# Computes block-bootstrap 95% CIs and one-sided p-values for all completed
# experiments, comparing Quantum vs each classical baseline.
#
# Usage:
#   bash scripts/04_run_bootstrap.sh                    # all completed exps
#   bash scripts/04_run_bootstrap.sh --exp A1_PA A1_PB  # selected experiments
#   bash scripts/04_run_bootstrap.sh --n-bootstrap 5000 # faster (less precise)
#
# Output per experiment:
#   results/exp_<ID>/bootstrap_ci_quantum_vs_slsqp.json
#   results/exp_<ID>/bootstrap_ci_quantum_vs_equal_weight.json
#   results/exp_<ID>/bootstrap_ci_summary.json
#
# Requirements: pip install arch  (circular block bootstrap)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "============================================================"
echo " STEP 4: BOOTSTRAP CONFIDENCE INTERVALS [T-009]"
echo " Project root : $PROJECT_ROOT"
echo "============================================================"

cd "$PROJECT_ROOT"

# Check arch library (circular block bootstrap)
if ! python -c "import arch" 2>/dev/null; then
    echo "WARNING: 'arch' library not installed."
    echo "  The bootstrap module has a native fallback implementation."
    echo "  For best results: pip install arch"
fi

N_BOOTSTRAP=10000
BLOCK_SIZE=21
EXTRA_EXPS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-bootstrap) N_BOOTSTRAP="$2"; shift 2 ;;
        --block-size)  BLOCK_SIZE="$2";  shift 2 ;;
        --exp)         shift; EXTRA_EXPS="$@"; break ;;
        *) shift ;;
    esac
done

echo ""
echo "Settings: n_bootstrap=$N_BOOTSTRAP, block_size=${BLOCK_SIZE}d"
echo ""

python - << PYEOF
import sys, json, os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(".")))

from src.phase_3.bootstrap_ci import run_bootstrap_suite

results_dir = Path("results")
n_bootstrap = int("$N_BOOTSTRAP")
block_size  = int("$BLOCK_SIZE")
exp_filter  = "$EXTRA_EXPS".split() if "$EXTRA_EXPS" else None

# Discover completed experiments
exp_dirs = sorted([d for d in results_dir.iterdir()
                   if d.is_dir() and d.name.startswith("exp_")])

if exp_filter:
    exp_dirs = [d for d in exp_dirs if any(f in d.name for f in exp_filter)]

print(f"Found {len(exp_dirs)} experiment directories to process")

all_summaries = {}

for exp_dir in exp_dirs:
    exp_id = exp_dir.name.replace("exp_", "")
    metrics_file = exp_dir / "tfm_comprehensive_metrics_latest.json"

    if not metrics_file.exists():
        print(f"  [{exp_id}] SKIP — no metrics file")
        continue

    try:
        with open(metrics_file) as fh:
            metrics = json.load(fh)

        # Extract OOS returns and weights
        oos_data  = metrics.get("metrics", {}).get("oos_validation", {})
        portfolios = oos_data.get("portfolios", {})
        returns_oos_path = metrics.get("metadata", {}).get("returns_oos_path")

        # Try to load OOS returns
        csv_path = metrics.get("metadata", {}).get("csv_path") or "data/returns_sp500_full.csv"
        oos_start = metrics.get("metadata", {}).get("oos_start_date")
        oos_end   = metrics.get("metadata", {}).get("oos_end_date")

        if not (csv_path and oos_start and oos_end):
            # Fallback: load from any available oos returns
            print(f"  [{exp_id}] SKIP — missing oos metadata")
            continue

        import pandas as pd
        returns_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        returns_oos = returns_df.loc[oos_start:oos_end]

        # Get quantum weights
        w_q_list = portfolios.get("Quantum CVaR", {}).get("weights")
        if not w_q_list:
            print(f"  [{exp_id}] SKIP — no quantum weights in metrics")
            continue
        w_quantum = np.array(w_q_list)

        # Build classical portfolio weights dict
        classical_weights = {}
        for label, pdata in portfolios.items():
            if label == "Quantum CVaR":
                continue
            w = pdata.get("weights")
            if w:
                classical_weights[label] = np.array(w)

        if not classical_weights:
            print(f"  [{exp_id}] SKIP — no classical weights")
            continue

        # Align returns to weight dimension
        if returns_oos.shape[1] != len(w_quantum):
            print(f"  [{exp_id}] SKIP — shape mismatch: "
                  f"returns {returns_oos.shape[1]} vs weights {len(w_quantum)}")
            continue

        # Run bootstrap
        print(f"  [{exp_id}] Running bootstrap ({n_bootstrap} samples)...")
        bs_results = run_bootstrap_suite(
            returns_oos=returns_oos.values,
            weights_quantum=w_quantum,
            classical_weights=classical_weights,
            n_bootstrap=n_bootstrap,
            block_size=block_size,
            seed=42,
        )

        # Save per-comparison JSON
        summary_entry = {}
        for comp_label, bs_res in bs_results.items():
            safe_label = comp_label.replace(" ", "_").replace("/", "-")
            out_path = exp_dir / f"bootstrap_ci_quantum_vs_{safe_label}.json"
            with open(out_path, "w") as fh:
                json.dump(bs_res.to_dict(), fh, indent=2)
            summary_entry[comp_label] = {
                k: {
                    "diff_mean": v.diff_mean,
                    "diff_ci_95": list(v.diff_ci),
                    "p_value": v.p_value,
                    "significant": v.is_significant(),
                }
                for k, v in bs_res.metrics.items()
            }

        # Save summary
        summary_path = exp_dir / "bootstrap_ci_summary.json"
        with open(summary_path, "w") as fh:
            json.dump({
                "experiment": exp_id,
                "n_bootstrap": n_bootstrap,
                "block_size": block_size,
                "comparisons": summary_entry,
            }, fh, indent=2)

        all_summaries[exp_id] = summary_entry
        print(f"  [{exp_id}] Done — saved to {summary_path.name}")

    except Exception as exc:
        import traceback
        print(f"  [{exp_id}] ERROR: {exc}")
        print(traceback.format_exc())

# Write global summary
global_out = results_dir / "bootstrap_summary_all.json"
with open(global_out, "w") as fh:
    json.dump(all_summaries, fh, indent=2)
print(f"\nGlobal summary: {global_out}")
PYEOF

echo ""
echo "============================================================"
echo " STEP 4 COMPLETE"
echo " Bootstrap CIs saved in each results/exp_*/bootstrap_ci_*.json"
echo " Global summary: results/bootstrap_summary_all.json"
echo "============================================================"
