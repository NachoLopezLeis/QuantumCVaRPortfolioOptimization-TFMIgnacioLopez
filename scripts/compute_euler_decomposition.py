#!/usr/bin/env python3
# =============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : April 2026
#
# Script  : compute_euler_decomposition.py
#
# Description
# -----------
# Computes the Euler CVaR risk decomposition for every experiment stored under
# results/exp_*/tfm_comprehensive_metrics_latest.json.
#
# Mathematical basis
# ------------------
# For a portfolio with weights w and return vector r_t, the historical CVaR
# at confidence level alpha satisfies the Euler property:
#
#   CVaR_alpha(w) = sum_i RC_i(w)
#
# where the risk contribution of asset i is:
#
#   RC_i = w_i * dCVaR/dw_i = w_i * E[r_i | r^T w <= VaR_alpha]
#
# This identity is exact (zero additivity error) for the historical estimator.
# The conditional expectation equals the sample mean of asset i's returns
# over the T*alpha worst portfolio return days.
#
# Outputs (per experiment)
# ------------------------
# results/euler_decomposition/{exp_id}_euler.json
#   Per-weight-set (initial/quantum/classical/oos_quantum) breakdown with:
#   - risk_contributions : RC_i (absolute, loss sign convention)
#   - rc_percentage       : RC_i / CVaR_total * 100
#   - marginal_cvar       : dCVaR/dw_i
#   - total_cvar          : portfolio CVaR (loss)
#   - additivity_error    : |sum(RC_i) - CVaR| / CVaR  (should be ~1e-14)
#   - herfindahl_index    : HHI = sum((RC_i/CVaR)^2)
#   - effective_n         : 1 / HHI  (effective number of risk contributors)
#   - euler_check_pass    : additivity_error < 1e-8
#
# results/euler_decomposition/summary.csv
#   One row per (experiment, weight_set) with key scalar metrics.
#
# results/euler_decomposition/euler_decomposition.log
#   Execution log (overwrites on each run).
#
# Usage
# -----
#   python scripts/compute_euler_decomposition.py
#   python scripts/compute_euler_decomposition.py --exp A1_PA --exp S1_PA_100
#   python scripts/compute_euler_decomposition.py --alpha 0.01
#
# References
# ----------
# [1] Tasche, D. (2002). Expected Shortfall and Beyond. Journal of Banking
#     and Finance 26(7), 1519-1533.
# [2] Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
#     Value-at-Risk. Journal of Risk 2(3), 21-41.
# [3] Euler, L. (1755). Institutiones Calculi Differentialis. Homogeneity
#     theorem for positively homogeneous functions of degree 1.
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "returns_sp500_full.csv"
RESULTS_ROOT = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "euler_decomposition"
LOG_FILE = OUTPUT_DIR / "euler_decomposition.log"


# ---------------------------------------------------------------------------
# Logger (overwrites on each run, per project convention)
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("euler_decomposition")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # File handler (overwrite)
    fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


logger = _setup_logger()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_euler_decomposition(
    weights: np.ndarray,
    returns: np.ndarray,
    asset_names: List[str],
    alpha: float = 0.05,
    label: str = "portfolio",
) -> Dict:
    """
    Compute exact Euler CVaR risk decomposition via conditional expectation.

    For historical CVaR the marginal risk contribution of asset i is:

        dCVaR/dw_i = E[r_i | r^T w <= VaR_alpha]

    Multiplying by w_i gives the risk contribution RC_i such that
    sum(RC_i) == CVaR_total exactly (additivity_error < 1e-12 numerically).

    Parameters
    ----------
    weights : np.ndarray, shape (N,)
        Portfolio weights (must sum to ~1, non-negative for long-only).
    returns : np.ndarray, shape (T, N)
        Asset return matrix (daily log or simple returns).
    asset_names : List[str]
        Asset ticker or name labels, length N.
    alpha : float
        CVaR tail probability (0 < alpha < 1). Default 0.05.
    label : str
        Descriptive label for this weight set (e.g. "quantum_optimal").

    Returns
    -------
    Dict with keys:
        label, alpha, n_assets, n_periods, n_tail_obs,
        total_cvar, total_var,
        marginal_cvar       : list[float]  (dCVaR/dw_i, loss sign)
        risk_contributions  : list[float]  (RC_i = w_i * marg_i, loss sign)
        rc_percentage       : list[float]  (RC_i / CVaR * 100)
        weights             : list[float]
        asset_names         : list[str]
        herfindahl_index    : float        (concentration in risk space)
        effective_n         : float        (1/HHI)
        max_rc_asset        : str
        max_rc_pct          : float
        min_rc_asset        : str
        min_rc_pct          : float
        additivity_error    : float        (should be < 1e-8)
        euler_check_pass    : bool
    """
    n_assets = len(weights)
    n_periods = len(returns)

    # --- Portfolio return series ---
    port_ret = returns @ weights  # shape (T,)

    # --- VaR (lower tail quantile) ---
    var = float(np.percentile(port_ret, alpha * 100.0))

    # --- Tail mask ---
    tail_mask = port_ret <= var
    n_tail = int(tail_mask.sum())

    if n_tail == 0:
        logger.warning(f"[{label}] No tail observations found at alpha={alpha}. Returning zeros.")
        return _empty_result(label, alpha, n_assets, n_periods, asset_names, weights)

    tail_returns = returns[tail_mask]   # shape (n_tail, N)
    tail_port = port_ret[tail_mask]     # shape (n_tail,)

    # --- CVaR ---
    cvar_portfolio = float(np.mean(tail_port))   # negative number (return in loss tail)

    # --- Marginal CVaR: dCVaR/dw_i = E[r_i | tail] ---
    # This is the conditional expectation of each asset's return in the tail.
    # Sign convention: returns are negative in loss tail, CVaR is negative.
    # We report everything in LOSS convention (positive = bad) by negating.
    marginal_cvar_ret = tail_returns.mean(axis=0)   # shape (N,), negative = loss
    marginal_cvar_loss = -marginal_cvar_ret         # positive = loss contribution

    # --- Risk contributions (Euler) ---
    # RC_i = w_i * dCVaR/dw_i  (in return sign: negative sum = CVaR)
    rc_ret = weights * marginal_cvar_ret            # shape (N,), sum = cvar_portfolio
    rc_loss = weights * marginal_cvar_loss          # shape (N,), positive = loss
    cvar_loss = -cvar_portfolio                     # positive loss magnitude

    # --- Additivity check (Euler theorem) ---
    # sum(RC_i) should equal cvar_portfolio exactly
    additivity_error = float(abs(np.sum(rc_ret) - cvar_portfolio) / (abs(cvar_portfolio) + 1e-15))
    euler_ok = additivity_error < 1e-8

    # --- Percentage contributions ---
    rc_pct = (rc_loss / cvar_loss * 100.0) if cvar_loss > 1e-15 else np.zeros(n_assets)

    # --- Concentration metrics ---
    # Normalize so positives sum to 100 for HHI computation
    rc_pct_pos = np.maximum(rc_pct, 0.0)
    rc_pct_sum = rc_pct_pos.sum()
    rc_pct_norm = rc_pct_pos / rc_pct_sum if rc_pct_sum > 1e-15 else np.ones(n_assets) / n_assets
    hhi = float(np.sum(rc_pct_norm ** 2))
    effective_n = float(1.0 / hhi) if hhi > 1e-15 else float(n_assets)

    # --- Top/bottom contributors ---
    sorted_idx = np.argsort(rc_pct)[::-1]
    max_idx = sorted_idx[0]
    min_idx = sorted_idx[-1]

    logger.debug(
        f"[{label}] CVaR={cvar_loss:.5f}, VaR={-var:.5f}, "
        f"n_tail={n_tail}, HHI={hhi:.4f}, eff_N={effective_n:.2f}, "
        f"additivity_err={additivity_error:.2e}"
    )

    return {
        "label": label,
        "alpha": alpha,
        "n_assets": n_assets,
        "n_periods": n_periods,
        "n_tail_obs": n_tail,
        "total_cvar": cvar_loss,
        "total_var": -var,
        "marginal_cvar": marginal_cvar_loss.tolist(),
        "risk_contributions": rc_loss.tolist(),
        "rc_percentage": rc_pct.tolist(),
        "weights": weights.tolist(),
        "asset_names": asset_names,
        "herfindahl_index": hhi,
        "effective_n": effective_n,
        "max_rc_asset": asset_names[max_idx],
        "max_rc_pct": float(rc_pct[max_idx]),
        "min_rc_asset": asset_names[min_idx],
        "min_rc_pct": float(rc_pct[min_idx]),
        "additivity_error": additivity_error,
        "euler_check_pass": euler_ok,
    }


def _empty_result(label, alpha, n_assets, n_periods, asset_names, weights) -> Dict:
    """Return a zero-filled result dict when computation is not possible."""
    return {
        "label": label,
        "alpha": alpha,
        "n_assets": n_assets,
        "n_periods": n_periods,
        "n_tail_obs": 0,
        "total_cvar": 0.0,
        "total_var": 0.0,
        "marginal_cvar": [0.0] * n_assets,
        "risk_contributions": [0.0] * n_assets,
        "rc_percentage": [0.0] * n_assets,
        "weights": weights.tolist(),
        "asset_names": asset_names,
        "herfindahl_index": 0.0,
        "effective_n": 0.0,
        "max_rc_asset": "",
        "max_rc_pct": 0.0,
        "min_rc_asset": "",
        "min_rc_pct": 0.0,
        "additivity_error": 0.0,
        "euler_check_pass": False,
    }


# ---------------------------------------------------------------------------
# Experiment processing
# ---------------------------------------------------------------------------

def discover_experiments(filter_ids: Optional[List[str]] = None) -> List[Path]:
    """
    Return sorted list of tfm_comprehensive_metrics_latest.json files.

    Parameters
    ----------
    filter_ids : Optional[List[str]]
        If provided, only experiments whose directory name contains one of
        these substrings are returned (e.g. ['A1_PA', 'S1_PA_100']).
    """
    pattern = "exp_*/tfm_comprehensive_metrics_latest.json"
    all_files = sorted(RESULTS_ROOT.glob(pattern))

    if filter_ids:
        all_files = [
            f for f in all_files
            if any(fid in f.parent.name for fid in filter_ids)
        ]

    logger.info(f"Discovered {len(all_files)} experiment files.")
    return all_files


def _extract_weight_sets(metrics: Dict, oos_portfolios: Dict) -> Dict[str, Optional[np.ndarray]]:
    """
    Extract all weight vectors to decompose from a metrics JSON.

    Returns a dict mapping label -> np.ndarray or None (if not available).
    """
    sets: Dict[str, Optional[np.ndarray]] = {}

    # 1. Initial equal-weight
    w_init = metrics.get("weights", {}).get("initial")
    if w_init:
        sets["initial_equal_weight"] = np.array(w_init, dtype=float)

    # 2. Quantum optimal (in-sample)
    w_opt = metrics.get("weights", {}).get("optimal")
    if w_opt:
        sets["quantum_optimal"] = np.array(w_opt, dtype=float)

    # 3. Classical SLSQP (in-sample)
    slsqp = (
        metrics.get("metrics", {})
        .get("benchmarks", {})
        .get("classical_results", {})
        .get("slsqp", {})
        .get("optimal_weights")
    )
    if slsqp:
        sets["classical_slsqp"] = np.array(slsqp, dtype=float)

    # 4. Classical Markowitz
    mkw = (
        metrics.get("metrics", {})
        .get("benchmarks", {})
        .get("classical_results", {})
        .get("markowitz", {})
        .get("optimal_weights")
    )
    if mkw:
        sets["classical_markowitz"] = np.array(mkw, dtype=float)

    # 5. Risk Parity
    rp = (
        metrics.get("metrics", {})
        .get("benchmarks", {})
        .get("classical_results", {})
        .get("risk_parity", {})
        .get("optimal_weights")
    )
    if rp:
        sets["classical_risk_parity"] = np.array(rp, dtype=float)

    # 6. OOS portfolios (Quantum CVaR, Classical CVaR, Inverse Vol, Equal Weight)
    for port_name, port_data in oos_portfolios.items():
        w_oos = port_data.get("weights")
        if w_oos:
            safe_name = port_name.lower().replace(" ", "_")
            sets[f"oos_{safe_name}"] = np.array(w_oos, dtype=float)

    return sets


def process_experiment(
    json_path: Path,
    returns_df: pd.DataFrame,
    alpha: float = 0.05,
) -> Optional[Dict]:
    """
    Process a single experiment JSON and return the Euler decomposition result.

    Parameters
    ----------
    json_path : Path
        Path to tfm_comprehensive_metrics_latest.json.
    returns_df : pd.DataFrame
        Full returns DataFrame indexed by date, columns = ticker symbols.
    alpha : float
        CVaR tail probability.

    Returns
    -------
    Dict with experiment metadata and per-weight-set decompositions,
    or None on error.
    """
    exp_id = json_path.parent.name.replace("exp_", "")
    logger.info(f"Processing experiment: {exp_id}")

    # --- Load JSON ---
    try:
        with open(json_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        logger.error(f"[{exp_id}] Failed to load JSON: {exc}")
        return None

    cfg = data.get("metadata", {}).get("configuration", {})
    phase0_cfg = cfg.get("phase0", cfg)  # some configs store under phase0

    # --- Extract tickers ---
    tickers: Optional[List[str]] = None
    # Try several paths the config might store tickers
    for path in [
        ("phase0", "data", "tickers"),
        ("data", "tickers"),
    ]:
        obj = cfg
        for key in path:
            obj = obj.get(key, {}) if isinstance(obj, dict) else None
        if isinstance(obj, list) and len(obj) > 0:
            tickers = obj
            break

    # Fallback: read from by_asset keys in weights section
    if not tickers:
        by_asset = data.get("weights", {}).get("by_asset", {})
        if by_asset:
            tickers = list(by_asset.keys())

    if not tickers:
        logger.warning(f"[{exp_id}] Could not determine tickers. Skipping.")
        return None

    # --- Verify all tickers in returns data ---
    available = [t for t in tickers if t in returns_df.columns]
    if len(available) < len(tickers):
        missing = set(tickers) - set(available)
        logger.warning(f"[{exp_id}] Missing {len(missing)} tickers: {missing}. Using {len(available)}.")
        tickers = available

    if len(tickers) < 2:
        logger.error(f"[{exp_id}] Fewer than 2 valid tickers. Skipping.")
        return None

    # --- Split periods ---
    oos_split = (
        cfg.get("data", {}).get("oos_split_date")
        or cfg.get("phase0", {}).get("data", {}).get("oos_split_date")
    )

    returns_all = returns_df[tickers].dropna()
    if oos_split:
        returns_train = returns_all[returns_all.index < oos_split]
        returns_oos = returns_all[returns_all.index >= oos_split]
    else:
        returns_train = returns_all
        returns_oos = None

    if len(returns_train) < 50:
        logger.warning(f"[{exp_id}] Only {len(returns_train)} training obs. Skipping.")
        return None

    logger.debug(
        f"[{exp_id}] Tickers={len(tickers)}, "
        f"Train={len(returns_train)}, "
        f"OOS={len(returns_oos) if returns_oos is not None else 0}"
    )

    # --- Extract weight sets ---
    oos_portfolios = (
        data.get("metrics", {}).get("oos_validation", {}).get("portfolios", {})
    )
    weight_sets = _extract_weight_sets(data, oos_portfolios)

    if not weight_sets:
        logger.warning(f"[{exp_id}] No weight sets found. Skipping.")
        return None

    # --- Compute decompositions ---
    decompositions: Dict[str, Dict] = {}

    for label, weights in weight_sets.items():
        if weights is None:
            continue

        # Align weights to available tickers
        if len(weights) != len(tickers):
            logger.warning(
                f"[{exp_id}][{label}] Weight dim={len(weights)} != tickers={len(tickers)}. Skipping."
            )
            continue

        # Clip and re-normalise (guard against tiny numerical negatives)
        weights = np.clip(weights, 0.0, None)
        w_sum = weights.sum()
        if w_sum < 1e-10:
            logger.warning(f"[{exp_id}][{label}] Weights sum to zero. Skipping.")
            continue
        weights = weights / w_sum

        # Determine which return matrix to use:
        # OOS weight-sets use OOS returns; in-sample sets use train returns
        is_oos_set = label.startswith("oos_")
        ret_matrix = (
            returns_oos.values
            if is_oos_set and returns_oos is not None and len(returns_oos) >= 20
            else returns_train.values
        )
        period_label = "oos" if is_oos_set else "train"

        result = compute_euler_decomposition(
            weights=weights,
            returns=ret_matrix,
            asset_names=tickers,
            alpha=alpha,
            label=f"{label}_{period_label}",
        )
        decompositions[label] = result

        if result["euler_check_pass"]:
            logger.debug(
                f"[{exp_id}][{label}] OK — CVaR={result['total_cvar']:.5f}, "
                f"HHI={result['herfindahl_index']:.4f}, "
                f"eff_N={result['effective_n']:.2f}, "
                f"top={result['max_rc_asset']} {result['max_rc_pct']:.1f}%"
            )
        else:
            logger.warning(
                f"[{exp_id}][{label}] Euler check FAILED — "
                f"additivity_err={result['additivity_error']:.2e}"
            )

    # --- Assemble output ---
    output = {
        "metadata": {
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "tickers": tickers,
            "n_assets": len(tickers),
            "n_train": len(returns_train),
            "n_oos": len(returns_oos) if returns_oos is not None else 0,
            "oos_split": oos_split,
            "alpha": alpha,
            "source_json": str(json_path),
        },
        "decompositions": decompositions,
    }

    return output


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def build_summary_row(exp_id: str, label: str, dec: Dict) -> Dict:
    """Convert one decomposition result into a flat summary dict."""
    return {
        "experiment": exp_id,
        "weight_set": label,
        "alpha": dec["alpha"],
        "n_assets": dec["n_assets"],
        "n_periods": dec["n_periods"],
        "n_tail_obs": dec["n_tail_obs"],
        "total_cvar": dec["total_cvar"],
        "total_var": dec["total_var"],
        "herfindahl_index": dec["herfindahl_index"],
        "effective_n": dec["effective_n"],
        "max_rc_asset": dec["max_rc_asset"],
        "max_rc_pct": dec["max_rc_pct"],
        "min_rc_asset": dec["min_rc_asset"],
        "min_rc_pct": dec["min_rc_pct"],
        "additivity_error": dec["additivity_error"],
        "euler_check_pass": dec["euler_check_pass"],
        # Top-3 contributors as separate columns for quick inspection
        **_top3_columns(dec),
    }


def _top3_columns(dec: Dict) -> Dict:
    """Return top-3 risk contributor columns from a decomposition."""
    names = dec["asset_names"]
    pcts = dec["rc_percentage"]
    if not names or not pcts:
        return {}
    paired = sorted(zip(names, pcts), key=lambda x: -x[1])
    result = {}
    for rank, (name, pct) in enumerate(paired[:3], start=1):
        result[f"top{rank}_asset"] = name
        result[f"top{rank}_pct"] = round(pct, 4)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    logger.info("=" * 70)
    logger.info("EULER CVaR DECOMPOSITION — TFM Batch Runner")
    logger.info(f"Started: {datetime.now().isoformat()}")
    logger.info(f"Alpha: {args.alpha}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 70)

    # --- Load returns once ---
    if not DATA_PATH.exists():
        logger.error(f"Returns data not found: {DATA_PATH}")
        sys.exit(1)

    logger.info(f"Loading returns from {DATA_PATH} ...")
    returns_df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    logger.info(f"Returns loaded: {returns_df.shape[0]} days × {returns_df.shape[1]} assets.")

    # --- Discover experiments ---
    filter_ids = args.exp if args.exp else None
    json_files = discover_experiments(filter_ids)

    if not json_files:
        logger.error("No experiment files found. Exiting.")
        sys.exit(1)

    # --- Process each experiment ---
    summary_rows: List[Dict] = []
    n_ok = 0
    n_fail = 0

    for json_path in json_files:
        exp_id = json_path.parent.name.replace("exp_", "")

        result = process_experiment(
            json_path=json_path,
            returns_df=returns_df,
            alpha=args.alpha,
        )

        if result is None:
            n_fail += 1
            continue

        # --- Save per-experiment JSON ---
        out_path = OUTPUT_DIR / f"{exp_id}_euler.json"
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, default=float)
            logger.info(f"[{exp_id}] Saved → {out_path.name}")
        except Exception as exc:
            logger.error(f"[{exp_id}] Failed to save JSON: {exc}")
            n_fail += 1
            continue

        # --- Accumulate summary rows ---
        for label, dec in result["decompositions"].items():
            summary_rows.append(build_summary_row(exp_id, label, dec))

        n_ok += 1

    # --- Save summary CSV ---
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        csv_path = OUTPUT_DIR / "summary.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info(f"Summary CSV saved → {csv_path} ({len(summary_df)} rows)")

        # --- Print quick overview to log ---
        logger.info("")
        logger.info("=== QUICK SUMMARY: QUANTUM OPTIMAL WEIGHTS (train period) ===")

        quantum_rows = summary_df[summary_df["weight_set"] == "quantum_optimal"].copy()
        quantum_rows = quantum_rows.sort_values("total_cvar")

        if not quantum_rows.empty:
            for _, row in quantum_rows.iterrows():
                logger.info(
                    f"  {row['experiment']:<20} "
                    f"CVaR={row['total_cvar']:.5f}  "
                    f"HHI={row['herfindahl_index']:.4f}  "
                    f"eff_N={row['effective_n']:.2f}  "
                    f"top1={row.get('top1_asset','?')} {row.get('top1_pct',0):.1f}%  "
                    f"top2={row.get('top2_asset','?')} {row.get('top2_pct',0):.1f}%"
                )
    else:
        logger.warning("No summary rows produced — no valid decompositions.")

    # --- Final report ---
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"DONE: {n_ok} experiments OK, {n_fail} failed.")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Finished: {datetime.now().isoformat()}")
    logger.info("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Euler CVaR risk decomposition for all TFM experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CVaR confidence level (tail probability). Use 0.05 for 95%% CVaR.",
    )
    parser.add_argument(
        "--exp",
        action="append",
        metavar="EXP_ID",
        help=(
            "Filter to specific experiment IDs (substring match). "
            "Can be repeated: --exp A1_PA --exp S1_PA_100"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
