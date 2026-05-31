#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Script  : run_universe_selection.py
# ============================================================================
#
# Description
# -----------
# Generates the four base universes used in all TFM experiments:
#
#   PA     (K=10)  : 10 peripheral assets selected via QUBO + SimAnneal/D-Wave.
#   PB     (K=10)  : 10 central assets ranked by eigenvector centrality.
#   PA_100 (K=100) : 100 peripheral assets selected via QUBO + SimAnneal/D-Wave.
#   PB_100 (K=100) : 100 central assets ranked by eigenvector centrality.
#
# PC (K=10) and PC_100 (K=100) are predefined ticker lists embedded in their
# respective config files (config_A1_n3.yaml and config_S1.yaml) and do not
# require selection here.
#
# Penalty calibration
# -------------------
# The QUBO cardinality penalty must dominate the objective range so that the
# annealer respects the equality constraint sum(x_i) = K.
#
# For correlation objective in [-1, 1]:
#   max_obj(K=10)  = K*(K-1)/2 = 45   -> P_card =   500  (11x margin)
#   max_obj(K=100) = K*(K-1)/2 = 4950 -> P_card = 100000 (20x margin)
#
# Post-processing repair
# ----------------------
# If the annealer still returns K +/- delta assets, a greedy repair is applied:
#   - Too many: drop assets with the lowest peripherality score until |S| = K.
#   - Too few : add assets with the highest peripherality score until |S| = K.
# This guarantees exactly K feasible assets regardless of annealer quality.
#
# Usage
# -----
#   cd PROJECT_ROOT
#   python scripts/run_universe_selection.py                    # all 4 universes
#   python scripts/run_universe_selection.py --universes PA PB  # subset
#   python scripts/run_universe_selection.py --backend hybrid   # D-Wave hybrid
#   python scripts/run_universe_selection.py --dry-run          # plan only
#
# Outputs
# -------
#   results/network_selection/selection_PA.json
#   results/network_selection/selection_PB.json
#   results/network_selection/selection_PA_100.json
#   results/network_selection/selection_PB_100.json
#   results/network_selection/selection_report_<timestamp>.json
#   logs/network_selection_run.txt
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


# ============================================================================
# PATHS
# ============================================================================

def find_project_root() -> Path:
    """Find project root by locating the config/ directory."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for p in candidates:
        if (p / "config").is_dir() and (p / "src").is_dir():
            return p
    raise FileNotFoundError(
        "Cannot find project root. Run from PROJECT_ROOT or scripts/."
    )


def setup_paths(project_root: Path):
    """Insert all src subdirectories at the front of sys.path."""
    paths_to_add = [
        str(project_root),
        str(project_root / "src"),
        str(project_root / "src" / "utils"),
        str(project_root / "src" / "phase_0"),
    ]
    for p in reversed(paths_to_add):
        if p not in sys.path:
            sys.path.insert(0, p)


# ============================================================================
# LOGGER
# ============================================================================

class SelectionLogger:
    """Simple file + stdout logger. Overwrites log on each run."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(
                f"UNIVERSE SELECTION RUN\n"
                f"Started: {datetime.now().isoformat()}\n"
                f"{'=' * 70}\n\n"
            )

    def __call__(self, msg: str):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ============================================================================
# PENALTY CALIBRATION
# ============================================================================

def calibrated_penalties(k: int) -> dict:
    """
    Return QUBO penalty coefficients calibrated for cardinality K.

    The cardinality penalty P_card must satisfy:
        P_card >> lambda_corr * K*(K-1)/2

    where lambda_corr = 1.0 and correlations in [-1, 1].

    Parameters
    ----------
    k : int
        Target portfolio cardinality.

    Returns
    -------
    dict with keys: penalty_cardinality, penalty_community,
                    lambda_correlation, lambda_peripherality, penalty_bridge
    """
    max_corr_term = k * (k - 1) / 2  # upper bound on correlation objective

    if k <= 15:
        # K=10: max_corr_term = 45  -> use 500 (11x margin)
        p_card = max(500.0, 10.0 * max_corr_term)
        p_comm = max(100.0, 2.0 * max_corr_term)
    else:
        # K=100: max_corr_term = 4950 -> use 100000 (20x margin)
        p_card = max(100000.0, 20.0 * max_corr_term)
        p_comm = max(10000.0, 2.0 * max_corr_term)

    return {
        "lambda_correlation": 1.0,
        "lambda_peripherality": 0.5,
        "penalty_cardinality": p_card,
        "penalty_community": p_comm,
        "penalty_bridge": 0.3,
    }


# ============================================================================
# FEASIBILITY REPAIR
# ============================================================================

def repair_to_k(
    selected_tickers: list,
    metrics_df: pd.DataFrame,
    target_k: int,
    log,
) -> list:
    """
    Greedy repair: trim or expand selected_tickers to exactly target_k.

    Trimming removes assets with the lowest peripherality score.
    Expansion adds assets with the highest peripherality score among
    those not already selected.

    Parameters
    ----------
    selected_tickers : list of str
    metrics_df : pd.DataFrame
        Must contain columns 'ticker' and 'peripherality_score'.
    target_k : int
    log : callable

    Returns
    -------
    list of str, length == target_k
    """
    current = set(selected_tickers)
    n = len(current)

    if n == target_k:
        return selected_tickers

    scores = dict(
        zip(metrics_df["ticker"], metrics_df["peripherality_score"])
    )

    if n > target_k:
        # Trim: remove lowest-peripherality assets
        ordered = sorted(current, key=lambda t: scores.get(t, 0.0), reverse=True)
        repaired = ordered[:target_k]
        log(
            f"  Repair: trimmed {n} -> {target_k} "
            f"(removed {n - target_k} low-peripherality assets)"
        )
        return repaired

    # Expand: add highest-peripherality assets not already in set
    candidates = [
        t for t in metrics_df["ticker"].tolist() if t not in current
    ]
    candidates_sorted = sorted(
        candidates, key=lambda t: scores.get(t, 0.0), reverse=True
    )
    to_add = candidates_sorted[: target_k - n]
    repaired = list(current) + to_add
    log(
        f"  Repair: expanded {n} -> {target_k} "
        f"(added {target_k - n} high-peripherality assets)"
    )
    return repaired


# ============================================================================
# CENTRAL RANKING SELECTION
# ============================================================================

def select_central(
    metrics_df: pd.DataFrame,
    target_k: int,
    max_per_community: int,
    log,
) -> list:
    """
    Select top-K central assets by eigenvector centrality, with a
    per-community cap to prevent sector concentration.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Columns: ticker, eigenvector_centrality, community.
    target_k : int
    max_per_community : int
    log : callable

    Returns
    -------
    list of str, length == target_k (guaranteed feasible)
    """
    ranked = metrics_df.sort_values(
        "eigenvector_centrality", ascending=False
    ).reset_index(drop=True)

    selected = []
    community_counts: dict = {}

    for _, row in ranked.iterrows():
        c = int(row["community"])
        if community_counts.get(c, 0) >= max_per_community:
            continue
        selected.append(row["ticker"])
        community_counts[c] = community_counts.get(c, 0) + 1
        if len(selected) == target_k:
            break

    log(
        f"  Central ranking: selected {len(selected)}/{target_k} "
        f"(communities represented: {len(community_counts)})"
    )
    return selected


# ============================================================================
# RESULT SERIALISATION
# ============================================================================

def build_result_dict(
    universe_id: str,
    method: str,
    selected: list,
    metrics_df: pd.DataFrame,
    n_universe: int,
    energy: float,
    feasible_before_repair: bool,
    timing_ms: float,
) -> dict:
    """Build a JSON-serialisable result dictionary."""

    sel_metrics = metrics_df[metrics_df["ticker"].isin(selected)]
    avg_periph = float(sel_metrics["peripherality_score"].mean()) \
        if not sel_metrics.empty else 0.0
    avg_central = float(sel_metrics["eigenvector_centrality"].mean()) \
        if not sel_metrics.empty else 0.0
    n_communities = int(sel_metrics["community"].nunique()) \
        if not sel_metrics.empty else 0

    return {
        "universe_id": universe_id,
        "method": method,
        "selected_tickers": sorted(selected),
        "energy": float(energy) if np.isfinite(energy) else None,
        "feasible_before_repair": feasible_before_repair,
        "feasible": True,  # always true after repair
        "n_universe": n_universe,
        "n_selected": len(selected),
        "avg_peripherality": avg_periph,
        "avg_centrality": avg_central,
        "n_communities_represented": n_communities,
        "timing_ms": round(timing_ms, 1),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate PA, PB, PA_100, PB_100 universe portfolios"
    )
    parser.add_argument(
        "--universes",
        nargs="*",
        default=["PA", "PB", "PA_100", "PB_100"],
        choices=["PA", "PB", "PA_100", "PB_100"],
        help="Which universes to generate (default: all four)",
    )
    parser.add_argument(
        "--backend",
        default="simulated_annealing",
        choices=["simulated_annealing", "hybrid", "qpu"],
        help="D-Wave backend for QUBO universes (default: simulated_annealing)",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=2000,
        help="Number of annealing reads (default: 2000)",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to full returns CSV (auto-detected if omitted)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    project_root = find_project_root()
    setup_paths(project_root)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = SelectionLogger(project_root / "logs" / "network_selection_run.txt")

    log("=" * 70)
    log("UNIVERSE SELECTION — TFM Quantum Portfolio Optimization")
    log("=" * 70)
    log(f"Project root  : {project_root}")
    log(f"Universes     : {args.universes}")
    log(f"D-Wave backend: {args.backend}")
    log(f"Num reads     : {args.num_reads}")

    # ------------------------------------------------------------------
    # Load returns
    # ------------------------------------------------------------------
    csv_path = args.csv_path
    if csv_path is None:
        csv_path = project_root / "data" / "returns_sp500_full.csv"
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path

    if not csv_path.exists():
        log(f"ERROR: Returns CSV not found: {csv_path}")
        log("Run first: python scripts/download_sp500_full.py")
        sys.exit(1)

    log(f"\nLoading returns: {csv_path.name}")
    returns_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    n_tickers = returns_df.shape[1]
    n_periods = returns_df.shape[0]
    log(f"Loaded: {n_periods} periods x {n_tickers} tickers")

    if args.dry_run:
        log("\n--- DRY RUN — plan only, no execution ---")
        for u in args.universes:
            k = 100 if "100" in u else 10
            method = "qubo_peripheral" if "PA" in u else "central_ranking"
            penalties = calibrated_penalties(k)
            log(f"  {u}: K={k}, method={method}, "
                f"P_card={penalties['penalty_cardinality']:.0f}")
        return

    # ------------------------------------------------------------------
    # Import network modules
    # ------------------------------------------------------------------
    from network_portfolio_selector import (
        CorrelationNetworkBuilder,
        NetworkFilter,
        NetworkAnalyzer,
        PortfolioSelectionQUBO,
        DWaveSolver,
    )

    # ------------------------------------------------------------------
    # Build network (shared across all universes)
    # ------------------------------------------------------------------
    log("\n--- Building correlation network ---")
    t0 = time.time()

    builder = CorrelationNetworkBuilder(distance_metric="mantegna")
    G_full, corr_matrix = builder.build(returns_df)
    log(f"Full graph: {G_full.number_of_nodes()} nodes, "
        f"{G_full.number_of_edges()} edges")

    net_filter = NetworkFilter(method="tmfg")
    G_filtered = net_filter.filter(G_full)
    log(f"Filtered graph (TMFG): {G_filtered.number_of_edges()} edges")

    analyzer = NetworkAnalyzer()
    metrics_df = analyzer.analyze(G_filtered)
    log(f"Metrics computed for {len(metrics_df)} assets")
    log(f"Communities detected: {metrics_df['community'].nunique()}")
    log(f"Network build time: {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Universe generation
    # ------------------------------------------------------------------
    output_dir = project_root / "results" / "network_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = returns_df.columns.tolist()
    results = {}

    for universe_id in args.universes:
        log(f"\n{'=' * 60}")
        log(f"Generating universe: {universe_id}")
        log(f"{'=' * 60}")

        k = 100 if "100" in universe_id else 10
        max_per_community = 15 if k == 100 else 3
        penalties = calibrated_penalties(k)

        log(f"  K={k}, max_per_community={max_per_community}")
        log(f"  P_card={penalties['penalty_cardinality']:.0f}, "
            f"P_comm={penalties['penalty_community']:.0f}")

        t_start = time.time()

        # ---- Central ranking (PB, PB_100) ----
        if "PB" in universe_id:
            selected = select_central(
                metrics_df=metrics_df,
                target_k=k,
                max_per_community=max_per_community,
                log=log,
            )
            energy = 0.0
            feasible_before_repair = True

        # ---- QUBO peripheral (PA, PA_100) ----
        else:
            qubo_builder = PortfolioSelectionQUBO(
                lambda_correlation=penalties["lambda_correlation"],
                lambda_peripherality=penalties["lambda_peripherality"],
                penalty_cardinality=penalties["penalty_cardinality"],
                penalty_community=penalties["penalty_community"],
                penalty_bridge=penalties["penalty_bridge"],
            )
            bqm = qubo_builder.build(
                metrics_df=metrics_df,
                corr_matrix=corr_matrix,
                tickers=tickers,
                target_k=k,
                max_per_community=max_per_community,
            )
            log(f"  QUBO built: {bqm.num_variables} variables, "
                f"{len(bqm.quadratic)} quadratic terms")

            solver = DWaveSolver(
                backend=args.backend,
                num_reads=args.num_reads,
            )
            raw_result = solver.solve(
                bqm=bqm,
                target_k=k,
                metrics_df=metrics_df,
                max_per_community=max_per_community,
            )

            selected_raw = raw_result.selected_tickers
            energy = raw_result.energy
            feasible_before_repair = raw_result.feasible

            log(f"  Annealer returned: {len(selected_raw)} assets "
                f"(target={k}), feasible={feasible_before_repair}, "
                f"energy={energy:.4f}")

            # Repair to exactly K
            selected = repair_to_k(
                selected_tickers=selected_raw,
                metrics_df=metrics_df,
                target_k=k,
                log=log,
            )

        timing_ms = (time.time() - t_start) * 1000
        method_label = "central_ranking" if "PB" in universe_id else "qubo_peripheral"

        result = build_result_dict(
            universe_id=universe_id,
            method=method_label,
            selected=selected,
            metrics_df=metrics_df,
            n_universe=n_tickers,
            energy=energy,
            feasible_before_repair=feasible_before_repair,
            timing_ms=timing_ms,
        )
        results[universe_id] = result

        # Save individual JSON
        out_file = output_dir / f"selection_{universe_id}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        log(f"  Saved: {out_file.name}")
        log(f"  Tickers: {', '.join(result['selected_tickers'][:10])}"
            + (" ..." if len(result["selected_tickers"]) > 10 else ""))
        log(f"  Avg peripherality: {result['avg_peripherality']:.4f}")
        log(f"  Communities represented: {result['n_communities_represented']}")
        log(f"  Time: {timing_ms:.0f}ms")

    # ------------------------------------------------------------------
    # Selection report (summary of all universes)
    # ------------------------------------------------------------------
    network_summary = {
        "n_assets_universe": n_tickers,
        "n_periods": n_periods,
        "network_edges_full": G_full.number_of_edges(),
        "network_edges_filtered": G_filtered.number_of_edges(),
        "n_communities": int(metrics_df["community"].nunique()),
        "community_sizes": {
            str(k): int(v)
            for k, v in metrics_df["community"].value_counts().items()
        },
        "top_10_peripheral": metrics_df.nlargest(10, "peripherality_score")[
            "ticker"
        ].tolist(),
        "top_10_central": metrics_df.nlargest(10, "eigenvector_centrality")[
            "ticker"
        ].tolist(),
    }

    report = {
        "timestamp": timestamp,
        "dwave_backend": args.backend,
        "network_summary": network_summary,
        "universes": results,
    }

    report_path = output_dir / f"selection_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SELECTION SUMMARY")
    log("=" * 70)
    for uid, res in results.items():
        feasible_tag = "OK" if res["feasible_before_repair"] else "REPAIRED"
        log(
            f"  {uid:8s} | K={res['n_selected']:3d} | {res['method']:18s} | "
            f"periph={res['avg_peripherality']:.3f} | "
            f"comm={res['n_communities_represented']:2d} | {feasible_tag}"
        )

    log(f"\nReport saved: {report_path.name}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    main()
