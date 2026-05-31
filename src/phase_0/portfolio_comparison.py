#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Module  : portfolio_comparison.py  (Phase 0 - Portfolio Comparison)
# ============================================================================
#
# Description
# -----------
# Compares the results of portfolios A (peripheral/D-Wave), B (central),
# C (random), and D (large peripheral) after each has been optimized
# through the full Phase 1-4 pipeline.
#
# Collects and aggregates:
#   - CVaR (in-sample and out-of-sample)
#   - VaR, Sharpe, Sortino, Max Drawdown
#   - Convergence speed and query count
#   - Network topology metrics of selected assets
#   - Community diversity index
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

# ── Passport system (BUG-005 fix) ──────────────────────────────────────────
try:
    from src.utils.passport_orchestrator import get_orchestrator
except ImportError:
    try:
        from utils.passport_orchestrator import get_orchestrator
    except ImportError:
        def get_orchestrator():
            class _D:
                def seal(self, p, **kw): return p
                def checkpoint(self, p, **kw): return p
                def assign_passport(self, *a, **kw): return None
                def link_child(self, *a): pass
            return _D()


# ============================================================================
# LOGGER SETUP
# ============================================================================

try:
    from src.utils.logger import get_logger
except ImportError:
    try:
        from logger import get_logger
    except ImportError:
        import logging

        def get_logger(name: str = __name__) -> logging.Logger:
            _logger = logging.getLogger(name)
            if not _logger.handlers:
                handler = logging.FileHandler(
                    "logs/portfolio_comparison.txt", mode="w"
                )
                handler.setFormatter(
                    logging.Formatter("%(levelname)s: %(message)s")
                )
                _logger.addHandler(handler)
                _logger.setLevel(logging.INFO)
            return _logger


logger = get_logger("portfolio_comparison")


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class PortfolioResult:
    """Aggregated results for a single portfolio after full pipeline."""

    portfolio_id: str
    method: str
    tickers: List[str]
    n_assets: int
    # Risk metrics (in-sample)
    cvar_initial: Optional[float] = None
    cvar_optimized: Optional[float] = None
    cvar_improvement_pct: Optional[float] = None
    var_optimized: Optional[float] = None
    # Performance metrics
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    annual_return: Optional[float] = None
    annual_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    # Out-of-sample
    cvar_oos: Optional[float] = None
    var_oos: Optional[float] = None
    sharpe_oos: Optional[float] = None
    # Optimization metrics
    total_iterations: Optional[int] = None
    converged: Optional[bool] = None
    total_iqae_queries: Optional[int] = None
    optimization_time_s: Optional[float] = None
    # Network metrics
    avg_peripherality: Optional[float] = None
    avg_centrality: Optional[float] = None
    n_communities_represented: Optional[int] = None
    community_diversity_index: Optional[float] = None


@dataclass
class ComparisonReport:
    """Full comparison across all portfolios."""

    timestamp: str
    portfolios: Dict[str, PortfolioResult] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to a comparison DataFrame (one row per portfolio)."""
        records = []
        for pid, pr in self.portfolios.items():
            records.append(
                {
                    "portfolio": pid,
                    "method": pr.method,
                    "n_assets": pr.n_assets,
                    "cvar_initial": pr.cvar_initial,
                    "cvar_optimized": pr.cvar_optimized,
                    "cvar_improvement_pct": pr.cvar_improvement_pct,
                    "var_optimized": pr.var_optimized,
                    "sharpe_ratio": pr.sharpe_ratio,
                    "sortino_ratio": pr.sortino_ratio,
                    "annual_return": pr.annual_return,
                    "annual_volatility": pr.annual_volatility,
                    "max_drawdown": pr.max_drawdown,
                    "cvar_oos": pr.cvar_oos,
                    "var_oos": pr.var_oos,
                    "sharpe_oos": pr.sharpe_oos,
                    "total_iterations": pr.total_iterations,
                    "converged": pr.converged,
                    "total_iqae_queries": pr.total_iqae_queries,
                    "optimization_time_s": pr.optimization_time_s,
                    "avg_peripherality": pr.avg_peripherality,
                    "avg_centrality": pr.avg_centrality,
                    "n_communities_represented": pr.n_communities_represented,
                    "community_diversity_index": pr.community_diversity_index,
                }
            )
        return pd.DataFrame(records).set_index("portfolio")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "timestamp": self.timestamp,
            "comparison_table": self.to_dataframe().reset_index().to_dict(
                orient="records"
            ),
        }


# ============================================================================
# COMPARISON ENGINE
# ============================================================================


class PortfolioComparison:
    """
    Collects results from multiple portfolio experiments and produces
    a unified comparison report.
    """

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        logger.info("PortfolioComparison initialized")

    def collect_results(
        self,
        portfolio_ids: List[str],
        selection_report_path: Optional[Path] = None,
    ) -> ComparisonReport:
        """
        Collect results for all portfolios from their experiment directories.

        Parameters
        ----------
        portfolio_ids : list of str
            Portfolio identifiers (e.g., ['A', 'B', 'C', 'D']).
        selection_report_path : Path, optional
            Path to the network selection report JSON.

        Returns
        -------
        report : ComparisonReport
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = ComparisonReport(timestamp=timestamp)

        # Load selection report for network metrics
        selection_data = None
        if selection_report_path and Path(selection_report_path).exists():
            with open(selection_report_path, "r") as f:
                selection_data = json.load(f)

        for pid in portfolio_ids:
            result = self._load_portfolio_result(pid, selection_data)
            if result:
                report.portfolios[pid] = result
                logger.info(
                    f"Portfolio {pid}: CVaR={result.cvar_optimized}, "
                    f"Sharpe={result.sharpe_ratio}"
                )
            else:
                logger.warning(f"Portfolio {pid}: no results found")

        return report

    def _load_portfolio_result(
        self, portfolio_id: str, selection_data: Optional[Dict]
    ) -> Optional[PortfolioResult]:
        """Load results for a single portfolio from its experiment directory."""
        results_dir = self.project_root / "results"

        # Find the experiment directory for this portfolio
        exp_dir = results_dir / f"exp_net_P{portfolio_id}"
        if not exp_dir.exists():
            logger.warning(f"Experiment directory not found: {exp_dir}")
            return None

        # Find the comprehensive metrics JSON
        metrics_files = sorted(exp_dir.glob("tfm_comprehensive_metrics_*.json"))
        if not metrics_files:
            logger.warning(f"No metrics JSON found in {exp_dir}")
            return None

        metrics_path = metrics_files[-1]  # Latest
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Extract selection info
        sel_info = {}
        if selection_data:
            sel_info = selection_data.get("portfolios", {}).get(portfolio_id, {})

        tickers = sel_info.get("selected_tickers", [])
        method = sel_info.get("method", "unknown")

        # Extract metrics
        opt_metrics = metrics.get("optimization", {})
        port_metrics = metrics.get("portfolio_metrics", {})
        oos_metrics = metrics.get("oos_triple_for_export", {})
        oos_pm = oos_metrics.get("oos_portfolio_metrics", {})

        result = PortfolioResult(
            portfolio_id=portfolio_id,
            method=method,
            tickers=tickers,
            n_assets=len(tickers),
            cvar_initial=opt_metrics.get("cvar_initial"),
            cvar_optimized=opt_metrics.get("cvar_optimized"),
            cvar_improvement_pct=opt_metrics.get("cvar_improvement_percent"),
            var_optimized=opt_metrics.get("var_optimized"),
            sharpe_ratio=port_metrics.get("sharpe_ratio"),
            sortino_ratio=port_metrics.get("sortino_ratio"),
            annual_return=port_metrics.get("annual_return"),
            annual_volatility=port_metrics.get("annual_volatility"),
            max_drawdown=port_metrics.get("max_drawdown"),
            cvar_oos=oos_pm.get("cvar"),
            var_oos=oos_pm.get("var"),
            sharpe_oos=oos_pm.get("sharpe_ratio"),
            total_iterations=opt_metrics.get("total_iterations"),
            converged=opt_metrics.get("converged"),
            total_iqae_queries=opt_metrics.get("total_queries_iqae"),
            optimization_time_s=opt_metrics.get("total_time_s"),
        )

        # Add network metrics from selection report
        if selection_data:
            self._enrich_with_network_metrics(result, selection_data)

        return result

    def _enrich_with_network_metrics(
        self, result: PortfolioResult, selection_data: Dict
    ) -> None:
        """Add network topology metrics to the portfolio result."""
        metrics_path = (
            self.project_root / "results" / "network_selection" / "network_metrics.csv"
        )
        if not metrics_path.exists():
            return

        all_metrics = pd.read_csv(metrics_path)
        selected = all_metrics[all_metrics["ticker"].isin(result.tickers)]

        if selected.empty:
            return

        result.avg_peripherality = float(selected["peripherality_score"].mean())
        result.avg_centrality = float(selected["eigenvector_centrality"].mean())
        result.n_communities_represented = int(selected["community"].nunique())

        # Shannon diversity index for community distribution
        comm_counts = selected["community"].value_counts(normalize=True)
        entropy = -np.sum(comm_counts * np.log(comm_counts + 1e-10))
        result.community_diversity_index = float(entropy)

    def save_report(
        self, report: ComparisonReport, filename: Optional[str] = None
    ) -> Path:
        """Save comparison report as JSON and CSV."""
        output_dir = self.project_root / "results" / "network_selection"
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        json_path = output_dir / (
            filename or f"comparison_report_{report.timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        # CSV
        csv_path = json_path.with_suffix(".csv")
        report.to_dataframe().to_csv(csv_path)

        logger.info(f"Comparison report saved: {json_path}")
        logger.info(f"Comparison CSV saved: {csv_path}")

        return json_path
