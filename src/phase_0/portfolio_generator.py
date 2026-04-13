#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Module  : portfolio_generator.py  (Phase 0 - Portfolio Generation Orchestrator)
# ============================================================================
#
# Description
# -----------
# Orchestrates the generation of four portfolios for comparative analysis:
#   A  : Peripheral assets selected via QUBO + D-Wave (K=10).
#   B  : Central assets ranked by eigenvector centrality (K=10).
#   C  : Randomly selected assets (K=10).
#   D  : Peripheral assets via QUBO + D-Wave (K=100, large-scale).
#
# Each portfolio generates a dedicated config YAML file that can be
# executed independently through the existing Phase 1-4 pipeline.
#
# References
# ----------
# [1] Tumminello et al. (2005). PMFG for filtering complex systems.
# [2] Massara et al. (2016). TMFG for network filtering.
# [3] Negre et al. (2019). Community detection via quantum annealing.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import yaml
import json
import copy

from .network_portfolio_selector import (
    CorrelationNetworkBuilder,
    NetworkFilter,
    NetworkAnalyzer,
    PortfolioSelectionQUBO,
    DWaveSolver,
    SelectionResult,
)

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
                    "logs/portfolio_generator.txt", mode="w"
                )
                handler.setFormatter(
                    logging.Formatter("%(levelname)s: %(message)s")
                )
                _logger.addHandler(handler)
                _logger.setLevel(logging.INFO)
            return _logger


logger = get_logger("portfolio_generator")


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class PortfolioSpec:
    """Specification for a single portfolio."""

    portfolio_id: str
    method: str  # qubo_peripheral, central_ranking, random
    size: int
    description: str = ""


@dataclass
class GenerationReport:
    """Complete report from the portfolio generation process."""

    timestamp: str
    n_assets_universe: int
    n_periods: int
    network_edges_full: int
    network_edges_filtered: int
    n_communities: int
    portfolios: Dict[str, SelectionResult] = field(default_factory=dict)
    network_metrics_summary: Dict[str, Any] = field(default_factory=dict)
    configs_generated: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = {
            "timestamp": self.timestamp,
            "n_assets_universe": self.n_assets_universe,
            "n_periods": self.n_periods,
            "network_edges_full": self.network_edges_full,
            "network_edges_filtered": self.network_edges_filtered,
            "n_communities": self.n_communities,
            "network_metrics_summary": self.network_metrics_summary,
            "configs_generated": self.configs_generated,
            "portfolios": {},
        }
        for pid, res in self.portfolios.items():
            d["portfolios"][pid] = {
                "portfolio_id": res.portfolio_id,
                "method": res.method,
                "selected_tickers": res.selected_tickers,
                "energy": res.energy if np.isfinite(res.energy) else None,
                "feasible": res.feasible,
                "num_reads": res.num_reads,
                "timing_ms": res.timing_ms,
                "solver_info": {
                    k: v
                    for k, v in res.solver_info.items()
                    if isinstance(v, (str, int, float, bool, list, dict))
                },
            }
        return d


# ============================================================================
# PORTFOLIO GENERATOR
# ============================================================================


class PortfolioGenerator:
    """
    Generates multiple portfolios using network analysis and quantum annealing.

    Workflow:
    1. Load full returns → build correlation network → filter (TMFG).
    2. Compute centrality + communities.
    3. Generate portfolios A, B, C, D.
    4. Write config YAML files for each.
    5. Produce a JSON selection report.
    """

    def __init__(self, config: Dict[str, Any], project_root: Path):
        """
        Parameters
        ----------
        config : dict
            The 'phase0.network_selection' section of config.yaml.
        project_root : Path
            Project root directory.
        """
        self.config = config
        self.project_root = Path(project_root)
        self.random_seed = config.get("random_seed", 42)

        # Network config
        net_cfg = config.get("network", {})
        self.distance_metric = net_cfg.get("distance_metric", "mantegna")
        self.filtering_method = net_cfg.get("filtering_method", "tmfg")

        # Community config
        comm_cfg = config.get("community", {})
        self.community_algorithm = comm_cfg.get("algorithm", "louvain")
        self.max_per_community_k10 = comm_cfg.get("max_assets_per_community_k10", 3)
        self.max_per_community_k100 = comm_cfg.get("max_assets_per_community_k100", 15)

        # QUBO config
        qubo_cfg = config.get("qubo", {})
        self.lambda_correlation = qubo_cfg.get("lambda_correlation", 1.0)
        self.lambda_peripherality = qubo_cfg.get("lambda_peripherality", 0.5)
        self.penalty_cardinality = qubo_cfg.get("penalty_cardinality", 10.0)
        self.penalty_community = qubo_cfg.get("penalty_community", 5.0)
        self.penalty_bridge = qubo_cfg.get("penalty_bridge", 0.3)

        # D-Wave config
        dw_cfg = config.get("dwave", {})
        self.dwave_backend = dw_cfg.get("backend", "simulated_annealing")
        self.num_reads = dw_cfg.get("num_reads", 1000)
        # API token: read ONLY from DWAVE_API_TOKEN env var (see .env.example)
        self.qpu_solver = dw_cfg.get("qpu_solver", None)

        logger.info(
            f"PortfolioGenerator initialized: "
            f"filter={self.filtering_method}, dwave={self.dwave_backend}"
        )

    def generate_all(
        self,
        returns_df: pd.DataFrame,
        base_config: Dict[str, Any],
        portfolios: Optional[Dict[str, PortfolioSpec]] = None,
    ) -> GenerationReport:
        """
        Generate all portfolios and write config files.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Full universe returns (T x N).
        base_config : dict
            Base config.yaml content to clone for each portfolio.
        portfolios : dict, optional
            Custom portfolio specs. If None, uses defaults (A, B, C, D).

        Returns
        -------
        report : GenerationReport
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tickers = returns_df.columns.tolist()
        n_assets = len(tickers)
        n_periods = len(returns_df)
        np.random.seed(self.random_seed)

        logger.info("=" * 70)
        logger.info("PORTFOLIO GENERATION — START")
        logger.info("=" * 70)
        logger.info(f"Universe: {n_assets} assets, {n_periods} periods")

        # --- Step 1: Build network ---
        builder = CorrelationNetworkBuilder(distance_metric=self.distance_metric)
        G_full, corr_matrix = builder.build(returns_df)
        n_edges_full = G_full.number_of_edges()

        # --- Step 2: Filter network ---
        nf = NetworkFilter(method=self.filtering_method)
        G_filtered = nf.filter(G_full)
        n_edges_filtered = G_filtered.number_of_edges()

        # --- Step 3: Analyze ---
        analyzer = NetworkAnalyzer(community_algorithm=self.community_algorithm)
        metrics_df = analyzer.analyze(G_filtered)
        n_communities = metrics_df["community"].nunique()

        # Initialize report
        report = GenerationReport(
            timestamp=timestamp,
            n_assets_universe=n_assets,
            n_periods=n_periods,
            network_edges_full=n_edges_full,
            network_edges_filtered=n_edges_filtered,
            n_communities=n_communities,
        )

        # Summary stats for report
        report.network_metrics_summary = {
            "avg_eigenvector_centrality": float(
                metrics_df["eigenvector_centrality"].mean()
            ),
            "avg_peripherality": float(metrics_df["peripherality_score"].mean()),
            "avg_betweenness": float(metrics_df["betweenness_centrality"].mean()),
            "community_sizes": (
                metrics_df.groupby("community")
                .size()
                .to_dict()
            ),
            "top_10_peripheral": (
                metrics_df.nlargest(10, "peripherality_score")["ticker"].tolist()
            ),
            "top_10_central": (
                metrics_df.nsmallest(10, "peripherality_score")["ticker"].tolist()
            ),
        }

        # --- Default portfolio specs ---
        if portfolios is None:
            cfg_portfolios = self.config.get("portfolios", {})
            portfolios = {
                "A": PortfolioSpec(
                    portfolio_id="A",
                    method=cfg_portfolios.get("A", {}).get(
                        "method", "qubo_peripheral"
                    ),
                    size=cfg_portfolios.get("A", {}).get("size", 10),
                    description="Peripheral assets via QUBO + D-Wave",
                ),
                "B": PortfolioSpec(
                    portfolio_id="B",
                    method=cfg_portfolios.get("B", {}).get(
                        "method", "central_ranking"
                    ),
                    size=cfg_portfolios.get("B", {}).get("size", 10),
                    description="Central assets by eigenvector centrality",
                ),
                "C": PortfolioSpec(
                    portfolio_id="C",
                    method=cfg_portfolios.get("C", {}).get("method", "random"),
                    size=cfg_portfolios.get("C", {}).get("size", 10),
                    description="Randomly selected assets",
                ),
                "D": PortfolioSpec(
                    portfolio_id="D",
                    method=cfg_portfolios.get("D", {}).get(
                        "method", "qubo_peripheral"
                    ),
                    size=cfg_portfolios.get("D", {}).get("size", 100),
                    description="Large-scale peripheral selection via QUBO + D-Wave",
                ),
            }

        # --- Generate each portfolio ---
        for pid, spec in portfolios.items():
            logger.info(f"--- Generating Portfolio {pid}: {spec.description} ---")
            result = self._generate_single(
                spec, metrics_df, corr_matrix, tickers
            )
            result.portfolio_id = pid
            result.network_metrics = metrics_df[
                metrics_df["ticker"].isin(result.selected_tickers)
            ].copy()
            report.portfolios[pid] = result

            # Write config
            config_path = self._write_config(
                pid, result.selected_tickers, base_config, spec
            )
            report.configs_generated.append(str(config_path))

            logger.info(
                f"Portfolio {pid}: {len(result.selected_tickers)} assets, "
                f"energy={result.energy:.4f}, feasible={result.feasible}"
            )

        # --- Save report ---
        report_path = self._save_report(report)
        logger.info(f"Report saved: {report_path}")

        # --- Save network metrics CSV ---
        metrics_path = (
            self.project_root / "results" / "network_selection" / "network_metrics.csv"
        )
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Network metrics saved: {metrics_path}")

        logger.info("=" * 70)
        logger.info("PORTFOLIO GENERATION — COMPLETE")
        logger.info("=" * 70)

        return report

    def _generate_single(
        self,
        spec: PortfolioSpec,
        metrics_df: pd.DataFrame,
        corr_matrix: np.ndarray,
        tickers: List[str],
    ) -> SelectionResult:
        """Generate a single portfolio based on its specification."""

        if spec.method == "qubo_peripheral":
            return self._generate_qubo(spec, metrics_df, corr_matrix, tickers)
        elif spec.method == "central_ranking":
            return self._generate_central(spec, metrics_df)
        elif spec.method == "random":
            return self._generate_random(spec, tickers)
        else:
            raise ValueError(f"Unknown method: {spec.method}")

    def _generate_qubo(
        self,
        spec: PortfolioSpec,
        metrics_df: pd.DataFrame,
        corr_matrix: np.ndarray,
        tickers: List[str],
    ) -> SelectionResult:
        """Generate portfolio via QUBO + D-Wave."""
        max_per_comm = (
            self.max_per_community_k10
            if spec.size <= 20
            else self.max_per_community_k100
        )

        qubo_builder = PortfolioSelectionQUBO(
            lambda_correlation=self.lambda_correlation,
            lambda_peripherality=self.lambda_peripherality,
            penalty_cardinality=self.penalty_cardinality,
            penalty_community=self.penalty_community,
            penalty_bridge=self.penalty_bridge,
        )

        bqm = qubo_builder.build(
            metrics_df=metrics_df,
            corr_matrix=corr_matrix,
            tickers=tickers,
            target_k=spec.size,
            max_per_community=max_per_comm,
        )

        # Choose backend: for large problems (K>=50), prefer hybrid
        backend = self.dwave_backend
        if spec.size >= 50 and backend == "qpu":
            logger.info(
                f"Portfolio {spec.portfolio_id}: K={spec.size}, "
                f"upgrading to hybrid solver for robustness"
            )
            backend = "hybrid"

        solver = DWaveSolver(
            backend=backend,
            num_reads=self.num_reads,
            qpu_solver=self.qpu_solver,
        )

        return solver.solve(
            bqm=bqm,
            target_k=spec.size,
            metrics_df=metrics_df,
            max_per_community=max_per_comm,
        )

    def _generate_central(
        self, spec: PortfolioSpec, metrics_df: pd.DataFrame
    ) -> SelectionResult:
        """Generate portfolio from the most central assets."""
        top_central = (
            metrics_df.nlargest(spec.size, "eigenvector_centrality")["ticker"]
            .tolist()
        )
        return SelectionResult(
            portfolio_id=spec.portfolio_id,
            method="central_ranking",
            selected_tickers=top_central,
            energy=0.0,
            feasible=True,
            num_reads=0,
            solver_info={"criterion": "eigenvector_centrality_descending"},
        )

    def _generate_random(
        self, spec: PortfolioSpec, tickers: List[str]
    ) -> SelectionResult:
        """Generate portfolio by random selection."""
        selected = list(
            np.random.choice(tickers, size=spec.size, replace=False)
        )
        return SelectionResult(
            portfolio_id=spec.portfolio_id,
            method="random",
            selected_tickers=selected,
            energy=0.0,
            feasible=True,
            num_reads=0,
            solver_info={"seed": self.random_seed},
        )

    def _write_config(
        self,
        portfolio_id: str,
        selected_tickers: List[str],
        base_config: Dict[str, Any],
        spec: PortfolioSpec,
    ) -> Path:
        """Write a config YAML file for this portfolio."""
        cfg = copy.deepcopy(base_config)

        # Update tickers
        cfg.setdefault("phase0", {}).setdefault("data", {})
        cfg["phase0"]["data"]["tickers"] = selected_tickers
        cfg["phase0"]["data"]["use_custom_tickers"] = True
        cfg["phase0"]["data"]["portfolio_name"] = f"net_P{portfolio_id}"
        cfg["phase0"]["data"]["csv_path"] = "data/returns_sp500_full.csv"

        # Update export directory
        cfg.setdefault("export", {})
        cfg["export"]["run_subdirectory"] = f"exp_net_P{portfolio_id}"

        # For portfolio D (large), adjust optimizer settings
        if spec.size >= 50:
            cfg.setdefault("global", {}).setdefault("constraints", {})
            cfg["global"]["constraints"]["max_weight"] = round(
                min(0.3, 3.0 / spec.size), 4
            )

        # Write
        config_dir = self.project_root / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"config_net_P{portfolio_id}.yaml"

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"Config written: {config_path}")
        return config_path

    def _save_report(self, report: GenerationReport) -> Path:
        """Save the generation report as JSON."""
        report_dir = self.project_root / "results" / "network_selection"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / f"selection_report_{report.timestamp}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)

        return report_path
