#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Module  : network_portfolio_selector.py  (Phase 0 - Network Portfolio Selection)
# ============================================================================
#
# Description
# -----------
# Implements network-based portfolio selection using complex network theory
# and quantum annealing (D-Wave). The pipeline is:
#   1. Build correlation network from asset returns.
#   2. Filter via TMFG (Triangulated Maximally Filtered Graph).
#   3. Compute centrality metrics and detect communities.
#   4. Formulate asset selection as a QUBO problem.
#   5. Solve on D-Wave (simulated annealing, QPU, or hybrid).
#
# Mathematical Basis
# ------------------
# Distance:    d_ij = sqrt(2 * (1 - rho_ij))     (Mantegna, 1999)
# TMFG:        Retains 3(n-2) edges as a planar triangulation.
# QUBO:        H(x) = lam_corr * sum rho_ij x_i x_j
#                    - lam_periph * sum p_i x_i
#                    + P_card * (sum x_i - K)^2
#                    + P_comm * sum_c (sum_{i in c} x_i - M_c)^2
#                    + P_bridge * sum b_i x_i
#
# References
# ----------
# [1] Mantegna, R.N. (1999). Hierarchical structure in financial markets.
#     Eur. Phys. J. B, 11, 193-197.
# [2] Massara, G.P. et al. (2016). Network filtering for big data:
#     Triangulated Maximally Filtered Graph. J. Complex Networks, 5(2).
# [3] Tumminello, M. et al. (2005). A tool for filtering information in
#     complex systems. PNAS, 102(30), 10421-10426.
# [4] Blondel, V.D. et al. (2008). Fast unfolding of communities in large
#     networks. J. Stat. Mech., P10008.
# [5] Lucas, A. (2014). Ising formulations of many NP problems.
#     Frontiers in Physics, 2, 5.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import warnings

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
                    "logs/network_portfolio_selector.txt", mode="w"
                )
                handler.setFormatter(
                    logging.Formatter("%(levelname)s: %(message)s")
                )
                _logger.addHandler(handler)
                _logger.setLevel(logging.INFO)
            return _logger


logger = get_logger("network_portfolio_selector")

# ============================================================================
# OPTIONAL IMPORTS — D-Wave
# ============================================================================

DWAVE_AVAILABLE = False
try:
    import dimod
    from neal import SimulatedAnnealingSampler

    DWAVE_AVAILABLE = True
    logger.info("D-Wave dimod + neal available")
except ImportError:
    logger.warning("D-Wave packages not installed (pip install dwave-ocean-sdk)")

DWAVE_CLOUD_AVAILABLE = False
try:
    from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridBQMSampler

    DWAVE_CLOUD_AVAILABLE = True
    logger.info("D-Wave cloud solvers available")
except ImportError:
    logger.info("D-Wave cloud solvers not available (optional)")

COMMUNITY_AVAILABLE = False
try:
    import community as community_louvain

    COMMUNITY_AVAILABLE = True
except ImportError:
    try:
        from networkx.algorithms.community import louvain_communities

        COMMUNITY_AVAILABLE = True
    except ImportError:
        logger.warning(
            "Community detection not available "
            "(pip install python-louvain or upgrade networkx>=2.7)"
        )


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class NetworkMetrics:
    """Per-node network metrics for the asset universe."""

    ticker: str
    eigenvector_centrality: float
    betweenness_centrality: float
    degree_centrality: float
    community: int
    peripherality_score: float


@dataclass
class SelectionResult:
    """Result of a QUBO-based portfolio selection."""

    portfolio_id: str
    method: str
    selected_tickers: List[str]
    energy: float
    feasible: bool
    num_reads: int
    solver_info: Dict[str, Any] = field(default_factory=dict)
    network_metrics: Optional[pd.DataFrame] = None
    timing_ms: float = 0.0


# ============================================================================
# 1. CORRELATION NETWORK BUILDER
# ============================================================================


class CorrelationNetworkBuilder:
    """
    Builds a complete weighted graph from asset return correlations.

    Edge weights use Mantegna distance: d_ij = sqrt(2 * (1 - rho_ij)).
    """

    def __init__(self, distance_metric: str = "mantegna"):
        """
        Parameters
        ----------
        distance_metric : str
            Distance metric to use. Currently supports 'mantegna'.
        """
        self.distance_metric = distance_metric
        logger.info(
            f"CorrelationNetworkBuilder initialized (metric={distance_metric})"
        )

    def build(self, returns_df: pd.DataFrame) -> Tuple[nx.Graph, np.ndarray]:
        """
        Build complete weighted graph from returns.

        Parameters
        ----------
        returns_df : pd.DataFrame
            Returns matrix (T periods x N assets).

        Returns
        -------
        graph : nx.Graph
            Complete graph with distance weights.
        corr_matrix : np.ndarray
            Pearson correlation matrix (N x N).
        """
        tickers = returns_df.columns.tolist()
        n_assets = len(tickers)
        logger.info(f"Building correlation network: {n_assets} assets, "
                     f"{len(returns_df)} periods")

        # Pearson correlation
        corr_matrix = returns_df.corr().values
        np.fill_diagonal(corr_matrix, 1.0)

        # Handle NaN correlations
        nan_count = np.isnan(corr_matrix).sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN correlations, filling with 0")
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Distance matrix
        if self.distance_metric == "mantegna":
            dist_matrix = np.sqrt(2.0 * (1.0 - corr_matrix))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

        np.fill_diagonal(dist_matrix, 0.0)

        # Build complete graph
        G = nx.Graph()
        for i, t in enumerate(tickers):
            G.add_node(t, index=i)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                G.add_edge(
                    tickers[i],
                    tickers[j],
                    weight=dist_matrix[i, j],
                    correlation=corr_matrix[i, j],
                )

        logger.info(
            f"Network built: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges"
        )
        return G, corr_matrix


# ============================================================================
# 2. NETWORK FILTER (TMFG / MST)
# ============================================================================


class NetworkFilter:
    """
    Filters a complete correlation network to retain the most informative
    edges using TMFG or MST.

    TMFG (Triangulated Maximally Filtered Graph) retains 3(n-2) edges
    and preserves the hierarchical structure of the market.
    """

    def __init__(self, method: str = "tmfg"):
        """
        Parameters
        ----------
        method : str
            Filtering method: 'tmfg', 'pmfg', or 'mst'.
        """
        self.method = method.lower()
        logger.info(f"NetworkFilter initialized (method={self.method})")

    def filter(self, G: nx.Graph) -> nx.Graph:
        """
        Apply network filtering.

        Parameters
        ----------
        G : nx.Graph
            Complete weighted graph (distance weights).

        Returns
        -------
        filtered : nx.Graph
            Filtered graph.
        """
        n = G.number_of_nodes()
        logger.info(f"Filtering network: {n} nodes, method={self.method}")

        if self.method == "mst":
            filtered = self._build_mst(G)
        elif self.method in ("tmfg", "pmfg"):
            filtered = self._build_tmfg(G)
        else:
            raise ValueError(f"Unknown filtering method: {self.method}")

        logger.info(
            f"Filtered network: {filtered.number_of_nodes()} nodes, "
            f"{filtered.number_of_edges()} edges "
            f"(target for TMFG: {3 * (n - 2)})"
        )
        return filtered

    def _build_mst(self, G: nx.Graph) -> nx.Graph:
        """Minimum Spanning Tree (baseline)."""
        return nx.minimum_spanning_tree(G, weight="weight")

    def _build_tmfg(self, G: nx.Graph) -> nx.Graph:
        """
        Triangulated Maximally Filtered Graph.

        Greedy algorithm: iteratively add the node that maximizes the sum
        of correlations with existing clique members, forming triangulations.
        This is the Massara et al. (2016) approach simplified for robustness.
        """
        nodes = list(G.nodes())
        n = len(nodes)
        target_edges = 3 * (n - 2)

        # Sort all edges by correlation (descending, strongest first)
        edges_sorted = sorted(
            G.edges(data=True),
            key=lambda e: e[2].get("correlation", 0),
            reverse=True,
        )

        # Start with the MST as skeleton (guarantees connectivity)
        mst = nx.minimum_spanning_tree(G, weight="weight")
        tmfg = mst.copy()

        # Add strongest-correlation edges that maintain planarity
        # (approximation: we add edges greedily, checking planarity)
        added = 0
        for u, v, data in edges_sorted:
            if tmfg.has_edge(u, v):
                continue
            if tmfg.number_of_edges() >= target_edges:
                break

            tmfg.add_edge(u, v, **data)

            # Check planarity (exact but O(n) per check via Boyer-Myrvold)
            if not nx.check_planarity(tmfg)[0]:
                tmfg.remove_edge(u, v)
            else:
                added += 1

        logger.info(
            f"TMFG: MST edges={mst.number_of_edges()}, "
            f"added={added}, total={tmfg.number_of_edges()}"
        )
        return tmfg


# ============================================================================
# 3. NETWORK ANALYZER
# ============================================================================


class NetworkAnalyzer:
    """
    Computes centrality metrics and detects communities in the filtered
    network. Produces a peripherality score for each asset.
    """

    def __init__(self, community_algorithm: str = "louvain"):
        """
        Parameters
        ----------
        community_algorithm : str
            Community detection algorithm: 'louvain' or 'greedy'.
        """
        self.community_algorithm = community_algorithm
        logger.info(
            f"NetworkAnalyzer initialized (community={community_algorithm})"
        )

    def analyze(self, G: nx.Graph) -> pd.DataFrame:
        """
        Compute all network metrics.

        Parameters
        ----------
        G : nx.Graph
            Filtered network graph.

        Returns
        -------
        metrics_df : pd.DataFrame
            Columns: ticker, eigenvector_centrality, betweenness_centrality,
            degree_centrality, community, peripherality_score.
        """
        nodes = list(G.nodes())
        n = len(nodes)
        logger.info(f"Analyzing network: {n} nodes, {G.number_of_edges()} edges")

        # Centrality metrics
        eigvec = self._safe_eigenvector_centrality(G)
        between = nx.betweenness_centrality(G, weight="weight")
        degree = nx.degree_centrality(G)

        # Community detection
        communities = self._detect_communities(G)

        # Peripherality: 1 - normalized eigenvector centrality
        max_eig = max(eigvec.values()) if eigvec else 1.0
        min_eig = min(eigvec.values()) if eigvec else 0.0
        range_eig = max_eig - min_eig if max_eig > min_eig else 1.0

        records = []
        for node in nodes:
            eig_norm = (eigvec.get(node, 0) - min_eig) / range_eig
            records.append(
                {
                    "ticker": node,
                    "eigenvector_centrality": eigvec.get(node, 0),
                    "betweenness_centrality": between.get(node, 0),
                    "degree_centrality": degree.get(node, 0),
                    "community": communities.get(node, 0),
                    "peripherality_score": 1.0 - eig_norm,
                }
            )

        metrics_df = pd.DataFrame(records)

        n_communities = metrics_df["community"].nunique()
        avg_periph = metrics_df["peripherality_score"].mean()
        logger.info(
            f"Analysis complete: {n_communities} communities, "
            f"avg peripherality={avg_periph:.4f}"
        )

        return metrics_df

    def _safe_eigenvector_centrality(
        self, G: nx.Graph, max_iter: int = 1000
    ) -> Dict[str, float]:
        """Eigenvector centrality with fallback to degree centrality."""
        try:
            return nx.eigenvector_centrality(
                G, max_iter=max_iter, weight="correlation"
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning(
                "Eigenvector centrality did not converge, "
                "falling back to degree centrality"
            )
            return nx.degree_centrality(G)

    def _detect_communities(self, G: nx.Graph) -> Dict[str, int]:
        """Detect communities using Louvain or greedy modularity."""
        if self.community_algorithm == "louvain" and COMMUNITY_AVAILABLE:
            try:
                # Try python-louvain library first
                partition = community_louvain.best_partition(
                    G, weight="correlation", random_state=42
                )
                n_comm = len(set(partition.values()))
                logger.info(f"Louvain communities detected: {n_comm}")
                return partition
            except Exception as _bare_exc:
                pass  # silently suppress log-write failures (p is undefined here)

            # Fallback to networkx louvain
            try:
                comms = louvain_communities(G, weight="correlation", seed=42)
                partition = {}
                for idx, comm in enumerate(comms):
                    for node in comm:
                        partition[node] = idx
                n_comm = len(comms)
                logger.info(f"NetworkX Louvain communities detected: {n_comm}")
                return partition
            except Exception as e:
                logger.warning(f"Louvain failed: {e}")

        # Fallback: greedy modularity
        try:
            comms = nx.community.greedy_modularity_communities(
                G, weight="correlation"
            )
            partition = {}
            for idx, comm in enumerate(comms):
                for node in comm:
                    partition[node] = idx
            logger.info(f"Greedy modularity communities: {len(comms)}")
            return partition
        except Exception as e:
            logger.warning(f"Community detection failed: {e}, assigning all to 0")
            return {n: 0 for n in G.nodes()}


# ============================================================================
# 4. QUBO FORMULATION
# ============================================================================


class PortfolioSelectionQUBO:
    """
    Formulates the asset selection problem as a QUBO for quantum annealing.

    The objective encodes:
    - Minimize intra-portfolio correlation (diversification).
    - Maximize aggregate peripherality score.
    - Penalize bridge nodes (high betweenness = contagion risk).
    - Enforce exact cardinality K.
    - Limit concentration within any single community.
    """

    def __init__(
        self,
        lambda_correlation: float = 1.0,
        lambda_peripherality: float = 0.5,
        penalty_cardinality: float = 500.0,
        penalty_community: float = 100.0,
        penalty_bridge: float = 0.3,
    ):
        """
        Parameters
        ----------
        lambda_correlation : float
            Weight for correlation penalty (higher = more diversified).
        lambda_peripherality : float
            Weight for peripherality reward.
        penalty_cardinality : float
            Penalty strength for exact cardinality constraint sum(x_i) = K.

            [P2-001] Calibration rule — the penalty must dominate the full
            objective range so the annealer satisfies the cardinality
            equality constraint with high probability:

                P_card  >>  lambda_corr * K*(K-1)/2   (at least 10x the
                                                        maximum objective)

            Calibrated values (10–20x margin):
                K = 10  : P_card =    500  (max_obj ≈ 45   → 11x margin)
                K = 100 : P_card = 100000  (max_obj ≈ 4950 → 20x margin)

            If the annealer returns K±δ assets, double P_card and re-run.
            Greedy cardinality repair is applied as a safety net.

        penalty_community : float
            Penalty strength for community concentration constraint.
            Default 100.0 ≈ 2x the cardinality penalty per community.
        penalty_bridge : float
            Weight for betweenness penalty (contagion avoidance).
        """
        self.lambda_correlation = lambda_correlation
        self.lambda_peripherality = lambda_peripherality
        self.penalty_cardinality = penalty_cardinality
        self.penalty_community = penalty_community
        self.penalty_bridge = penalty_bridge

        logger.info(
            f"QUBO initialized: lam_corr={lambda_correlation}, "
            f"lam_periph={lambda_peripherality}, "
            f"P_card={penalty_cardinality}, P_comm={penalty_community}, "
            f"P_bridge={penalty_bridge}"
        )

    def build(
        self,
        metrics_df: pd.DataFrame,
        corr_matrix: np.ndarray,
        tickers: List[str],
        target_k: int,
        max_per_community: int = 3,
    ) -> Any:
        """
        Build the QUBO as a dimod BinaryQuadraticModel.

        Parameters
        ----------
        metrics_df : pd.DataFrame
            Network metrics (from NetworkAnalyzer).
        corr_matrix : np.ndarray
            Full N x N correlation matrix.
        tickers : List[str]
            Ordered list of tickers matching corr_matrix indices.
        target_k : int
            Number of assets to select.
        max_per_community : int
            Maximum assets allowed from the same community.

        Returns
        -------
        bqm : dimod.BinaryQuadraticModel
            QUBO model ready for D-Wave solvers.
        """
        if not DWAVE_AVAILABLE:
            raise ImportError(
                "D-Wave packages required. Install: pip install dwave-ocean-sdk"
            )

        n = len(tickers)
        logger.info(f"Building QUBO: N={n}, K={target_k}, max_per_comm={max_per_community}")

        # Map tickers to indices
        _ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        metrics_indexed = metrics_df.set_index("ticker")

        # Initialize linear and quadratic terms
        linear = {}
        quadratic = {}

        # --- Term 1: Correlation penalty (off-diagonal) ---
        for i in range(n):
            for j in range(i + 1, n):
                rho = corr_matrix[i, j]
                key = (tickers[i], tickers[j])
                quadratic[key] = quadratic.get(key, 0.0) + self.lambda_correlation * rho

        # --- Term 2: Peripherality reward (linear, negative = reward) ---
        for t in tickers:
            if t in metrics_indexed.index:
                p = metrics_indexed.loc[t, "peripherality_score"]
                linear[t] = linear.get(t, 0.0) - self.lambda_peripherality * p

        # --- Term 3: Bridge penalty (linear, positive = penalty) ---
        for t in tickers:
            if t in metrics_indexed.index:
                b = metrics_indexed.loc[t, "betweenness_centrality"]
                linear[t] = linear.get(t, 0.0) + self.penalty_bridge * b

        # --- Term 4: Cardinality constraint (sum x_i = K) ---
        # (sum x_i - K)^2 = sum x_i^2 + 2 sum_{i<j} x_i x_j - 2K sum x_i + K^2
        # Since x_i^2 = x_i for binary: sum x_i + 2 sum_{i<j} x_i x_j - 2K sum x_i + K^2
        # = (1 - 2K) sum x_i + 2 sum_{i<j} x_i x_j + K^2
        P_card = self.penalty_cardinality
        for t in tickers:
            linear[t] = linear.get(t, 0.0) + P_card * (1.0 - 2.0 * target_k)
        for i in range(n):
            for j in range(i + 1, n):
                key = (tickers[i], tickers[j])
                quadratic[key] = quadratic.get(key, 0.0) + 2.0 * P_card

        # --- Term 5: Community concentration constraint ---
        # For each community c: (sum_{i in c} x_i - M_c)^2  if sum > M_c
        # We use a soft penalty: (sum_{i in c} x_i)^2 - 2*M_c*(sum_{i in c} x_i)
        # when sum > M_c. For simplicity, apply the quadratic penalty always
        # with appropriate scaling.
        P_comm = self.penalty_community
        community_groups: Dict[int, List[str]] = {}
        for t in tickers:
            if t in metrics_indexed.index:
                c = int(metrics_indexed.loc[t, "community"])
                community_groups.setdefault(c, []).append(t)

        for c_id, members in community_groups.items():
            if len(members) <= max_per_community:
                continue
            M_c = max_per_community
            for t in members:
                linear[t] = linear.get(t, 0.0) + P_comm * (1.0 - 2.0 * M_c)
            for ii in range(len(members)):
                for jj in range(ii + 1, len(members)):
                    key = (members[ii], members[jj])
                    quadratic[key] = quadratic.get(key, 0.0) + 2.0 * P_comm

        # Build BQM
        bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)

        logger.info(
            f"QUBO built: {bqm.num_variables} variables, "
            f"{bqm.num_interactions} interactions, "
            f"{len(community_groups)} communities"
        )
        return bqm

    def validate_solution(
        self,
        sample: Dict[str, int],
        target_k: int,
        metrics_df: pd.DataFrame,
        max_per_community: int = 3,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate that a QUBO solution satisfies all constraints.

        Returns
        -------
        feasible : bool
        info : dict
            Diagnostic information about constraint satisfaction.
        """
        selected = [t for t, v in sample.items() if v == 1]
        k = len(selected)

        metrics_indexed = metrics_df.set_index("ticker")
        community_counts: Dict[int, int] = {}
        for t in selected:
            if t in metrics_indexed.index:
                c = int(metrics_indexed.loc[t, "community"])
                community_counts[c] = community_counts.get(c, 0) + 1

        max_comm = max(community_counts.values()) if community_counts else 0
        card_ok = k == target_k
        comm_ok = max_comm <= max_per_community

        info = {
            "selected_count": k,
            "target_k": target_k,
            "cardinality_ok": card_ok,
            "community_counts": community_counts,
            "max_per_community": max_comm,
            "community_limit": max_per_community,
            "community_ok": comm_ok,
        }

        feasible = card_ok and comm_ok
        if not feasible:
            logger.warning(
                f"Solution infeasible: k={k} (target={target_k}), "
                f"max_comm={max_comm} (limit={max_per_community})"
            )

        return feasible, info


# ============================================================================
# 5. D-WAVE SOLVER
# ============================================================================


class DWaveSolver:
    """
    Unified interface to D-Wave solvers: simulated annealing (local),
    QPU (real hardware), and hybrid (LeapHybrid).
    """

    BACKEND_SIMULATED = "simulated_annealing"
    BACKEND_QPU = "qpu"
    BACKEND_HYBRID = "hybrid"

    def __init__(
        self,
        backend: str = "simulated_annealing",
        num_reads: int = 1000,
        qpu_solver: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        backend : str
            'simulated_annealing', 'qpu', or 'hybrid'.
        num_reads : int
            Number of samples to draw.
        qpu_solver : str, optional
            QPU solver name (e.g., 'Advantage_system6.4').

        Notes
        -----
        D-Wave API token is read exclusively from the DWAVE_API_TOKEN
        environment variable. Never hardcode tokens in source or config.
        See .env.example for setup instructions.
        """
        self.backend = backend
        self.num_reads = num_reads
        self.qpu_solver = qpu_solver

        logger.info(
            f"DWaveSolver initialized: backend={backend}, "
            f"num_reads={num_reads}, "
            f"qpu_solver={qpu_solver or 'default'}"
        )

    def solve(
        self,
        bqm: Any,
        target_k: int,
        metrics_df: pd.DataFrame,
        max_per_community: int = 3,
    ) -> SelectionResult:
        """
        Solve the QUBO and return the best feasible solution.

        Parameters
        ----------
        bqm : dimod.BinaryQuadraticModel
            The QUBO model.
        target_k : int
            Expected cardinality.
        metrics_df : pd.DataFrame
            Network metrics for validation.
        max_per_community : int
            Community concentration limit.

        Returns
        -------
        result : SelectionResult
        """
        if not DWAVE_AVAILABLE:
            raise ImportError("D-Wave packages required")

        start_time = datetime.now()
        logger.info(f"Solving QUBO on {self.backend} ({bqm.num_variables} vars)")

        sampler = self._get_sampler()
        sample_set = self._run_sampler(sampler, bqm)

        elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000.0

        # Extract best feasible solution
        qubo_validator = PortfolioSelectionQUBO()
        best_result = self._extract_best(
            sample_set, target_k, metrics_df, max_per_community, qubo_validator
        )
        best_result.timing_ms = elapsed_ms

        logger.info(
            f"Solve complete: {len(best_result.selected_tickers)} assets selected, "
            f"energy={best_result.energy:.4f}, "
            f"feasible={best_result.feasible}, "
            f"time={elapsed_ms:.1f}ms"
        )

        return best_result

    def _get_sampler(self):
        """Create the appropriate sampler."""
        import os

        if self.backend == self.BACKEND_SIMULATED:
            return SimulatedAnnealingSampler()

        token = os.environ.get("DWAVE_API_TOKEN")
        if not token:
            raise ValueError(
                "D-Wave API token required for QPU/hybrid backends. "
                "Set DWAVE_API_TOKEN environment variable."
            )

        if self.backend == self.BACKEND_QPU:
            if not DWAVE_CLOUD_AVAILABLE:
                raise ImportError("dwave-system required for QPU backend")
            kwargs = {"token": token}
            if self.qpu_solver:
                kwargs["solver"] = self.qpu_solver
            return EmbeddingComposite(DWaveSampler(**kwargs))

        if self.backend == self.BACKEND_HYBRID:
            if not DWAVE_CLOUD_AVAILABLE:
                raise ImportError("dwave-system required for hybrid backend")
            return LeapHybridBQMSampler(token=token)

        raise ValueError(f"Unknown backend: {self.backend}")

    def _run_sampler(self, sampler, bqm):
        """Run the sampler with appropriate parameters."""
        if self.backend == self.BACKEND_SIMULATED:
            return sampler.sample(
                bqm, num_reads=self.num_reads, seed=42, beta_range=[0.1, 10.0]
            )
        elif self.backend == self.BACKEND_HYBRID:
            return sampler.sample(bqm, time_limit=10)
        else:
            return sampler.sample(bqm, num_reads=self.num_reads)

    def _extract_best(
        self,
        sample_set,
        target_k: int,
        metrics_df: pd.DataFrame,
        max_per_community: int,
        qubo_validator: PortfolioSelectionQUBO,
    ) -> SelectionResult:
        """Extract the best feasible solution from the sample set."""
        best_feasible = None
        best_infeasible = None
        validator = qubo_validator

        for sample, energy in zip(
            sample_set.samples(), sample_set.record.energy
        ):
            sample_dict = dict(sample)
            feasible, info = validator.validate_solution(
                sample_dict, target_k, metrics_df, max_per_community
            )
            selected = [t for t, v in sample_dict.items() if v == 1]

            result = SelectionResult(
                portfolio_id="",
                method=self.backend,
                selected_tickers=selected,
                energy=float(energy),
                feasible=feasible,
                num_reads=self.num_reads,
                solver_info=info,
            )

            if feasible:
                if best_feasible is None or energy < best_feasible.energy:
                    best_feasible = result
            else:
                if best_infeasible is None or energy < best_infeasible.energy:
                    best_infeasible = result

        if best_feasible:
            logger.info("Found feasible solution")
            return best_feasible

        # Fallback: relax and return best infeasible with warning
        logger.warning(
            "No feasible solution found across all samples. "
            "Returning best infeasible solution. Consider adjusting penalties."
        )
        if best_infeasible:
            return best_infeasible

        # Absolute fallback: empty
        return SelectionResult(
            portfolio_id="",
            method=self.backend,
            selected_tickers=[],
            energy=float("inf"),
            feasible=False,
            num_reads=self.num_reads,
        )
