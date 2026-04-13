#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : oos_types.py  (Phase 4 - OOS Comparison Types)
# ============================================================================
#
# Description
# -----------
# Shared dataclasses for the Out-of-Sample triple comparison between
# Parametric, Monte Carlo, and Quantum CVaR prediction methods.
# All loss values follow the loss convention: positive = bad.
#
# Types
# -----
# OOSPrediction       : CVaR prediction and distribution from one method.
# OOSMethodComparison : Accuracy metrics for one method vs. realized OOS.
# OOSComparisonResult : Aggregated result from all three methods.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# ============================================================================
# LOGGER
# ============================================================================

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str = __name__) -> logging.Logger:
        _log = logging.getLogger(name)
        if not _log.handlers:
            handler = logging.FileHandler("logs/oos_types.txt", mode="w")
            handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(handler)
            _log.setLevel(logging.INFO)
        return _log

logger = get_logger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class OOSPrediction:
    """
    passport_id: str = ""  # BUG-006: links OOS result to data passport
    data_hash: str = ""   # hash of input returns used
    CVaR prediction and distribution produced by one estimation method.

    All risk values use the loss convention: positive means loss.

    Attributes
    ----------
    method : str
        Identifier for the method: 'parametric', 'monte_carlo', or 'quantum'.
    cvar_predicted : float
        Predicted CVaR at level alpha in loss convention (positive).
    var_predicted : float
        Predicted VaR at level alpha in loss convention (positive).
    density_x : np.ndarray
        Loss values at which the predicted density is evaluated (for plotting).
    density_y : np.ndarray
        Density / probability mass at each point in density_x.
    execution_time_s : float
        Wall-clock time taken to produce this prediction, in seconds.
    metadata : dict
        Method-specific diagnostic information (e.g. n_scenarios, n_qubits).
    """

    method: str
    cvar_predicted: float
    var_predicted: float
    density_x: np.ndarray
    density_y: np.ndarray
    execution_time_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary (arrays converted to lists)."""
        return {
            "method":           self.method,
            "cvar_predicted":   float(self.cvar_predicted),
            "var_predicted":    float(self.var_predicted),
            "density_x":        self.density_x.tolist(),
            "density_y":        self.density_y.tolist(),
            "execution_time_s": float(self.execution_time_s),
            "metadata":         self.metadata,
        }


@dataclass
class OOSMethodComparison:
    """
    Accuracy metrics comparing one method's CVaR prediction to the OOS reality.

    Attributes
    ----------
    method : str
        Identifier for the method: 'parametric', 'monte_carlo', or 'quantum'.
    cvar_predicted : float
        Predicted CVaR in loss convention.
    cvar_realized : float
        Realized CVaR computed on the test period in loss convention.
    cvar_relative_error : float
        |cvar_predicted - cvar_realized| / |cvar_realized|.
    ks_statistic : float
        Kolmogorov-Smirnov D-statistic between predicted and realized CDFs.
    ks_pvalue : float
        p-value of the KS test.
    wasserstein_distance : float
        Earth Mover's Distance between predicted and realized loss distributions.
    kl_divergence : float
        KL divergence KL(predicted || realized) on discretized histograms.
        Uses small epsilon for numerical stability.
    """

    method: str
    cvar_predicted: float
    cvar_realized: float
    cvar_relative_error: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    kl_divergence: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "method":               self.method,
            "cvar_predicted":       float(self.cvar_predicted),
            "cvar_realized":        float(self.cvar_realized),
            "cvar_relative_error":  float(self.cvar_relative_error),
            "ks_statistic":         float(self.ks_statistic),
            "ks_pvalue":            float(self.ks_pvalue),
            "wasserstein_distance": float(self.wasserstein_distance),
            "kl_divergence":        float(self.kl_divergence),
        }

    def passes_ks(self, significance: float = 0.05) -> bool:
        """Return True if the KS test does not reject the predicted distribution."""
        return self.ks_pvalue >= significance


@dataclass
class OOSComparisonResult:
    """
    master_passport_id: str = ""  # BUG-006: links to pipeline master passport
    Aggregated result from the three-way OOS comparison experiment.

    Attributes
    ----------
    alpha : float
        CVaR confidence level used in all methods.
    n_train : int
        Number of observations in the training period.
    n_test : int
        Number of observations in the test / OOS period.
    cvar_realized : float
        Empirical CVaR on the test period in loss convention.
    var_realized : float
        Empirical VaR on the test period in loss convention.
    predictions : dict
        Map from method name to OOSPrediction for each method.
    comparisons : dict
        Map from method name to OOSMethodComparison for each method.
    oos_metrics : dict
        Portfolio performance metrics (Sharpe, Sortino, MDD) on the test period,
        as returned by MetricsComputation.compute_all_metrics.
    execution_time_s : float
        Total wall-clock time for the full comparison pipeline, in seconds.
    """

    alpha: float
    n_train: int
    n_test: int
    cvar_realized: float
    var_realized: float
    predictions: Dict[str, OOSPrediction]
    comparisons: Dict[str, OOSMethodComparison]
    oos_metrics: Dict[str, Any]
    execution_time_s: float = 0.0

    # ----------------------------------------------------------------
    # Convenience accessors
    # ----------------------------------------------------------------

    def best_method_by_cvar_error(self) -> Optional[str]:
        """Return the method name with the lowest CVaR relative error."""
        if not self.comparisons:
            return None
        return min(
            self.comparisons,
            key=lambda m: self.comparisons[m].cvar_relative_error,
        )

    def ranking_by_cvar_error(self) -> List[str]:
        """Return method names sorted from lowest to highest CVaR relative error."""
        return sorted(
            self.comparisons,
            key=lambda m: self.comparisons[m].cvar_relative_error,
        )

    def ranking_by_wasserstein(self) -> List[str]:
        """Return method names sorted from lowest to highest Wasserstein distance."""
        return sorted(
            self.comparisons,
            key=lambda m: self.comparisons[m].wasserstein_distance,
        )

    # ----------------------------------------------------------------
    # Serialization
    # ----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "alpha":            float(self.alpha),
            "n_train":          int(self.n_train),
            "n_test":           int(self.n_test),
            "cvar_realized":    float(self.cvar_realized),
            "var_realized":     float(self.var_realized),
            "predictions":      {m: p.to_dict() for m, p in self.predictions.items()},
            "comparisons":      {m: c.to_dict() for m, c in self.comparisons.items()},
            "oos_metrics":      self.oos_metrics,
            "execution_time_s": float(self.execution_time_s),
            "best_method":      self.best_method_by_cvar_error(),
            "ranking_cvar":     self.ranking_by_cvar_error(),
            "ranking_wass":     self.ranking_by_wasserstein(),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return human-readable summary for logging."""
        lines = [
            "=" * 70,
            "OOS TRIPLE COMPARISON SUMMARY",
            "=" * 70,
            f"  Alpha          : {self.alpha}",
            f"  Train periods  : {self.n_train}",
            f"  Test periods   : {self.n_test}",
            f"  CVaR realized  : {self.cvar_realized:.6f}",
            f"  VaR realized   : {self.var_realized:.6f}",
            "",
            f"  {'Method':<16} {'CVaR pred':>10} {'Rel. err':>10} "
            f"{'KS stat':>10} {'Wasserstein':>12}",
            f"  {'-'*60}",
        ]
        for method in self.ranking_by_cvar_error():
            c = self.comparisons[method]
            lines.append(
                f"  {method:<16} {c.cvar_predicted:>10.6f} {c.cvar_relative_error:>10.4f} "
                f"{c.ks_statistic:>10.4f} {c.wasserstein_distance:>12.6f}"
            )
        lines.append("")
        lines.append(f"  Best method    : {self.best_method_by_cvar_error()}")
        lines.append(f"  Total time     : {self.execution_time_s:.2f}s")
        lines.append("=" * 70)
        return "\n".join(lines)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "OOSPrediction",
    "OOSMethodComparison",
    "OOSComparisonResult",
]
