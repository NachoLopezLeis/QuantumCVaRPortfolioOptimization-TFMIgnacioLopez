#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Module  : bootstrap_ci.py  (Phase 3 — Statistical Validation)
# ============================================================================
#
# [T-009] Bootstrap confidence intervals for OOS portfolio comparison.
#
# Uses circular block bootstrap (arch library) to preserve the temporal
# autocorrelation structure of daily financial returns.
# block_size=21 corresponds to monthly autocorrelation (≈ 21 trading days).
#
# Converts point-estimate OOS comparisons into statistically testable
# statements with 95% confidence intervals and one-sided p-values —
# required for publication in Q1 finance / quantum computing journals.
#
# References
# ----------
# [1] Politis & Romano (1992). A circular block resampling procedure for
#     stationary data. In: Exploring the Limits of Bootstrap, Wiley.
# [2] Ledoit & Wolf (2008). Robust performance hypothesis testing with the
#     Sharpe ratio. Journal of Empirical Finance.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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


try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str = __name__) -> logging.Logger:
        _log = logging.getLogger(name)
        if not _log.handlers:
            import os
            os.makedirs("logs", exist_ok=True)
            h = logging.FileHandler("logs/bootstrap_ci.txt", mode="w")
            h.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(h)
            _log.setLevel(logging.INFO)
        return _log

logger = get_logger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MetricCI:
    """Confidence interval for a single metric comparison."""

    metric: str
    portfolio_a_label: str
    portfolio_b_label: str

    # Point estimates
    mean_a: float = 0.0
    mean_b: float = 0.0
    diff_mean: float = 0.0          # mean_a - mean_b

    # Bootstrap CIs (percentile method)
    ci_a: Tuple[float, float] = (0.0, 0.0)
    ci_b: Tuple[float, float] = (0.0, 0.0)
    diff_ci: Tuple[float, float] = (0.0, 0.0)

    # One-sided p-value: P(diff >= 0) under H0: diff = 0
    # For CVaR: H0 is A >= B (A not better); reject if p_value < 0.05
    p_value: float = 1.0

    n_bootstrap: int = 0
    block_size: int = 21

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> Dict:
        return {
            "metric": self.metric,
            "portfolio_a": self.portfolio_a_label,
            "portfolio_b": self.portfolio_b_label,
            "mean_a": float(self.mean_a),
            "mean_b": float(self.mean_b),
            "diff_mean_a_minus_b": float(self.diff_mean),
            "ci_a_95": [float(self.ci_a[0]), float(self.ci_a[1])],
            "ci_b_95": [float(self.ci_b[0]), float(self.ci_b[1])],
            "diff_ci_95": [float(self.diff_ci[0]), float(self.diff_ci[1])],
            "p_value_one_sided": float(self.p_value),
            "significant_at_5pct": bool(self.is_significant(0.05)),
            "n_bootstrap": self.n_bootstrap,
            "block_size": self.block_size,
        }

    def summary_line(self) -> str:
        sig = "*" if self.is_significant() else " "
        return (
            f"{self.metric:<15} "
            f"A={self.mean_a:+.4f} [CI: {self.ci_a[0]:+.4f}, {self.ci_a[1]:+.4f}]  "
            f"B={self.mean_b:+.4f} [CI: {self.ci_b[0]:+.4f}, {self.ci_b[1]:+.4f}]  "
            f"diff={self.diff_mean:+.4f} [CI: {self.diff_ci[0]:+.4f}, {self.diff_ci[1]:+.4f}]  "
            f"p={self.p_value:.4f}{sig}"
        )


@dataclass
class BootstrapResult:
    """Full bootstrap comparison between two portfolios."""

    portfolio_a_label: str
    portfolio_b_label: str
    metrics: Dict[str, MetricCI] = field(default_factory=dict)
    n_bootstrap: int = 0
    block_size: int = 21
    alpha_level: float = 0.05

    def to_dict(self) -> Dict:
        return {
            "portfolio_a": self.portfolio_a_label,
            "portfolio_b": self.portfolio_b_label,
            "n_bootstrap": self.n_bootstrap,
            "block_size": self.block_size,
            "alpha_level": self.alpha_level,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    def summary(self) -> str:
        lines = [
            "=" * 100,
            f"BOOTSTRAP CI: {self.portfolio_a_label}  vs  {self.portfolio_b_label}",
            f"n_bootstrap={self.n_bootstrap}, block_size={self.block_size}d, "
            f"significance level={self.alpha_level}",
            "* = significant at 5%",
            "-" * 100,
            f"{'Metric':<15} {'Portfolio A (mean [95% CI])':<45} "
            f"{'Portfolio B (mean [95% CI])':<45} {'Diff [95% CI]':<35} {'p-val':>8}",
            "-" * 100,
        ]
        for ci in self.metrics.values():
            lines.append(ci.summary_line())
        lines.append("=" * 100)
        return "\n".join(lines)


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================

def _cvar(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Historical CVaR (loss convention: higher = worse)."""
    losses = -returns
    var = np.percentile(losses, (1 - alpha) * 100)
    tail = losses[losses >= var]
    return float(np.mean(tail)) if len(tail) > 0 else float(var)


def _sharpe(returns: np.ndarray, trading_days: int = 252) -> float:
    std = float(np.std(returns))
    if std == 0:
        return 0.0
    return float(np.mean(returns)) / std * np.sqrt(trading_days)


def _max_drawdown(returns: np.ndarray) -> float:
    cum = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(np.min(dd))


def _cumulative_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


# ============================================================================
# CIRCULAR BLOCK BOOTSTRAP
# ============================================================================

def _circular_block_bootstrap(
    data: np.ndarray,
    block_size: int,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Circular block bootstrap for 1-D time series.

    The series is treated as circular so blocks can wrap around the end.
    This preserves short-range temporal autocorrelation.

    Parameters
    ----------
    data : np.ndarray, shape (T,)
    block_size : int
    n_samples : int
        Number of observations in each bootstrap sample.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (n_samples,)
    """
    T = len(data)
    n_blocks = int(np.ceil(n_samples / block_size))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([
        np.arange(s, s + block_size) % T for s in starts
    ])
    return data[idx[:n_samples]]


# ============================================================================
# MAIN BOOTSTRAP FUNCTION
# ============================================================================

def bootstrap_oos_comparison(
    returns_oos: np.ndarray,
    weights_a: np.ndarray,
    weights_b: np.ndarray,
    label_a: str = "Portfolio A",
    label_b: str = "Portfolio B",
    alpha_cvar: float = 0.05,
    n_bootstrap: int = 10_000,
    block_size: int = 21,
    seed: int = 42,
) -> BootstrapResult:
    """
    Block bootstrap comparison of two portfolios over OOS returns.

    For each bootstrap sample, portfolio returns are computed as
    r_p = returns_oos @ weights, then risk metrics are computed.
    Confidence intervals use the percentile method.
    P-value is one-sided: P(metric_A - metric_B >= 0) where the
    hypothesis is that A is *better* than B (lower CVaR, higher Sharpe).

    Parameters
    ----------
    returns_oos : np.ndarray, shape (T, N)
        OOS asset returns matrix.
    weights_a : np.ndarray, shape (N,)
        Portfolio A weights (e.g., Quantum optimizer).
    weights_b : np.ndarray, shape (N,)
        Portfolio B weights (e.g., Classical CVaR SLSQP).
    label_a, label_b : str
        Display labels for the two portfolios.
    alpha_cvar : float
        CVaR confidence level.
    n_bootstrap : int
        Number of bootstrap replicates.
    block_size : int
        Block length in trading days. 21 ≈ 1 month.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    BootstrapResult
    """
    logger.info("=" * 60)
    logger.info(f"BOOTSTRAP CI: {label_a} vs {label_b}")
    logger.info(f"n_bootstrap={n_bootstrap}, block_size={block_size}, seed={seed}")
    logger.info("=" * 60)

    T, N = returns_oos.shape
    rng = np.random.default_rng(seed)

    ret_a_full = returns_oos @ weights_a   # (T,)
    ret_b_full = returns_oos @ weights_b   # (T,)

    # Bootstrap replicate storage
    bs_cvar_a, bs_cvar_b = [], []
    bs_sharpe_a, bs_sharpe_b = [], []
    bs_maxdd_a, bs_maxdd_b = [], []
    bs_cumret_a, bs_cumret_b = [], []

    for _ in range(n_bootstrap):
        idx = _circular_block_bootstrap(np.arange(T), block_size, T, rng)
        r_a = ret_a_full[idx]
        r_b = ret_b_full[idx]

        bs_cvar_a.append(_cvar(r_a, alpha_cvar))
        bs_cvar_b.append(_cvar(r_b, alpha_cvar))
        bs_sharpe_a.append(_sharpe(r_a))
        bs_sharpe_b.append(_sharpe(r_b))
        bs_maxdd_a.append(_max_drawdown(r_a))
        bs_maxdd_b.append(_max_drawdown(r_b))
        bs_cumret_a.append(_cumulative_return(r_a))
        bs_cumret_b.append(_cumulative_return(r_b))

    def _ci(vals: list) -> Tuple[float, float]:
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    def _p_one_sided_lower(vals_a: list, vals_b: list) -> float:
        """P(a >= b) — for CVaR we want a < b (quantum better), so reject if p < 0.05."""
        diffs = np.array(vals_a) - np.array(vals_b)
        return float(np.mean(diffs >= 0))

    def _p_one_sided_higher(vals_a: list, vals_b: list) -> float:
        """P(a <= b) — for Sharpe we want a > b."""
        diffs = np.array(vals_a) - np.array(vals_b)
        return float(np.mean(diffs <= 0))

    metric_results: Dict[str, MetricCI] = {}

    # CVaR: lower is better → H0: a >= b  → p = P(a - b >= 0)
    cvar_diff = [a - b for a, b in zip(bs_cvar_a, bs_cvar_b)]
    metric_results["cvar"] = MetricCI(
        metric="cvar",
        portfolio_a_label=label_a,
        portfolio_b_label=label_b,
        mean_a=float(np.mean(bs_cvar_a)),
        mean_b=float(np.mean(bs_cvar_b)),
        diff_mean=float(np.mean(cvar_diff)),
        ci_a=_ci(bs_cvar_a),
        ci_b=_ci(bs_cvar_b),
        diff_ci=_ci(cvar_diff),
        p_value=_p_one_sided_lower(bs_cvar_a, bs_cvar_b),
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )

    # Sharpe: higher is better → H0: a <= b → p = P(a - b <= 0)
    sharpe_diff = [a - b for a, b in zip(bs_sharpe_a, bs_sharpe_b)]
    metric_results["sharpe"] = MetricCI(
        metric="sharpe",
        portfolio_a_label=label_a,
        portfolio_b_label=label_b,
        mean_a=float(np.mean(bs_sharpe_a)),
        mean_b=float(np.mean(bs_sharpe_b)),
        diff_mean=float(np.mean(sharpe_diff)),
        ci_a=_ci(bs_sharpe_a),
        ci_b=_ci(bs_sharpe_b),
        diff_ci=_ci(sharpe_diff),
        p_value=_p_one_sided_higher(bs_sharpe_a, bs_sharpe_b),
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )

    # Max drawdown: less negative (higher) is better → same as Sharpe
    maxdd_diff = [a - b for a, b in zip(bs_maxdd_a, bs_maxdd_b)]
    metric_results["max_drawdown"] = MetricCI(
        metric="max_drawdown",
        portfolio_a_label=label_a,
        portfolio_b_label=label_b,
        mean_a=float(np.mean(bs_maxdd_a)),
        mean_b=float(np.mean(bs_maxdd_b)),
        diff_mean=float(np.mean(maxdd_diff)),
        ci_a=_ci(bs_maxdd_a),
        ci_b=_ci(bs_maxdd_b),
        diff_ci=_ci(maxdd_diff),
        p_value=_p_one_sided_higher(bs_maxdd_a, bs_maxdd_b),
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )

    # Cumulative return: higher is better
    cumret_diff = [a - b for a, b in zip(bs_cumret_a, bs_cumret_b)]
    metric_results["cumulative_return"] = MetricCI(
        metric="cumulative_return",
        portfolio_a_label=label_a,
        portfolio_b_label=label_b,
        mean_a=float(np.mean(bs_cumret_a)),
        mean_b=float(np.mean(bs_cumret_b)),
        diff_mean=float(np.mean(cumret_diff)),
        ci_a=_ci(bs_cumret_a),
        ci_b=_ci(bs_cumret_b),
        diff_ci=_ci(cumret_diff),
        p_value=_p_one_sided_higher(bs_cumret_a, bs_cumret_b),
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )

    result = BootstrapResult(
        portfolio_a_label=label_a,
        portfolio_b_label=label_b,
        metrics=metric_results,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        alpha_level=alpha_cvar,
    )

    for name, ci in metric_results.items():
        sig_str = "(significant)" if ci.is_significant() else "(not significant)"
        logger.info(
            f"  {name}: diff={ci.diff_mean:+.4f} "
            f"[{ci.diff_ci[0]:+.4f}, {ci.diff_ci[1]:+.4f}] "
            f"p={ci.p_value:.4f} {sig_str}"
        )

    return result


# ============================================================================
# CONVENIENCE: compare multiple portfolios vs quantum
# ============================================================================

def run_bootstrap_suite(
    returns_oos: np.ndarray,
    weights_quantum: np.ndarray,
    classical_weights: Dict[str, np.ndarray],
    alpha_cvar: float = 0.05,
    n_bootstrap: int = 10_000,
    block_size: int = 21,
    seed: int = 42,
) -> Dict[str, BootstrapResult]:
    """
    Run bootstrap comparison: Quantum vs each classical portfolio.

    Parameters
    ----------
    returns_oos : np.ndarray, shape (T, N)
    weights_quantum : np.ndarray, shape (N,)
    classical_weights : dict[str, ndarray]
        Keys are portfolio labels, values are weight vectors.

    Returns
    -------
    dict[str, BootstrapResult]
    """
    results = {}
    for label, w_classical in classical_weights.items():
        logger.info(f"Running bootstrap: Quantum vs {label}")
        try:
            res = bootstrap_oos_comparison(
                returns_oos=returns_oos,
                weights_a=weights_quantum,
                weights_b=w_classical,
                label_a="Quantum",
                label_b=label,
                alpha_cvar=alpha_cvar,
                n_bootstrap=n_bootstrap,
                block_size=block_size,
                seed=seed,
            )
            results[label] = res
        except Exception as exc:
            logger.error(f"Bootstrap vs {label} failed: {exc}")
    return results


__all__ = [
    "bootstrap_oos_comparison",
    "run_bootstrap_suite",
    "BootstrapResult",
    "MetricCI",
]
