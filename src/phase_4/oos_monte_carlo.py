#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : oos_monte_carlo.py  (Phase 4 - OOS Monte Carlo Estimator)
# ============================================================================
#
# Description
# -----------
# Monte Carlo CVaR prediction via Cholesky decomposition.
# Uses the same (mu, Sigma) estimated on the training period and
# generates N multivariate normal scenarios from which portfolio
# losses are computed.
#
# CVaR and VaR computation delegates entirely to compute_cvar and
# compute_var from cvar_computation.py (Phase 1) to avoid duplication.
#
# All loss values use the loss convention: positive = bad.
#
# References
# ----------
# [1] Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
# [2] Glasserman (2003). Monte Carlo Methods in Financial Engineering.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import time
import numpy as np
from scipy.stats import gaussian_kde
from typing import Tuple

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
    from oos_types import OOSPrediction
except ImportError:
    from src.phase_4.oos_types import OOSPrediction

# ============================================================================
# PHASE 1 IMPORTS  (compute_cvar / compute_var from cvar_computation.py)
# ============================================================================

try:
    from cvar_computation import compute_cvar, compute_var
except ImportError:
    try:
        from src.phase_1.cvar_computation import compute_cvar, compute_var
    except ImportError:
        # Inline fallback - preserves identical semantics
        def compute_var(returns: np.ndarray, alpha: float = 0.05, **_) -> float:
            return float(np.percentile(returns, alpha * 100))

        def compute_cvar(returns: np.ndarray, alpha: float = 0.05, **_) -> float:
            var_t = compute_var(returns, alpha)
            tail = returns[returns <= var_t]
            return float(np.mean(tail)) if len(tail) > 0 else float(var_t)

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
            handler = logging.FileHandler("logs/oos_monte_carlo.txt", mode="w")
            handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(handler)
            _log.setLevel(logging.INFO)
        return _log

logger = get_logger(__name__)

# ============================================================================
# SCENARIO GENERATION
# ============================================================================

def _simulate_portfolio_losses(
    mu: np.ndarray,
    Sigma: np.ndarray,
    weights: np.ndarray,
    n_scenarios: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate portfolio loss scenarios via Cholesky decomposition.

    Algorithm
    ---------
    1. L = cholesky(Sigma)  such that L L^T = Sigma.
    2. For each scenario k: z_k ~ N(0, I_n), r_k = mu + L z_k.
    3. Portfolio return: rp_k = w^T r_k.
    4. Portfolio loss:   loss_k = -rp_k.

    Parameters
    ----------
    mu : np.ndarray
        Shape (n_assets,). Mean daily return vector from training.
    Sigma : np.ndarray
        Shape (n_assets, n_assets). Covariance matrix from training.
    weights : np.ndarray
        Shape (n_assets,). Portfolio weights.
    n_scenarios : int
        Number of simulated scenarios.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    portfolio_returns : np.ndarray
        Shape (n_scenarios,). Simulated portfolio returns (return convention).
    portfolio_losses : np.ndarray
        Shape (n_scenarios,). Simulated portfolio losses = -portfolio_returns.
    """
    rng = np.random.default_rng(seed)

    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    weights = np.asarray(weights, dtype=float).flatten()

    # Cholesky factor -- use lower triangle: Sigma = L L^T
    L = np.linalg.cholesky(Sigma)                        # (n, n)

    # z_k ~ N(0, I_n),  shape (n, N)
    z = rng.standard_normal((len(mu), n_scenarios))

    # r_k = mu + L z_k,  shape (n, N)
    asset_returns = mu[:, np.newaxis] + L @ z            # (n, N)

    # Portfolio returns, shape (N,)
    portfolio_returns = weights @ asset_returns           # w^T r_k

    portfolio_losses = -portfolio_returns

    return portfolio_returns, portfolio_losses


# ============================================================================
# MONTE CARLO ESTIMATOR
# ============================================================================

def estimate_monte_carlo(
    mu: np.ndarray,
    Sigma: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
    n_scenarios: int = 1000000,
    seed: int = 42,
    n_plot_points: int = 500,
) -> OOSPrediction:
    """
    Predict CVaR and VaR via Monte Carlo simulation under the Gaussian model.

    CVaR and VaR are computed by delegating to compute_cvar and compute_var
    from cvar_computation.py (Phase 1). Those functions operate in return
    convention (negative = loss); results are negated to obtain the loss
    convention (positive = bad) used throughout Phase 4.

    Parameters
    ----------
    mu : np.ndarray
        Shape (n_assets,). Mean daily return per asset from training data.
    Sigma : np.ndarray
        Shape (n_assets, n_assets). Covariance matrix from training data.
    weights : np.ndarray
        Shape (n_assets,). Portfolio weights (must sum to 1).
    alpha : float, optional
        CVaR confidence level. Default 0.05 (95% CVaR).
    n_scenarios : int, optional
        Number of simulated loss scenarios. Default 100 000.
    seed : int, optional
        Random seed for reproducibility. Default 42.
    n_plot_points : int, optional
        Number of evaluation points for the KDE density curve. Default 500.

    Returns
    -------
    OOSPrediction
        Prediction with method='monte_carlo', cvar_predicted, var_predicted,
        and a KDE-smoothed density curve over the simulated losses.
    """
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("MONTE CARLO CVaR ESTIMATION")
    logger.info("=" * 60)
    logger.info(f"n_scenarios={n_scenarios}, alpha={alpha}, seed={seed}")

    # ------------------------------------------------------------------
    # Simulate scenarios
    # ------------------------------------------------------------------
    portfolio_returns, portfolio_losses = _simulate_portfolio_losses(
        mu=mu,
        Sigma=Sigma,
        weights=weights,
        n_scenarios=n_scenarios,
        seed=seed,
    )

    logger.info(
        f"Simulated {n_scenarios} scenarios: "
        f"mean_loss={np.mean(portfolio_losses):.6f}, "
        f"std_loss={np.std(portfolio_losses):.6f}"
    )

    # ------------------------------------------------------------------
    # CVaR and VaR via Phase 1 functions (return convention)
    # compute_cvar returns a negative number (tail of returns)
    # Negate to get loss convention (positive = bad)
    # ------------------------------------------------------------------
    cvar_return = compute_cvar(portfolio_returns, alpha)   # negative
    var_return = compute_var(portfolio_returns, alpha)     # negative

    cvar_loss = -float(cvar_return)
    var_loss = -float(var_return)

    logger.info(f"VaR_loss={var_loss:.6f}, CVaR_loss={cvar_loss:.6f}")

    # ------------------------------------------------------------------
    # KDE density of portfolio losses for plotting
    # ------------------------------------------------------------------
    kde = gaussian_kde(portfolio_losses)
    x_lo = float(np.percentile(portfolio_losses, 0.1))
    x_hi = float(np.percentile(portfolio_losses, 99.9))
    density_x = np.linspace(x_lo, x_hi, n_plot_points)
    density_y = kde(density_x)

    execution_time_s = time.time() - t_start
    logger.info(f"Monte Carlo estimation completed in {execution_time_s:.4f}s")

    return OOSPrediction(
        method="monte_carlo",
        cvar_predicted=cvar_loss,
        var_predicted=var_loss,
        density_x=density_x,
        density_y=density_y,
        execution_time_s=execution_time_s,
        metadata={
            "n_scenarios":    n_scenarios,
            "seed":           seed,
            "alpha":          float(alpha),
            "mean_loss_sim":  float(np.mean(portfolio_losses)),
            "std_loss_sim":   float(np.std(portfolio_losses)),
            "kde_bandwidth":  float(kde.factor),
        },
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "estimate_monte_carlo",
]
