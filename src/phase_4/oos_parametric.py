#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : oos_parametric.py  (Phase 4 - OOS Parametric Estimator)
# ============================================================================
#
# Description
# -----------
# Parametric CVaR prediction under the Gaussian assumption.
# Fits mu and Sigma from the training period, projects onto portfolio
# weights to obtain (mu_p, sigma_p), and applies the closed-form formula:
#
#   CVaR_loss = -mu_p + sigma_p * phi(z_alpha) / alpha
#
# where z_alpha = Phi^{-1}(alpha) and phi is the standard normal PDF.
# VaR_loss = -mu_p + sigma_p * Phi^{-1}(1 - alpha).
#
# All loss values returned use the loss convention: positive = bad.
#
# References
# ----------
# [1] Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
#     Journal of Risk, 2(3), 21–41.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import time
import numpy as np
from scipy import stats
from typing import Tuple, Dict, Any

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
# LOGGER
# ============================================================================

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str = __name__) -> logging.Logger:
        _log = logging.getLogger(name)
        if not _log.handlers:
            handler = logging.FileHandler("logs/oos_parametric.txt", mode="w")
            handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(handler)
            _log.setLevel(logging.INFO)
        return _log

logger = get_logger(__name__)

# ============================================================================
# PORTFOLIO PARAMETER ESTIMATION
# ============================================================================

def compute_portfolio_params(
    returns_train: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Estimate mu, Sigma, mu_p, and sigma_p from the training return matrix.

    All three OOS modules (parametric, Monte Carlo, quantum) call this function
    to guarantee they start from identical model estimates.

    Parameters
    ----------
    returns_train : np.ndarray
        Shape (T, n_assets). Each row is a daily return vector.
        Convention: positive = gain.
    weights : np.ndarray
        Shape (n_assets,). Portfolio weights. Must sum to 1.

    Returns
    -------
    mu : np.ndarray
        Shape (n_assets,). Mean daily return per asset.
    Sigma : np.ndarray
        Shape (n_assets, n_assets). Sample covariance matrix.
    mu_p : float
        Scalar portfolio mean return: w^T mu.
    sigma_p : float
        Scalar portfolio return volatility: sqrt(w^T Sigma w).

    Raises
    ------
    ValueError
        If returns_train has fewer rows than columns (under-determined Sigma).
    """
    returns_train = np.asarray(returns_train, dtype=float)
    weights = np.asarray(weights, dtype=float).flatten()

    T, n = returns_train.shape

    if T < n:
        raise ValueError(
            f"Training period has {T} rows and {n} assets. "
            f"Need T >= n for a full-rank covariance matrix."
        )

    mu = np.mean(returns_train, axis=0)                 # (n,)
    Sigma = np.cov(returns_train, rowvar=False)          # (n, n)
    mu_p = float(weights @ mu)
    sigma_p = float(np.sqrt(weights @ Sigma @ weights))

    logger.info(
        f"Portfolio params: mu_p={mu_p:.6f}, sigma_p={sigma_p:.6f}, "
        f"T={T}, n_assets={n}"
    )

    return mu, Sigma, mu_p, sigma_p


# ============================================================================
# PARAMETRIC ESTIMATOR
# ============================================================================

def estimate_parametric(
    mu_p: float,
    sigma_p: float,
    alpha: float = 0.05,
    n_plot_points: int = 500,
) -> OOSPrediction:
    """
    Predict CVaR and VaR in closed form under the normal distribution assumption.

    Mathematical derivation
    -----------------------
    Let r_p ~ N(mu_p, sigma_p^2) be the portfolio daily return.
    The daily loss L = -r_p ~ N(-mu_p, sigma_p^2).

    Closed-form results at confidence level alpha:
        z_alpha   = Phi^{-1}(alpha)            # negative for alpha < 0.5
        VaR_loss  = -mu_p + sigma_p * Phi^{-1}(1 - alpha)
        CVaR_loss = -mu_p + sigma_p * phi(z_alpha) / alpha

    where phi = standard normal PDF (symmetric, so phi(z_alpha) = phi(-z_alpha)).

    Parameters
    ----------
    mu_p : float
        Portfolio mean daily return estimated from training data.
    sigma_p : float
        Portfolio daily return volatility estimated from training data.
    alpha : float, optional
        CVaR confidence level. Default 0.05 (95% CVaR).
    n_plot_points : int, optional
        Number of points on the loss axis for the density curve. Default 500.

    Returns
    -------
    OOSPrediction
        Prediction with method='parametric', cvar_predicted, var_predicted,
        and the normal loss density curve (density_x, density_y).
    """
    t_start = time.time()

    logger.info("=" * 60)
    logger.info("PARAMETRIC CVaR ESTIMATION")
    logger.info("=" * 60)
    logger.info(f"mu_p={mu_p:.6f}, sigma_p={sigma_p:.6f}, alpha={alpha}")

    # ------------------------------------------------------------------
    # Closed-form CVaR and VaR (loss convention, positive = bad)
    # ------------------------------------------------------------------
    z_alpha = stats.norm.ppf(alpha)                      # e.g. -1.645 for alpha=0.05
    phi_z_alpha = stats.norm.pdf(z_alpha)                # phi is symmetric, always > 0

    var_loss = -mu_p + sigma_p * stats.norm.ppf(1.0 - alpha)
    cvar_loss = -mu_p + sigma_p * phi_z_alpha / alpha

    logger.info(f"z_alpha={z_alpha:.6f}, phi(z_alpha)={phi_z_alpha:.6f}")
    logger.info(f"VaR_loss={var_loss:.6f}, CVaR_loss={cvar_loss:.6f}")

    # ------------------------------------------------------------------
    # Density of L = -r_p ~ N(-mu_p, sigma_p^2) for plotting
    # ------------------------------------------------------------------
    loss_mean = -mu_p
    x_lo = loss_mean - 4.0 * sigma_p
    x_hi = loss_mean + 4.0 * sigma_p

    density_x = np.linspace(x_lo, x_hi, n_plot_points)
    density_y = stats.norm.pdf(density_x, loc=loss_mean, scale=sigma_p)

    execution_time_s = time.time() - t_start
    logger.info(f"Parametric estimation completed in {execution_time_s:.4f}s")

    return OOSPrediction(
        method="parametric",
        cvar_predicted=float(cvar_loss),
        var_predicted=float(var_loss),
        density_x=density_x,
        density_y=density_y,
        execution_time_s=execution_time_s,
        metadata={
            "mu_p":       float(mu_p),
            "sigma_p":    float(sigma_p),
            "alpha":      float(alpha),
            "z_alpha":    float(z_alpha),
            "phi_z":      float(phi_z_alpha),
            "assumption": "Normal(mu_p, sigma_p^2)",
        },
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "compute_portfolio_params",
    "estimate_parametric",
]
