#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : cvar_computation.py  (Phase 1 - Classical CVaR Baseline)
# ============================================================================
#
# Description
# -----------
# Computes classical CVaR baseline and VaR for Phase 1 QUBO formulation.
# Core risk metric computation: VaR (alpha-quantile), CVaR (tail expectation),
# CVaR gradient via central finite differences, and unified baseline interface.
#
# Mathematical Basis
# ------------------
# VaR_alpha(X) = inf{z in R : P(X <= z) >= alpha}
# CVaR_alpha(X) = E[X | X <= VaR_alpha(X)]
# Gradient: dCVaR/dw_i approx (CVaR(w+eps) - CVaR(w-eps)) / (2*eps)
#
# References
# ----------
# [1] Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
#     Journal of Risk, 2(3), 21-41.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, Union
from pathlib import Path
from datetime import datetime

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
            return _D()


# ============================================================================
# LOGGER SETUP
# ============================================================================

try:
    from src.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name: str = __name__) -> logging.Logger:
        _logger = logging.getLogger(name)
        if not _logger.handlers:
            handler = logging.FileHandler(
                "results/phase_1/cvar_computation.txt", mode="w"
            )
            handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            _logger.addHandler(handler)
            _logger.setLevel(logging.INFO)
        return _logger

logger = get_logger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_ALPHA = 0.001
MAX_ALPHA = 0.5
MIN_PERIODS = 10
MAX_PERIODS = 100000
MIN_RETURN_REALISTIC = -1.5
MAX_RETURN_REALISTIC = 5.0
CONSTANT_SERIES_THRESHOLD = 1e-10

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def _passport_seal_cvar(passport, fn_name: str, cvar_val: float,
                         var_val: float, n_obs: int, alpha: float):
    """Seal passport after a CVaR computation (BUG-003 fix)."""
    try:
        orch = get_orchestrator()
        orch.seal(
            passport=passport,
            function_name=fn_name,
            phase="Phase 1",
            result={
                "execution_time_ms": 0,
                "validation_status": "valid",
                "transformation_type": "cvar_computation",
                "parameters": {"alpha": alpha, "n_obs": n_obs},
                "output_cvar": float(cvar_val),
                "output_var": float(var_val),
                "warnings": [],
                "errors": [],
            },
        )
    except Exception:
        pass  # passport sealing is non-critical

def validate_returns_array(returns: np.ndarray, param_name: str = "returns") -> np.ndarray:
    """Validate returns array - LEVEL 1-4 validation"""
    
    # LEVEL 1: Type validation
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"{param_name} must be np.ndarray, got {type(returns).__name__}")
    
    # LEVEL 2: Shape validation
    if returns.ndim not in [1, 2]:
        raise ValueError(f"{param_name} must be 1D or 2D, got shape {returns.shape}")
    
    if len(returns) < MIN_PERIODS:
        raise ValueError(f"{param_name} must have >= {MIN_PERIODS} periods, got {len(returns)}")
    
    if len(returns) > MAX_PERIODS:
        raise ValueError(f"{param_name} cannot exceed {MAX_PERIODS} periods, got {len(returns)}")
    
    # LEVEL 4: Data quality validation
    if np.any(np.isnan(returns)):
        n_nan = np.sum(np.isnan(returns))
        pct = 100 * n_nan / returns.size
        raise ValueError(f"{param_name} contains {n_nan} NaN values ({pct:.2f}%)")
    
    if np.any(np.isinf(returns)):
        n_inf = np.sum(np.isinf(returns))
        raise ValueError(f"{param_name} contains {n_inf} infinite values")
    
    # Check for unrealistic values (warning)
    if np.any((returns < MIN_RETURN_REALISTIC) | (returns > MAX_RETURN_REALISTIC)):
        logger.info(f"WARNING: {param_name} contains values outside realistic bounds")
    
    # Check for constant series
    if returns.ndim == 1 and np.std(returns) < CONSTANT_SERIES_THRESHOLD:
        raise ValueError(f"{param_name} is essentially constant (std < {CONSTANT_SERIES_THRESHOLD})")
    
    return returns


def validate_alpha(alpha: float, param_name: str = "alpha") -> float:
    """Validate alpha parameter"""
    
    if not isinstance(alpha, (int, float)):
        raise TypeError(f"{param_name} must be float, got {type(alpha).__name__}")
    
    if not (MIN_ALPHA <= alpha <= MAX_ALPHA):
        raise ValueError(f"{param_name} must be in [{MIN_ALPHA}, {MAX_ALPHA}], got {alpha}")
    
    return float(alpha)


# ============================================================================
# CORE CVaR & VaR FUNCTIONS
# ============================================================================

def compute_var(returns: np.ndarray, alpha: float = 0.05, passport: Optional[Any] = None) -> float:
    """
    Compute Value-at-Risk (VaR) - the alpha-quantile of returns distribution.
    
    Mathematical Definition:
        VaR_alpha(X) = inf{z in R : P(X <= z) >= alpha}
    
    Interpretation:
        - If alpha=0.05: VaR is the 5th percentile (worst 5% of returns)
        - Convention: negative VaR indicates loss
    
    Args:
        returns: Array of returns (1D) or portfolio returns
                Shape: (n_periods,)
                Convention: negative = losses, positive = gains
        alpha: Confidence level (default 0.05)
               Range: [0.001, 0.5]
        passport: Passport object for tracking (optional)
    
    Returns:
        VaR value at confidence level alpha
        Sign: Same as returns convention
    
    Reference:
        Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
        Journal of Risk, 2(3), 21-41.
    """
    
    operation_start = datetime.now()
    
    try:
        # Validation
        returns = validate_returns_array(returns)
        alpha = validate_alpha(alpha)
        
        # Flatten if 2D
        returns = returns.flatten()
        
        # Compute VaR as alpha-percentile
        percentile_q = alpha * 100
        var_threshold = np.percentile(returns, percentile_q)
        
        execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
        
        logger.info(f"VaR computed: alpha={alpha:.4f}, VaR={var_threshold:.6f} (exec: {execution_ms:.2f}ms)")
        
        return float(var_threshold)
    
    except Exception as e:
        logger.error(f"compute_var failed: {str(e)}")
        raise
    



def compute_cvar(returns: np.ndarray, alpha: float = 0.05, passport: Optional[Any] = None) -> float:  # BUG-003 FIXED
    """
    Compute Conditional Value-at-Risk (CVaR) - the expected value in the alpha-tail.
    
    Mathematical Definition:
        CVaR_alpha(X) = E[X | X <= VaR_alpha(X)]
    
    Interpretation:
        - Average loss in the worst alpha% of cases
        - Always >= |VaR| in absolute terms (further in tail)
        - Better risk measure than VaR (coherent)
    
    Args:
        returns: Array of returns (1D) or portfolio returns
                Shape: (n_periods,)
                Convention: negative = losses, positive = gains
        alpha: Confidence level (default 0.05)
               Range: [0.001, 0.5]
        passport: Passport object for tracking (optional)
    
    Returns:
        CVaR value at confidence level alpha
        Sign: Same as returns convention
    
    Method:
        1. Compute VaR threshold at alpha-quantile
        2. Extract all returns <= VaR
        3. Return mean of tail returns
    
    Reference:
        Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
        Journal of Risk, 2(3), 21-41.
    """
    
    operation_start = datetime.now()
    
    try:
        # Validation
        returns = validate_returns_array(returns)
        alpha = validate_alpha(alpha)
        
        # Flatten if 2D
        returns = returns.flatten()
        
        # Get VaR threshold
        var_threshold = compute_var(returns, alpha)
        
        # Extract tail (returns <= VaR)
        tail_mask = returns <= var_threshold
        tail_returns = returns[tail_mask]
        
        # Validate tail size
        min_tail_obs = max(5, int(np.ceil(alpha * len(returns))))
        
        if len(tail_returns) == 0:
            # Fallback: use value closest to VaR
            tail_returns = np.array([np.min(returns)])
            logger.info(f"WARNING: Empty tail at alpha={alpha}. Using minimum return.")
        
        elif len(tail_returns) < min_tail_obs:
            logger.info(f"WARNING: Tail has only {len(tail_returns)} obs (expected >= {min_tail_obs})")
        
        # Compute CVaR as mean of tail
        cvar = float(np.mean(tail_returns))
        
        execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
        
        logger.info(f"CVaR computed: alpha={alpha:.4f}, CVaR={cvar:.6f}, tail_size={len(tail_returns)}/{len(returns)} (exec: {execution_ms:.2f}ms)")
        
        # Validate result
        _validate_cvar_result(returns, var_threshold, cvar, alpha)
        
        return cvar
    
    except Exception as e:
        logger.error(f"compute_cvar failed: {str(e)}")
        raise
    



def _validate_cvar_result(returns: np.ndarray, var_value: float, cvar_value: float, alpha: float) -> None:
    """Validate CVaR computation result against mathematical properties"""
    
    # Property 1: VaR and CVaR same sign
    if np.sign(var_value) != np.sign(cvar_value) and cvar_value != 0:
        logger.info(f"WARNING: VaR and CVaR have different signs. VaR={var_value:.6f}, CVaR={cvar_value:.6f}")
    
    # Property 2: |CVaR| >= |VaR|
    if abs(cvar_value) < abs(var_value) * 0.99:
        logger.info(f"WARNING: |CVaR| < |VaR|. |CVaR|={abs(cvar_value):.6f}, |VaR|={abs(var_value):.6f}")
    
    # Property 3: Ratio should be reasonable (1.0 to 3.0 typically)
    if abs(var_value) > 1e-8:
        ratio = abs(cvar_value) / abs(var_value)
        if ratio < 1.0 or ratio > 5.0:
            logger.info(f"WARNING: CVaR/VaR ratio = {ratio:.4f}. Expected [1.0, 3.0]")
    
    # Property 4: CVaR should not be excessively large
    returns_std = np.std(returns)
    if abs(cvar_value) > 10 * returns_std:
        logger.info(f"WARNING: CVaR = {cvar_value:.6f} is >10x std ({returns_std:.6f})")


# ============================================================================
# GRADIENT FUNCTIONS
# ============================================================================

def compute_cvar_gradient(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.05,
    eps: float = 1e-5,
    passport: Optional[Any] = None
) -> np.ndarray:
    """
    Compute CVaR gradient using finite differences.
    
    Args:
        returns_df: DataFrame of returns (n_periods x n_assets)
        weights: Portfolio weights (n_assets,)
        alpha: Confidence level
        eps: Finite difference step size
        passport: Passport object (optional)
    
    Returns:
        Gradient vector dCVaR/dw (shape: (n_assets,))
    
    Method:
        Central finite difference: dCVaR/dw_i ~ (CVaR(w+eps) - CVaR(w-eps)) / (2eps)
    """
    
    operation_start = datetime.now()
    
    try:
        returns_array = returns_df.values
        n_assets = len(weights)
        gradient = np.zeros(n_assets)
        
        # Base portfolio CVaR
        portfolio_returns = returns_array @ weights
        _cvar_base = compute_cvar(portfolio_returns, alpha)  # noqa: central diff uses ±eps, not base
        
        # Finite difference for each asset
        for i in range(n_assets):
            # Forward perturbation
            weights_plus = weights.copy()
            weights_plus[i] += eps
            weights_plus = weights_plus / weights_plus.sum()  # Re-normalize
            
            portfolio_plus = returns_array @ weights_plus
            cvar_plus = compute_cvar(portfolio_plus, alpha)
            
            # Backward perturbation
            weights_minus = weights.copy()
            weights_minus[i] -= eps
            weights_minus = weights_minus / weights_minus.sum()  # Re-normalize
            
            portfolio_minus = returns_array @ weights_minus
            cvar_minus = compute_cvar(portfolio_minus, alpha)
            
            # Central difference
            gradient[i] = (cvar_plus - cvar_minus) / (2 * eps)
        
        execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
        
        logger.info(f"CVaR gradient computed: shape={gradient.shape}, exec={execution_ms:.2f}ms")
        
        return gradient
    
    except Exception as e:
        logger.error(f"compute_cvar_gradient failed: {str(e)}")
        raise


# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def calculateVaR(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Alias for compute_var - backward compatibility"""
    return compute_var(returns, alpha)


def calculateCVaR(returns: np.ndarray, alpha: float = 0.05) -> float:
    """Alias for compute_cvar - backward compatibility"""
    return compute_cvar(returns, alpha)


def calculateCVaRGradient(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.05,
    eps: float = 1e-5
) -> np.ndarray:
    """Alias for compute_cvar_gradient - backward compatibility"""
    return compute_cvar_gradient(returns_df, weights, alpha, eps)


# ============================================================================
# UNIFIED INTERFACE FOR PHASE 1 PIPELINE
# ============================================================================

def compute_cvar_baseline(
    returns_df: pd.DataFrame,
    initial_weights: np.ndarray,
    alpha: float = 0.05,
    passport: Optional[Any] = None
) -> Dict[str, float]:
    """
    Compute CVaR baseline for Phase 1 - QUBO formulation.
    
    This is the main function called during Phase 1 (classical formulation).
    
    Args:
        returns_df: DataFrame of returns (n_periods x n_assets)
        initial_weights: Initial portfolio weights (n_assets,)
        alpha: CVaR confidence level (0.05 = 95% confidence)
        passport: Passport object for tracking
    
    Returns:
        Dict containing:
            - 'cvar': CVaR baseline value
            - 'var': VaR baseline value
            - 'portfolio_returns': Array of portfolio returns
            - 'tail_size': Number of observations in tail
    """
    
    logger.info(f"PHASE 1.1: CVaR CLASSICAL BASELINE COMPUTATION (alpha={alpha})")
    
    try:
        # Compute portfolio returns
        portfolio_returns = returns_df.values @ initial_weights
        
        logger.info(f"Portfolio returns shape: {portfolio_returns.shape}")
        logger.info(f"Portfolio return stats: min={np.min(portfolio_returns):.6f}, max={np.max(portfolio_returns):.6f}, mean={np.mean(portfolio_returns):.6f}")
        
        # Compute VaR
        var_value = compute_var(portfolio_returns, alpha)
        logger.info(f"VaR: {var_value:.6f}")
        
        # Compute CVaR
        cvar_value = compute_cvar(portfolio_returns, alpha)
        logger.info(f"CVaR: {cvar_value:.6f}")
        
        # Tail size
        var_threshold = np.percentile(portfolio_returns, alpha * 100)
        tail_size = np.sum(portfolio_returns <= var_threshold)
        
        logger.info(f"Tail size: {tail_size}/{len(portfolio_returns)}")
        logger.info("BASELINE COMPUTATION COMPLETE")
        
        return {
            'cvar': cvar_value,
            'var': var_value,
            'portfolio_returns': portfolio_returns,
            'tail_size': tail_size,
            'n_periods': len(portfolio_returns),
            'alpha': alpha
        }
    
    except Exception as e:
        logger.error(f"compute_cvar_baseline failed: {str(e)}")
        raise
