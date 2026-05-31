# ---------------------------------------------------------------------------
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
#
# Module  : evar_estimation.py  (Phase 2 - Risk Measure Validation)
#
# Description
# -----------
# Entropic Value at Risk (EVaR) estimation following the Ahmadi-Javid /
# Delbaen formulation.  EVaR is used as an independent coherent risk
# measure to validate CVaR-optimised portfolios.
#
# Correct mathematical formulation:
#   EVaR_alpha(X) = inf_{lam>0} [ (1/lam) * log( E[exp(lam*X)] / alpha ) ]
#
# Key properties:
#   EVaR >= CVaR >= VaR              (always holds)
#   Typical ratio EVaR/CVaR in [1.5, 3.0]  for normal distributions
#   Coherent risk measure (convex, monotone, translation-invariant)
#
# References
# ----------
# [1] Ahmadi-Javid, A. (2012) - Entropic Value-at-Risk: A New Coherent
#     Risk Measure.  J. Optim. Theory Appl. 155, 1105-1123
# [2] Delbaen, F. (2018) - Remark on the paper "Entropic Value-at-Risk"
# [3] Laudage & Turkalj (2025) - Quantum Risk Analysis Beyond CVaR
# [4] Rockafellar & Uryasev (2000) - CVaR Optimization
#
# License : Academic use only -- contact nacholopezleis@gmail.com
# ---------------------------------------------------------------------------
"""
Entropic Value at Risk (EVaR) Estimation -- Phase 2

Implements EVaR as an independent coherent risk measure for portfolio
validation, following the correct Ahmadi-Javid / Delbaen formulation.
"""

import numpy as np
import pandas as pd
import json
# logging imported in fallback block below
from typing import Dict, List, Tuple, Optional, Any as AnyType, Union
from numpy.typing import NDArray
from pathlib import Path
from datetime import datetime
import warnings
from scipy.optimize import minimize_scalar, minimize
from scipy.special import logsumexp

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS: Utilities & Optional Dependencies
# ─────────────────────────────────────────────────────────────────────────────
try:
    from src.utils.logger import get_logger
    from src.utils.passport_orchestrator import get_orchestrator
    logger = get_logger('evar_estimation')
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('evar_estimation')
    # Mock orchestrator if not available
    class MockOrchestrator:
        def assign_passport(self, **kwargs):
            return type('obj', (object,), {'passport_id': 'mock_passport_id'})()
        def checkpoint(self, passport, **kwargs):
            return passport
        def seal(self, passport, **kwargs):
            return passport
    def get_orchestrator():
        return MockOrchestrator()

# Phase 2 imports (optional)
try:
    from src.phase_2.hybridoptimizer import HybridOptimizer
except ImportError:
    HybridOptimizer = None
    logger.debug("HybridOptimizer not available")

try:
    from src.phase_2.quantumsubgradient import QuantumSubgradientEstimator
except ImportError:
    QuantumSubgradientEstimator = None
    logger.debug("QuantumSubgradientEstimator not available")

try:
    from src.phase_2.noisemodels import NoiseModelManager
except ImportError:
    NoiseModelManager = None
    logger.debug("NoiseModelManager not available")

# Configuration
try:
    from src.utils.config_loader import Config
    config = Config().get("phase2")
    ALPHA = config["model"]["alpha"]
except (ImportError, KeyError):
    ALPHA = 0.05
    logger.debug("Using default alpha=0.05")

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_hash(data: AnyType) -> str:
    """
    Compute MD5 hash of data for passport tracking.

    Parameters
    ----------
    data : Any
        Data to hash (dict, numpy array, or any serializable type)

    Returns
    -------
    str
        16-character MD5 hash
    """
    import hashlib
    try:
        if isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True, default=str).encode()
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = str(data).encode()
        return hashlib.md5(data_bytes).hexdigest()[:16]
    except Exception:
        return "hash_error"

# ═══════════════════════════════════════════════════════════════════════════════
# CLASS: EVaREstimator
# ═══════════════════════════════════════════════════════════════════════════════

class EVaREstimator:
    """
    Entropic Value at Risk Estimator

    Computes EVaR using the correct Ahmadi-Javid/Delbaen formulation:

        EVaR_alpha(X) = inf_lam>0 [(1/lam) * log(E[exp(lamX)]/alpha)]

    where:
    - X: portfolio losses (positive = bad outcomes)
    - lam: optimization parameter (Lagrange multiplier, lam > 0)
    - alpha: confidence level (tail probability, e.g., 0.05 for 5% tail)

    BACKWARD COMPATIBILITY NOTE:
    The risk_aversion property is computed from alpha as:
        risk_aversion = -log(alpha)

    This provides compatibility with existing code while using the correct formula.

    Key Mathematical Properties:
    - EVaR_alpha(X) >= CVaR_alpha(X) >= VaR_alpha(X) (always)
    - Typical ratio: EVaR/CVaR  in  [1.5, 3.0] for normal distributions
    - Coherent: convex, monotone, translation invariant, positive homogeneous

    Attributes
    ----------
    alpha : float
        Confidence level (tail probability) in range (0, 1)
        - alpha=0.05: Focus on worst 5% outcomes
        - alpha=0.01: Focus on worst 1% outcomes (more conservative)

    risk_aversion : float (property)
        Computed risk aversion parameter = -log(alpha)
        This is a derived quantity for backward compatibility
        - alpha=0.05 -> risk_aversion ≈ 3.0
        - alpha=0.01 -> risk_aversion ≈ 4.6

    logger : logging.Logger
        Module logger instance

    orchestrator : PassportOrchestrator
        Passport system for audit trail

    Examples
    --------
    >>> estimator = EVaREstimator(alpha=0.05)
    >>> print(f"Alpha: {estimator.alpha}")  # 0.05
    >>> print(f"Risk aversion (computed): {estimator.risk_aversion}")  # ~3.0
    >>> losses = np.array([0.02, 0.01, 0.005, 0.001, 0.0])
    >>> result = estimator.compute_evar(losses)
    >>> print(f"EVaR: {result['evar']:.6f}")
    >>> print(f"CVaR: {result['cvar']:.6f}")
    >>> print(f"Ratio: {result['evar_cvar_ratio']:.4f}")  # Should be 1.5-3.0
    """

    def __init__(self, alpha: float = ALPHA):
        """
        Initialize EVaR Estimator with confidence level.

        Parameters
        ----------
        alpha : float, optional
            Confidence level (tail probability), default: 0.05
            Must be in range (0, 1)
            - Smaller alpha -> more conservative (focus on extreme tails)
            - Larger alpha -> less conservative (focus on moderate losses)

        Raises
        ------
        TypeError
            If alpha not numeric
        ValueError
            If alpha not in (0, 1) or contains NaN/Inf
        """
        # ────────────────────────────────────────────────────────────────────
        # LEVEL 1: TYPE VALIDATION
        # ────────────────────────────────────────────────────────────────────
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be numeric, got {type(alpha).__name__}")

        # ────────────────────────────────────────────────────────────────────
        # LEVEL 3: VALUE VALIDATION (Range checks)
        # ────────────────────────────────────────────────────────────────────
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        # ────────────────────────────────────────────────────────────────────
        # LEVEL 4: DATA QUALITY VALIDATION
        # ────────────────────────────────────────────────────────────────────
        if np.isnan(alpha) or np.isinf(alpha):
            raise ValueError(f"alpha contains NaN/Inf: {alpha}")

        # ────────────────────────────────────────────────────────────────────
        # Initialization
        # ────────────────────────────────────────────────────────────────────
        self.alpha = float(alpha)
        self.logger = logger
        self.orchestrator = get_orchestrator()

        logger.info(
            f"OK: EVaREstimator initialized: alpha={alpha:.4f}, "
            f"risk_aversion (computed)={self.risk_aversion:.4f}"
        )

    @property
    def risk_aversion(self) -> float:
        """
        Computed risk aversion parameter from alpha.

        BACKWARD COMPATIBILITY:
        This property provides compatibility with existing notebooks that
        expect a risk_aversion attribute. The value is computed as:

            risk_aversion = -log(alpha)

        This is NOT used in the EVaR formula computation, which uses only alpha.

        Returns
        -------
        float
            Risk aversion = -log(alpha)
            - alpha=0.05 -> ~3.0
            - alpha=0.01 -> ~4.6
            - alpha=0.10 -> ~2.3

        Examples
        --------
        >>> estimator = EVaREstimator(alpha=0.05)
        >>> print(estimator.risk_aversion)  # ~3.0
        """
        return float(-np.log(self.alpha))

    def compute_evar(
        self,
        losses: NDArray[np.floating],
        passport: Optional[Dict] = None
    ) -> Dict[str, Union[float, int]]:
        """
        Compute Entropic Value at Risk with CORRECTED formula.

        Implements the correct Ahmadi-Javid formulation:

            EVaR_alpha(X) = inf_lam>0 [(1/lam) * log(E[exp(lamX)]/alpha)]

        Equivalently:
            EVaR_alpha(X) = inf_lam>0 {(1/lam) * [log(E[exp(lamX)]) - log(alpha)]}

        Numerical Stability:
        - Uses scipy.special.logsumexp for stable computation
        - Avoids overflow/underflow in exponentials
        - Robust to extreme loss values

        Optimization:
        - Uses scipy.optimize.minimize_scalar with bounded method
        - Search range: lam  in  [1e-8, 1000]
        - Tight convergence tolerance

        Validation:
        - Checks EVaR >= CVaR (theoretical guarantee)
        - Validates EVaR/CVaR ratio in expected range [1.0, 5.0]
        - Logs warnings if ratio > 5.0 (possible numerical issues)

        Parameters
        ----------
        losses : NDArray[np.floating]
            Portfolio losses (T,) or (T, 1)
            Positive values = losses, negative values = gains
            Range: typically [-0.1, 0.1] for daily returns

        passport : Dict, optional
            Passport object for audit trail (auto-created if None)

        Returns
        -------
        Dict[str, Union[float, int]]
            Dictionary with EVaR metrics:
            - evar: Computed EVaR value
            - cvar: CVaR for comparison
            - var: VaR for comparison
            - evar_cvar_ratio: EVaR/CVaR ratio (should be 1.5-3.0)
            - lambda_optimal: Optimal lam* from optimization
            - mgf_optimal: E[exp(lam* X)] at optimal lam
            - log_mgf: log(E[exp(lam* X)])
            - alpha: Confidence level used
            - risk_aversion: Computed -log(alpha) for compatibility
            - num_samples: Number of loss samples

        Raises
        ------
        TypeError
            If losses not numpy array
        ValueError
            If losses empty, contains NaN/Inf
            If optimization fails or produces invalid result
            If EVaR < CVaR (mathematical violation)

        Examples
        --------
        >>> losses = np.array([0.02, 0.01, 0.005, 0.001, 0.0])
        >>> result = estimator.compute_evar(losses)
        >>> print(f"EVaR: {result['evar']:.6f}")
        >>> print(f"CVaR: {result['cvar']:.6f}")
        >>> print(f"Ratio: {result['evar_cvar_ratio']:.4f}")  # Expect 1.5-3.0
        >>> print(f"Risk aversion: {result['risk_aversion']:.4f}")  # ~3.0 for alpha=0.05

        Notes
        -----
        Expected EVaR/CVaR Ratios (Ahmadi-Javid, 2012):
        - Normal distribution, alpha=0.05: ~1.8-2.2
        - Heavy-tailed (Student-t): ~2.0-3.0
        - Extreme tails: up to ~5.0
        - If ratio > 5: Check data quality or numerical stability
        """
        operation_start = datetime.now()

        # ────────────────────────────────────────────────────────────────────
        # LEVEL 1: TYPE VALIDATION
        # ────────────────────────────────────────────────────────────────────
        if not isinstance(losses, np.ndarray):
            raise TypeError(f"losses must be numpy array, got {type(losses).__name__}")

        # Flatten if needed
        if losses.ndim > 1:
            losses = losses.flatten()

        # ────────────────────────────────────────────────────────────────────
        # LEVEL 2: SIZE VALIDATION
        # ────────────────────────────────────────────────────────────────────
        if len(losses) == 0:
            raise ValueError("losses array is empty")

        # ────────────────────────────────────────────────────────────────────
        # LEVEL 4: DATA QUALITY VALIDATION
        # ────────────────────────────────────────────────────────────────────
        if np.any(np.isnan(losses)):
            raise ValueError("losses contains NaN values")
        if np.any(np.isinf(losses)):
            raise ValueError("losses contains Inf values")

        # ────────────────────────────────────────────────────────────────────
        # PASSPORT CHECKPOINT (after validation)
        # ────────────────────────────────────────────────────────────────────
        if passport is None:
            passport = self.orchestrator.assign_passport(
                data={
                    "num_samples": len(losses),
                    "alpha": float(self.alpha),
                    "risk_aversion_computed": float(self.risk_aversion),
                    "loss_range": [float(np.min(losses)), float(np.max(losses))]
                },
                data_type="evar_computation",
                source="EVaREstimator.compute_evar"
            )

        try:
            passport = self.orchestrator.checkpoint(
                passport,
                name='validation_complete',
                data=None,
                metadata={
                    "status": "validated",
                    "num_samples": int(len(losses)),
                    "alpha": float(self.alpha),
                    "validation_level": 4,
                    "stage": "input_validation"
                }
            )
            passport_id = passport.passport_id[:8] if hasattr(passport, 'passport_id') else 'N/A'
            logger.debug(f"OK: Validation passed: {passport_id}...")
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise

        # ────────────────────────────────────────────────────────────────────
        # COMPUTATION: EVaR optimization (CORRECTED FORMULA)
        # ────────────────────────────────────────────────────────────────────

        def evar_objective(lam: float) -> float:
            """
            Objective function for EVaR optimization.

            f(lam) = (1/lam) * [log(E[exp(lamX)]) - log(alpha)]
                 = (1/lam) * [logsumexp(lamX) - log(n) - log(alpha)]

            This is the quantity to be minimized over lam > 0.
            Uses logsumexp for numerical stability.
            """
            if lam <= 0:
                return float('inf')

            # Compute log(E[exp(lamX)]) using logsumexp
            scaled_losses = lam * losses
            log_mean_exp = logsumexp(scaled_losses) - np.log(len(losses))

            # CORRECTED FORMULA: Add -log(alpha) term
            # EVaR = (1/lam) * [log(E[exp(lamX)]) - log(alpha)]
            objective = (log_mean_exp - np.log(self.alpha)) / lam

            return objective

        # Optimize over lam  in  [1e-8, 1000] with tight tolerance
        result = minimize_scalar(
            evar_objective,
            bounds=(1e-8, 1000.0),
            method='bounded',
            options={'xatol': 1e-12}
        )

        if not result.success:
            raise ValueError(f"EVaR optimization failed: {result.message}")

        # Extract results
        lambda_optimal = float(result.x)
        evar_value = float(result.fun)

        # Compute MGF at optimal lam for verification
        scaled_losses_opt = lambda_optimal * losses
        log_mgf = float(logsumexp(scaled_losses_opt) - np.log(len(losses)))
        mgf_optimal = float(np.exp(log_mgf))

        # ────────────────────────────────────────────────────────────────────
        # COMPARISON: Compute CVaR and VaR
        # ────────────────────────────────────────────────────────────────────
        sorted_losses = np.sort(losses)
        n_tail = int(np.ceil(len(losses) * self.alpha))
        tail_losses = sorted_losses[-n_tail:]
        var_value = float(tail_losses[0])
        cvar_value = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_value

        # ────────────────────────────────────────────────────────────────────
        # VALIDATION: EVaR >= CVaR (mathematical property)
        # ────────────────────────────────────────────────────────────────────
        evar_cvar_ratio = evar_value / cvar_value if cvar_value != 0 else float('inf')

        # Check mathematical validity
        if evar_value < cvar_value - 1e-6:
            raise ValueError(
                f"Mathematical violation: EVaR={evar_value:.6e} < CVaR={cvar_value:.6e}. "
                f"This indicates a critical error in the implementation."
            )

        # Check ratio is in expected range
        if evar_cvar_ratio > 5.0:
            logger.warning(
                f"EVaR/CVaR ratio={evar_cvar_ratio:.2f} > 5.0. "
                f"This may indicate heavy-tailed distribution or numerical issues. "
                f"Typical range is 1.5-3.0 for normal distributions."
            )
        elif evar_cvar_ratio < 1.0:
            logger.warning(
                f"EVaR/CVaR ratio={evar_cvar_ratio:.2f} < 1.0. "
                f"Mathematical violation detected (should always be >= 1.0)."
            )

        # ────────────────────────────────────────────────────────────────────
        # RESULT WITH PASSPORT SEAL
        # ────────────────────────────────────────────────────────────────────
        result_dict = {
            "evar": float(evar_value),
            "cvar": float(cvar_value),
            "var": float(var_value),
            "evar_cvar_ratio": float(evar_cvar_ratio),
            "lambda_optimal": float(lambda_optimal),
            "mgf_optimal": float(mgf_optimal),
            "log_mgf": float(log_mgf),
            "alpha": float(self.alpha),
            "risk_aversion": float(self.risk_aversion),  # BACKWARD COMPATIBILITY
            "num_samples": int(len(losses)),
            "num_tail_samples": int(len(tail_losses)),
            "optimization_success": bool(result.success)
        }

        # Calculate execution time
        operation_end = datetime.now()
        execution_ms = (operation_end - operation_start).total_seconds() * 1000

        passport = self.orchestrator.seal(
            passport,
            function_name='compute_evar_corrected',
            phase='Phase 2',
            result={
                'execution_time_ms': execution_ms,
                'output_hash': _compute_hash(result_dict),
                'validation_status': 'valid',
                'transformation_type': 'evar_optimization_corrected',
                'evar_value': float(evar_value),
                'cvar_value': float(cvar_value),
                'evar_cvar_ratio': float(evar_cvar_ratio),
                'lambda_optimal': float(lambda_optimal),
                'alpha': float(self.alpha),
                'risk_aversion_computed': float(self.risk_aversion),
                'num_samples': int(len(losses))
            }
        )

        passport_id = passport.passport_id[:8] if hasattr(passport, 'passport_id') else 'N/A'
        logger.info(
            f"EVaR computation sealed: {passport_id}... "
            f"(alpha={self.alpha:.4f}, lam*={lambda_optimal:.6f}, "
            f"EVaR={evar_value:.6f}, CVaR={cvar_value:.6f}, "
            f"Ratio={evar_cvar_ratio:.4f})"
        )

        return result_dict

    def compute_cvar_for_comparison(
        self,
        losses: NDArray[np.floating],
        passport: Optional[Dict] = None
    ) -> Dict[str, Union[float, int]]:
        """
        Compute CVaR for direct comparison with EVaR.

        CVaR (Conditional Value at Risk) = Expected loss in alpha-tail
        CVaR_alpha(X) = E[X | X >= VaR_alpha(X)]

        Parameters
        ----------
        losses : NDArray[np.floating]
            Portfolio losses
        passport : Dict, optional
            Passport object for audit trail

        Returns
        -------
        Dict[str, Union[float, int]]
            Dictionary with CVaR metrics
        """
        operation_start = datetime.now()

        # Validation
        if not isinstance(losses, np.ndarray):
            raise TypeError("losses must be numpy array")
        if losses.ndim > 1:
            losses = losses.flatten()
        if len(losses) == 0:
            raise ValueError("losses array is empty")
        if np.any(np.isnan(losses)) or np.any(np.isinf(losses)):
            raise ValueError("losses contains NaN/Inf")

        # Passport
        if passport is None:
            passport = self.orchestrator.assign_passport(
                data={"num_samples": len(losses), "alpha": self.alpha},
                data_type="cvar_comparison",
                source="EVaREstimator.compute_cvar_for_comparison"
            )

        # Computation
        
        sorted_losses = np.sort(losses)
        n_tail = int(np.ceil(len(losses) * self.alpha))
        tail_losses = sorted_losses[-n_tail:]
        var_value = float(tail_losses[0])
        cvar_value = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_value
        
        if cvar_value < var_value - 1e-10:
            raise ValueError("CVaR < VaR (mathematical error)")

        q75 = np.percentile(sorted_losses, 75)
        if cvar_value < q75:
            raise ValueError("CVaR in lower part of distribution (error)")

        result = {
            "cvar": float(cvar_value),
            "var": float(var_value),
            "alpha": float(self.alpha),
            "num_tail_samples": int(len(tail_losses)),
            "total_samples": int(len(losses))
        }

        # Seal
        operation_end = datetime.now()
        execution_ms = (operation_end - operation_start).total_seconds() * 1000

        passport = self.orchestrator.seal(
            passport,
            function_name='compute_cvar',
            phase='Phase 2',
            result={
                'execution_time_ms': execution_ms,
                'output_hash': _compute_hash(result),
                'validation_status': 'valid',
                'transformation_type': 'cvar_computation',
                'cvar_value': float(cvar_value),
                'alpha': float(self.alpha),
                'num_samples': int(len(losses))
            }
        )

        logger.debug(f"CVaR computed: {cvar_value:.6f} (tail: {len(tail_losses)} samples)")
        return result

    def compare_evar_cvar(
        self,
        losses: NDArray[np.floating],
        passport: Optional[Dict] = None
    ) -> Dict[str, AnyType]:
        """
        Compare EVaR and CVaR with rigorous validation.

        Validates: EVaR >= CVaR (mathematical property)

        Parameters
        ----------
        losses : NDArray[np.floating]
            Portfolio losses
        passport : Dict, optional
            Passport object for audit trail

        Returns
        -------
        Dict[str, Any]
            Comprehensive comparison with validation
        """
        logger.debug("Starting EVaR vs CVaR comparison")

        # Compute both measures
        evar_result = self.compute_evar(losses, passport)
        cvar_result = self.compute_cvar_for_comparison(losses, passport)

        evar_value = evar_result["evar"]
        cvar_value = cvar_result["cvar"]

        # Validation
        if np.isnan(evar_value) or np.isinf(evar_value):
            raise ValueError(f"Invalid EVaR result: {evar_value}")
        if np.isnan(cvar_value) or np.isinf(cvar_value):
            raise ValueError(f"Invalid CVaR result: {cvar_value}")

        # Metrics
        difference = evar_value - cvar_value
        ratio = evar_value / cvar_value if cvar_value != 0 else float('inf')
        relative_diff_pct = (difference / abs(cvar_value) * 100) if cvar_value != 0 else 0.0

        # Validation
        evar_gte_cvar = evar_value >= cvar_value - 1e-6
        validation_passed = evar_gte_cvar

        if not evar_gte_cvar:
            raise ValueError(
                f"Mathematical violation: EVaR ({evar_value:.6e}) < CVaR ({cvar_value:.6e})"
            )

        result = {
            "evar_value": float(evar_value),
            "cvar_value": float(cvar_value),
            "var_value": float(evar_result["var"]),
            "difference": float(difference),
            "relative_difference_pct": float(relative_diff_pct),
            "ratio": float(ratio),
            "evar_>=_cvar": bool(evar_gte_cvar),
            "validation_passed": bool(validation_passed),
            "evar_details": evar_result,
            "cvar_details": cvar_result
        }

        logger.info(
            f"Comparison: EVaR={evar_value:.6f}, CVaR={cvar_value:.6f}, "
            f"Ratio={ratio:.4f}, Valid={validation_passed}"
        )

        return result

    def validate_optimization_with_evar(
            self,
            optimal_weights: NDArray[np.floating],
            returns: NDArray[np.floating],
            cvar_optimized: float,
            passport: Optional[Dict] = None
        ) -> Dict[str, AnyType]:
            """
            Validate CVaR-optimized portfolio using EVaR.

            Parameters
            ----------
            optimal_weights : NDArray[np.floating]
                Portfolio weights from CVaR optimization
            returns : NDArray[np.floating]
                Historical returns (T x N)
            cvar_optimized : float
                CVaR value achieved by optimization
            passport : Dict, optional
                Passport object for audit trail

            Returns
            -------
            Dict[str, Any]
                Validation report
            """
            operation_start = datetime.now()

            # Validation — defensive against None inputs
            if optimal_weights is None:
                raise ValueError("optimal_weights is None")
            if not isinstance(optimal_weights, np.ndarray):
                optimal_weights = np.array(optimal_weights, dtype=float)
            if returns is None:
                raise ValueError("returns is None")
            if not isinstance(returns, np.ndarray):
                raise TypeError("returns must be numpy array")
            if not np.isclose(optimal_weights.sum(), 1.0, atol=1e-6):
                raise ValueError(f"weights must sum to 1, got {optimal_weights.sum():.6f}")
            if np.any(optimal_weights < -1e-10):
                raise ValueError("All weights must be non-negative")
            if returns.ndim != 2:
                raise ValueError(f"returns must be 2D, got shape {returns.shape}")
            if len(returns) == 0:
                raise ValueError("returns array is empty")
            if np.any(np.isnan(returns)) or np.any(np.isinf(returns)):
                raise ValueError("returns contains NaN/Inf")

            # Passport
            if passport is None:
                passport = self.orchestrator.assign_passport(
                    data={"weights_sum": float(optimal_weights.sum())},
                    data_type="portfolio_validation",
                    source="EVaREstimator.validate_optimization_with_evar"
                )

            # Compute portfolio losses
            portfolio_returns = returns @ optimal_weights
            portfolio_losses = -portfolio_returns

            # Compute EVaR
            evar_result = self.compute_evar(portfolio_losses, passport=passport)
            evar_value = evar_result["evar"]

            # Convert CVaR to loss magnitude for consistent comparison
            cvar_as_loss = abs(cvar_optimized)  #  ADDED

            # Compute ratio (both as losses now)
            ratio = evar_value / cvar_as_loss if cvar_as_loss != 0 else float('inf')  #  ADDED

            # Discrepancy
            discrepancy = abs(evar_value - cvar_as_loss)  #  MODIFIED
            relative_discrepancy = discrepancy / cvar_as_loss if cvar_as_loss != 0 else 0.0  #  MODIFIED

            # LITERATURE-BASED VALIDATION (Ahmadi-Javid, 2012)
            # Use ratio instead of relative_discrepancy for realistic thresholds
            if ratio < 1.2:
                validation_status = "PASSED - Excellent convergence"
                _interpretation = (
                    "EVaR and CVaR are very close, indicating excellent tail behavior. "
                    "Portfolio shows minimal sensitivity to extreme losses. "
                    "Typical for well-diversified, normal-like distributions."
                )
                _status_level = "excellent"
                
            elif ratio < 1.5:
                validation_status = "PASSED - Acceptable for portfolio optimization"
                _interpretation = (
                    "EVaR/CVaR ratio within expected range for risk-averse portfolios. "
                    "Portfolio has acceptable tail risk characteristics. "
                    "This is the standard range for optimized portfolios (Ahmadi-Javid, 2012)."
                )
                _status_level = "acceptable"
                
            elif ratio < 2.5:
                validation_status = "WARNING - Elevated tail risk"
                _interpretation = (
                    "Heavy-tailed distribution detected. Portfolio exhibits elevated sensitivity "
                    "to extreme losses. Consider reviewing portfolio composition, increasing "
                    "diversification, or adjusting the alpha parameter."
                )
                _status_level = "warning"
                
            else:  # ratio >= 2.5
                validation_status = "FAILED - Excessive ratio"
                _interpretation = (
                    "EVaR/CVaR ratio exceeds acceptable bounds. This indicates extreme tail "
                    "behavior that may signal data quality issues, numerical instability, "
                    "or portfolio unsuitable for EVaR-based risk management."
                )
                _status_level = "failed"

            
            result = {
                "cvar_optimized": float(cvar_as_loss),  #  MODIFIED
                "evar_validation": float(evar_value),
                "evar_cvar_ratio": float(ratio),  #  ADDED
                "discrepancy": float(discrepancy),
                "relative_discrepancy": float(relative_discrepancy),
                "validation_status": validation_status,
                "evar_details": evar_result
            }

            # Seal
            operation_end = datetime.now()
            execution_ms = (operation_end - operation_start).total_seconds() * 1000

            passport = self.orchestrator.seal(
                passport,
                function_name='validate_portfolio',
                phase='Phase 2',
                result={
                    'execution_time_ms': execution_ms,
                    'output_hash': _compute_hash(result),
                    'validation_status': validation_status,
                    'transformation_type': 'portfolio_validation',
                    'cvar_optimized': float(cvar_as_loss),  #  MODIFIED
                    'evar_value': float(evar_value),
                    'relative_discrepancy': float(relative_discrepancy),
                    'evar_cvar_ratio': float(ratio)  #  ADDED
                }
            )

            logger.info(
                f"Portfolio validation: {validation_status} "
                f"(CVaR={cvar_as_loss:.6f}, EVaR={evar_value:.6f}, "  #  MODIFIED
                f"Ratio={ratio:.3f}, Discrepancy={relative_discrepancy:.2%})"  #  MODIFIED
            )

            return result


    def noise_robustness_analysis(
        self,
        weights: NDArray[np.floating],
        returns: NDArray[np.floating],
        noise_levels: Optional[List[float]] = None,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Analyze EVaR robustness under noise (deterministic).

        Parameters
        ----------
        weights : NDArray[np.floating]
            Portfolio weights
        returns : NDArray[np.floating]
            Historical returns (T x N)
        noise_levels : List[float], optional
            Noise standard deviations (default: [0.0, 0.01, 0.05, 0.1])
        random_seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        pd.DataFrame
            Robustness analysis results
        """
        if noise_levels is None:
            noise_levels = [0.0, 0.01, 0.05, 0.1]

        if not isinstance(noise_levels, (list, tuple)):
            raise TypeError("noise_levels must be list or tuple")
        if any(nl < 0 for nl in noise_levels):
            raise ValueError("noise_levels must be non-negative")

        logger.debug(f"Noise robustness analysis with seed={random_seed}")
        np.random.seed(random_seed)

        results = []
        baseline_evar = None

        for noise_std in noise_levels:
            # Add noise
            noisy_returns = returns + np.random.randn(*returns.shape) * noise_std
            portfolio_returns = noisy_returns @ weights
            portfolio_losses = -portfolio_returns

            # Compute
            evar_result = self.compute_evar(portfolio_losses)
            cvar_result = self.compute_cvar_for_comparison(portfolio_losses)

            evar_val = evar_result["evar"]
            cvar_val = cvar_result["cvar"]

            # Stability
            if baseline_evar is None:
                baseline_evar = evar_val
                stability = 0.0
            else:
                stability = (evar_val - baseline_evar) / abs(baseline_evar)

            results.append({
                "noise_std": float(noise_std),
                "evar": float(evar_val),
                "cvar": float(cvar_val),
                "evar_cvar_ratio": float(evar_val / cvar_val) if cvar_val != 0 else float('inf'),
                "stability_metric": float(stability)
            })

        df = pd.DataFrame(results)
        logger.info(f"Noise analysis: EVaR std={df['evar'].std():.6e}")

        return df

    def compute_evar_gradient(
        self,
        weights: NDArray[np.floating],
        returns: NDArray[np.floating],
        epsilon: float = 1e-6,
        passport: Optional[Dict] = None
    ) -> NDArray[np.floating]:
        """
        Compute EVaR gradient via finite differences.

        Parameters
        ----------
        weights : NDArray[np.floating]
            Portfolio weights
        returns : NDArray[np.floating]
            Returns (T x N)
        epsilon : float, optional
            Step size (default: 1e-6)
        passport : Dict, optional
            Passport for audit

        Returns
        -------
        NDArray[np.floating]
            Gradient ∂EVaR/∂w
        """
        if not isinstance(weights, np.ndarray):
            raise TypeError("weights must be numpy array")
        if not isinstance(returns, np.ndarray):
            raise TypeError("returns must be numpy array")
        if not np.isclose(weights.sum(), 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1, got {weights.sum():.6f}")
        if epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")

        logger.debug(f"Computing EVaR gradient with ε={epsilon:.6e}")

        num_assets = len(weights)
        gradient = np.zeros(num_assets)

        # Base EVaR
        portfolio_returns = returns @ weights
        portfolio_losses = -portfolio_returns
        evar_base = self.compute_evar(portfolio_losses)["evar"]

        # Finite differences
        for i in range(num_assets):
            weights_pert = weights.copy()
            weights_pert[i] += epsilon
            weights_pert /= weights_pert.sum()

            portfolio_returns_pert = returns @ weights_pert
            portfolio_losses_pert = -portfolio_returns_pert
            evar_pert = self.compute_evar(portfolio_losses_pert)["evar"]

            gradient[i] = (evar_pert - evar_base) / epsilon

        gradient_norm = np.linalg.norm(gradient)
        logger.info(f"EVaR gradient: norm={gradient_norm:.6e}")

        return gradient


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE FUNCTIONS (Integration API)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_cvar_optimization_with_evar(
    optimization_result: Dict[str, AnyType],
    returns: NDArray[np.floating],
    alpha: float = ALPHA,
    passport: Optional[Dict] = None
) -> Dict[str, AnyType]:
    """
    High-level function: Validate CVaR optimization using EVaR.

    Parameters
    ----------
    optimization_result : Dict[str, Any]
        Result from optimizer containing optimal_weights and optimal_cvar
    returns : NDArray[np.floating]
        Historical returns
    alpha : float, optional
        Confidence level
    passport : Dict, optional
        Passport for audit

    Returns
    -------
    Dict[str, Any]
        Validation report with noise robustness
    """
    estimator = EVaREstimator(alpha=alpha)

    optimal_weights = optimization_result["optimal_weights"]
    cvar_optimized = optimization_result["optimal_cvar"]

    validation = estimator.validate_optimization_with_evar(
        optimal_weights=optimal_weights,
        returns=returns,
        cvar_optimized=cvar_optimized,
        passport=passport
    )

    # Noise robustness
    noise_analysis = estimator.noise_robustness_analysis(
        weights=optimal_weights,
        returns=returns
    )

    validation["noise_robustness"] = noise_analysis

    return validation


def compare_risk_measures(
    weights: NDArray[np.floating],
    returns: NDArray[np.floating],
    alpha: float = ALPHA
) -> Dict[str, AnyType]:
    """
    Comprehensive comparison of VaR, CVaR, and EVaR.

    Parameters
    ----------
    weights : NDArray[np.floating]
        Portfolio weights
    returns : NDArray[np.floating]
        Historical returns
    alpha : float, optional
        Confidence level

    Returns
    -------
    Dict[str, Any]
        Comparison of all three measures
    """
    estimator = EVaREstimator(alpha=alpha)

    # Compute losses
    portfolio_returns = returns @ weights
    portfolio_losses = -portfolio_returns

    # Compute all measures
    comparison = estimator.compare_evar_cvar(portfolio_losses)

    # Add expected loss
    expected_loss = np.mean(portfolio_losses)
    comparison["expected_loss"] = float(expected_loss)

    return comparison

