#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Module: hybridoptimizer.py
 HYBRID QUANTUM-CLASSICAL CVaR PORTFOLIO OPTIMIZER
================================================================================

 Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
           Master's Thesis (TFM) - MSc in Quantum Computing
 Author  : Ignacio Lopez Leis
 Affil.  : Universidad Autonoma de Madrid (UAM)
 Date    : February 2026
 Version : 6.4

 Description
 -----------
 Implements the outer hybrid quantum-classical optimization loop (Phase 2)
 for CVaR-based portfolio optimization. Uses quantum subgradient estimates
 (from quantumsubgradient.py) within an Adam-based subgradient descent
 scheme. Supports softmax reparametrization to avoid simplex projection
 fixed points, and dual convergence criteria: gradient-norm and relative
 CVaR change over a sliding window.

 The simplex projection has spurious fixed points when the subgradient is
 approximately uniform. With equal weights w = [1/N, ..., 1/N] and uniform
 subgradient g = [c, ..., c], the update w' = w - alpha*g followed by
 normalization returns w'' = w (fixed point). The softmax reparametrization
 solves this: w_i = exp(theta_i) / sum_j exp(theta_j), where the Jacobian
 is non-singular, so uniform dCVaR/dw does NOT imply zero dCVaR/dtheta.

 References
 ----------
 [1] Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional
     Value-at-Risk." Journal of Risk, 2(3), 21-41.
 [2] Woerner, S. & Egger, D.J. (2019). "Quantum Risk Analysis."
     npj Quantum Information, 5, 15.
 [3] Brassard, G. et al. (2002). "Quantum Amplitude Amplification and
     Estimation." Contemporary Mathematics, 305, 53-74.
 [4] Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance,
     7(1), 77-91.

 License
 -------
 Academic use only. This code is part of a Master's Thesis and is provided
 for academic and research purposes. For any non-academic or commercial use,
 please contact: nacholopezleis@gmail.com

================================================================================
"""

import sys
import numpy as np
import pandas as pd
import time
from typing import Optional, Dict, List, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# IMPORTS - QUANTUM MODULES
# ==============================================================================

QUANTUM_AVAILABLE = False
QAE_AVAILABLE = False

try:
    from src.phase_2.quantumsubgradient import (
        QuantumSubgradientEstimator,
        SubgradientResult,
        QAE_AVAILABLE as SUBGRAD_QAE_AVAILABLE
    )
    QUANTUM_AVAILABLE = True
except ImportError:
    try:
        from phase_2.quantumsubgradient import (
            QuantumSubgradientEstimator,
            SubgradientResult,
            QAE_AVAILABLE as SUBGRAD_QAE_AVAILABLE
        )
        QUANTUM_AVAILABLE = True
    except ImportError:
        SUBGRAD_QAE_AVAILABLE = False

try:
    from src.phase_2.qae_circuits import QuantumAmplitudeEstimator
    QAE_AVAILABLE = True
except ImportError:
    try:
        from phase_2.qae_circuits import QuantumAmplitudeEstimator
        QAE_AVAILABLE = True
    except ImportError:
        pass


# ==============================================================================
# LOGGER
# ==============================================================================

class FileLogger:
    """File-based logger for the hybrid optimizer. Truncates the log file
    on initialization (overwrite-mode contract from CLAUDE.md)."""

    def __init__(self, log_dir: Path = None):
        self.log_dir = Path(log_dir or './logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'hybrid_optimizer.txt'
        self._write_header()

    def _write_header(self):
        """Truncate the log and write the session header."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"HYBRID OPTIMIZER - {datetime.now()}\n\n")

    def reset_log(self):
        """Truncate and rewrite header. Called once at start of optimize()."""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"HYBRID OPTIMIZER - {datetime.now()}\n\n")

    def _write(self, level: str, msg: str):
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {level}: {msg}\n")
        except Exception as exc:
            sys.stderr.write(f"[hybrid_optimizer LOGGER_ERROR] {exc}\n")

    def info(self, msg: str): self._write("INFO", msg)
    def warning(self, msg: str): self._write("WARN", msg)
    def error(self, msg: str): self._write("ERROR", msg)
    def debug(self, msg: str): self._write("DEBUG", msg)


_logger = None

def get_logger() -> FileLogger:
    """Return module-level singleton logger."""
    global _logger
    if _logger is None:
        _logger = FileLogger()
    return _logger

logger = get_logger()


# ==============================================================================
# SOFTMAX REPARAMETRIZATION
# ==============================================================================

class SoftmaxParameterization:
    """
    Softmax reparametrization for portfolio weight optimization.

    Transforms the constrained optimization on the simplex to an
    unconstrained problem in R^n:

        w_i = softmax(theta)_i = exp(theta_i / tau) / sum_j exp(theta_j / tau)

    Properties:
        - Bijective mapping between R^n and interior of the simplex.
        - Differentiable everywhere with non-singular Jacobian.
        - No spurious fixed points from simplex projection.
        - Temperature tau controls exploration vs exploitation.

    Parameters
    ----------
    n_assets : int
        Number of portfolio assets.
    temperature : float
        Softmax temperature (higher = more uniform, lower = more concentrated).
    """

    def __init__(self, n_assets: int, temperature: float = 1.0):
        self.n_assets = n_assets
        self.temperature = temperature
        logger.info(f"SoftmaxParameterization: n_assets={n_assets}, temperature={temperature}")

    def theta_to_weights(self, theta: np.ndarray) -> np.ndarray:
        """Convert unconstrained parameters theta to portfolio weights w."""
        scaled = theta / self.temperature
        scaled_stable = scaled - np.max(scaled)  # log-sum-exp trick
        exp_theta = np.exp(scaled_stable)
        return exp_theta / np.sum(exp_theta)

    def weights_to_theta(self, weights: np.ndarray) -> np.ndarray:
        """Convert portfolio weights w to unconstrained parameters theta."""
        weights_safe = np.clip(weights, 1e-10, 1.0)
        weights_safe = weights_safe / np.sum(weights_safe)
        return self.temperature * np.log(weights_safe)

    def transform_gradient(self, grad_w: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Transform gradient from weight space to theta space via chain rule.

        dL/dtheta_i = sum_j (dL/dw_j) * (dw_j/dtheta_i)

        The Jacobian of softmax is:
            dw_j/dtheta_i = (1/tau) * w_j * (delta_ij - w_i)

        Efficient vectorized form:
            dL/dtheta = (1/tau) * (w * grad_w - w * (w . grad_w))
        """
        w_dot_g = np.dot(weights, grad_w)
        return (weights * grad_w - weights * w_dot_g) / self.temperature

    def get_initial_theta(self, initial_weights: np.ndarray = None) -> np.ndarray:
        """Get initial theta (zeros -> uniform weights if None)."""
        if initial_weights is None:
            return np.zeros(self.n_assets)
        return self.weights_to_theta(initial_weights)


# ==============================================================================
# ENUMS AND DATA CLASSES
# ==============================================================================

class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"
    SGDM = "sgdm"


@dataclass
class AdamState:
    """State for Adam optimizer."""
    m: np.ndarray
    v: np.ndarray
    t: int = 0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8


@dataclass
class IterationMetrics:
    """Metrics collected at each optimization iteration."""
    iteration: int
    cvar: float
    var: float
    gradient_norm: float
    weight_change_norm: float
    learning_rate: float
    queries_iqae: int = 0
    queries_woerner: int = 0
    quantum_time_ms: float = 0.0
    is_quantum: bool = False
    theta_gradient_norm: float = 0.0


@dataclass
class OptimizationResult:
    """Complete result from the hybrid optimization run."""
    optimal_weights: np.ndarray
    optimal_cvar: float
    optimal_var: float
    initial_cvar: float
    initial_var: float
    cvar_improvement_pct: float
    converged: bool
    num_iterations: int
    early_stopped: bool
    early_stop_iteration: Optional[int]
    final_gradient_norm: float
    total_queries_iqae: int
    total_queries_woerner: int
    quantum_success_rate: float
    avg_subgradient_error: float
    total_time_seconds: float
    total_quantum_time_ms: float
    avg_iteration_time_ms: float
    cvar_history: List[float]
    iteration_history: List[IterationMetrics]
    node_id: Optional[str] = None
    is_virtual_assets: bool = False
    asset_labels: Optional[List[str]] = None
    used_softmax: bool = False
    final_theta: Optional[np.ndarray] = None
    convergence_reason: Optional[str] = None

    def summary(self) -> str:
        """Return human-readable optimization summary."""
        softmax_str = " [SOFTMAX]" if self.used_softmax else ""
        conv_str = f" ({self.convergence_reason})" if self.convergence_reason else ""
        lines = [
            "=" * 70,
            f"HYBRID QUANTUM-CLASSICAL OPTIMIZATION RESULT{softmax_str}",
            "=" * 70,
            f"CVaR: {self.initial_cvar:.6f} -> {self.optimal_cvar:.6f} ({self.cvar_improvement_pct:+.2f}%)",
            f"VaR:  {self.initial_var:.6f} -> {self.optimal_var:.6f}",
            f"Iterations: {self.num_iterations} (converged={self.converged}{conv_str})",
            f"Total IQAE queries: {self.total_queries_iqae}",
            f"Total Woerner queries: {self.total_queries_woerner}",
            f"Quantum success rate: {self.quantum_success_rate:.2%}",
            f"Total time: {self.total_time_seconds:.2f}s",
            "=" * 70
        ]
        return "\n".join(lines)


# ==============================================================================
# CLASSICAL SUBGRADIENT (FALLBACK)
# ==============================================================================

def classical_cvar_subgradient(
    weights: np.ndarray,
    returns: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, Dict]:
    """
    Compute CVaR subgradient using classical Monte Carlo.

    g_j = -E[r_j | L(w) >= VaR_alpha(w)]

    This is the conditional expectation of the negative return given
    that the portfolio loss exceeds VaR.

    Reference: Rockafellar & Uryasev (2000), Proposition 2.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (N,).
    returns : np.ndarray
        Returns matrix (T, N).
    alpha : float
        CVaR confidence level.

    Returns
    -------
    subgradient : np.ndarray
        CVaR subgradient (N,).
    metrics : dict
        Auxiliary metrics (tail_prob, var, etc.).
    """
    weights = np.asarray(weights).flatten()
    returns = np.asarray(returns)

    portfolio_losses = -(returns @ weights)
    var = np.percentile(portfolio_losses, (1 - alpha) * 100)
    tail_mask = portfolio_losses >= var
    tail_count = np.sum(tail_mask)

    subgradient = np.zeros(len(weights))
    if tail_count > 0:
        subgradient = -np.mean(returns[tail_mask], axis=0)

    return subgradient, {
        'tail_prob': float(np.mean(tail_mask)),
        'tail_count': int(tail_count),
        'var': float(var),
        'is_classical': True,
        'queries_iqae': 0,
        'queries_woerner': 0,
        'quantum_time_ms': 0.0
    }


# ==============================================================================
# MAIN CLASS: HYBRID OPTIMIZER
# ==============================================================================

class HybridOptimizer:
    """
    Hybrid Quantum-Classical CVaR Portfolio Optimizer.

    Uses QuantumSubgradientEstimator for quantum subgradient computation
    with a classical Adam optimizer for weight updates. Supports softmax
    reparametrization and dual convergence criteria.

    Parameters
    ----------
    n_assets : int
        Number of portfolio assets.
    alpha : float
        CVaR confidence level (e.g. 0.05 for 95% CVaR).
    max_iterations : int
        Maximum optimization iterations.
    learning_rate : float
        Initial learning rate for Adam.
    learning_rate_decay : float
        Multiplicative learning rate decay per iteration.
    convergence_threshold : float
        Gradient norm threshold for convergence.
    patience : int
        Early stopping patience (iterations without improvement).
    use_softmax : bool
        If True, use softmax reparametrization instead of simplex projection.
    rel_convergence_threshold : float
        Threshold for relative CVaR convergence criterion.
    rel_convergence_window : int
        Sliding window size for relative CVaR convergence check.
    """

    def __init__(
        self,
        n_assets: int,
        alpha: float = 0.05,
        max_iterations: int = 100,
        learning_rate: float = 0.01,
        learning_rate_decay: float = 0.995,
        convergence_threshold: float = 1e-6,
        patience: int = 20,
        optimizer_type: str = "adam",
        # QAE parameters
        num_distribution_qubits: int = 8,
        epsilon: float = 0.01,
        use_iqae: bool = True,
        use_woerner: bool = True,
        shots: int = 1024,
        # Bounds
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        # Backend
        backend: str = "aer_simulator",
        passport: Any = None,
        # ZNE
        use_zne: bool = False,
        zne_scale_factors: Optional[List[float]] = None,
        # Circuit export
        export_circuits: bool = False,
        circuit_export_dir: Optional[str] = None,
        # Recursive tree support
        is_virtual_assets: bool = False,
        asset_labels: Optional[List[str]] = None,
        node_id: Optional[str] = None,
        # Softmax reparametrization
        use_softmax: bool = True,
        softmax_temperature: float = 1.0,
        # Relative convergence criterion
        rel_convergence_threshold: float = 1e-3,
        rel_convergence_window: int = 5,
        # Noise model for optimizer's internal QAE (E-group experiments)
        noise_model_config: Optional[Dict] = None
    ):
        self.n_assets = n_assets
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.convergence_threshold = convergence_threshold
        self.patience = patience
        self.optimizer_type = OptimizerType(optimizer_type.lower())
        self.weight_bounds = weight_bounds

        # Relative convergence
        self.rel_convergence_threshold = rel_convergence_threshold
        self.rel_convergence_window = rel_convergence_window

        # QAE configuration
        self.num_distribution_qubits = num_distribution_qubits
        self.epsilon = epsilon
        self.use_iqae = use_iqae
        self.use_woerner = use_woerner
        self.shots = shots
        self.use_zne = use_zne
        self.zne_scale_factors = zne_scale_factors

        # Recursive tree metadata
        self.is_virtual_assets = is_virtual_assets
        self.asset_labels = asset_labels or [f"A{i}" for i in range(n_assets)]
        self.node_id = node_id

        # Softmax reparametrization
        self.use_softmax = use_softmax
        self.softmax_temperature = softmax_temperature
        self.softmax_param = None
        if use_softmax:
            self.softmax_param = SoftmaxParameterization(n_assets, softmax_temperature)

        # Initialize quantum subgradient estimator
        self.subgradient_estimator = None
        self._use_classical_fallback = not QUANTUM_AVAILABLE

        if QUANTUM_AVAILABLE and (use_iqae or use_woerner):
            try:
                self.subgradient_estimator = QuantumSubgradientEstimator(
                    num_distribution_qubits=num_distribution_qubits,
                    epsilon=epsilon,
                    alpha=alpha,
                    use_iqae=use_iqae,
                    use_woerner=use_woerner,
                    backend=backend,
                    passport=passport,
                    use_zne=use_zne,
                    zne_scale_factors=zne_scale_factors,
                    export_circuits=export_circuits,
                    circuit_export_dir=circuit_export_dir,
                    shots=shots,
                    noise_model_config=noise_model_config
                )
                self._use_classical_fallback = False
                logger.info(f"Quantum subgradient estimator initialized: "
                            f"IQAE={use_iqae}, Woerner={use_woerner}")
            except Exception as e:
                import traceback
                logger.error(f"Quantum init failed: {e}\n{traceback.format_exc()}")
                self._use_classical_fallback = True

        self._optimizer_state = None
        self._reset_tracking()

        node_str = f" [{node_id}]" if node_id else ""
        softmax_str = " [SOFTMAX]" if use_softmax else ""
        logger.info(
            f"HybridOptimizer{node_str}{softmax_str}: {n_assets} assets, "
            f"quantum={not self._use_classical_fallback}, "
            f"rel_conv_threshold={self.rel_convergence_threshold}, "
            f"rel_conv_window={self.rel_convergence_window}"
        )

    # --------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------

    def _reset_tracking(self):
        """Reset per-run tracking variables."""
        self._iteration_history: List[IterationMetrics] = []
        self._cvar_history: List[float] = []
        self._weight_history: List[np.ndarray] = []
        self._total_queries_iqae = 0
        self._total_queries_woerner = 0
        self._quantum_successes = 0
        self._quantum_attempts = 0
        self._total_quantum_time_ms = 0.0
        self._subgradient_errors: List[float] = []

    def _init_optimizer_state(self, n: int):
        """Initialize Adam optimizer state."""
        self._optimizer_state = AdamState(m=np.zeros(n), v=np.zeros(n))

    def _compute_adam_update(self, gradient: np.ndarray, lr: float) -> np.ndarray:
        """Compute Adam update step."""
        s = self._optimizer_state
        s.t += 1
        s.m = s.beta1 * s.m + (1 - s.beta1) * gradient
        s.v = s.beta2 * s.v + (1 - s.beta2) * (gradient ** 2)
        m_hat = s.m / (1 - s.beta1 ** s.t)
        v_hat = s.v / (1 - s.beta2 ** s.t)
        return lr * m_hat / (np.sqrt(v_hat) + s.epsilon)

    def _project_weights(self, weights: np.ndarray) -> np.ndarray:
        """Project weights to simplex (used when softmax is disabled)."""
        w = np.clip(weights, self.weight_bounds[0], self.weight_bounds[1])
        w = np.maximum(w, 0)
        w_sum = np.sum(w)
        return w / w_sum if w_sum > 0 else np.ones_like(w) / len(w)

    def _compute_subgradient(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        classical_var: float,
        classical_cvar: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute CVaR subgradient using quantum or classical method.

        Returns gradient in WEIGHT space (dCVaR/dw).
        """
        if self._use_classical_fallback or self.subgradient_estimator is None:
            return classical_cvar_subgradient(weights, returns, self.alpha)

        self._quantum_attempts += 1

        try:
            result: SubgradientResult = self.subgradient_estimator.estimate_subgradient(
                weights=weights,
                returns=returns,
                classical_var=classical_var,
                classical_cvar=classical_cvar
            )
            self._quantum_successes += 1
            self._subgradient_errors.append(result.subgradient_l2_error)

            metrics = {
                'tail_prob_iqae': result.tail_probability_iqae,
                'tail_prob_woerner': result.tail_probability_woerner,
                'queries_iqae': result.total_oracle_queries_iqae,
                'queries_woerner': result.total_oracle_queries_woerner,
                'quantum_time_ms': result.execution_time_ms,
                'iqae_converged': result.iqae_converged,
                'woerner_converged': result.woerner_converged,
                'subgradient_error': result.subgradient_l2_error,
                'cosine_similarity': result.subgradient_cosine_similarity,
                'is_classical': False
            }

            logger.debug(
                f"Quantum subgradient: queries_iqae={metrics['queries_iqae']}, "
                f"queries_woerner={metrics['queries_woerner']}, "
                f"error={metrics['subgradient_error']:.6f}"
            )
            return result.subgradient, metrics

        except Exception as e:
            logger.warning(f"Quantum subgradient failed: {e}")
            return classical_cvar_subgradient(weights, returns, self.alpha)

    # --------------------------------------------------------------------------
    # Main optimization loop
    # --------------------------------------------------------------------------

    def optimize(
        self,
        returns: Union[np.ndarray, pd.DataFrame],
        compute_var_func: Callable[[np.ndarray, np.ndarray, float], float],
        compute_cvar_func: Callable[[np.ndarray, np.ndarray, float], float],
        initial_weights: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio to minimize CVaR.

        Parameters
        ----------
        returns : np.ndarray or DataFrame
            Returns matrix (T, N).
        compute_var_func : Callable
            Function(weights, returns, alpha) -> VaR in loss convention.
        compute_cvar_func : Callable
            Function(weights, returns, alpha) -> CVaR in loss convention.
        initial_weights : np.ndarray, optional
            Starting weights (default: equal weights).
        constraints : dict, optional
            Reserved for future constraint enforcement.
        progress_callback : Callable, optional
            Callback(iteration, cvar, weights) for progress updates.

        Returns
        -------
        OptimizationResult
            Complete optimization results.
        """
        start_time = time.time()
        logger.reset_log()

        if isinstance(returns, pd.DataFrame):
            returns = returns.values
        returns = np.asarray(returns)
        T, N = returns.shape

        if N != self.n_assets:
            self.n_assets = N
            if self.use_softmax:
                self.softmax_param = SoftmaxParameterization(N, self.softmax_temperature)

        node_str = f" [{self.node_id}]" if self.node_id else ""
        softmax_str = " [SOFTMAX]" if self.use_softmax else ""
        logger.info(f"Starting optimization{node_str}{softmax_str}: ({T}, {N})")

        # Initialize weights / theta
        current_theta = None
        if self.use_softmax:
            if initial_weights is not None:
                current_theta = self.softmax_param.weights_to_theta(
                    self._project_weights(np.asarray(initial_weights).flatten())
                )
            else:
                current_theta = np.zeros(N)
            current_weights = self.softmax_param.theta_to_weights(current_theta)
            logger.info(f"Softmax initialized: theta_norm={np.linalg.norm(current_theta):.6f}")
        else:
            if initial_weights is not None:
                current_weights = self._project_weights(np.asarray(initial_weights).flatten())
            else:
                current_weights = np.ones(N) / N
            current_theta = None

        self._init_optimizer_state(N)
        self._reset_tracking()

        # Initial metrics
        initial_var = compute_var_func(current_weights, returns, self.alpha)
        initial_cvar = compute_cvar_func(current_weights, returns, self.alpha)
        logger.info(f"Initial: VaR={initial_var:.6f}, CVaR={initial_cvar:.6f}")

        best_weights = current_weights.copy()
        best_theta = current_theta.copy() if current_theta is not None else None
        best_cvar = initial_cvar
        best_var = initial_var

        patience_counter = 0
        converged = False
        early_stopped = False
        early_stop_iteration = None
        last_gradient_norm = 0.0
        current_lr = self.learning_rate
        convergence_reason = None

        self._cvar_history.append(initial_cvar)
        self._weight_history.append(current_weights.copy())

        # ======================================================================
        # Main optimization loop
        # ======================================================================
        for iteration in range(self.max_iterations):
            iter_start = time.time()

            current_var = compute_var_func(current_weights, returns, self.alpha)
            current_cvar = compute_cvar_func(current_weights, returns, self.alpha)

            # Compute subgradient in weight space
            subgradient_w, quantum_metrics = self._compute_subgradient(
                weights=current_weights,
                returns=returns,
                classical_var=current_var,
                classical_cvar=current_cvar
            )

            gradient_norm_w = np.linalg.norm(subgradient_w)
            last_gradient_norm = gradient_norm_w

            # Track quantum metrics
            if not quantum_metrics.get('is_classical', True):
                self._total_queries_iqae += quantum_metrics.get('queries_iqae', 0)
                self._total_queries_woerner += quantum_metrics.get('queries_woerner', 0)
                self._total_quantum_time_ms += quantum_metrics.get('quantum_time_ms', 0.0)

            # Update weights via softmax or simplex projection
            gradient_norm_theta = 0.0
            if self.use_softmax:
                subgradient_theta = self.softmax_param.transform_gradient(
                    subgradient_w, current_weights
                )
                gradient_norm_theta = np.linalg.norm(subgradient_theta)

                update_theta = self._compute_adam_update(subgradient_theta, current_lr)
                new_theta = current_theta - update_theta
                new_weights = self.softmax_param.theta_to_weights(new_theta)

                # Enforce upper weight bound if needed
                w_min, w_max = self.weight_bounds
                if w_max < 1.0:
                    new_weights = np.minimum(new_weights, w_max)
                    w_sum = np.sum(new_weights)
                    if w_sum > 0:
                        new_weights = new_weights / w_sum
                    new_theta = self.softmax_param.weights_to_theta(new_weights)

                weight_change = np.linalg.norm(new_weights - current_weights)
                current_theta = new_theta
                current_weights = new_weights
            else:
                update = self._compute_adam_update(subgradient_w, current_lr)
                new_weights = current_weights - update
                new_weights = self._project_weights(new_weights)
                weight_change = np.linalg.norm(new_weights - current_weights)
                current_weights = new_weights

            # Decay learning rate
            current_lr *= self.learning_rate_decay

            # Record iteration metrics
            iter_metrics = IterationMetrics(
                iteration=iteration,
                cvar=current_cvar,
                var=current_var,
                gradient_norm=gradient_norm_w,
                weight_change_norm=weight_change,
                learning_rate=current_lr,
                queries_iqae=quantum_metrics.get('queries_iqae', 0),
                queries_woerner=quantum_metrics.get('queries_woerner', 0),
                quantum_time_ms=quantum_metrics.get('quantum_time_ms', 0.0),
                is_quantum=not quantum_metrics.get('is_classical', True),
                theta_gradient_norm=gradient_norm_theta
            )
            self._iteration_history.append(iter_metrics)
            self._cvar_history.append(current_cvar)
            self._weight_history.append(current_weights.copy())

            if progress_callback:
                progress_callback(iteration, current_cvar, current_weights)

            # Check for improvement
            if current_cvar < best_cvar:
                best_cvar = current_cvar
                best_var = current_var
                best_weights = current_weights.copy()
                best_theta = current_theta.copy() if current_theta is not None else None
                patience_counter = 0
            else:
                patience_counter += 1

            # Periodic logging
            if iteration % 5 == 0 or iteration == self.max_iterations - 1:
                logger.info(
                    f"Iter {iteration}: CVaR={current_cvar:.6f}, "
                    f"grad_norm={gradient_norm_w:.6f}, "
                    f"queries_iqae={self._total_queries_iqae}, "
                    f"queries_woerner={self._total_queries_woerner}"
                )

            # ==================================================================
            # Convergence check (dual criterion)
            # ==================================================================

            # Criterion 1: Gradient norm (works for smooth problems)
            convergence_metric = gradient_norm_theta if self.use_softmax else gradient_norm_w
            if convergence_metric < self.convergence_threshold:
                logger.info(
                    f"Converged at iteration {iteration} "
                    f"(gradient norm {convergence_metric:.2e} < "
                    f"{self.convergence_threshold:.2e})"
                )
                converged = True
                convergence_reason = "gradient_norm"
                break

            # Criterion 2: Relative CVaR improvement over sliding window.
            # Correct criterion for subgradient methods where the gradient
            # norm does NOT vanish at the optimum.
            W = self.rel_convergence_window
            if len(self._cvar_history) > W and current_cvar > 0:
                recent = self._cvar_history[-W:]
                max_rel_change = max(
                    abs(recent[i] - recent[i - 1]) / abs(recent[i - 1])
                    for i in range(1, len(recent))
                    if abs(recent[i - 1]) > 1e-15
                )
                if max_rel_change < self.rel_convergence_threshold:
                    logger.info(
                        f"Converged at iteration {iteration} "
                        f"(relative CVaR change {max_rel_change:.2e} < "
                        f"{self.rel_convergence_threshold:.2e} "
                        f"over last {W} iterations)"
                    )
                    converged = True
                    convergence_reason = "relative_cvar"
                    break

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at iteration {iteration}")
                early_stopped = True
                early_stop_iteration = iteration
                break

        # ======================================================================
        # Build result
        # ======================================================================
        total_time = time.time() - start_time
        num_iterations = len(self._iteration_history)
        improvement_pct = (
            (initial_cvar - best_cvar) / initial_cvar * 100
            if initial_cvar > 0 else 0
        )
        quantum_success_rate = self._quantum_successes / max(1, self._quantum_attempts)
        avg_subgradient_error = (
            float(np.mean(self._subgradient_errors))
            if self._subgradient_errors else 0.0
        )
        avg_iteration_time_ms = (total_time * 1000) / max(1, num_iterations)

        logger.info(
            f"Optimization complete: CVaR {initial_cvar:.6f} -> "
            f"{best_cvar:.6f} ({improvement_pct:+.2f}%)"
        )
        logger.info(
            f"Total queries: IQAE={self._total_queries_iqae}, "
            f"Woerner={self._total_queries_woerner}"
        )

        return OptimizationResult(
            optimal_weights=best_weights,
            optimal_cvar=best_cvar,
            optimal_var=best_var,
            initial_cvar=initial_cvar,
            initial_var=initial_var,
            cvar_improvement_pct=improvement_pct,
            converged=converged,
            num_iterations=num_iterations,
            early_stopped=early_stopped,
            early_stop_iteration=early_stop_iteration,
            final_gradient_norm=last_gradient_norm,
            total_queries_iqae=self._total_queries_iqae,
            total_queries_woerner=self._total_queries_woerner,
            quantum_success_rate=quantum_success_rate,
            avg_subgradient_error=avg_subgradient_error,
            total_time_seconds=total_time,
            total_quantum_time_ms=self._total_quantum_time_ms,
            avg_iteration_time_ms=avg_iteration_time_ms,
            cvar_history=self._cvar_history,
            iteration_history=self._iteration_history,
            node_id=self.node_id,
            is_virtual_assets=self.is_virtual_assets,
            asset_labels=self.asset_labels,
            used_softmax=self.use_softmax,
            final_theta=best_theta,
            convergence_reason=convergence_reason
        )


# ==============================================================================
# FACTORY FUNCTION FOR RECURSIVE OPTIMIZER
# ==============================================================================

def create_optimizer_for_node(
    n_assets: int,
    node_id: str,
    asset_labels: List[str],
    alpha: float = 0.05,
    num_qubits: int = 4,
    epsilon: float = 0.05,
    max_iterations: int = 100,
    learning_rate: float = 0.01,
    learning_rate_decay: float = 0.995,
    convergence_threshold: float = 1e-6,
    patience: int = 20,
    use_quantum: bool = True,
    use_woerner: bool = False,
    use_softmax: bool = True,
    softmax_temperature: float = 1.0,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    shots: int = 8192,
    use_zne: bool = False,
    zne_scale_factors: Optional[List[float]] = None,
    export_circuits: bool = False,
    circuit_export_dir: str = 'results/circuits',
    passport: object = None,
    rel_convergence_threshold: float = 1e-3,
    rel_convergence_window: int = 5
) -> HybridOptimizer:
    """Factory function to create optimizer for a recursive tree node."""
    return HybridOptimizer(
        n_assets=n_assets,
        alpha=alpha,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        convergence_threshold=convergence_threshold,
        patience=patience,
        optimizer_type="adam",
        num_distribution_qubits=num_qubits,
        epsilon=epsilon,
        use_iqae=use_quantum,
        use_woerner=use_woerner,
        shots=shots,
        weight_bounds=weight_bounds,
        backend="aer_simulator",
        passport=passport,
        use_zne=use_zne,
        zne_scale_factors=zne_scale_factors,
        export_circuits=export_circuits,
        circuit_export_dir=circuit_export_dir,
        is_virtual_assets=True,
        asset_labels=asset_labels,
        node_id=node_id,
        use_softmax=use_softmax,
        softmax_temperature=softmax_temperature,
        rel_convergence_threshold=rel_convergence_threshold,
        rel_convergence_window=rel_convergence_window
    )


# ==============================================================================
# CONVENIENCE FUNCTION
# ==============================================================================

def optimize_portfolio(
    returns: np.ndarray,
    alpha: float = 0.05,
    max_iterations: int = 100,
    use_quantum: bool = True,
    use_softmax: bool = True,
    **kwargs
) -> OptimizationResult:
    """High-level convenience function for portfolio optimization."""
    T, N = returns.shape

    optimizer = HybridOptimizer(
        n_assets=N,
        alpha=alpha,
        max_iterations=max_iterations,
        use_iqae=use_quantum,
        use_woerner=use_quantum,
        use_softmax=use_softmax,
        **kwargs
    )

    def compute_var(w, r, a):
        losses = -(r @ w)
        return float(np.percentile(losses, (1 - a) * 100))

    def compute_cvar(w, r, a):
        losses = -(r @ w)
        var = np.percentile(losses, (1 - a) * 100)
        tail = losses[losses >= var]
        return float(np.mean(tail)) if len(tail) > 0 else float(var)

    return optimizer.optimize(
        returns=returns,
        compute_var_func=compute_var,
        compute_cvar_func=compute_cvar
    )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    'HybridOptimizer',
    'OptimizationResult',
    'IterationMetrics',
    'SoftmaxParameterization',
    'create_optimizer_for_node',
    'optimize_portfolio',
    'classical_cvar_subgradient',
    'QUANTUM_AVAILABLE',
]
