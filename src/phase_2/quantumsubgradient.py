#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Module: quantumsubgradient.py
 QUANTUM SUBGRADIENT ESTIMATION FOR CVaR OPTIMIZATION
================================================================================

 Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
           Master's Thesis (TFM) - MSc in Quantum Computing
 Author  : Ignacio Lopez Leis
 Affil.  : Universidad Autonoma de Madrid (UAM)
 Date    : February 2026
 Version : 5.2

 Description
 -----------
 Implements quantum subgradient estimation for the CVaR objective within
 the hybrid optimization loop (Phase 2). The CVaR subgradient for each
 asset j is computed as:

     g_j = -E[r_j | L >= VaR]

 where:
   - P(L >= VaR)      is estimated via QAE with a threshold oracle.
   - E[r_j | L >= VaR] is estimated via QAE with conditional amplitude encoding.

 The estimate_conditional_expectation() method in qae_circuits.py returns
 the already-conditioned E[r_j | tail], so no division by P(tail) is
 needed in the subgradient formula (Rockafellar & Uryasev, 2000, Prop. 2).

 Provides full backward-compatible API for integration with the hybrid
 optimizer and the main notebook pipeline.

 References
 ----------
 [1] Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional
     Value-at-Risk." Journal of Risk, 2(3), 21-41.
 [2] Brassard, G. et al. (2002). "Quantum Amplitude Amplification and
     Estimation." Contemporary Mathematics, 305, 53-74.
 [3] Woerner, S. & Egger, D.J. (2019). "Quantum Risk Analysis."
     npj Quantum Information, 5, 15.

 License
 -------
 Academic use only. This code is part of a Master's Thesis and is provided
 for academic and research purposes. For any non-academic or commercial use,
 please contact: nacholopezleis@gmail.com

================================================================================
"""

import sys
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
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


warnings.filterwarnings('ignore')


# ==============================================================================
# IMPORTS
# ==============================================================================

QAE_AVAILABLE = False
QISKIT_AVAILABLE = False
QISKIT_ALGORITHMS_AVAILABLE = False

try:
    from qae_circuits import (
        QuantumAmplitudeEstimator as QAEEstimator,
        QuantumCVaRCircuits,
        QAEResult,
        QuantumCVaRResult,
        get_circuit_exporter,
        reset_circuit_exporter,
        QISKIT_AVAILABLE as _QISKIT,
        QISKIT_ALGORITHMS_AVAILABLE as _QISKIT_ALGO
    )
    QAE_AVAILABLE = True
    QISKIT_AVAILABLE = _QISKIT
    QISKIT_ALGORITHMS_AVAILABLE = _QISKIT_ALGO
except ImportError:
    try:
        from phase_2.qae_circuits import (
            QuantumAmplitudeEstimator as QAEEstimator,
            QuantumCVaRCircuits,
            QAEResult,
            QuantumCVaRResult,
            get_circuit_exporter,
            reset_circuit_exporter,
            QISKIT_AVAILABLE as _QISKIT,
            QISKIT_ALGORITHMS_AVAILABLE as _QISKIT_ALGO
        )
        QAE_AVAILABLE = True
        QISKIT_AVAILABLE = _QISKIT
        QISKIT_ALGORITHMS_AVAILABLE = _QISKIT_ALGO
    except ImportError:
        pass


# ==============================================================================
# LOGGER
# ==============================================================================

class FileLogger:
    """File-based logger for the quantum subgradient module."""

    def __init__(self, log_dir: str = './logs', log_name: str = 'quantum_subgradient.txt'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_name
        self.start_time = time.time()
        self._initialize()

    def _initialize(self):
        """Write session header (overwrites previous log)."""
        header = [
            "=" * 80,
            "QUANTUM SUBGRADIENT ESTIMATOR",
            "=" * 80,
            f"Started: {datetime.now().isoformat()}",
            f"QAE Available: {QAE_AVAILABLE}",
            f"Qiskit Available: {QISKIT_AVAILABLE}",
            f"Qiskit Algorithms: {QISKIT_ALGORITHMS_AVAILABLE}",
            "=" * 80, ""
        ]
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(header) + '\n')

    def _write(self, level: str, msg: str):
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        elapsed = time.time() - self.start_time
        line = f"[{timestamp}] [{elapsed:7.3f}s] [{level:7s}] {msg}"
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except Exception as exc:
            sys.stderr.write(f"[quantum_subgradient LOGGER_ERROR] {exc}\n")

    def info(self, msg: str): self._write("INFO", msg)
    def warning(self, msg: str): self._write("WARNING", msg)
    def error(self, msg: str): self._write("ERROR", msg)
    def debug(self, msg: str): self._write("DEBUG", msg)

    def section(self, title: str):
        self._write("", "")
        self._write("", "-" * 70)
        self._write("", f"  {title}")
        self._write("", "-" * 70)

    def metric(self, name: str, value):
        if isinstance(value, float):
            self._write("METRIC", f"{name}: {value:.6f}")
        else:
            self._write("METRIC", f"{name}: {value}")


logger = FileLogger()


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class SubgradientResult:
    """
    Complete result from quantum subgradient estimation.

    Contains all fields expected by the hybrid optimizer and notebook API.
    """
    # Core results
    subgradient: np.ndarray
    subgradient_norm: float

    # Quantum tail probability (per method)
    tail_probability_iqae: float
    tail_probability_woerner: float
    tail_probability_classical: float

    # Conditional expectations (per method)
    conditional_expectations: np.ndarray
    conditional_expectations_iqae: np.ndarray
    conditional_expectations_woerner: np.ndarray

    # Oracle queries
    total_oracle_queries_iqae: int
    total_oracle_queries_woerner: int

    # Convergence flags
    iqae_converged: bool
    woerner_converged: bool

    # Error metrics vs classical
    subgradient_l2_error: float
    subgradient_cosine_similarity: float
    tail_prob_error_iqae: float
    tail_prob_error_woerner: float

    # Classical reference values
    classical_var: float
    classical_cvar: float
    classical_tail_prob: float
    classical_subgradient: np.ndarray

    # Metadata
    num_assets: int
    num_periods: int
    alpha: float
    epsilon: float
    num_distribution_qubits: int

    # Execution info
    execution_time_ms: float
    method: str  # 'iqae', 'woerner', or 'classical_fallback'
    circuits_exported: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Serialize result to dictionary."""
        return {
            'subgradient': self.subgradient.tolist(),
            'subgradient_norm': float(self.subgradient_norm),
            'tail_probability_iqae': float(self.tail_probability_iqae),
            'tail_probability_woerner': float(self.tail_probability_woerner),
            'tail_probability_classical': float(self.tail_probability_classical),
            'conditional_expectations': self.conditional_expectations.tolist(),
            'conditional_expectations_iqae': self.conditional_expectations_iqae.tolist(),
            'conditional_expectations_woerner': self.conditional_expectations_woerner.tolist(),
            'total_oracle_queries_iqae': int(self.total_oracle_queries_iqae),
            'total_oracle_queries_woerner': int(self.total_oracle_queries_woerner),
            'iqae_converged': bool(self.iqae_converged),
            'woerner_converged': bool(self.woerner_converged),
            'subgradient_l2_error': float(self.subgradient_l2_error),
            'subgradient_cosine_similarity': float(self.subgradient_cosine_similarity),
            'tail_prob_error_iqae': float(self.tail_prob_error_iqae),
            'tail_prob_error_woerner': float(self.tail_prob_error_woerner),
            'classical_var': float(self.classical_var),
            'classical_cvar': float(self.classical_cvar),
            'classical_tail_prob': float(self.classical_tail_prob),
            'classical_subgradient': self.classical_subgradient.tolist(),
            'num_assets': int(self.num_assets),
            'num_periods': int(self.num_periods),
            'alpha': float(self.alpha),
            'epsilon': float(self.epsilon),
            'num_distribution_qubits': int(self.num_distribution_qubits),
            'execution_time_ms': float(self.execution_time_ms),
            'method': self.method,
            'circuits_exported': self.circuits_exported
        }


# ==============================================================================
# QUANTUM SUBGRADIENT ESTIMATOR
# ==============================================================================

class QuantumSubgradientEstimator:
    """
    Quantum subgradient estimator for CVaR optimization.

    Uses real quantum amplitude estimation for:
        1. Tail probability P(L >= VaR)
        2. Conditional expectations E[r_j | L >= VaR]

    Parameters
    ----------
    num_distribution_qubits : int
        Number of qubits for loss distribution encoding.
    epsilon : float
        QAE precision target.
    alpha : float
        CVaR confidence level (e.g. 0.05 for 95% CVaR).
    use_iqae : bool
        Use Iterative QAE method.
    use_woerner : bool
        Use Woerner canonical QAE method.
    shots : int
        Number of shots per circuit execution.
    use_zne : bool
        Enable Zero Noise Extrapolation.
    zne_scale_factors : list of float
        Scale factors for ZNE.
    """

    def __init__(
        self,
        num_distribution_qubits: int = 4,
        epsilon: float = 0.1,
        alpha: float = 0.05,
        use_iqae: bool = True,
        use_woerner: bool = True,
        backend: str = 'aer_simulator',
        export_circuits: bool = True,
        circuit_export_dir: str = None,
        passport: Any = None,
        use_zne: bool = False,
        zne_scale_factors: Optional[List[float]] = None,
        shots: int = 1024,
        noise_model_config: Optional[Dict] = None
    ):
        logger.section("INITIALIZING QUANTUM SUBGRADIENT ESTIMATOR")

        self.num_qubits = num_distribution_qubits
        self.num_levels = 2 ** num_distribution_qubits
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_iqae = use_iqae
        self.use_woerner = use_woerner
        self.passport = passport
        self.use_zne = use_zne
        self.zne_scale_factors = zne_scale_factors or [1.0, 1.5, 2.0]
        self.shots = shots
        self.noise_model_config = noise_model_config

        # Initialize QAE estimator
        self.qae_estimator = None
        if QAE_AVAILABLE and QISKIT_ALGORITHMS_AVAILABLE:
            try:
                self.qae_estimator = QAEEstimator(
                    num_distribution_qubits=num_distribution_qubits,
                    epsilon=epsilon,
                    export_circuits=export_circuits,
                    circuit_export_dir=circuit_export_dir or './results/circuits',
                    shots=shots,
                    use_zne=use_zne,
                    zne_scale_factors=zne_scale_factors,
                    noise_model_config=noise_model_config
                )
                logger.info("QAE estimator initialized (TRUE QUANTUM)")
                logger.metric("Shots", shots)
                logger.metric("ZNE enabled", use_zne)
            except Exception as e:
                import traceback
                logger.error(f"QAE initialization failed: {e}")
                logger.error(traceback.format_exc())
                self.qae_estimator = None
        else:
            logger.warning("QAE not available - will use classical fallback")

        # Statistics
        self.stats = {
            'total_calls': 0,
            'total_iqae_queries': 0,
            'total_woerner_queries': 0,
            'total_time_ms': 0
        }

        logger.metric("Qubits", num_distribution_qubits)
        logger.metric("Levels", self.num_levels)
        logger.metric("Epsilon", epsilon)
        logger.metric("Alpha", alpha)
        logger.metric("IQAE enabled", use_iqae)
        logger.metric("Woerner enabled", use_woerner)
        logger.info("Initialization complete")

    def estimate_subgradient(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        classical_var: float,
        classical_cvar: float
    ) -> SubgradientResult:
        """
        Estimate CVaR subgradient using quantum amplitude estimation.

        Parameters
        ----------
        weights : np.ndarray
            Current portfolio weights (N,).
        returns : np.ndarray
            Returns matrix (T, N).
        classical_var : float
            Classical VaR value (loss convention) used as threshold.
        classical_cvar : float
            Classical CVaR value (loss convention) for reference.

        Returns
        -------
        SubgradientResult
            Complete subgradient estimation results.
        """
        start_time = time.time()
        self.stats['total_calls'] += 1

        logger.section("QUANTUM SUBGRADIENT ESTIMATION")

        T, N = returns.shape
        logger.metric("Assets (N)", N)
        logger.metric("Periods (T)", T)
        logger.metric("Classical VaR (loss)", classical_var)
        logger.metric("Classical CVaR (loss)", classical_cvar)

        # Compute portfolio losses
        portfolio_losses = -(returns @ weights)
        var_threshold = classical_var

        # Classical reference calculations
        tail_mask = portfolio_losses >= var_threshold
        classical_tail_prob = max(float(np.mean(tail_mask)), 1e-8)
        tail_count = int(np.sum(tail_mask))

        logger.metric("VaR threshold", var_threshold)
        logger.metric("Classical tail prob", classical_tail_prob)
        logger.metric("Tail count", f"{tail_count} / {T}")

        # Classical conditional expectations and subgradient
        if np.any(tail_mask):
            classical_cond_exp = np.mean(returns[tail_mask, :], axis=0)
        else:
            classical_cond_exp = np.mean(returns, axis=0)

        # g_j = -E[r_j | L >= VaR] (no division by P(tail) needed,
        # since the conditioning already restricts to the tail).
        classical_subgradient = -classical_cond_exp
        logger.metric("Classical subgradient norm", np.linalg.norm(classical_subgradient))

        # Initialize result accumulators
        tail_prob_iqae = 0.0
        tail_prob_woerner = 0.0
        queries_iqae = 0
        queries_woerner = 0
        iqae_converged = False
        woerner_converged = False
        cond_exp_iqae = np.zeros(N)
        cond_exp_woerner = np.zeros(N)
        circuits_exported = []

        # Select QAE methods
        methods = []
        if self.use_iqae:
            methods.append('iqae')
        if self.use_woerner:
            methods.append('woerner')
        if not methods:
            methods.append('iqae')

        # Run quantum estimation pipeline
        if self.qae_estimator is not None and methods:
            # Step 1: Tail probability
            logger.section("STEP 1: QUANTUM TAIL PROBABILITY")
            try:
                tail_results = self.qae_estimator.estimate_tail_probability(
                    portfolio_losses, var_threshold, methods
                )
                if 'iqae' in tail_results:
                    tail_prob_iqae = tail_results['iqae'].amplitude
                    queries_iqae += tail_results['iqae'].num_oracle_queries
                    iqae_converged = tail_results['iqae'].converged
                    logger.metric("IQAE tail prob", tail_prob_iqae)
                if 'woerner' in tail_results:
                    tail_prob_woerner = tail_results['woerner'].amplitude
                    queries_woerner += tail_results['woerner'].num_oracle_queries
                    woerner_converged = tail_results['woerner'].converged
                    logger.metric("Woerner tail prob", tail_prob_woerner)
            except Exception as e:
                logger.error(f"Quantum tail probability failed: {e}")
                tail_prob_iqae = classical_tail_prob
                tail_prob_woerner = classical_tail_prob

            # Step 2: Conditional expectations
            logger.section("STEP 2: QUANTUM CONDITIONAL EXPECTATIONS")
            try:
                primary_method = 'iqae' if self.use_iqae else 'woerner'
                for j in range(N):
                    logger.debug(f"Processing asset {j + 1}/{N}")
                    exp_j, result_j = self.qae_estimator.estimate_conditional_expectation(
                        returns[:, j], portfolio_losses, var_threshold, j, primary_method
                    )
                    if primary_method == 'iqae':
                        cond_exp_iqae[j] = exp_j
                        if result_j:
                            queries_iqae += result_j.num_oracle_queries
                    else:
                        cond_exp_woerner[j] = exp_j
                        if result_j:
                            queries_woerner += result_j.num_oracle_queries
                logger.info("Conditional expectations computed")
            except Exception as e:
                logger.error(f"Quantum conditional expectations failed: {e}")
                cond_exp_iqae = classical_cond_exp.copy()
                cond_exp_woerner = classical_cond_exp.copy()

            # Collect exported circuit IDs
            if hasattr(self.qae_estimator, 'exporter'):
                circuits_exported = [
                    e.get('circuit_id', '') for e in self.qae_estimator.exporter.export_log
                ]
        else:
            logger.warning("Using classical fallback (QAE not available)")
            tail_prob_iqae = classical_tail_prob
            tail_prob_woerner = classical_tail_prob
            cond_exp_iqae = classical_cond_exp.copy()
            cond_exp_woerner = classical_cond_exp.copy()

        # Step 3: Compute quantum subgradient
        logger.section("STEP 3: COMPUTE SUBGRADIENT")

        if self.use_iqae and iqae_converged and tail_prob_iqae > 0:
            primary_cond_exp = cond_exp_iqae
            method_used = 'iqae'
        elif self.use_woerner and woerner_converged and tail_prob_woerner > 0:
            primary_cond_exp = cond_exp_woerner
            method_used = 'woerner'
        else:
            primary_cond_exp = classical_cond_exp
            method_used = 'classical_fallback'

        quantum_subgradient = -primary_cond_exp
        logger.info(f"Primary method: {method_used}")
        logger.metric("Quantum subgradient norm", np.linalg.norm(quantum_subgradient))

        # Step 4: Error metrics vs classical
        logger.section("STEP 4: ERROR METRICS")

        l2_error = float(np.linalg.norm(quantum_subgradient - classical_subgradient))

        norm_q = np.linalg.norm(quantum_subgradient)
        norm_c = np.linalg.norm(classical_subgradient)
        if norm_q > 1e-10 and norm_c > 1e-10:
            cosine_sim = float(np.dot(quantum_subgradient, classical_subgradient) / (norm_q * norm_c))
        else:
            cosine_sim = 1.0

        tail_prob_error_iqae = abs(tail_prob_iqae - classical_tail_prob) / max(classical_tail_prob, 1e-10)
        tail_prob_error_woerner = abs(tail_prob_woerner - classical_tail_prob) / max(classical_tail_prob, 1e-10)

        logger.metric("L2 error", l2_error)
        logger.metric("Cosine similarity", cosine_sim)
        logger.metric("Tail prob error (IQAE)", tail_prob_error_iqae)
        logger.metric("Tail prob error (Woerner)", tail_prob_error_woerner)

        # Update statistics
        execution_time = (time.time() - start_time) * 1000
        self.stats['total_iqae_queries'] += queries_iqae
        self.stats['total_woerner_queries'] += queries_woerner
        self.stats['total_time_ms'] += execution_time

        logger.section("QUANTUM SUBGRADIENT COMPLETE")
        logger.metric("Total IQAE queries", queries_iqae)
        logger.metric("Total Woerner queries", queries_woerner)
        logger.metric("Execution time (ms)", execution_time)

        return SubgradientResult(
            subgradient=quantum_subgradient,
            subgradient_norm=float(np.linalg.norm(quantum_subgradient)),
            tail_probability_iqae=tail_prob_iqae,
            tail_probability_woerner=tail_prob_woerner,
            tail_probability_classical=classical_tail_prob,
            conditional_expectations=primary_cond_exp,
            conditional_expectations_iqae=cond_exp_iqae,
            conditional_expectations_woerner=cond_exp_woerner,
            total_oracle_queries_iqae=queries_iqae,
            total_oracle_queries_woerner=queries_woerner,
            iqae_converged=iqae_converged,
            woerner_converged=woerner_converged,
            subgradient_l2_error=l2_error,
            subgradient_cosine_similarity=cosine_sim,
            tail_prob_error_iqae=tail_prob_error_iqae,
            tail_prob_error_woerner=tail_prob_error_woerner,
            classical_var=classical_var,
            classical_cvar=classical_cvar,
            classical_tail_prob=classical_tail_prob,
            classical_subgradient=classical_subgradient,
            num_assets=N,
            num_periods=T,
            alpha=self.alpha,
            epsilon=self.epsilon,
            num_distribution_qubits=self.num_qubits,
            execution_time_ms=execution_time,
            method=method_used,
            circuits_exported=circuits_exported
        )

    def get_statistics(self) -> Dict:
        """Get estimator statistics."""
        return self.stats.copy()


# ==============================================================================
# CONVENIENCE FUNCTION (BACKWARD COMPATIBILITY)
# ==============================================================================

def estimate_cvar_subgradient(
    weights: np.ndarray,
    returns: np.ndarray,
    alpha: float = 0.05,
    num_qubits: int = 4,
    epsilon: float = 0.1,
    use_iqae: bool = True,
    use_woerner: bool = False
) -> Tuple[np.ndarray, SubgradientResult]:
    """
    Convenience function for quantum CVaR subgradient estimation.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    returns : np.ndarray
        Returns matrix (T, N).
    alpha : float
        CVaR confidence level.
    num_qubits : int
        Distribution qubits.
    epsilon : float
        QAE precision.
    use_iqae : bool
        Use IQAE method.
    use_woerner : bool
        Use Woerner method.

    Returns
    -------
    Tuple of (subgradient array, SubgradientResult).
    """
    T, N = returns.shape
    portfolio_losses = -(returns @ weights)
    sorted_losses = np.sort(portfolio_losses)
    var_idx = int((1 - alpha) * T)
    classical_var = float(sorted_losses[var_idx])
    classical_cvar = float(np.mean(sorted_losses[var_idx:]))

    estimator = QuantumSubgradientEstimator(
        num_distribution_qubits=num_qubits,
        epsilon=epsilon,
        alpha=alpha,
        use_iqae=use_iqae,
        use_woerner=use_woerner
    )

    result = estimator.estimate_subgradient(weights, returns, classical_var, classical_cvar)
    return result.subgradient, result


# ==============================================================================
# CLASSICAL COMPARISON FUNCTION
# ==============================================================================

def compute_classical_subgradient(
    weights: np.ndarray,
    returns: np.ndarray,
    alpha: float = 0.05
) -> Tuple[np.ndarray, Dict]:
    """
    Compute classical CVaR subgradient for comparison.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights.
    returns : np.ndarray
        Returns matrix (T, N).
    alpha : float
        CVaR confidence level.

    Returns
    -------
    Tuple of (subgradient array, metrics dict).
    """
    T, N = returns.shape
    portfolio_losses = -(returns @ weights)
    sorted_losses = np.sort(portfolio_losses)
    var_idx = int((1 - alpha) * T)
    var_threshold = float(sorted_losses[var_idx])

    tail_mask = portfolio_losses >= var_threshold
    tail_prob = max(float(np.mean(tail_mask)), 1e-8)

    if np.any(tail_mask):
        cond_exp = np.mean(returns[tail_mask, :], axis=0)
    else:
        cond_exp = np.mean(returns, axis=0)

    subgradient = -cond_exp

    metrics = {
        'subgradient': subgradient.tolist(),
        'subgradient_norm': float(np.linalg.norm(subgradient)),
        'var_threshold': var_threshold,
        'tail_prob': tail_prob,
        'tail_count': int(np.sum(tail_mask)),
        'conditional_expectations': cond_exp.tolist()
    }
    return subgradient, metrics


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def create_quantum_subgradient_estimator(
    num_qubits: int = 4,
    epsilon: float = 0.1,
    alpha: float = 0.05,
    use_iqae: bool = True,
    use_woerner: bool = False,
    export_circuits: bool = True,
    circuit_export_dir: str = None
) -> QuantumSubgradientEstimator:
    """Factory function to create quantum subgradient estimator."""
    return QuantumSubgradientEstimator(
        num_distribution_qubits=num_qubits,
        epsilon=epsilon,
        alpha=alpha,
        use_iqae=use_iqae,
        use_woerner=use_woerner,
        export_circuits=export_circuits,
        circuit_export_dir=circuit_export_dir
    )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    'QuantumSubgradientEstimator',
    'SubgradientResult',
    'estimate_cvar_subgradient',
    'compute_classical_subgradient',
    'create_quantum_subgradient_estimator',
    'QAE_AVAILABLE',
    'QISKIT_AVAILABLE',
    'QISKIT_ALGORITHMS_AVAILABLE'
]
