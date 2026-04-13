#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : method_comparison.py  (Phase 3 - QAE Method Comparison)
# ============================================================================
#
# Description
# -----------
# Compares different QAE methods (IQAE, Woerner, QSP) on accuracy,
# query complexity, runtime, noise resilience, and ZNE compatibility.
# Produces rankings and recommendations for optimal method selection.
#
# References
# ----------
# [1] Brassard et al. (2002). Quantum Amplitude Amplification and Estimation.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
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


warnings.filterwarnings('ignore')

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
            handler = logging.FileHandler("logs/method_comparison.txt", mode="w")
            handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(handler)
            _log.setLevel(logging.INFO)
        return _log

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MethodMetrics:
    """Metrics for a single QAE method."""
    
    method_name: str
    enabled: bool = False
    
    # Accuracy
    amplitude: float = 0.0
    amplitude_classical: float = 0.0
    absolute_error: float = 0.0
    relative_error: float = 0.0
    
    # Complexity
    oracle_queries: int = 0
    circuit_depth: int = 0
    num_qubits: int = 0
    
    # Runtime
    execution_time_ms: float = 0.0
    
    # Convergence
    converged: bool = False
    iterations: int = 0
    epsilon_achieved: float = 0.0
    
    # ZNE
    zne_applied: bool = False
    zne_improvement: float = 0.0
    
    # Derived metrics
    queries_per_epsilon: float = 0.0  # queries * epsilon (lower is better)
    accuracy_per_query: float = 0.0   # (1 - rel_error) / queries
    
    def compute_derived(self, epsilon: float = 0.01):
        """Compute derived efficiency metrics."""
        if self.oracle_queries > 0:
            self.queries_per_epsilon = self.oracle_queries * epsilon
            self.accuracy_per_query = (1 - self.relative_error) / self.oracle_queries * 1000
    
    def to_dict(self) -> Dict:
        return {
            'method_name': self.method_name,
            'enabled': self.enabled,
            'amplitude': float(self.amplitude),
            'amplitude_classical': float(self.amplitude_classical),
            'absolute_error': float(self.absolute_error),
            'relative_error': float(self.relative_error),
            'oracle_queries': int(self.oracle_queries),
            'circuit_depth': int(self.circuit_depth),
            'execution_time_ms': float(self.execution_time_ms),
            'converged': bool(self.converged),
            'zne_applied': bool(self.zne_applied),
            'zne_improvement': float(self.zne_improvement),
            'queries_per_epsilon': float(self.queries_per_epsilon),
            'accuracy_per_query': float(self.accuracy_per_query)
        }

@dataclass
class ComparisonResult:
    """Result from comparing multiple QAE methods."""
    
    # Method metrics
    methods: Dict[str, MethodMetrics] = field(default_factory=dict)
    
    # Rankings (method name -> rank, 1 = best)
    accuracy_ranking: Dict[str, int] = field(default_factory=dict)
    efficiency_ranking: Dict[str, int] = field(default_factory=dict)
    speed_ranking: Dict[str, int] = field(default_factory=dict)
    overall_ranking: Dict[str, int] = field(default_factory=dict)
    
    # Best method per category
    best_accuracy: str = ""
    best_efficiency: str = ""
    best_speed: str = ""
    best_overall: str = ""
    
    # Comparison metadata
    n_methods_compared: int = 0
    epsilon_used: float = 0.0
    classical_baseline: float = 0.0
    
    # Recommendations
    recommendation: str = ""
    recommendation_reason: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'methods': {k: v.to_dict() for k, v in self.methods.items()},
            'rankings': {
                'accuracy': self.accuracy_ranking,
                'efficiency': self.efficiency_ranking,
                'speed': self.speed_ranking,
                'overall': self.overall_ranking
            },
            'best': {
                'accuracy': self.best_accuracy,
                'efficiency': self.best_efficiency,
                'speed': self.best_speed,
                'overall': self.best_overall
            },
            'n_methods_compared': self.n_methods_compared,
            'epsilon_used': float(self.epsilon_used),
            'classical_baseline': float(self.classical_baseline),
            'recommendation': self.recommendation,
            'recommendation_reason': self.recommendation_reason
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 70,
            "QAE METHOD COMPARISON",
            "=" * 70,
            f"Methods compared: {self.n_methods_compared}",
            f"Epsilon: {self.epsilon_used}",
            f"Classical baseline: {self.classical_baseline:.6f}",
            "",
            "-" * 70,
            "COMPARISON TABLE",
            "-" * 70,
            f"{'Method':<12} {'Amplitude':>10} {'Rel.Error':>10} {'Queries':>10} {'Time(ms)':>10} {'Rank':>6}",
            "-" * 70,
        ]
        
        for name in sorted(self.methods.keys()):
            m = self.methods[name]
            if m.enabled:
                rank = self.overall_ranking.get(name, '-')
                lines.append(
                    f"{name:<12} {m.amplitude:>10.6f} {m.relative_error:>9.4f} "
                    f"{m.oracle_queries:>10,} {m.execution_time_ms:>10.1f} {rank:>6}"
                )
        
        lines.extend([
            "",
            "-" * 70,
            "RANKINGS BY CATEGORY",
            "-" * 70,
            f"  Best Accuracy:   {self.best_accuracy}",
            f"  Best Efficiency: {self.best_efficiency}",
            f"  Best Speed:      {self.best_speed}",
            f"  Best Overall:    {self.best_overall}",
            "",
            "-" * 70,
            "RECOMMENDATION",
            "-" * 70,
            f"  Method: {self.recommendation}",
            f"  Reason: {self.recommendation_reason}",
            "=" * 70,
        ])
        
        return "\n".join(lines)

# ============================================================================
# MAIN CLASS: MethodComparator
# ============================================================================

class MethodComparator:
    """
    Compare QAE methods on accuracy, efficiency, and runtime.
    
    This class analyzes results from multiple QAE methods and provides:
    1. Side-by-side comparison metrics
    2. Rankings by different criteria
    3. Recommendations for which method to use
    """
    
    def __init__(self, epsilon: float = 0.01):
        """
        Parameters
        ----------
        epsilon : float
            Target precision used for QAE
        """
        self.epsilon = epsilon
        self.logger = get_logger(__name__)
        self.logger.info(f"MethodComparator initialized: epsilon={epsilon}")
    
    def extract_method_metrics(
        self,
        method_name: str,
        result: Optional[Dict],
        classical_amplitude: float
    ) -> MethodMetrics:
        """
        Extract metrics from a QAE method result.
        
        Parameters
        ----------
        method_name : str
            Name of the method
        result : Dict
            Result dictionary from QAE estimation
        classical_amplitude : float
            Classical baseline for comparison
        
        Returns
        -------
        MethodMetrics
            Extracted metrics
        """
        metrics = MethodMetrics(
            method_name=method_name,
            enabled=result is not None,
            amplitude_classical=classical_amplitude
        )
        
        if result is None:
            return metrics
        
        # Extract amplitude
        metrics.amplitude = result.get('amplitude', 0.0)
        
        # Compute errors
        metrics.absolute_error = abs(metrics.amplitude - classical_amplitude)
        if abs(classical_amplitude) > 1e-10:
            metrics.relative_error = metrics.absolute_error / abs(classical_amplitude)
        else:
            metrics.relative_error = metrics.absolute_error
        
        # Extract complexity metrics
        metrics.oracle_queries = result.get('num_oracle_queries', 
                                    result.get('queries', 0))
        metrics.circuit_depth = result.get('circuit_depth', 0)
        metrics.num_qubits = result.get('num_qubits', 0)
        
        # Runtime
        metrics.execution_time_ms = result.get('execution_time_ms',
                                       result.get('time_ms', 0))
        
        # Convergence
        metrics.converged = result.get('converged', True)
        metrics.iterations = result.get('iterations', 0)
        metrics.epsilon_achieved = result.get('epsilon_achieved', self.epsilon)
        
        # ----------------------------------------------------------------------
        # ZNE improvement calculation
        # ----------------------------------------------------------------------
        # Problem: dict.get('amplitude_raw', fallback) returns None when
        # the key EXISTS with value None, causing TypeError on subtraction.
        #
        # Fix: use `or` so that both missing key AND explicit None fall
        # back to metrics.amplitude (always a valid float).
        # ----------------------------------------------------------------------
        metrics.zne_applied = result.get('zne_applied', False)
        if metrics.zne_applied:
            raw_amp = result.get('amplitude_raw') or metrics.amplitude
            if raw_amp is not None and classical_amplitude is not None:
                raw_error = abs(raw_amp - classical_amplitude)
                if raw_error > 1e-10:
                    metrics.zne_improvement = (raw_error - metrics.absolute_error) / raw_error
                else:
                    metrics.zne_improvement = 0.0
            else:
                metrics.zne_improvement = 0.0
            
            self.logger.debug(
                f"{method_name} ZNE: amp_raw={raw_amp:.6f}, "
                f"amp_zne={metrics.amplitude:.6f}, "
                f"improvement={metrics.zne_improvement:.4f}"
            )
        
        # Compute derived metrics
        metrics.compute_derived(self.epsilon)
        
        return metrics
    
    def compare(
        self,
        qae_results: Dict,
        classical_amplitude: float,
        weights: Optional[Dict[str, float]] = None
    ) -> ComparisonResult:
        """
        Compare all available QAE methods.
        
        Parameters
        ----------
        qae_results : Dict
            Results from all QAE methods
        classical_amplitude : float
            Classical baseline amplitude
        weights : Dict[str, float], optional
            Weights for overall ranking:
            {'accuracy': 0.4, 'efficiency': 0.3, 'speed': 0.3}
        
        Returns
        -------
        ComparisonResult
            Complete comparison result
        """
        self.logger.info("=" * 60)
        self.logger.info("COMPARING QAE METHODS")
        self.logger.info("=" * 60)
        
        if weights is None:
            weights = {'accuracy': 0.5, 'efficiency': 0.3, 'speed': 0.2}
        
        result = ComparisonResult(
            epsilon_used=self.epsilon,
            classical_baseline=classical_amplitude
        )
        
        # Extract metrics for each method
        method_names = ['iqae', 'woerner', 'qsp']
        
        for name in method_names:
            method_result = qae_results.get(name, None)
            metrics = self.extract_method_metrics(name, method_result, classical_amplitude)
            result.methods[name] = metrics
            
            if metrics.enabled:
                result.n_methods_compared += 1
                self.logger.info(f"{name}: amplitude={metrics.amplitude:.6f}, "
                               f"error={metrics.relative_error:.4f}, queries={metrics.oracle_queries}")
        
        if result.n_methods_compared == 0:
            self.logger.warning("No methods to compare")
            result.recommendation = "none"
            result.recommendation_reason = "No QAE methods were enabled"
            return result
        
        # Rank by accuracy (lower error = better = rank 1)
        enabled_methods = [n for n, m in result.methods.items() if m.enabled]
        
        accuracy_sorted = sorted(enabled_methods, 
                                key=lambda n: result.methods[n].relative_error)
        for rank, name in enumerate(accuracy_sorted, 1):
            result.accuracy_ranking[name] = rank
        result.best_accuracy = accuracy_sorted[0] if accuracy_sorted else ""
        
        # Rank by efficiency (lower queries = better)
        efficiency_sorted = sorted(enabled_methods,
                                  key=lambda n: result.methods[n].oracle_queries)
        for rank, name in enumerate(efficiency_sorted, 1):
            result.efficiency_ranking[name] = rank
        result.best_efficiency = efficiency_sorted[0] if efficiency_sorted else ""
        
        # Rank by speed (lower time = better)
        speed_sorted = sorted(enabled_methods,
                             key=lambda n: result.methods[n].execution_time_ms)
        for rank, name in enumerate(speed_sorted, 1):
            result.speed_ranking[name] = rank
        result.best_speed = speed_sorted[0] if speed_sorted else ""
        
        # Overall ranking (weighted combination)
        def overall_score(name):
            acc_rank = result.accuracy_ranking.get(name, len(enabled_methods))
            eff_rank = result.efficiency_ranking.get(name, len(enabled_methods))
            spd_rank = result.speed_ranking.get(name, len(enabled_methods))
            return (weights['accuracy'] * acc_rank + 
                   weights['efficiency'] * eff_rank + 
                   weights['speed'] * spd_rank)
        
        overall_sorted = sorted(enabled_methods, key=overall_score)
        for rank, name in enumerate(overall_sorted, 1):
            result.overall_ranking[name] = rank
        result.best_overall = overall_sorted[0] if overall_sorted else ""
        
        # Generate recommendation
        result.recommendation = result.best_overall
        
        best_metrics = result.methods[result.best_overall]
        reasons = []
        
        if result.accuracy_ranking[result.best_overall] == 1:
            reasons.append(f"lowest error ({best_metrics.relative_error:.2%})")
        if result.efficiency_ranking[result.best_overall] == 1:
            reasons.append(f"fewest queries ({best_metrics.oracle_queries:,})")
        if result.speed_ranking[result.best_overall] == 1:
            reasons.append(f"fastest ({best_metrics.execution_time_ms:.1f}ms)")
        
        if not reasons:
            reasons.append("best overall weighted score")
        
        result.recommendation_reason = ", ".join(reasons)
        
        self.logger.info(f"Recommendation: {result.recommendation}")
        self.logger.info(f"Reason: {result.recommendation_reason}")
        
        return result
    
    def generate_comparison_table(
        self,
        result: ComparisonResult,
        include_disabled: bool = False
    ) -> str:
        """
        Generate a formatted comparison table.
        
        Parameters
        ----------
        result : ComparisonResult
            Comparison result
        include_disabled : bool
            Include disabled methods in table
        
        Returns
        -------
        str
            Formatted table
        """
        lines = [
            "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*8 + "+",
            "|" + " Method".ljust(12) + "|" + " Amplitude".ljust(12) + "|" + " Error %".ljust(10) + "|" + " Queries".ljust(10) + "|" + " Time(ms)".ljust(10) + "|" + " Rank".ljust(8) + "|",
            "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*8 + "+",
        ]
        
        for name in ['iqae', 'woerner', 'qsp']:
            m = result.methods.get(name)
            if m is None:
                continue
            if not m.enabled and not include_disabled:
                continue
            
            if m.enabled:
                rank = str(result.overall_ranking.get(name, '-'))
                amp = f"{m.amplitude:.6f}"
                err = f"{m.relative_error*100:.2f}%"
                queries = f"{m.oracle_queries:,}"
                time = f"{m.execution_time_ms:.1f}"
            else:
                rank = "-"
                amp = "-"
                err = "-"
                queries = "-"
                time = "-"
            
            lines.append(
                "|" + f" {name}".ljust(12) + "|" + f" {amp}".ljust(12) + "|" + 
                f" {err}".ljust(10) + "|" + f" {queries}".ljust(10) + "|" + 
                f" {time}".ljust(10) + "|" + f" {rank}".ljust(8) + "|"
            )
        
        lines.append("+" + "-"*12 + "+" + "-"*12 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*10 + "+" + "-"*8 + "+")
        
        return "\n".join(lines)

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compare_qae_methods(
    qae_results: Dict,
    classical_amplitude: float,
    epsilon: float = 0.01
) -> ComparisonResult:
    """
    High-level function to compare QAE methods.
    
    Parameters
    ----------
    qae_results : Dict
        Results from QAE estimation
    classical_amplitude : float
        Classical baseline
    epsilon : float
        Target epsilon
    
    Returns
    -------
    ComparisonResult
        Comparison result
    """
    comparator = MethodComparator(epsilon=epsilon)
    return comparator.compare(qae_results, classical_amplitude)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'MethodComparator',
    'ComparisonResult',
    'MethodMetrics',
    'compare_qae_methods'
]
