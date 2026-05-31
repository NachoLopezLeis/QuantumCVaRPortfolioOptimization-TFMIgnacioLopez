#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : classical_benchmark.py  (Phase 3 - Classical Benchmarking)
# ============================================================================
#
# Description
# -----------
# Benchmarks quantum portfolio optimization against classical methods.
# Fixes [P0-001]: BENCHMARKING_ENABLED default corrected; explicit error
#                 logging in except blocks.
# Adds  [P1-001]: Markowitz minimum-variance and Risk-Parity baselines.
# Adds  [T-002]:  Classical CVaR subgradient — apples-to-apples comparison
#                 isolating the contribution of the quantum component.
#
# Methods: SLSQP, COBYLA, L-BFGS-B, Monte Carlo, Subgradient,
#          Markowitz Min-Variance, Risk Parity.
#
# References
# ----------
# [1] Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
# [2] Markowitz (1952). Portfolio Selection. Journal of Finance.
# [3] Maillard, Roncalli & Teiletche (2010). The Properties of Equally
#     Weighted Risk Contribution Portfolios. Journal of Portfolio Management.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FuturesTimeout
import warnings
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

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

SCIPY_AVAILABLE = False
CVXPY_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    pass

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    pass

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
            import os
            os.makedirs("logs", exist_ok=True)
            h = logging.FileHandler("logs/classical_benchmark.txt", mode="w")
            h.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(h)
            _log.setLevel(logging.INFO)
        return _log


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ClassicalResult:
    """Result from a single classical optimizer run."""

    method_name: str
    success: bool = False
    optimal_weights: Optional[np.ndarray] = None
    optimal_cvar: float = float('inf')
    optimal_var: float = float('inf')
    execution_time_s: float = 0.0
    iterations: int = 0
    function_evaluations: int = 0
    converged: bool = False
    message: str = ""

    def to_dict(self) -> Dict:
        return {
            "method_name": self.method_name,
            "success": bool(self.success),
            "optimal_weights": (self.optimal_weights.tolist()
                                if self.optimal_weights is not None else None),
            "optimal_cvar": float(self.optimal_cvar),
            "optimal_var": float(self.optimal_var),
            "execution_time_s": float(self.execution_time_s),
            "iterations": int(self.iterations),
            "function_evaluations": int(self.function_evaluations),
            "converged": bool(self.converged),
            "message": self.message,
        }


@dataclass
class BenchmarkResult:
    """Full benchmark comparison: quantum vs all classical methods."""

    # Quantum
    quantum_cvar: float = 0.0
    quantum_var: float = 0.0
    quantum_weights: Optional[np.ndarray] = None
    quantum_time_s: float = 0.0
    quantum_iterations: int = 0
    quantum_queries: int = 0

    # Classical
    classical_results: Dict[str, ClassicalResult] = field(default_factory=dict)

    # Best classical summary
    best_classical_method: str = ""
    best_classical_cvar: float = float('inf')
    best_classical_time_s: float = 0.0

    # Derived metrics
    cvar_improvement_vs_best: float = 0.0
    speedup_vs_best: float = 0.0

    # Metadata
    n_assets: int = 0
    n_periods: int = 0
    alpha: float = 0.05
    computed: bool = True          # False only when benchmark is skipped

    def to_dict(self) -> Dict:
        return {
            "computed": self.computed,
            "quantum": {
                "cvar": float(self.quantum_cvar),
                "var": float(self.quantum_var),
                "weights": (self.quantum_weights.tolist()
                            if self.quantum_weights is not None else None),
                "time_s": float(self.quantum_time_s),
                "iterations": int(self.quantum_iterations),
                "queries": int(self.quantum_queries),
            },
            "classical": {k: v.to_dict() for k, v in self.classical_results.items()},
            "best_classical": {
                "method": self.best_classical_method,
                "cvar": float(self.best_classical_cvar),
                "time_s": float(self.best_classical_time_s),
            },
            "comparison": {
                "cvar_improvement_vs_best": float(self.cvar_improvement_vs_best),
                "speedup_vs_best": float(self.speedup_vs_best),
            },
            "metadata": {
                "n_assets": self.n_assets,
                "n_periods": self.n_periods,
                "alpha": self.alpha,
            },
        }

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "BENCHMARK COMPARISON: QUANTUM vs CLASSICAL",
            "=" * 70,
            f"Problem size: {self.n_assets} assets, {self.n_periods} periods",
            f"Alpha: {self.alpha}",
            "",
            f"{'Method':<22} {'CVaR':>12} {'Time (s)':>10} {'Iters':>8} {'OK?':>6}",
            "-" * 62,
        ]
        # Quantum row
        lines.append(
            f"{'Quantum Hybrid':<22} {self.quantum_cvar:>12.6f} "
            f"{self.quantum_time_s:>10.3f} {self.quantum_iterations:>8} {'Yes':>6}"
        )
        lines.append("-" * 62)
        for name, res in self.classical_results.items():
            status = "Yes" if res.success else "No"
            lines.append(
                f"{name:<22} {res.optimal_cvar:>12.6f} "
                f"{res.execution_time_s:>10.3f} {res.iterations:>8} {status:>6}"
            )
        lines.extend([
            "=" * 70,
            f"Best classical: {self.best_classical_method} "
            f"(CVaR={self.best_classical_cvar:.6f})",
            f"CVaR improvement vs best: {self.cvar_improvement_vs_best:.2%}",
            f"Speedup vs best:          {self.speedup_vs_best:.2f}x",
            "=" * 70,
        ])
        return "\n".join(lines)


# ============================================================================
# CLASSICAL OPTIMIZERS
# ============================================================================

class ClassicalOptimizers:
    """
    Collection of classical CVaR portfolio optimization methods.

    All methods work in the *loss* convention: L = -(r @ w) is positive
    for losses. CVaR is the expected loss in the worst-alpha tail.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        max_iterations: int = 200,
        tolerance: float = 1e-7,
    ):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _compute_cvar(self, weights: np.ndarray, returns: np.ndarray) -> float:
        """CVaR in loss convention (positive = bad)."""
        losses = -(returns @ weights)
        var = np.percentile(losses, (1 - self.alpha) * 100)
        tail = losses[losses >= var]
        return float(np.mean(tail)) if len(tail) > 0 else float(var)

    def _compute_var(self, weights: np.ndarray, returns: np.ndarray) -> float:
        losses = -(returns @ weights)
        return float(np.percentile(losses, (1 - self.alpha) * 100))

    def _project_simplex(
        self,
        weights: np.ndarray,
        lb: float,
        ub: float,
    ) -> np.ndarray:
        """Clip to [lb, ub] then renormalize to sum=1."""
        w = np.clip(weights, lb, ub)
        s = w.sum()
        return w / s if s > 0 else np.ones_like(w) / len(w)

    # ------------------------------------------------------------------
    # SLSQP — direct CVaR minimization (primary classical baseline)
    # ------------------------------------------------------------------

    def optimize_slsqp(
        self,
        returns: np.ndarray,
        initial_weights: np.ndarray,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> ClassicalResult:
        """
        Minimize CVaR via SLSQP (Rockafellar-Uryasev, 2000).

        This is the primary apples-to-apples classical baseline: same
        objective function as the quantum optimizer, solved classically.
        """
        if not SCIPY_AVAILABLE:
            return ClassicalResult("slsqp", message="scipy not available")

        self.logger.info("Running SLSQP-CVaR optimization")
        N = len(initial_weights)
        n_eval = [0]

        def obj(w):
            n_eval[0] += 1
            return self._compute_cvar(w, returns)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds_list = [bounds] * N
        t0 = time.time()
        try:
            res = minimize(
                obj,
                initial_weights,
                method="SLSQP",
                bounds=bounds_list,
                constraints=constraints,
                options={"maxiter": self.max_iterations, "ftol": self.tolerance},
            )
            elapsed = time.time() - t0
            w = self._project_simplex(res.x, bounds[0], bounds[1])
            return ClassicalResult(
                method_name="slsqp",
                success=bool(res.success),
                optimal_weights=w,
                optimal_cvar=self._compute_cvar(w, returns),
                optimal_var=self._compute_var(w, returns),
                execution_time_s=elapsed,
                iterations=int(res.nit),
                function_evaluations=int(n_eval[0]),
                converged=bool(res.success),
                message=str(res.message),
            )
        except Exception as exc:
            self.logger.error(f"SLSQP failed: {exc}\n{traceback.format_exc()}")
            return ClassicalResult("slsqp", message=str(exc))

    # ------------------------------------------------------------------
    # COBYLA
    # ------------------------------------------------------------------

    def optimize_cobyla(
        self,
        returns: np.ndarray,
        initial_weights: np.ndarray,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> ClassicalResult:
        """Minimize CVaR via COBYLA (derivative-free)."""
        if not SCIPY_AVAILABLE:
            return ClassicalResult("cobyla", message="scipy not available")

        self.logger.info("Running COBYLA-CVaR optimization")
        N = len(initial_weights)
        n_eval = [0]

        def obj(w):
            n_eval[0] += 1
            return self._compute_cvar(w, returns)

        constraints = [
            {"type": "ineq", "fun": lambda w: 1.0 - np.sum(w) + 1e-8},
            {"type": "ineq", "fun": lambda w: np.sum(w) - 1.0 + 1e-8},
        ]
        for i in range(N):
            constraints.append({"type": "ineq", "fun": lambda w, i=i: w[i] - bounds[0]})
            constraints.append({"type": "ineq", "fun": lambda w, i=i: bounds[1] - w[i]})

        t0 = time.time()
        try:
            res = minimize(
                obj,
                initial_weights,
                method="COBYLA",
                constraints=constraints,
                options={"maxiter": self.max_iterations, "rhobeg": 0.1},
            )
            elapsed = time.time() - t0
            w = self._project_simplex(res.x, bounds[0], bounds[1])
            return ClassicalResult(
                method_name="cobyla",
                success=bool(res.success),
                optimal_weights=w,
                optimal_cvar=self._compute_cvar(w, returns),
                optimal_var=self._compute_var(w, returns),
                execution_time_s=elapsed,
                iterations=int(getattr(res, "nfev", n_eval[0])),
                function_evaluations=int(n_eval[0]),
                converged=bool(res.success),
                message=str(getattr(res, "message", "")),
            )
        except Exception as exc:
            self.logger.error(f"COBYLA failed: {exc}\n{traceback.format_exc()}")
            return ClassicalResult("cobyla", message=str(exc))

    # ------------------------------------------------------------------
    # Subgradient — critical baseline (same algorithm, classical subgrad)
    # ------------------------------------------------------------------

    def optimize_subgradient(
        self,
        returns: np.ndarray,
        initial_weights: np.ndarray,
        learning_rate: float = 0.01,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> ClassicalResult:
        """
        Classical CVaR subgradient descent.

        Formula: g_j = -E[r_j | L >= VaR]
        Note: conditional mean is computed directly (mean over tail
        observations), which implicitly divides by P(tail). No explicit
        division is needed.
        Ref: Rockafellar & Uryasev (2000), Proposition 2.

        This is the critical apples-to-apples baseline: same iteration
        structure and objective as the quantum hybrid optimizer, but using
        a classically-computed subgradient. Any performance difference vs
        the quantum optimizer is attributable solely to the quantum component.
        """
        self.logger.info("Running classical subgradient-CVaR optimization")

        weights = self._project_simplex(initial_weights.copy(), bounds[0], bounds[1])
        best_weights = weights.copy()
        best_cvar = self._compute_cvar(weights, returns)

        t0 = time.time()
        for it in range(self.max_iterations):
            losses = -(returns @ weights)
            var = np.percentile(losses, (1 - self.alpha) * 100)
            tail_mask = losses >= var

            if tail_mask.sum() > 0:
                # g_j = -E[r_j | L >= VaR]  (Prop. 2, Rockafellar-Uryasev 2000)
                subgrad = -np.mean(returns[tail_mask], axis=0)
            else:
                subgrad = np.zeros(len(weights))

            lr = learning_rate * (0.995 ** it)
            weights = self._project_simplex(weights - lr * subgrad, bounds[0], bounds[1])

            cvar = self._compute_cvar(weights, returns)
            if cvar < best_cvar:
                best_cvar = cvar
                best_weights = weights.copy()

        elapsed = time.time() - t0
        return ClassicalResult(
            method_name="subgradient",
            success=True,
            optimal_weights=best_weights,
            optimal_cvar=best_cvar,
            optimal_var=self._compute_var(best_weights, returns),
            execution_time_s=elapsed,
            iterations=self.max_iterations,
            function_evaluations=self.max_iterations,
            converged=True,
            message="Classical subgradient descent completed",
        )

    # ------------------------------------------------------------------
    # Monte Carlo random search
    # ------------------------------------------------------------------

    def optimize_monte_carlo(
        self,
        returns: np.ndarray,
        n_samples: int = 10_000,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> ClassicalResult:
        """Random-portfolio sampling baseline."""
        self.logger.info(f"Running Monte Carlo search ({n_samples} samples)")
        N = returns.shape[1]
        best_cvar = float("inf")
        best_w = np.ones(N) / N
        t0 = time.time()
        try:
            from src.utils.config_loader import get_config
        except ImportError:
            from utils.config_loader import get_config
        seed = int(get_config()["global"]["random_seed"])
        rng = np.random.default_rng(seed)
        for _ in range(n_samples):
            w = rng.dirichlet(np.ones(N))
            w = self._project_simplex(w, bounds[0], bounds[1])
            cvar = self._compute_cvar(w, returns)
            if cvar < best_cvar:
                best_cvar = cvar
                best_w = w.copy()
        elapsed = time.time() - t0
        return ClassicalResult(
            method_name="monte_carlo",
            success=True,
            optimal_weights=best_w,
            optimal_cvar=best_cvar,
            optimal_var=self._compute_var(best_w, returns),
            execution_time_s=elapsed,
            iterations=n_samples,
            function_evaluations=n_samples,
            converged=True,
            message=f"Sampled {n_samples} portfolios",
        )

    # ------------------------------------------------------------------
    # Markowitz minimum-variance (P1-001 / T-002)
    # ------------------------------------------------------------------

    def optimize_mean_variance(
        self,
        returns: np.ndarray,
        initial_weights: np.ndarray,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> ClassicalResult:
        """
        Markowitz (1952) minimum-variance portfolio.

        Minimizes w^T Sigma w  s.t. sum(w)=1, lb <= w <= ub.

        Included as a canonical reference baseline. CVaR is computed
        post-optimisation using the Rockafellar-Uryasev formula.
        """
        self.logger.info("Running Markowitz minimum-variance optimization")
        if not SCIPY_AVAILABLE:
            return ClassicalResult("markowitz", message="scipy not available")

        N = returns.shape[1]
        Sigma = np.cov(returns.T)  # (N, N)
        n_eval = [0]

        def obj(w):
            n_eval[0] += 1
            return float(w @ Sigma @ w)

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds_list = [bounds] * N
        t0 = time.time()
        try:
            res = minimize(
                obj,
                initial_weights,
                method="SLSQP",
                bounds=bounds_list,
                constraints=constraints,
                options={"maxiter": self.max_iterations, "ftol": self.tolerance},
            )
            elapsed = time.time() - t0
            w = self._project_simplex(res.x, bounds[0], bounds[1])
            return ClassicalResult(
                method_name="markowitz",
                success=bool(res.success),
                optimal_weights=w,
                optimal_cvar=self._compute_cvar(w, returns),
                optimal_var=self._compute_var(w, returns),
                execution_time_s=elapsed,
                iterations=int(res.nit),
                function_evaluations=int(n_eval[0]),
                converged=bool(res.success),
                message=str(res.message),
            )
        except Exception as exc:
            self.logger.error(f"Markowitz failed: {exc}\n{traceback.format_exc()}")
            return ClassicalResult("markowitz", message=str(exc))

    # ------------------------------------------------------------------
    # Risk Parity (P1-001 / T-002)
    # ------------------------------------------------------------------

    def optimize_risk_parity(
        self,
        returns: np.ndarray,
        initial_weights: np.ndarray,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> ClassicalResult:
        """
        Equal Risk Contribution (Risk Parity) portfolio.

        Minimizes sum_i (RC_i - RC_j)^2 where RC_i = w_i * (Sigma @ w)_i
        is the contribution of asset i to total portfolio variance.
        Reference: Maillard, Roncalli & Teiletche (2010).

        Included as a modern industry-standard baseline. CVaR is computed
        post-optimisation.
        """
        self.logger.info("Running Risk Parity (equal risk contribution) optimization")
        if not SCIPY_AVAILABLE:
            return ClassicalResult("risk_parity", message="scipy not available")

        N = returns.shape[1]
        Sigma = np.cov(returns.T)
        target = 1.0 / N          # equal risk share
        n_eval = [0]

        def obj(w):
            n_eval[0] += 1
            port_var = w @ Sigma @ w
            if port_var <= 0:
                return 1e10
            rc = w * (Sigma @ w) / port_var   # relative risk contributions
            return float(np.sum((rc - target) ** 2))

        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds_list = [(max(bounds[0], 1e-6), bounds[1])] * N  # strictly positive for RC
        t0 = time.time()
        try:
            res = minimize(
                obj,
                initial_weights,
                method="SLSQP",
                bounds=bounds_list,
                constraints=constraints,
                options={"maxiter": self.max_iterations, "ftol": self.tolerance},
            )
            elapsed = time.time() - t0
            w = self._project_simplex(res.x, bounds[0], bounds[1])
            return ClassicalResult(
                method_name="risk_parity",
                success=bool(res.success),
                optimal_weights=w,
                optimal_cvar=self._compute_cvar(w, returns),
                optimal_var=self._compute_var(w, returns),
                execution_time_s=elapsed,
                iterations=int(res.nit),
                function_evaluations=int(n_eval[0]),
                converged=bool(res.success),
                message=str(res.message),
            )
        except Exception as exc:
            self.logger.error(f"Risk Parity failed: {exc}\n{traceback.format_exc()}")
            return ClassicalResult("risk_parity", message=str(exc))


# ============================================================================
# MAIN BENCHMARKER CLASS
# ============================================================================

class Benchmarker:
    """
    Orchestrates all classical optimizer runs and compares against quantum.

    Fix [P0-001]: The `run_benchmark` method now always runs when called.
    The BENCHMARKING_ENABLED guard belongs in the notebook, not here.
    Error handling in each optimizer uses traceback.format_exc() so failures
    are visible in logs.
    """

    AVAILABLE_METHODS = [
        "slsqp", "cobyla", "subgradient",
        "monte_carlo", "markowitz", "risk_parity",
    ]

    def __init__(self, alpha: float = 0.05, max_iterations: int = 200,
                 timeout_per_method_seconds: int = 120):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.timeout_per_method = int(timeout_per_method_seconds)
        self.optimizers = ClassicalOptimizers(alpha=alpha, max_iterations=max_iterations)
        self.logger = get_logger(__name__)
        self.logger.info(
            f"Benchmarker initialized: alpha={alpha}, max_iter={max_iterations}, "
            f"timeout={timeout_per_method_seconds}s"
        )

    def run_benchmark(
        self,
        returns: np.ndarray,
        quantum_result: Dict,
        initial_weights: np.ndarray,
        methods: Optional[List[str]] = None,
        bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> BenchmarkResult:
        """
        Run all requested classical optimizers and compare with quantum result.

        Parameters
        ----------
        returns : np.ndarray, shape (T, N)
            Historical returns matrix.
        quantum_result : dict
            Result dict from the hybrid optimizer.
        initial_weights : np.ndarray, shape (N,)
            Starting point for gradient-based solvers.
        methods : list[str], optional
            Subset of AVAILABLE_METHODS to run. Default: all.
        bounds : tuple[float, float]
            (lower, upper) weight bounds per asset.

        Returns
        -------
        BenchmarkResult
        """
        self.logger.info("=" * 60)
        self.logger.info("CLASSICAL BENCHMARKING — START")
        self.logger.info("=" * 60)

        if methods is None:
            methods = self.AVAILABLE_METHODS

        T, N = returns.shape
        result = BenchmarkResult(n_assets=N, n_periods=T, alpha=self.alpha, computed=True)

        # Extract quantum metrics
        # BUG-002 FIX: exhaustive key search across all naming conventions used
        # by different versions of HybridOptimizer result dicts
        _q_cvar = (
            quantum_result.get("optimal_cvar_loss")
            or quantum_result.get("optimal_cvar")
            or quantum_result.get("final_cvar_loss")
            or quantum_result.get("final_cvar")
            or quantum_result.get("cvar_loss")
            or quantum_result.get("cvar")
            or 0.0
        )
        result.quantum_cvar = float(_q_cvar) if _q_cvar else 0.0
        result.quantum_var = quantum_result.get(
            "optimal_var_loss", quantum_result.get("optimal_var", 0.0)
        )
        result.quantum_weights = np.array(quantum_result.get("optimal_weights", []))
        result.quantum_time_s = quantum_result.get(
            "execution_time_s", quantum_result.get("total_time_seconds", 0.0)
        )
        result.quantum_iterations = quantum_result.get("num_iterations", 0)
        result.quantum_queries = quantum_result.get("total_queries", 0)

        self.logger.info(
            f"Quantum baseline: CVaR={result.quantum_cvar:.6f}, "
            f"time={result.quantum_time_s:.2f}s"
        )

        # Dispatch
        dispatcher = {
            "slsqp":       lambda: self.optimizers.optimize_slsqp(returns, initial_weights, bounds),
            "cobyla":      lambda: self.optimizers.optimize_cobyla(returns, initial_weights, bounds),
            "subgradient": lambda: self.optimizers.optimize_subgradient(returns, initial_weights, bounds=bounds),
            "monte_carlo": lambda: self.optimizers.optimize_monte_carlo(returns, bounds=bounds),
            "markowitz":   lambda: self.optimizers.optimize_mean_variance(returns, initial_weights, bounds),
            "risk_parity": lambda: self.optimizers.optimize_risk_parity(returns, initial_weights, bounds),
        }

        for method in methods:
            if method not in dispatcher:
                self.logger.warning(f"Unknown method '{method}' — skipping")
                continue
            self.logger.info(f"Running {method} (timeout={self.timeout_per_method}s)...")
            try:
                # Run each method in a thread so we can enforce a wall-clock timeout.
                # This prevents a single slow method from blocking the full benchmark.
                with ThreadPoolExecutor(max_workers=1) as _ex:
                    _fut = _ex.submit(dispatcher[method])
                    res = _fut.result(timeout=self.timeout_per_method)
                result.classical_results[method] = res
                self.logger.info(
                    f"  {method}: CVaR={res.optimal_cvar:.6f}, "
                    f"time={res.execution_time_s:.3f}s, ok={res.success}"
                )
            except _FuturesTimeout:
                self.logger.warning(
                    f"  {method} timed out after {self.timeout_per_method}s — "
                    f"storing partial result"
                )
                result.classical_results[method] = ClassicalResult(
                    method_name=method,
                    message=f"timeout>{self.timeout_per_method}s",
                )
            except Exception as exc:
                self.logger.error(
                    f"  {method} CRASHED: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                result.classical_results[method] = ClassicalResult(
                    method_name=method, message=str(exc)
                )

        # Find best classical (by CVaR)
        for name, res in result.classical_results.items():
            if res.optimal_cvar < result.best_classical_cvar:
                result.best_classical_method = name
                result.best_classical_cvar = res.optimal_cvar
                result.best_classical_time_s = res.execution_time_s

        # Derived comparison metrics
        if result.best_classical_cvar not in (0.0, float("inf")):
            result.cvar_improvement_vs_best = (
                (result.best_classical_cvar - result.quantum_cvar)
                / result.best_classical_cvar
            )
        if result.quantum_time_s > 0 and result.best_classical_time_s > 0:
            result.speedup_vs_best = result.best_classical_time_s / result.quantum_time_s

        self.logger.info("=" * 60)
        self.logger.info("CLASSICAL BENCHMARKING — COMPLETE")
        self.logger.info(
            f"Best classical: {result.best_classical_method} "
            f"CVaR={result.best_classical_cvar:.6f}"
        )
        self.logger.info(f"CVaR improvement: {result.cvar_improvement_vs_best:.2%}")
        self.logger.info("=" * 60)

        return result


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def run_classical_benchmark(
    returns: np.ndarray,
    quantum_result: Dict,
    initial_weights: np.ndarray,
    alpha: float = 0.05,
    methods: Optional[List[str]] = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
) -> BenchmarkResult:
    """High-level convenience wrapper for Benchmarker.run_benchmark."""
    bm = Benchmarker(alpha=alpha)
    return bm.run_benchmark(
        returns=returns,
        quantum_result=quantum_result,
        initial_weights=initial_weights,
        methods=methods,
        bounds=bounds,
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "Benchmarker",
    "ClassicalOptimizers",
    "BenchmarkResult",
    "ClassicalResult",
    "run_classical_benchmark",
    "SCIPY_AVAILABLE",
    "CVXPY_AVAILABLE",
]
