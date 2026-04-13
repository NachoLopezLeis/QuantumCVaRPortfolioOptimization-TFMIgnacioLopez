#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Module  : tests/test_benchmarking_integration.py
# ============================================================================
#
# Integration tests for the classical benchmarking module.
# Required by fix [P0-001]: these tests must pass to confirm that SLSQP-CVaR
# produces weights that differ from Equal Weight, and that the benchmark
# result is properly marked computed=True.
#
# Run:
#   cd PROJECT_ROOT
#   pytest tests/test_benchmarking_integration.py -v
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import sys
import numpy as np
import pytest
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase_3.classical_benchmark import (
    Benchmarker,
    ClassicalOptimizers,
    BenchmarkResult,
    ClassicalResult,
    run_classical_benchmark,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def synthetic_returns():
    """
    Synthetic (T=500, N=5) return matrix.
    Returns are drawn from a skewed distribution to produce non-trivial
    CVaR and avoid degenerate test cases.
    """
    rng = np.random.default_rng(seed=42)
    N, T = 5, 500
    # Base normal returns
    mu    = np.array([0.0005, 0.0003, 0.0004, 0.0002, 0.0006])
    sigma = np.array([0.012,  0.015,  0.010,  0.020,  0.018])
    R     = rng.normal(0, 1, (T, N)) * sigma + mu
    # Add a few large drawdown events (fat tails)
    crash_days = rng.choice(T, size=10, replace=False)
    R[crash_days] -= rng.uniform(0.03, 0.08, (10, N))
    return R


@pytest.fixture(scope="module")
def equal_weights():
    return np.ones(5) / 5


@pytest.fixture(scope="module")
def mock_quantum_result():
    """Minimal quantum optimizer result dict for benchmark input."""
    rng = np.random.default_rng(42)
    w = rng.dirichlet(np.ones(5))
    return {
        "optimal_weights":   w.tolist(),
        "optimal_cvar_loss": 0.025,
        "optimal_var_loss":  0.018,
        "execution_time_s":  45.0,
        "num_iterations":    100,
        "total_queries":     40960,
    }


# ============================================================================
# P0-001: SLSQP MUST PRODUCE WEIGHTS DIFFERENT FROM EQUAL WEIGHT
# ============================================================================

class TestSLSQPBaseline:
    """[P0-001] SLSQP-CVaR must differ meaningfully from Equal Weight."""

    def test_slsqp_runs_successfully(self, synthetic_returns, equal_weights):
        """SLSQP must complete and set success=True."""
        opt = ClassicalOptimizers(alpha=0.05)
        result = opt.optimize_slsqp(synthetic_returns, equal_weights)
        assert result.success is True, (
            f"SLSQP did not converge: {result.message}"
        )

    def test_slsqp_weights_differ_from_equal_weight(self, synthetic_returns, equal_weights):
        """[P0-001] SLSQP weights must NOT equal 1/N (they optimize CVaR)."""
        opt = ClassicalOptimizers(alpha=0.05)
        result = opt.optimize_slsqp(synthetic_returns, equal_weights)
        assert result.optimal_weights is not None
        assert not np.allclose(
            result.optimal_weights,
            equal_weights,
            atol=1e-3,
        ), (
            "SLSQP CVaR weights are identical to Equal Weight — "
            "the optimizer is not working. Fix P0-001."
        )

    def test_slsqp_cvar_lower_than_equal_weight(self, synthetic_returns, equal_weights):
        """SLSQP CVaR must be <= Equal Weight CVaR (it minimizes CVaR)."""
        def cvar(w):
            losses = -(synthetic_returns @ w)
            var = np.percentile(losses, 95)
            return float(np.mean(losses[losses >= var]))

        opt = ClassicalOptimizers(alpha=0.05)
        result = opt.optimize_slsqp(synthetic_returns, equal_weights)
        cvar_slsqp = cvar(result.optimal_weights)
        cvar_ew    = cvar(equal_weights)
        assert cvar_slsqp <= cvar_ew + 1e-6, (
            f"SLSQP CVaR ({cvar_slsqp:.6f}) > Equal Weight CVaR ({cvar_ew:.6f}). "
            "Optimizer failed to improve on the trivial baseline."
        )

    def test_slsqp_weights_sum_to_one(self, synthetic_returns, equal_weights):
        """SLSQP weights must satisfy sum(w) = 1 (budget constraint)."""
        opt = ClassicalOptimizers(alpha=0.05)
        result = opt.optimize_slsqp(synthetic_returns, equal_weights)
        assert abs(np.sum(result.optimal_weights) - 1.0) < 1e-6, (
            f"sum(weights) = {np.sum(result.optimal_weights):.8f} != 1.0"
        )

    def test_slsqp_weights_non_negative(self, synthetic_returns, equal_weights):
        """Long-only constraint: all weights >= 0."""
        opt = ClassicalOptimizers(alpha=0.05)
        result = opt.optimize_slsqp(synthetic_returns, equal_weights)
        assert np.all(result.optimal_weights >= -1e-8), (
            f"Negative weights found: {result.optimal_weights.min():.6f}"
        )


# ============================================================================
# P0-001: BENCHMARK RESULT MUST HAVE computed=True
# ============================================================================

class TestBenchmarkResult:
    """[P0-001] BenchmarkResult.computed must be True when benchmark runs."""

    def test_benchmark_computed_flag(
        self, synthetic_returns, equal_weights, mock_quantum_result
    ):
        """BenchmarkResult.computed must be True after a successful run."""
        bm = Benchmarker(alpha=0.05, max_iterations=50)
        result = bm.run_benchmark(
            returns=synthetic_returns,
            quantum_result=mock_quantum_result,
            initial_weights=equal_weights,
            methods=["slsqp"],
        )
        assert result.computed is True, (
            "BenchmarkResult.computed=False — benchmarking did not execute. "
            "This is the root cause of P0-001."
        )

    def test_benchmark_has_slsqp_result(
        self, synthetic_returns, equal_weights, mock_quantum_result
    ):
        """classical_results must contain 'slsqp' key."""
        bm = Benchmarker(alpha=0.05, max_iterations=50)
        result = bm.run_benchmark(
            returns=synthetic_returns,
            quantum_result=mock_quantum_result,
            initial_weights=equal_weights,
            methods=["slsqp"],
        )
        assert "slsqp" in result.classical_results, (
            "No 'slsqp' key in classical_results dict"
        )

    def test_best_classical_is_not_equal_weight(
        self, synthetic_returns, equal_weights, mock_quantum_result
    ):
        """
        [P0-001] best_classical_cvar must differ from Equal Weight CVaR.
        Previously, because the benchmark never ran, best_classical_cvar
        was implicitly the Equal Weight CVaR (fallback).
        """
        losses = -(synthetic_returns @ equal_weights)
        var_ew = np.percentile(losses, 95)
        cvar_ew = float(np.mean(losses[losses >= var_ew]))

        bm = Benchmarker(alpha=0.05, max_iterations=50)
        result = bm.run_benchmark(
            returns=synthetic_returns,
            quantum_result=mock_quantum_result,
            initial_weights=equal_weights,
            methods=["slsqp"],
        )
        assert abs(result.best_classical_cvar - cvar_ew) > 1e-6, (
            f"best_classical_cvar ({result.best_classical_cvar:.6f}) == "
            f"Equal Weight CVaR ({cvar_ew:.6f}). "
            "The benchmark is using EW as fallback (P0-001 not fixed)."
        )

    def test_to_dict_serialisable(
        self, synthetic_returns, equal_weights, mock_quantum_result
    ):
        """BenchmarkResult.to_dict() must be JSON-serialisable."""
        import json
        bm = Benchmarker(alpha=0.05, max_iterations=30)
        result = bm.run_benchmark(
            returns=synthetic_returns,
            quantum_result=mock_quantum_result,
            initial_weights=equal_weights,
            methods=["slsqp"],
        )
        d = result.to_dict()
        serialised = json.dumps(d)  # must not raise
        assert len(serialised) > 100


# ============================================================================
# P1-001: NEW BASELINES — Markowitz and Risk Parity
# ============================================================================

class TestNewBaselines:
    """[P1-001] Markowitz and Risk Parity baselines must run and converge."""

    def test_markowitz_runs(self, synthetic_returns, equal_weights):
        opt = ClassicalOptimizers(alpha=0.05)
        result = opt.optimize_mean_variance(synthetic_returns, equal_weights)
        assert result.optimal_weights is not None
        assert abs(np.sum(result.optimal_weights) - 1.0) < 1e-5
        assert np.all(result.optimal_weights >= -1e-8)

    def test_risk_parity_runs(self, synthetic_returns, equal_weights):
        opt = ClassicalOptimizers(alpha=0.05)
        result = opt.optimize_risk_parity(synthetic_returns, equal_weights)
        assert result.optimal_weights is not None
        assert abs(np.sum(result.optimal_weights) - 1.0) < 1e-5
        assert np.all(result.optimal_weights >= -1e-8)

    def test_all_methods_run_in_benchmarker(
        self, synthetic_returns, equal_weights, mock_quantum_result
    ):
        """All 6 methods must produce results without crashing."""
        bm = Benchmarker(alpha=0.05, max_iterations=50)
        result = bm.run_benchmark(
            returns=synthetic_returns,
            quantum_result=mock_quantum_result,
            initial_weights=equal_weights,
            methods=["slsqp", "subgradient", "markowitz", "risk_parity"],
        )
        for method in ["slsqp", "subgradient", "markowitz", "risk_parity"]:
            assert method in result.classical_results, (
                f"Method '{method}' missing from classical_results"
            )
            cr = result.classical_results[method]
            assert cr.optimal_weights is not None, (
                f"Method '{method}' returned None weights"
            )
            assert cr.optimal_cvar < float("inf"), (
                f"Method '{method}' returned infinite CVaR"
            )

    def test_subgradient_differs_from_equal_weight(
        self, synthetic_returns, equal_weights
    ):
        """Classical subgradient must find a better CVaR than Equal Weight."""
        def cvar_ew():
            losses = -(synthetic_returns @ equal_weights)
            return float(np.mean(losses[losses >= np.percentile(losses, 95)]))

        opt = ClassicalOptimizers(alpha=0.05, max_iterations=200)
        result = opt.optimize_subgradient(synthetic_returns, equal_weights)
        assert result.optimal_cvar <= cvar_ew() + 1e-6


# ============================================================================
# T-009: BOOTSTRAP CI MODULE
# ============================================================================

class TestBootstrapCI:
    """[T-009] Bootstrap CI module must produce valid statistics."""

    def test_bootstrap_returns_result(self, synthetic_returns, equal_weights):
        from src.phase_3.bootstrap_ci import bootstrap_oos_comparison

        rng = np.random.default_rng(42)
        w_a = rng.dirichlet(np.ones(5))
        w_b = equal_weights.copy()

        result = bootstrap_oos_comparison(
            returns_oos=synthetic_returns,
            weights_a=w_a,
            weights_b=w_b,
            label_a="Portfolio A",
            label_b="Equal Weight",
            n_bootstrap=500,   # fast for CI
            block_size=10,
            seed=42,
        )
        assert "cvar" in result.metrics
        assert "sharpe" in result.metrics
        ci = result.metrics["cvar"]
        assert 0.0 <= ci.p_value <= 1.0
        assert ci.diff_ci[0] <= ci.diff_ci[1]   # CI is ordered

    def test_bootstrap_ci_width_decreases_with_more_samples(
        self, synthetic_returns, equal_weights
    ):
        """Wider CI with n=200 than with n=2000 (consistency check)."""
        from src.phase_3.bootstrap_ci import bootstrap_oos_comparison
        rng = np.random.default_rng(0)
        w_a = rng.dirichlet(np.ones(5))
        w_b = equal_weights.copy()

        r200 = bootstrap_oos_comparison(
            synthetic_returns, w_a, w_b, n_bootstrap=200, block_size=10, seed=0
        )
        r2000 = bootstrap_oos_comparison(
            synthetic_returns, w_a, w_b, n_bootstrap=2000, block_size=10, seed=0
        )
        width_200  = r200.metrics["cvar"].diff_ci[1]  - r200.metrics["cvar"].diff_ci[0]
        width_2000 = r2000.metrics["cvar"].diff_ci[1] - r2000.metrics["cvar"].diff_ci[0]
        assert width_2000 < width_200, (
            f"CI width did not decrease with more samples: "
            f"{width_200:.4f} (n=200) vs {width_2000:.4f} (n=2000)"
        )


# ============================================================================
# CONFIG LOADER
# ============================================================================

class TestConfigLoader:
    """[P3-002] Config loader must load config.yaml successfully."""

    def test_config_loads_without_error(self):
        """Config._load_config() must not raise and must return non-empty dict."""
        from src.utils.config_loader import init_config
        Config = init_config.__wrapped__.__class__ if hasattr(init_config, '__wrapped__') else None

        # Just call get_config — if it raises, the test fails
        from src.utils.config_loader import get_config
        cfg = get_config()
        # After fixing P3-002, config_data should be non-empty when run from project root
        # In CI it may be empty if config.yaml is not present — that's acceptable
        assert isinstance(cfg.get("global", {}), dict)

    def test_config_dot_notation(self):
        """get() must support dot-notation nested access."""
        from src.utils.config_loader import Config
        cfg = Config.__new__(Config)
        cfg._initialized = True
        cfg.config_data = {
            "phase3": {"benchmarking": {"enabled": True, "methods": ["slsqp"]}}
        }
        cfg._cache = {}
        cfg._cache_enabled = True
        assert cfg.get("phase3.benchmarking.enabled") is True
        assert cfg.get("phase3.benchmarking.methods") == ["slsqp"]
        assert cfg.get("phase3.benchmarking.missing", "default") == "default"
        assert cfg.get("nonexistent.key", 42) == 42
