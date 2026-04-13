#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : oos_comparison.py  (Phase 4 - OOS Comparison Orchestrator)
# ============================================================================
#
# Description
# -----------
# Orchestrates the three-way OOS comparison (Parametric vs. Monte Carlo
# vs. Quantum) and produces the accuracy metrics and visualization described
# in the TFM document:
#   - CVaR relative error vs. realized OOS CVaR
#   - Kolmogorov-Smirnov test between predicted and realized CDFs
#   - Wasserstein distance between predicted and realized distributions
#   - KL divergence KL(predicted || realized) on discretized histograms
#
# Realized CVaR on the test period is computed via compute_cvar from
# cvar_computation.py (Phase 1). Portfolio performance metrics (Sharpe,
# Sortino, MDD) on the test period are computed via MetricsComputation
# (Phase 2). No risk or metrics logic is reimplemented here.
#
# The plot_oos_comparison function returns a matplotlib Figure ready for
# display in the notebook or export to file; it does not call plt.show().
#
# All loss values use the loss convention: positive = bad.
#
# References
# ----------
# [1] Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
# [2] Villani (2009). Optimal Transport. Springer.
# [3] Massey (1951). The Kolmogorov-Smirnov Test for Goodness of Fit.
#     Journal of the American Statistical Association, 46(253), 68–78.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import time
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats, special

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
    from oos_types import OOSPrediction, OOSMethodComparison, OOSComparisonResult
    from oos_parametric import compute_portfolio_params, estimate_parametric
    from oos_monte_carlo import estimate_monte_carlo
    from oos_quantum import estimate_quantum
except ImportError:
    from src.phase_4.oos_types import OOSPrediction, OOSMethodComparison, OOSComparisonResult
    from src.phase_4.oos_parametric import compute_portfolio_params, estimate_parametric
    from src.phase_4.oos_monte_carlo import estimate_monte_carlo
    from src.phase_4.oos_quantum import estimate_quantum

# ============================================================================
# PHASE 1 IMPORTS  (realized CVaR / VaR on the test period)
# ============================================================================

try:
    from cvar_computation import compute_cvar, compute_var
except ImportError:
    try:
        from src.phase_1.cvar_computation import compute_cvar, compute_var
    except ImportError:
        def compute_var(returns: np.ndarray, alpha: float = 0.05, **_) -> float:
            return float(np.percentile(returns, alpha * 100))

        def compute_cvar(returns: np.ndarray, alpha: float = 0.05, **_) -> float:
            var_t = compute_var(returns, alpha)
            tail = returns[returns <= var_t]
            return float(np.mean(tail)) if len(tail) > 0 else float(var_t)

# ============================================================================
# PHASE 2 IMPORTS  (portfolio performance metrics on the test period)
# ============================================================================

try:
    from metricscomputation import MetricsComputation
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

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
            handler = logging.FileHandler("logs/oos_comparison.txt", mode="w")
            handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(handler)
            _log.setLevel(logging.INFO)
        return _log

logger = get_logger(__name__)

# ============================================================================
# REALIZED RISK COMPUTATION
# ============================================================================

def compute_realized_risk(
    returns_test: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute realized CVaR, VaR, and loss series from the OOS test period.

    Delegates to compute_cvar and compute_var from cvar_computation.py (Phase 1).
    Those functions work in return convention (negative = loss); results are
    negated to produce loss-convention values (positive = bad).

    Parameters
    ----------
    returns_test : np.ndarray
        Shape (T_test, n_assets). Realized returns in the test period.
    weights : np.ndarray
        Shape (n_assets,). Portfolio weights.
    alpha : float
        CVaR confidence level.

    Returns
    -------
    cvar_realized : float
        Realized CVaR in loss convention (positive).
    var_realized : float
        Realized VaR in loss convention (positive).
    losses_test : np.ndarray
        Shape (T_test,). Realized portfolio losses (positive = bad).
    """
    portfolio_returns = returns_test @ weights           # (T_test,)
    losses_test = -portfolio_returns

    cvar_return = compute_cvar(portfolio_returns, alpha)
    var_return = compute_var(portfolio_returns, alpha)

    cvar_realized = -float(cvar_return)
    var_realized = -float(var_return)

    logger.info(
        f"Realized OOS: CVaR_loss={cvar_realized:.6f}, "
        f"VaR_loss={var_realized:.6f}, n_test={len(portfolio_returns)}"
    )

    return cvar_realized, var_realized, losses_test


# ============================================================================
# ACCURACY METRICS PER METHOD
# ============================================================================

def _compare_method(
    prediction: OOSPrediction,
    losses_test: np.ndarray,
    cvar_realized: float,
    alpha: float,
    n_hist_bins: int = 50,
    kl_epsilon: float = 1e-10,
) -> OOSMethodComparison:
    """
    Compute accuracy metrics comparing one method's prediction to the OOS reality.

    Statistical tests
    -----------------
    KS test  : scipy.stats.ks_2samp between losses_test and a sample drawn from
               the predicted distribution (density_x / density_y resampled).
    Wasserstein : scipy.stats.wasserstein_distance between losses_test and the
               predicted sample.
    KL divergence : KL(P_pred || P_real) on a shared histogram grid.

    Parameters
    ----------
    prediction : OOSPrediction
        Output of one of the three estimator modules.
    losses_test : np.ndarray
        Realized portfolio losses from the test period (positive = bad).
    cvar_realized : float
        Realized CVaR in loss convention.
    alpha : float
        CVaR confidence level (used only for metadata, not recomputed).
    n_hist_bins : int
        Number of bins used to discretize both distributions for KL divergence.
    kl_epsilon : float
        Small constant added to histogram bins before computing KL to avoid
        division by zero or log(0).

    Returns
    -------
    OOSMethodComparison
        All accuracy metrics for this method.
    """
    # ------------------------------------------------------------------
    # CVaR relative error
    # ------------------------------------------------------------------
    cvar_rel_err = (
        abs(prediction.cvar_predicted - cvar_realized) / abs(cvar_realized)
        if abs(cvar_realized) > 1e-12
        else 0.0
    )

    # ------------------------------------------------------------------
    # Build a sample from the predicted distribution for KS / Wasserstein
    # For continuous densities (parametric, MC-KDE): use inverse CDF sampling
    # For discrete distributions (quantum): use the bin weights directly
    # ------------------------------------------------------------------
    density_x = np.asarray(prediction.density_x)
    density_y = np.asarray(prediction.density_y)

    # Normalize to obtain a valid PMF / PDF
    total = density_y.sum()
    if total > 0:
        probs_norm = density_y / total
    else:
        probs_norm = np.ones_like(density_y) / len(density_y)

    # Sample from predicted distribution (reproducible)
    rng = np.random.default_rng(0)
    n_sample = max(len(losses_test), 10_000)
    predicted_sample = rng.choice(density_x, size=n_sample, p=probs_norm)

    # ------------------------------------------------------------------
    # Kolmogorov-Smirnov test
    # ------------------------------------------------------------------
    ks_stat, ks_pval = stats.ks_2samp(predicted_sample, losses_test)

    # ------------------------------------------------------------------
    # Wasserstein distance (Earth Mover's Distance)
    # ------------------------------------------------------------------
    wass = float(stats.wasserstein_distance(predicted_sample, losses_test))

    # ------------------------------------------------------------------
    # KL divergence KL(P_pred || P_real) on a shared histogram grid
    # ------------------------------------------------------------------
    x_lo = min(density_x.min(), losses_test.min())
    x_hi = max(density_x.max(), losses_test.max())
    bin_edges = np.linspace(x_lo, x_hi, n_hist_bins + 1)

    hist_pred, _ = np.histogram(predicted_sample, bins=bin_edges)
    hist_real, _ = np.histogram(losses_test, bins=bin_edges)

    p_pred = hist_pred.astype(float) + kl_epsilon
    p_real = hist_real.astype(float) + kl_epsilon
    p_pred /= p_pred.sum()
    p_real /= p_real.sum()

    kl_div = float(np.sum(special.kl_div(p_pred, p_real)))

    logger.info(
        f"[{prediction.method}] CVaR_err={cvar_rel_err:.4f}, "
        f"KS={ks_stat:.4f} (p={ks_pval:.4f}), "
        f"Wasserstein={wass:.6f}, KL={kl_div:.6f}"
    )

    return OOSMethodComparison(
        method=prediction.method,
        cvar_predicted=prediction.cvar_predicted,
        cvar_realized=cvar_realized,
        cvar_relative_error=cvar_rel_err,
        ks_statistic=float(ks_stat),
        ks_pvalue=float(ks_pval),
        wasserstein_distance=wass,
        kl_divergence=kl_div,
    )


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def run_oos_triple_comparison(
    weights: np.ndarray,
    returns_train_df: pd.DataFrame,
    returns_test_df: pd.DataFrame,
    estimator: Any,
    alpha: float = 0.05,
    n_scenarios_mc: int = 100_000,
    seed: int = 42,
    qae_methods: Optional[List[str]] = None,
    risk_free_rate: float = 0.02,
    n_plot_points: int = 500,
) -> OOSComparisonResult:
    """
    Run the full three-way OOS comparison and return all results.

    Pipeline
    --------
    1. Fit (mu, Sigma, mu_p, sigma_p) on returns_train_df.
    2. Generate MC loss scenarios from N(mu_p, sigma_p²) via Cholesky.
    3. Run parametric estimation (closed-form).
    4. Run Monte Carlo estimation (reuses MC scenarios for consistency).
    5. Run quantum estimation (passes MC scenarios to QAE estimator).
    6. Compute realized CVaR / VaR on returns_test_df.
    7. Compute accuracy metrics (KS, Wasserstein, KL, CVaR error) per method.
    8. Compute OOS portfolio performance metrics via MetricsComputation.

    Parameters
    ----------
    weights : np.ndarray
        Shape (n_assets,). Optimized portfolio weights (quantum output).
    returns_train_df : pd.DataFrame
        Shape (T_train, n_assets). Daily returns in the training period.
    returns_test_df : pd.DataFrame
        Shape (T_test, n_assets). Daily returns in the OOS test period.
    estimator : QuantumAmplitudeEstimator
        Fully configured QAE estimator from Phase 2 notebook cells.
    alpha : float, optional
        CVaR confidence level. Default 0.05.
    n_scenarios_mc : int, optional
        Number of Monte Carlo scenarios. Default 100 000.
    seed : int, optional
        Random seed for MC simulation. Default 42.
    qae_methods : list of str, optional
        QAE methods to enable. Default ['iqae'].
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe / Sortino in OOS metrics. Default 0.02.
    n_plot_points : int, optional
        Density curve resolution for parametric and MC predictions. Default 500.

    Returns
    -------
    OOSComparisonResult
        Full result including predictions, comparisons, OOS metrics, and timing.
    """
    if qae_methods is None:
        qae_methods = ["iqae"]

    t_pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("OOS TRIPLE COMPARISON PIPELINE")
    logger.info("=" * 70)
    logger.info(
        f"alpha={alpha}, n_scenarios_mc={n_scenarios_mc}, seed={seed}, "
        f"T_train={len(returns_train_df)}, T_test={len(returns_test_df)}"
    )

    returns_train = returns_train_df.values
    returns_test = returns_test_df.values
    weights = np.asarray(weights, dtype=float).flatten()

    # ------------------------------------------------------------------
    # Step 1: Fit model parameters on training data (shared by all methods)
    # ------------------------------------------------------------------
    logger.info("Step 1: Fitting model parameters on training data")
    mu, Sigma, mu_p, sigma_p = compute_portfolio_params(returns_train, weights)

    # ------------------------------------------------------------------
    # Step 2: Parametric prediction (closed-form, no simulation needed)
    # ------------------------------------------------------------------
    logger.info("Step 2: Parametric CVaR estimation")
    pred_param = estimate_parametric(
        mu_p=mu_p,
        sigma_p=sigma_p,
        alpha=alpha,
        n_plot_points=n_plot_points,
    )

    # ------------------------------------------------------------------
    # Step 3: Monte Carlo prediction
    # ------------------------------------------------------------------
    logger.info("Step 3: Monte Carlo CVaR estimation")
    pred_mc = estimate_monte_carlo(
        mu=mu,
        Sigma=Sigma,
        weights=weights,
        alpha=alpha,
        n_scenarios=n_scenarios_mc,
        seed=seed,
        n_plot_points=n_plot_points,
    )

    # ------------------------------------------------------------------
    # Step 4: Quantum prediction
    # Pass the same MC loss scenarios so both methods encode the same model
    # ------------------------------------------------------------------
    logger.info("Step 4: Quantum CVaR estimation")

    # Reconstruct MC loss array from the MC prediction's density support
    # by re-running the simulation (same seed → identical draws)
    from oos_monte_carlo import _simulate_portfolio_losses
    _, losses_mc_reference = _simulate_portfolio_losses(
        mu=mu,
        Sigma=Sigma,
        weights=weights,
        n_scenarios=n_scenarios_mc,
        seed=seed,
    )

    pred_quantum = estimate_quantum(
        estimator=estimator,
        losses_reference=losses_mc_reference,
        alpha=alpha,
        methods=qae_methods,
    )

    # ------------------------------------------------------------------
    # Step 5: Realized risk on the test period
    # ------------------------------------------------------------------
    logger.info("Step 5: Computing realized OOS CVaR")
    cvar_realized, var_realized, losses_test = compute_realized_risk(
        returns_test=returns_test,
        weights=weights,
        alpha=alpha,
    )

    # ------------------------------------------------------------------
    # Step 6: Accuracy metrics per method
    # ------------------------------------------------------------------
    logger.info("Step 6: Computing accuracy metrics per method")
    predictions: Dict[str, OOSPrediction] = {
        "parametric":   pred_param,
        "monte_carlo":  pred_mc,
        "quantum":      pred_quantum,
    }
    comparisons: Dict[str, OOSMethodComparison] = {}
    for method_name, prediction in predictions.items():
        comparisons[method_name] = _compare_method(
            prediction=prediction,
            losses_test=losses_test,
            cvar_realized=cvar_realized,
            alpha=alpha,
        )

    # ------------------------------------------------------------------
    # Step 7: OOS portfolio performance metrics via MetricsComputation
    # ------------------------------------------------------------------
    oos_metrics: Dict[str, Any] = {}
    if _METRICS_AVAILABLE:
        logger.info("Step 7: Computing OOS portfolio performance metrics")
        try:
            mc = MetricsComputation(risk_free_rate=risk_free_rate)
            oos_metrics = mc.compute_all_metrics(
                returns_df=returns_test_df,
                weights=weights,
            )
        except Exception as exc:
            logger.warning(f"MetricsComputation failed on test period: {exc}")
    else:
        logger.warning(
            "MetricsComputation not available; OOS performance metrics skipped."
        )

    execution_time_s = time.time() - t_pipeline_start

    result = OOSComparisonResult(
        alpha=float(alpha),
        n_train=len(returns_train_df),
        n_test=len(returns_test_df),
        cvar_realized=float(cvar_realized),
        var_realized=float(var_realized),
        predictions=predictions,
        comparisons=comparisons,
        oos_metrics=oos_metrics,
        execution_time_s=execution_time_s,
    )

    logger.info(result.summary())

    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_oos_comparison(
    result: OOSComparisonResult,
    losses_test: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Any:
    """
    Generate the four-curve OOS comparison plot described in the TFM document.

    Curves
    ------
    - Gray histogram : realized OOS loss distribution.
    - Blue line      : parametric (normal) density.
    - Green line     : Monte Carlo KDE density.
    - Red stems      : quantum discrete probability mass (2^n_qubits bins).
    - Vertical lines : CVaR for each method + realized CVaR (black dashed).

    Parameters
    ----------
    result : OOSComparisonResult
        Output of run_oos_triple_comparison.
    losses_test : np.ndarray, optional
        Realized OOS loss series for the histogram background.
        If None, the histogram is omitted.
    figsize : tuple, optional
        Figure size in inches. Default (12, 6).

    Returns
    -------
    matplotlib.figure.Figure
        The figure object. Caller is responsible for display / saving;
        this function does NOT call plt.show().
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        logger.error(f"matplotlib not available: {exc}")
        raise

    fig, ax = plt.subplots(figsize=figsize)

    # Method display configuration: (color, linestyle, label)
    method_style = {
        "parametric":  ("steelblue",   "-",  "Parametric (Normal)"),
        "monte_carlo": ("forestgreen", "-",  "Monte Carlo KDE"),
        "quantum":     ("firebrick",   "-",  "Quantum (QAE, discrete)"),
    }
    cvar_colors = {
        "parametric":  "steelblue",
        "monte_carlo": "forestgreen",
        "quantum":     "firebrick",
    }

    # ------------------------------------------------------------------
    # Background: realized OOS histogram
    # ------------------------------------------------------------------
    if losses_test is not None:
        ax.hist(
            losses_test,
            bins=40,
            density=True,
            color="lightgray",
            edgecolor="gray",
            alpha=0.5,
            label="Realized OOS losses",
            zorder=1,
        )

    # ------------------------------------------------------------------
    # Predicted densities
    # ------------------------------------------------------------------
    for method_name, prediction in result.predictions.items():
        color, ls, label = method_style.get(method_name, ("black", "-", method_name))

        if method_name == "quantum":
            # Discrete PMF: stems at bin centers
            if method_name == "quantum":
                markerline, stemlines, baseline = ax.stem(
                    prediction.density_x,
                    prediction.density_y,
                    basefmt=" ",
                    label=label,
                )
                plt.setp(stemlines, color=color, linewidth=1.2)
                plt.setp(markerline, color=color, marker="D", markersize=4)

        else:
            ax.plot(
                prediction.density_x,
                prediction.density_y,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                label=label,
                zorder=3,
            )

    # ------------------------------------------------------------------
    # CVaR vertical lines
    # ------------------------------------------------------------------
    y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0

    for method_name, comparison in result.comparisons.items():
        color = cvar_colors.get(method_name, "black")
        ax.axvline(
            x=comparison.cvar_predicted,
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
        )

    # Realized CVaR (black dashed, prominent)
    ax.axvline(
        x=result.cvar_realized,
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=f"CVaR realized ({result.cvar_realized:.4f})",
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Labels, legend, title
    # ------------------------------------------------------------------
    ax.set_xlabel("Portfolio daily loss (loss convention: positive = bad)", fontsize=11)
    ax.set_ylabel("Density / Probability mass", fontsize=11)
    ax.set_title(
        f"OOS Triple Comparison — CVaR at α={result.alpha:.2f}  "
        f"(train={result.n_train}, test={result.n_test} days)",
        fontsize=12,
    )

    # Add relative error annotation per method
    annotation_lines = []
    for method_name in result.ranking_by_cvar_error():
        c = result.comparisons[method_name]
        annotation_lines.append(
            f"{method_name}: rel. err = {c.cvar_relative_error:.3f}"
        )
    ax.annotate(
        "\n".join(annotation_lines),
        xy=(0.02, 0.97),
        xycoords="axes fraction",
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    logger.info("OOS comparison plot generated")

    return fig


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "compute_realized_risk",
    "run_oos_triple_comparison",
    "plot_oos_comparison",
]
