#!/usr/bin/env python3
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : risk_contributions.py  (Phase 3 - Risk Contribution Analysis)
# ============================================================================
"""
Phase 3 Risk Contribution Analysis.

Computes marginal and component risk contributions to CVaR for portfolio
risk decomposition, concentration analysis, and sector aggregation.
    MC_i = -E[r_i | L >= VaR]
    RC_i = w_i * MC_i

References
----------
[1] Rockafellar & Uryasev (2000). Optimization of Conditional Value-at-Risk.
[2] Euler decomposition: sum(RC_i) = CVaR for homogeneous risk measures.
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
            handler = logging.FileHandler("logs/risk_contributions.txt", mode="w")
            handler.setFormatter(logging.Formatter("[%(levelname)-7s] %(message)s"))
            _log.addHandler(handler)
            _log.setLevel(logging.INFO)
        return _log

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RiskContributionResult:
    """Result from risk contribution analysis."""
    
    # Core metrics
    total_cvar: float
    total_var: float
    
    # Marginal contributions: dCVaR/dw_i
    marginal_contributions: np.ndarray
    
    # Component contributions: RC_i = w_i * MC_i
    component_contributions: np.ndarray
    
    # Percentage contributions: RC_i / CVaR
    percentage_contributions: np.ndarray
    
    # Weights used
    weights: np.ndarray
    
    # Asset info
    asset_names: Optional[List[str]] = None
    n_assets: int = 0
    
    # Concentration metrics
    herfindahl_index: float = 0.0  # HHI of component contributions
    effective_n_assets: float = 0.0  # 1/HHI
    max_contribution_pct: float = 0.0
    
    # Validation
    euler_check: float = 0.0  # |sum RC_i - CVaR| / CVaR (should be ~0)
    
    def __post_init__(self):
        self.n_assets = len(self.weights)
    
    def to_dict(self) -> Dict:
        return {
            'total_cvar': float(self.total_cvar),
            'total_var': float(self.total_var),
            'marginal_contributions': self.marginal_contributions.tolist(),
            'component_contributions': self.component_contributions.tolist(),
            'percentage_contributions': self.percentage_contributions.tolist(),
            'weights': self.weights.tolist(),
            'asset_names': self.asset_names,
            'n_assets': self.n_assets,
            'herfindahl_index': float(self.herfindahl_index),
            'effective_n_assets': float(self.effective_n_assets),
            'max_contribution_pct': float(self.max_contribution_pct),
            'euler_check': float(self.euler_check)
        }
    
    def summary(self) -> str:
        lines = [
            "=" * 70,
            "RISK CONTRIBUTION ANALYSIS",
            "=" * 70,
            f"Total CVaR: {self.total_cvar:.6f} ({self.total_cvar*100:.4f}%)",
            f"Total VaR:  {self.total_var:.6f} ({self.total_var*100:.4f}%)",
            "",
            f"Concentration Metrics:",
            f"  Herfindahl Index (HHI): {self.herfindahl_index:.4f}",
            f"  Effective N assets:    {self.effective_n_assets:.2f}",
            f"  Max contribution:      {self.max_contribution_pct:.2f}%",
            f"  Euler check error:     {self.euler_check:.6f}",
            "",
            "-" * 70,
            "RISK CONTRIBUTIONS BY ASSET",
            "-" * 70,
            f"{'Asset':<12} {'Weight':>10} {'Marginal':>12} {'Component':>12} {'% of CVaR':>10}",
            "-" * 70,
        ]
        
        for i in range(self.n_assets):
            name = self.asset_names[i] if self.asset_names else f"Asset_{i}"
            lines.append(
                f"{name:<12} {self.weights[i]:>10.4f} "
                f"{self.marginal_contributions[i]:>12.6f} "
                f"{self.component_contributions[i]:>12.6f} "
                f"{self.percentage_contributions[i]*100:>9.2f}%"
            )
        
        lines.append("-" * 70)
        lines.append(
            f"{'TOTAL':<12} {np.sum(self.weights):>10.4f} "
            f"{'':>12} "
            f"{np.sum(self.component_contributions):>12.6f} "
            f"{np.sum(self.percentage_contributions)*100:>9.2f}%"
        )
        lines.append("=" * 70)
        
        return "\n".join(lines)

@dataclass 
class SectorRiskResult:
    """Risk contributions aggregated by sector."""
    
    sector_contributions: Dict[str, float]
    sector_weights: Dict[str, float]
    sector_percentage: Dict[str, float]
    sector_assets: Dict[str, List[str]]
    
    def to_dict(self) -> Dict:
        return {
            'sector_contributions': self.sector_contributions,
            'sector_weights': self.sector_weights,
            'sector_percentage': self.sector_percentage,
            'sector_assets': self.sector_assets
        }

# ============================================================================
# MAIN CLASS: RiskContributionsAnalyzer
# ============================================================================

class RiskContributionsAnalyzer:
    """
    Analyze risk contributions to CVaR for portfolio decomposition.
    
    This class computes:
    1. Marginal contributions: How much CVaR changes per unit weight change
    2. Component contributions: Each asset's portion of total CVaR
    3. Concentration metrics: HHI, effective N
    4. Sector aggregations: Risk by sector/region
    
    Mathematical basis:
        For CVaR, the marginal contribution is the negative conditional
        expectation of returns given losses exceed VaR:
        
        MC_i = dCVaR/dw_i = -E[r_i | L >= VaR] / alpha
        
        This is the CVaR subgradient, which we also use for optimization.
        
        The component contribution is:
        RC_i = w_i * MC_i
        
        By Euler's theorem for homogeneous functions:
        sum RC_i = CVaR (exactly, for continuous distributions)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Parameters
        ----------
        alpha : float
            CVaR confidence level (e.g., 0.05 for 95% CVaR)
        """
        self.alpha = alpha
        self.logger = get_logger(__name__)
        self.logger.info(f"RiskContributionsAnalyzer initialized: alpha={alpha}")
    
    def compute_risk_contributions(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        compute_var: bool = True
    ) -> RiskContributionResult:
        """
        Compute marginal and component risk contributions to CVaR.
        
        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights (N,)
        returns : np.ndarray
            Asset returns matrix (T, N)
        asset_names : List[str], optional
            Names of assets
        compute_var : bool
            Also compute VaR contributions
        
        Returns
        -------
        RiskContributionResult
            Complete risk contribution analysis
        """
        self.logger.info("=" * 60)
        self.logger.info("COMPUTING RISK CONTRIBUTIONS")
        self.logger.info("=" * 60)
        
        weights = np.asarray(weights).flatten()
        returns = np.asarray(returns)
        T, N = returns.shape
        
        self.logger.metric("Assets", N)
        self.logger.metric("Periods", T)
        self.logger.metric("Alpha", self.alpha)
        
        # Compute portfolio returns and losses
        portfolio_returns = returns @ weights
        portfolio_losses = -portfolio_returns
        
        # Compute VaR (quantile of losses)
        var_index = int(np.ceil((1 - self.alpha) * T)) - 1
        sorted_losses = np.sort(portfolio_losses)
        var_loss = sorted_losses[var_index] if var_index < T else sorted_losses[-1]
        
        # Also compute as percentile for consistency
        var_loss_pct = np.percentile(portfolio_losses, (1 - self.alpha) * 100)
        
        self.logger.metric("VaR (loss)", var_loss)
        
        # Identify tail scenarios (losses >= VaR)
        tail_mask = portfolio_losses >= var_loss
        n_tail = np.sum(tail_mask)
        
        if n_tail == 0:
            self.logger.warning("No tail scenarios found, using worst scenarios")
            n_tail = max(1, int(self.alpha * T))
            tail_indices = np.argsort(portfolio_losses)[-n_tail:]
            tail_mask = np.zeros(T, dtype=bool)
            tail_mask[tail_indices] = True
            n_tail = np.sum(tail_mask)
        
        self.logger.metric("Tail scenarios", f"{n_tail} / {T}")
        
        # Compute CVaR (mean of tail losses)
        cvar_loss = np.mean(portfolio_losses[tail_mask])
        self.logger.metric("CVaR (loss)", cvar_loss)
        
        # ----------------------------------------------------------------------
        # MARGINAL CONTRIBUTIONS
        # ----------------------------------------------------------------------
        # MC_i = dCVaR/dw_i = -E[r_i | L >= VaR] / alpha
        # 
        # For discrete data, this is simply the negative mean of asset returns
        # in the tail scenarios, divided by alpha.
        # 
        # Actually, the standard formula is:
        # MC_i = -E[r_i | L >= VaR]
        # 
        # The division by alpha comes from the full subgradient formula for
        # optimization, but for risk decomposition we use the simpler form.
        # ----------------------------------------------------------------------
        
        tail_returns = returns[tail_mask]
        marginal_contributions = -np.mean(tail_returns, axis=0)
        
        self.logger.info(f"Marginal contributions computed")
        self.logger.metric("MC range", f"[{marginal_contributions.min():.6f}, {marginal_contributions.max():.6f}]")
        
        # ----------------------------------------------------------------------
        # COMPONENT CONTRIBUTIONS
        # ----------------------------------------------------------------------
        # RC_i = w_i * MC_i
        # 
        # This represents asset i's contribution to total portfolio CVaR.
        # By Euler's theorem: sum RC_i = CVaR (for homogeneous risk measures)
        # ----------------------------------------------------------------------
        
        component_contributions = weights * marginal_contributions
        
        self.logger.info(f"Component contributions computed")
        self.logger.metric("RC range", f"[{component_contributions.min():.6f}, {component_contributions.max():.6f}]")
        
        # Euler check: sum of component contributions should equal CVaR
        rc_sum = np.sum(component_contributions)
        euler_error = abs(rc_sum - cvar_loss) / max(abs(cvar_loss), 1e-10)
        
        self.logger.metric("Euler check", f"sum RC = {rc_sum:.6f}, CVaR = {cvar_loss:.6f}, error = {euler_error:.6f}")
        
        # ----------------------------------------------------------------------
        # PERCENTAGE CONTRIBUTIONS
        # ----------------------------------------------------------------------
        # What fraction of total CVaR comes from each asset
        # ----------------------------------------------------------------------
        
        if abs(cvar_loss) > 1e-10:
            percentage_contributions = component_contributions / cvar_loss
        else:
            percentage_contributions = np.zeros(N)
        
        # ----------------------------------------------------------------------
        # CONCENTRATION METRICS
        # ----------------------------------------------------------------------
        
        # Herfindahl-Hirschman Index of risk contributions
        # HHI = sum (RC_i / CVaR)^2 = sum pct_i^2
        # Range: [1/N, 1], where 1/N = perfectly diversified, 1 = concentrated
        hhi = np.sum(percentage_contributions ** 2)
        
        # Effective number of assets (inverse HHI)
        effective_n = 1.0 / max(hhi, 1e-10)
        
        # Maximum single contribution
        max_contrib_pct = np.max(np.abs(percentage_contributions)) * 100
        
        self.logger.metric("HHI", hhi)
        self.logger.metric("Effective N", effective_n)
        self.logger.metric("Max contribution %", max_contrib_pct)
        
        # Build result
        result = RiskContributionResult(
            total_cvar=float(cvar_loss),
            total_var=float(var_loss),
            marginal_contributions=marginal_contributions,
            component_contributions=component_contributions,
            percentage_contributions=percentage_contributions,
            weights=weights,
            asset_names=asset_names,
            herfindahl_index=float(hhi),
            effective_n_assets=float(effective_n),
            max_contribution_pct=float(max_contrib_pct),
            euler_check=float(euler_error)
        )
        
        self.logger.info("Risk contribution analysis complete")
        
        return result
    
    def analyze_by_sector(
        self,
        risk_result: RiskContributionResult,
        sector_mapping: Dict[str, str]
    ) -> SectorRiskResult:
        """
        Aggregate risk contributions by sector.
        
        Parameters
        ----------
        risk_result : RiskContributionResult
            Result from compute_risk_contributions
        sector_mapping : Dict[str, str]
            Mapping from asset name to sector name
            e.g., {'AAPL': 'Technology', 'JPM': 'Financials'}
        
        Returns
        -------
        SectorRiskResult
            Risk contributions aggregated by sector
        """
        self.logger.info("Analyzing risk by sector")
        
        sector_contributions = {}
        sector_weights = {}
        sector_assets = {}
        
        # Aggregate by sector
        for i, name in enumerate(risk_result.asset_names or [f"Asset_{i}" for i in range(risk_result.n_assets)]):
            sector = sector_mapping.get(name, 'Unknown')
            
            if sector not in sector_contributions:
                sector_contributions[sector] = 0.0
                sector_weights[sector] = 0.0
                sector_assets[sector] = []
            
            sector_contributions[sector] += risk_result.component_contributions[i]
            sector_weights[sector] += risk_result.weights[i]
            sector_assets[sector].append(name)
        
        # Compute percentages
        total_rc = sum(sector_contributions.values())
        sector_percentage = {
            k: v / max(total_rc, 1e-10) 
            for k, v in sector_contributions.items()
        }
        
        self.logger.info(f"Aggregated into {len(sector_contributions)} sectors")
        
        return SectorRiskResult(
            sector_contributions=sector_contributions,
            sector_weights=sector_weights,
            sector_percentage=sector_percentage,
            sector_assets=sector_assets
        )
    
    def compute_risk_parity_target(
        self,
        weights: np.ndarray,
        returns: np.ndarray
    ) -> np.ndarray:
        """
        Compute target weights for equal risk contribution (risk parity).
        
        In a risk parity portfolio, each asset contributes equally to total risk:
        RC_i = CVaR / N for all i
        
        Parameters
        ----------
        weights : np.ndarray
            Current weights
        returns : np.ndarray
            Returns matrix
        
        Returns
        -------
        np.ndarray
            Suggested weight adjustments to move toward risk parity
        """
        self.logger.info("Computing risk parity targets")
        
        result = self.compute_risk_contributions(weights, returns)
        
        N = len(weights)
        target_rc = result.total_cvar / N
        
        # How far is each asset from target?
        rc_gap = result.component_contributions - target_rc
        
        # Suggested adjustment: reduce weight of over-contributing assets
        # This is a simple heuristic, not a full optimization
        mc = result.marginal_contributions
        mc_safe = np.where(np.abs(mc) > 1e-10, mc, 1e-10)
        
        weight_adjustment = -rc_gap / mc_safe
        
        # Normalize to keep sum = 0 (budget neutral adjustment)
        weight_adjustment = weight_adjustment - np.mean(weight_adjustment)
        
        self.logger.info(f"Risk parity adjustment range: [{weight_adjustment.min():.4f}, {weight_adjustment.max():.4f}]")
        
        return weight_adjustment

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def compute_risk_contributions(
    weights: np.ndarray,
    returns: np.ndarray,
    alpha: float = 0.05,
    asset_names: Optional[List[str]] = None
) -> RiskContributionResult:
    """
    High-level function to compute risk contributions.
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    returns : np.ndarray
        Returns matrix (T, N)
    alpha : float
        CVaR confidence level
    asset_names : List[str], optional
        Asset names
    
    Returns
    -------
    RiskContributionResult
        Risk contribution analysis
    """
    analyzer = RiskContributionsAnalyzer(alpha=alpha)
    return analyzer.compute_risk_contributions(
        weights=weights,
        returns=returns,
        asset_names=asset_names
    )

def compute_marginal_cvar(
    weights: np.ndarray,
    returns: np.ndarray,
    alpha: float = 0.05
) -> np.ndarray:
    """
    Compute only marginal CVaR contributions.
    
    This is the CVaR subgradient: MC_i = -E[r_i | L >= VaR]
    
    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights
    returns : np.ndarray
        Returns matrix
    alpha : float
        CVaR confidence level
    
    Returns
    -------
    np.ndarray
        Marginal contributions (N,)
    """
    weights = np.asarray(weights).flatten()
    returns = np.asarray(returns)
    T, N = returns.shape
    
    portfolio_losses = -(returns @ weights)
    var_loss = np.percentile(portfolio_losses, (1 - alpha) * 100)
    
    tail_mask = portfolio_losses >= var_loss
    if np.sum(tail_mask) == 0:
        tail_mask = portfolio_losses >= np.percentile(portfolio_losses, 95)
    
    tail_returns = returns[tail_mask]
    marginal_contributions = -np.mean(tail_returns, axis=0)
    
    return marginal_contributions

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'RiskContributionsAnalyzer',
    'RiskContributionResult',
    'SectorRiskResult',
    'compute_risk_contributions',
    'compute_marginal_cvar'
]
