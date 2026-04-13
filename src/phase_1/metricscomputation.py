#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : metricscomputation.py  (Phase 1 - Portfolio Metrics)
# ============================================================================
#
# Description
# -----------
# Centralized portfolio metrics computation with validation and passport
# tracking.  Computes returns, volatility, Sharpe, Sortino, maximum drawdown,
# and Calmar ratio with full data lineage for audit compliance.
#
# References
# ----------
# [1] Sharpe (1966). Mutual Fund Performance. Journal of Business, 39(1).
# [2] Sortino & van der Meer (1991). Downside risk.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import hashlib
import json

# Internal imports with fallback
try:
    from src.utils.logger import get_logger as get_builtin_logger
    from src.utils.passport_orchestrator import get_orchestrator, PassportOrchestrator
    from src.utils.passport_types import DataPassport
except ImportError:
    # Fallbacks for standalone testing
    def get_builtin_logger(name):
        class DummyLogger:
            def info(self, msg, extra=None): pass
            def debug(self, msg, extra=None): pass
            def warning(self, msg, extra=None): pass
            def error(self, msg, extra=None): pass
        return DummyLogger()
    
    def get_orchestrator():
        """Dummy orchestrator for standalone mode"""
        class DummyOrchestrator:
            def assign_passport(self, data, data_type, source, metadata=None):
                return None
            def seal(self, passport, function_name, phase, result):
                return passport
            def checkpoint(self, passport, name, data, metadata=None):
                return passport
            def export_passport(self, passport, filepath):
                return filepath
            def get_journey_visualization(self, passport):
                return ""
        return DummyOrchestrator()

# ============================================================================
# CONSTANTS
# ============================================================================

MIN_PERIODS = 2
MAX_PERIODS = 100000
MIN_RETURN = -1.0
MAX_RETURN = 1.0

# Module-level singletons for standalone functions
_module_logger = get_builtin_logger('metricscomputation')
_module_orchestrator = get_orchestrator()

MIN_VOLATILITY = 1e-8
MAX_VOLATILITY = 10.0
MIN_SHARPE = -20.0
MAX_SHARPE = 20.0
TRADING_DAYS_PER_YEAR = 252

# ============================================================================
# FILE LOGGER (ENHANCED)
# ============================================================================

class FileLogger:
    """Enhanced logger that writes only to .txt file, no console output."""
    
    def __init__(self, log_dir: Path = None):
        """Initialize file logger."""
        if log_dir is None:
            log_dir = Path(__file__).parent.parent.parent / 'results' / 'phase_1'
        
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / 'metricscomputation.txt'
        
        # Clear previous log
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('')
        
        # Initialize metrics
        self.metrics = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_execution_ms': 0.0,
            'errors': []
        }
    
    def log(self, message: str):
        """Write message to log file."""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')
    
    def log_section(self, title: str):
        """Write section header."""
        self.log("")
        self.log("=" * 80)
        self.log(title)
        self.log("=" * 80)
    
    def log_error(self, error_msg: str):
        """Write error message."""
        self.log(f"ERROR: {error_msg}")
        self.metrics['errors'].append(error_msg)
    
    def log_success(self, success_msg: str):
        """Write success message."""
        self.log(f"OK: {success_msg}")
    
    def log_info(self, label: str, value: Any, decimals: int = 6):
        """Log key-value pair."""
        if isinstance(value, float):
            self.log(f" {label}: {value:.{decimals}f}")
        else:
            self.log(f" {label}: {value}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get operation metrics summary."""
        return {
            'total_operations': self.metrics['total_operations'],
            'successful': self.metrics['successful_operations'],
            'failed': self.metrics['failed_operations'],
            'success_rate': (
                self.metrics['successful_operations'] / self.metrics['total_operations'] * 100
                if self.metrics['total_operations'] > 0 else 0
            ),
            'total_execution_ms': self.metrics['total_execution_ms'],
            'errors': self.metrics['errors']
        }

# Global logger instance
_logger = None

def get_logger() -> FileLogger:
    """Get or create global logger instance."""
    global _logger
    if _logger is None:
        _logger = FileLogger()
    return _logger

# ============================================================================
# HELPER FUNCTIONS FOR PASSPORT TRACKING
# ============================================================================

def _compute_hash(data: Any) -> str:
    """Compute hash of data for passport tracking."""
    try:
        if isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        elif isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True, default=str).encode()
        elif isinstance(data, (int, float)):
            data_bytes = str(data).encode()
        else:
            data_bytes = str(data).encode()
        return hashlib.md5(data_bytes).hexdigest()[:16]
    except Exception:
        return "hash_error"

# ============================================================================
# VALIDATION HELPER FUNCTIONS (LEVEL 1-4)
# ============================================================================

def _validate_returns_array(returns: np.ndarray, param_name: str = "returns") -> np.ndarray:
    """
    Validate returns array (LEVEL 1-4 validation).
    
    Parameters
    ----------
    returns : np.ndarray
        Returns array to validate
    param_name : str
        Parameter name for error messages
    
    Returns
    -------
    np.ndarray
        Validated returns array
    
    Raises
    ------
    TypeError, ValueError
        If validation fails
    """
    # LEVEL 1: Type validation
    if not isinstance(returns, np.ndarray):
        raise TypeError(f"{param_name} must be np.ndarray, got {type(returns).__name__}")
    
    # LEVEL 2: Shape validation
    if returns.ndim not in [1, 2]:
        raise ValueError(f"{param_name} must be 1D or 2D, got shape {returns.shape}")
    
    if len(returns) < MIN_PERIODS:
        raise ValueError(
            f"{param_name} must have at least {MIN_PERIODS} periods, got {len(returns)}"
        )
    
    if len(returns) > MAX_PERIODS:
        raise ValueError(
            f"{param_name} cannot exceed {MAX_PERIODS} periods, got {len(returns)}"
        )
    
    # LEVEL 4: Data quality validation
    if np.any(np.isnan(returns)):
        n_nan = np.sum(np.isnan(returns))
        pct = 100 * n_nan / returns.size
        raise ValueError(f"{param_name} contains {n_nan} NaN values ({pct:.2f}%)")
    
    if np.any(np.isinf(returns)):
        n_inf = np.sum(np.isinf(returns))
        raise ValueError(f"{param_name} contains {n_inf} infinite values")
    
    return returns

def _validate_weights_vector(weights: np.ndarray, n_assets: int = None) -> np.ndarray:
    """Validate portfolio weights vector."""
    # LEVEL 1: Type validation
    if not isinstance(weights, np.ndarray):
        raise TypeError(f"weights must be np.ndarray, got {type(weights).__name__}")
    
    # LEVEL 2: Shape validation
    if weights.ndim != 1:
        raise ValueError(f"weights must be 1D, got shape {weights.shape}")
    
    if len(weights) < 2:
        raise ValueError(f"weights must have at least 2 assets, got {len(weights)}")
    
    if n_assets is not None and len(weights) != n_assets:
        raise ValueError(f"weights length ({len(weights)}) != n_assets ({n_assets})")
    
    # LEVEL 3: Value validation
    if np.any(weights < 0):
        raise ValueError("weights contain negative values")
    
    if np.any(weights > 1):
        raise ValueError("weights contain values > 1")
    
    # LEVEL 4: Sum validation
    weight_sum = np.sum(weights)
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise ValueError(f"weights do not sum to 1.0 (sum={weight_sum:.10f})")
    
    return weights

# ============================================================================
# METRICS COMPUTATION CLASS
# ============================================================================

class MetricsComputation:
    """
    Portfolio metrics computation with full validation and passport tracking.
    
    Computes comprehensive portfolio performance metrics including returns,
    volatility, Sharpe ratio, Sortino ratio, and maximum drawdown. All operations
    tracked with complete data lineage for audit compliance.
    
    Parameters
    ----------
    risk_free_rate : float, default=0.02
        Annual risk-free rate for Sharpe/Sortino calculations
    
    Attributes
    ----------
    logger : FileLogger
        File-only logger instance
    orchestrator : PassportOrchestrator
        Central passport orchestrator
    risk_free_rate : float
        Annual risk-free rate
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """Initialize MetricsComputation with risk-free rate."""
        if not isinstance(risk_free_rate, (int, float)):
            raise TypeError(f"risk_free_rate must be float, got {type(risk_free_rate).__name__}")
        
        if not (0 <= risk_free_rate <= 1):
            raise ValueError(f"risk_free_rate must be in [0, 1], got {risk_free_rate}")
        
        self.risk_free_rate = float(risk_free_rate)
        self.logger = get_logger()
        self.orchestrator = get_orchestrator()
        
        self.logger.log_section("MetricsComputation Initialized")
        self.logger.log(f"Risk-free rate: {self.risk_free_rate * 100:.2f}%")
        self.logger.log(f"Passport System: ENABLED")
    
    def compute_returns(
        self,
        portfolio_returns: np.ndarray,
        passport: Optional[DataPassport] = None
    ) -> float:
        """
        Compute annualized portfolio returns.
        
        Parameters
        ----------
        portfolio_returns : np.ndarray
            Daily portfolio returns time series
        passport : DataPassport, optional
            Data passport for lineage tracking
        
        Returns
        -------
        float
            Annualized returns (decimal, e.g., 0.10 for 10%)
        """
        operation_start = datetime.now()
        
        try:
            # Validate inputs
            portfolio_returns = _validate_returns_array(portfolio_returns)
            
            # Checkpoint: after validation
            if passport is not None:
                passport = self.orchestrator.checkpoint(
                    passport,
                    name='returns_after_validation',
                    data=portfolio_returns,
                    metadata={'n_periods': len(portfolio_returns)}
                )
            
            # Calculate mean daily return
            mean_daily_return = np.mean(portfolio_returns)
            
            # Annualize (assuming 252 trading days)
            annual_return = mean_daily_return * TRADING_DAYS_PER_YEAR
            
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            self.logger.metrics['successful_operations'] += 1
            self.logger.metrics['total_execution_ms'] += execution_ms
            
            # Seal function execution
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_returns',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'input_hash': _compute_hash(portfolio_returns),
                        'output_hash': _compute_hash(annual_return),
                        'validation_status': 'valid',
                        'transformation_type': 'returns_computation',
                        'parameters': {'mean_daily': mean_daily_return, 'annualized': annual_return}
                    }
                )
            
            self.logger.log_success(f"Computed returns: {annual_return*100:.2f}%")
            return annual_return
        
        except Exception as e:
            self.logger.log_error(f"compute_returns: {str(e)}")
            self.logger.metrics['failed_operations'] += 1
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_returns',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'error',
                        'errors': [str(e)]
                    }
                )
            raise
        
        finally:
            self.logger.metrics['total_operations'] += 1
    
    def compute_volatility(
        self,
        portfolio_returns: np.ndarray,
        passport: Optional[DataPassport] = None
    ) -> float:
        """
        Compute annualized portfolio volatility.
        
        Parameters
        ----------
        portfolio_returns : np.ndarray
            Daily portfolio returns time series
        passport : DataPassport, optional
            Data passport for lineage tracking
        
        Returns
        -------
        float
            Annualized volatility (decimal, e.g., 0.15 for 15%)
        """
        operation_start = datetime.now()
        
        try:
            # Validate inputs
            portfolio_returns = _validate_returns_array(portfolio_returns)
            
            # Calculate daily volatility
            daily_volatility = np.std(portfolio_returns)
            
            # Annualize
            annual_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            # Validate volatility
            if not (MIN_VOLATILITY <= annual_volatility <= MAX_VOLATILITY):
                raise ValueError(
                    f"Volatility out of reasonable range: {annual_volatility}"
                )
            
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            self.logger.metrics['successful_operations'] += 1
            self.logger.metrics['total_execution_ms'] += execution_ms
            
            # Seal function execution
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_volatility',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'input_hash': _compute_hash(portfolio_returns),
                        'output_hash': _compute_hash(annual_volatility),
                        'validation_status': 'valid',
                        'transformation_type': 'volatility_computation',
                        'parameters': {'daily_vol': daily_volatility, 'annualized': annual_volatility}
                    }
                )
            
            self.logger.log_success(f"Computed volatility: {annual_volatility*100:.2f}%")
            return annual_volatility
        
        except Exception as e:
            self.logger.log_error(f"compute_volatility: {str(e)}")
            self.logger.metrics['failed_operations'] += 1
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_volatility',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'error',
                        'errors': [str(e)]
                    }
                )
            raise
        
        finally:
            self.logger.metrics['total_operations'] += 1
    
    def compute_sharpe_ratio(
        self,
        annual_return: float,
        annual_volatility: float,
        passport: Optional[DataPassport] = None
    ) -> float:
        """
        Compute Sharpe ratio (risk-adjusted returns).
        
        Parameters
        ----------
        annual_return : float
            Annualized portfolio return
        annual_volatility : float
            Annualized portfolio volatility
        passport : DataPassport, optional
            Data passport for lineage tracking
        
        Returns
        -------
        float
            Sharpe ratio
        
        Formula
        -------
        Sharpe = (annual_return - risk_free_rate) / annual_volatility
        """
        operation_start = datetime.now()
        
        try:
            # Validate inputs
            if not isinstance(annual_return, (int, float)):
                raise TypeError(f"annual_return must be float, got {type(annual_return).__name__}")
            
            if not isinstance(annual_volatility, (int, float)):
                raise TypeError(f"annual_volatility must be float, got {type(annual_volatility).__name__}")
            
            if annual_volatility <= 0:
                raise ValueError(f"annual_volatility must be > 0, got {annual_volatility}")
            
            # Calculate Sharpe ratio
            excess_return = annual_return - self.risk_free_rate
            sharpe_ratio = excess_return / annual_volatility
            
            # Validate Sharpe ratio
            if not (MIN_SHARPE <= sharpe_ratio <= MAX_SHARPE):
                raise ValueError(f"Sharpe ratio out of reasonable range: {sharpe_ratio}")
            
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            self.logger.metrics['successful_operations'] += 1
            self.logger.metrics['total_execution_ms'] += execution_ms
            
            # Seal function execution
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_sharpe_ratio',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'input_hash': _compute_hash({'ret': annual_return, 'vol': annual_volatility}),
                        'output_hash': _compute_hash(sharpe_ratio),
                        'validation_status': 'valid',
                        'transformation_type': 'sharpe_computation',
                        'parameters': {'excess_return': excess_return, 'sharpe': sharpe_ratio}
                    }
                )
            
            self.logger.log_success(f"Computed Sharpe ratio: {sharpe_ratio:.4f}")
            return sharpe_ratio
        
        except Exception as e:
            self.logger.log_error(f"compute_sharpe_ratio: {str(e)}")
            self.logger.metrics['failed_operations'] += 1
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_sharpe_ratio',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'error',
                        'errors': [str(e)]
                    }
                )
            raise
        
        finally:
            self.logger.metrics['total_operations'] += 1
    
    def compute_sortino_ratio(
        self,
        portfolio_returns: np.ndarray,
        passport: Optional[DataPassport] = None
    ) -> float:
        """
        Compute Sortino ratio (downside risk-adjusted returns).
        
        Parameters
        ----------
        portfolio_returns : np.ndarray
            Daily portfolio returns time series
        passport : DataPassport, optional
            Data passport for lineage tracking
        
        Returns
        -------
        float
            Sortino ratio
        
        Formula
        -------
        Sortino = (annual_return - risk_free_rate) / downside_deviation
        where downside_deviation only considers negative returns
        """
        operation_start = datetime.now()
        
        try:
            # Validate inputs
            portfolio_returns = _validate_returns_array(portfolio_returns)
            
            # Calculate annual return
            mean_daily_return = np.mean(portfolio_returns)
            annual_return = mean_daily_return * TRADING_DAYS_PER_YEAR
            
            # Calculate downside deviation (only for returns below daily risk-free rate)
            daily_rfr = self.risk_free_rate / TRADING_DAYS_PER_YEAR
            downside_returns = np.minimum(portfolio_returns - daily_rfr, 0)
            downside_deviation_daily = np.sqrt(np.mean(downside_returns ** 2))
            downside_deviation = downside_deviation_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
            
            # Calculate Sortino ratio
            excess_return = annual_return - self.risk_free_rate
            if downside_deviation > 0:
                sortino_ratio = excess_return / downside_deviation
            else:
                sortino_ratio = 0.0
            
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            self.logger.metrics['successful_operations'] += 1
            self.logger.metrics['total_execution_ms'] += execution_ms
            
            # Seal function execution
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_sortino_ratio',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'input_hash': _compute_hash(portfolio_returns),
                        'output_hash': _compute_hash(sortino_ratio),
                        'validation_status': 'valid',
                        'transformation_type': 'sortino_computation',
                        'parameters': {
                            'annual_return': annual_return,
                            'downside_dev': downside_deviation,
                            'sortino': sortino_ratio
                        }
                    }
                )
            
            self.logger.log_success(f"Computed Sortino ratio: {sortino_ratio:.4f}")
            return sortino_ratio
        
        except Exception as e:
            self.logger.log_error(f"compute_sortino_ratio: {str(e)}")
            self.logger.metrics['failed_operations'] += 1
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_sortino_ratio',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'error',
                        'errors': [str(e)]
                    }
                )
            raise
        
        finally:
            self.logger.metrics['total_operations'] += 1
    
    def compute_max_drawdown(
        self,
        portfolio_returns: np.ndarray,
        passport: Optional[DataPassport] = None
    ) -> Tuple[float, int, int]:
        """
        Compute maximum drawdown and duration.
        
        Parameters
        ----------
        portfolio_returns : np.ndarray
            Daily portfolio returns time series
        passport : DataPassport, optional
            Data passport for lineage tracking
        
        Returns
        -------
        Tuple[float, int, int]
            (max_drawdown, start_index, end_index)
        """
        operation_start = datetime.now()
        
        try:
            # Validate inputs
            portfolio_returns = _validate_returns_array(portfolio_returns)
            
            # Compute cumulative returns
            cumulative_returns = np.cumprod(1 + portfolio_returns)
            
            # Compute running maximum
            running_max = np.maximum.accumulate(cumulative_returns)
            
            # Compute drawdown
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Find maximum drawdown
            max_dd_idx = np.argmin(drawdown)
            max_drawdown = drawdown[max_dd_idx]
            
            # Find start of drawdown (previous peak)
            start_idx = np.argmax(running_max[:max_dd_idx])
            
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            self.logger.metrics['successful_operations'] += 1
            self.logger.metrics['total_execution_ms'] += execution_ms
            
            # Seal function execution
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_max_drawdown',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'input_hash': _compute_hash(portfolio_returns),
                        'output_hash': _compute_hash(max_drawdown),
                        'validation_status': 'valid',
                        'transformation_type': 'drawdown_computation',
                        'parameters': {
                            'max_drawdown': max_drawdown,
                            'start_idx': start_idx,
                            'end_idx': max_dd_idx,
                            'duration': max_dd_idx - start_idx
                        }
                    }
                )
            
            self.logger.log_success(f"Computed max drawdown: {max_drawdown*100:.2f}% (duration: {max_dd_idx - start_idx} days)")
            return max_drawdown, start_idx, max_dd_idx
        
        except Exception as e:
            self.logger.log_error(f"compute_max_drawdown: {str(e)}")
            self.logger.metrics['failed_operations'] += 1
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_max_drawdown',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'error',
                        'errors': [str(e)]
                    }
                )
            raise
        
        finally:
            self.logger.metrics['total_operations'] += 1
    
    def compute_all_metrics(
        self,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        passport: Optional[DataPassport] = None
    ) -> Dict[str, Any]:
        """
        Compute all portfolio metrics in one call.
        
        Parameters
        ----------
        returns_df : pd.DataFrame
            Historical returns (dates x assets)
        weights : np.ndarray
            Portfolio weights
        passport : DataPassport, optional
            Data passport for lineage tracking
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with all computed metrics
        """
        operation_start = datetime.now()
        
        try:
            # Validate inputs
            if not isinstance(returns_df, pd.DataFrame):
                raise TypeError(f"returns_df must be pd.DataFrame, got {type(returns_df).__name__}")
            
            weights = _validate_weights_vector(weights)
            
            # Checkpoint: after input validation
            if passport is not None:
                passport = self.orchestrator.checkpoint(
                    passport,
                    name='before_all_metrics',
                    data=returns_df.values,
                    metadata={'n_periods': len(returns_df), 'n_assets': len(returns_df.columns)}
                )
            
            # Calculate portfolio returns
            portfolio_returns = returns_df.values @ weights
            
            # Compute all metrics
            annual_return = self.compute_returns(portfolio_returns, passport)
            annual_volatility = self.compute_volatility(portfolio_returns, passport)
            sharpe_ratio = self.compute_sharpe_ratio(annual_return, annual_volatility, passport)
            sortino_ratio = self.compute_sortino_ratio(portfolio_returns, passport)
            max_drawdown, dd_start, dd_end = self.compute_max_drawdown(portfolio_returns, passport)
            
            # Compile results
            all_metrics = {
                'annual_return': float(annual_return),
                'annual_volatility': float(annual_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'sortino_ratio': float(sortino_ratio),
                'max_drawdown': float(max_drawdown),
                'drawdown_start_idx': int(dd_start),
                'drawdown_end_idx': int(dd_end),
                'drawdown_duration': int(dd_end - dd_start),
                'risk_free_rate': float(self.risk_free_rate)
            }
            
            # Checkpoint: after computation
            if passport is not None:
                passport = self.orchestrator.checkpoint(
                    passport,
                    name='after_all_metrics',
                    data=all_metrics,
                    metadata={'n_metrics': len(all_metrics)}
                )
            
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            self.logger.metrics['successful_operations'] += 1
            self.logger.metrics['total_execution_ms'] += execution_ms
            
            # Seal function execution
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_all_metrics',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'output_hash': _compute_hash(all_metrics),
                        'validation_status': 'valid',
                        'transformation_type': 'comprehensive_metrics',
                        'parameters': {'n_metrics': len(all_metrics)}
                    }
                )
            
            self.logger.log_success(f"Computed all metrics (n={len(all_metrics)})")
            return all_metrics
        
        except Exception as e:
            self.logger.log_error(f"compute_all_metrics: {str(e)}")
            self.logger.metrics['failed_operations'] += 1
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='compute_all_metrics',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'error',
                        'errors': [str(e)]
                    }
                )
            raise
        
        finally:
            self.logger.metrics['total_operations'] += 1
    
    def validate_metrics(
        self,
        metrics_dict: Dict[str, Any],
        passport: Optional[DataPassport] = None
    ) -> Dict[str, Any]:
        """
        Validate computed metrics for reasonableness.
        
        Parameters
        ----------
        metrics_dict : Dict[str, Any]
            Dictionary of computed metrics
        passport : DataPassport, optional
            Data passport for lineage tracking
        
        Returns
        -------
        Dict[str, Any]
            Validation results
        """
        operation_start = datetime.now()
        validation_results = {}
        
        try:
            self.logger.log_section("METRICS VALIDATION TESTS")
            
            # Test 1: Return in reasonable range
            ret = metrics_dict.get('annual_return', 0)
            ret_valid = MIN_RETURN <= ret <= MAX_RETURN
            validation_results['return_valid'] = ret_valid
            self.logger.log(f"{'OK' if ret_valid else 'FAIL'} Annual return in range: {ret*100:.2f}%")
            
            # Test 2: Volatility positive and reasonable
            vol = metrics_dict.get('annual_volatility', 0)
            vol_valid = MIN_VOLATILITY <= vol <= MAX_VOLATILITY
            validation_results['volatility_valid'] = vol_valid
            self.logger.log(f"{'OK' if vol_valid else 'FAIL'} Annual volatility in range: {vol*100:.2f}%")
            
            # Test 3: Sharpe ratio reasonable
            sharpe = metrics_dict.get('sharpe_ratio', 0)
            sharpe_valid = MIN_SHARPE <= sharpe <= MAX_SHARPE
            validation_results['sharpe_valid'] = sharpe_valid
            self.logger.log(f"{'OK' if sharpe_valid else 'FAIL'} Sharpe ratio in range: {sharpe:.4f}")
            
            # Test 4: Max drawdown negative
            dd = metrics_dict.get('max_drawdown', 0)
            dd_valid = dd <= 0
            validation_results['drawdown_valid'] = dd_valid
            self.logger.log(f"{'OK' if dd_valid else 'FAIL'} Max drawdown negative: {dd*100:.2f}%")
            
            # Test 5: Sortino >= Sharpe (usually)
            sortino = metrics_dict.get('sortino_ratio', 0)
            sortino_sharpe_compare = sortino >= sharpe - 0.5  # Allow small difference
            validation_results['sortino_sharpe_compare'] = sortino_sharpe_compare
            self.logger.log(f"{'OK' if sortino_sharpe_compare else 'WARN'} Sortino >= Sharpe: {sortino:.4f} >= {sharpe:.4f}")
            
            validation_results['all_tests_passed'] = all([
                ret_valid, vol_valid, sharpe_valid, dd_valid, sortino_sharpe_compare
            ])
            
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            self.logger.metrics['successful_operations'] += 1
            self.logger.metrics['total_execution_ms'] += execution_ms
            
            # Seal function execution
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='validate_metrics',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'valid' if validation_results['all_tests_passed'] else 'warning',
                        'transformation_type': 'metrics_validation',
                        'parameters': {'tests_passed': sum([ret_valid, vol_valid, sharpe_valid, dd_valid, sortino_sharpe_compare])}
                    }
                )
            
            status = "OK PASSED" if validation_results['all_tests_passed'] else "WARN PARTIAL"
            self.logger.log_success(f"Metrics validation completed: {status}")
            
            return validation_results
        
        except Exception as e:
            self.logger.log_error(f"validate_metrics: {str(e)}")
            self.logger.metrics['failed_operations'] += 1
            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
            
            if passport is not None:
                passport = self.orchestrator.seal(
                    passport,
                    function_name='validate_metrics',
                    phase='Phase 1',
                    result={
                        'execution_time_ms': execution_ms,
                        'validation_status': 'error',
                        'errors': [str(e)]
                    }
                )
            raise
        
        finally:
            self.logger.metrics['total_operations'] += 1

# ============================================================================
# MODULE-LEVEL WRAPPER FUNCTIONS
# ============================================================================
# These functions delegate to MetricsComputation to keep logic centralized.
# ============================================================================

# Global instance for module-level functions
_default_metrics_calculator = None

def get_default_metrics_calculator(risk_free_rate: float = 0.02) -> MetricsComputation:
    """Get or create default metrics calculator instance."""
    global _default_metrics_calculator
    if _default_metrics_calculator is None:
        _default_metrics_calculator = MetricsComputation(risk_free_rate=risk_free_rate)
    return _default_metrics_calculator

def compute_sharpe_ratio(
    annual_return: float,
    annual_volatility: float,
    risk_free_rate: float = 0.02,
    passport: Optional[DataPassport] = None
) -> float:
    """
    Module-level wrapper for Sharpe ratio computation.
    
    Calls MetricsComputation.compute_sharpe_ratio() internally.
    No code duplication - delegates to class method.
    
    Parameters
    ----------
    annual_return : float
        Annualized portfolio return
    annual_volatility : float
        Annualized portfolio volatility
    risk_free_rate : float, default=0.02
        Annual risk-free rate
    passport : DataPassport, optional
        Data passport for lineage tracking
        
    Returns
    -------
    float
        Sharpe ratio
    """
    calculator = get_default_metrics_calculator(risk_free_rate=risk_free_rate)
    return calculator.compute_sharpe_ratio(annual_return, annual_volatility, passport)

def compute_sortino_ratio(
    portfolio_returns: np.ndarray,
    risk_free_rate: float = 0.02,
    passport: Optional[DataPassport] = None
) -> float:
    """
    Module-level wrapper for Sortino ratio computation.
    
    Calls MetricsComputation.compute_sortino_ratio() internally.
    No code duplication - delegates to class method.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Daily portfolio returns time series
    risk_free_rate : float, default=0.02
        Annual risk-free rate
    passport : DataPassport, optional
        Data passport for lineage tracking
        
    Returns
    -------
    float
        Sortino ratio
    """
    calculator = get_default_metrics_calculator(risk_free_rate=risk_free_rate)
    return calculator.compute_sortino_ratio(portfolio_returns, passport)

def compute_returns(
    portfolio_returns: np.ndarray,
    passport: Optional[DataPassport] = None
) -> float:
    """
    Module-level wrapper for returns computation.
    
    Calls MetricsComputation.compute_returns() internally.
    No code duplication - delegates to class method.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Daily portfolio returns time series
    passport : DataPassport, optional
        Data passport for lineage tracking
        
    Returns
    -------
    float
        Annualized returns
    """
    calculator = get_default_metrics_calculator()
    return calculator.compute_returns(portfolio_returns, passport)

def compute_volatility(
    portfolio_returns: np.ndarray,
    passport: Optional[DataPassport] = None
) -> float:
    """
    Module-level wrapper for volatility computation.
    
    Calls MetricsComputation.compute_volatility() internally.
    No code duplication - delegates to class method.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Daily portfolio returns time series
    passport : DataPassport, optional
        Data passport for lineage tracking
        
    Returns
    -------
    float
        Annualized volatility
    """
    calculator = get_default_metrics_calculator()
    return calculator.compute_volatility(portfolio_returns, passport)

def compute_max_drawdown(
    portfolio_returns: np.ndarray,
    passport: Optional[DataPassport] = None
) -> Tuple[float, int, int]:
    """
    Module-level wrapper for maximum drawdown computation.
    
    Calls MetricsComputation.compute_max_drawdown() internally.
    No code duplication - delegates to class method.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Daily portfolio returns time series
    passport : DataPassport, optional
        Data passport for lineage tracking
        
    Returns
    -------
    Tuple[float, int, int]
        (max_drawdown, start_index, end_index)
    """
    calculator = get_default_metrics_calculator()
    return calculator.compute_max_drawdown(portfolio_returns, passport)
def compute_calmar_ratio(
    portfolio_returns: np.ndarray,
    risk_free_rate: float = 0.02,
    passport: Optional[DataPassport] = None
) -> float:
    """
    Compute Calmar ratio (annual return / max drawdown).
    
    Calmar ratio is a risk-adjusted performance metric similar to Sharpe ratio
    but uses maximum drawdown instead of volatility in the denominator.
    
    Higher values indicate better risk-adjusted returns.
    
    Parameters
    ----------
    portfolio_returns : np.ndarray
        Daily portfolio returns time series
    risk_free_rate : float, default=0.02
        Annual risk-free rate (not used in Calmar but kept for consistency)
    passport : DataPassport, optional
        Data passport for lineage tracking
    
    Returns
    -------
    float
        Calmar ratio (annual_return / abs(max_drawdown))
        Returns 0.0 if max_drawdown is near zero to avoid division issues
    
    Notes
    -----
    Calmar Ratio = Annual Return / Absolute Maximum Drawdown
    
    Example
    -------
    >>> returns = np.array([0.01, 0.02, -0.01, 0.015, 0.02])
    >>> calmar = compute_calmar_ratio(returns)
    >>> print(f"Calmar Ratio: {calmar:.4f}")
    """
    
    operation_start = datetime.now()
    
    try:
        # Validate input
        portfolio_returns = _validate_returns_array(portfolio_returns, "portfolio_returns")
        
        # Compute annual return
        annual_return = compute_returns(portfolio_returns)
        
        # Compute max drawdown
        max_dd, _, _ = compute_max_drawdown(portfolio_returns)
        max_drawdown = abs(max_dd)
        
        # Avoid division by zero
        if max_drawdown < MIN_VOLATILITY:
            calmar_ratio = 0.0
        else:
            calmar_ratio = annual_return / max_drawdown
        
        execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
        
        # Seal execution if passport available
        if passport is not None:
            passport = _module_orchestrator.seal(
                passport,
                function_name='compute_calmar_ratio',
                phase='Phase 1',
                result={
                    'execution_time_ms': execution_ms,
                    'validation_status': 'valid',
                    'transformation_type': 'ratio_computation',
                    'parameters': {'annual_return': annual_return, 'max_drawdown': max_drawdown},
                    'output': calmar_ratio
                }
            )
        
        return float(calmar_ratio)
        
    except Exception as e:
        _module_logger.error(f"compute_calmar_ratio: {str(e)}")
        execution_ms = (datetime.now() - operation_start).total_seconds() * 1000
        
        if passport is not None:
            passport = _module_orchestrator.seal(
                passport,
                function_name='compute_calmar_ratio',
                phase='Phase 1',
                result={
                    'execution_time_ms': execution_ms,
                    'validation_status': 'error',
                    'errors': [str(e)]
                }
            )
        raise

def compute_all_metrics(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    risk_free_rate: float = 0.02,
    passport: Optional[DataPassport] = None
) -> Dict[str, Any]:
    """
    Module-level wrapper for computing all metrics at once.
    
    Calls MetricsComputation.compute_all_metrics() internally.
    No code duplication - delegates to class method.
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        Historical returns (dates x assets)
    weights : np.ndarray
        Portfolio weights
    risk_free_rate : float, default=0.02
        Annual risk-free rate
    passport : DataPassport, optional
        Data passport for lineage tracking
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with all computed metrics
    """
    calculator = get_default_metrics_calculator(risk_free_rate=risk_free_rate)
    return calculator.compute_all_metrics(returns_df, weights, passport)

def validate_metrics(
    metrics_dict: Dict[str, Any],
    passport: Optional[DataPassport] = None
) -> Dict[str, Any]:
    """
    Module-level wrapper for metrics validation.
    
    Calls MetricsComputation.validate_metrics() internally.
    No code duplication - delegates to class method.
    
    Parameters
    ----------
    metrics_dict : Dict[str, Any]
        Dictionary of computed metrics
    passport : DataPassport, optional
        Data passport for lineage tracking
        
    Returns
    -------
    Dict[str, Any]
        Validation results
    """
    calculator = get_default_metrics_calculator()
    return calculator.validate_metrics(metrics_dict, passport)

# ============================================================================
# END OF MODULE-LEVEL WRAPPER FUNCTIONS
# ============================================================================

# ============================================================================
# MAIN EXECUTION
