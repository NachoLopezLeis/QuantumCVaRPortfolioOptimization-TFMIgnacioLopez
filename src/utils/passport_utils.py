#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : passport_utils.py  (Utils - Passport Decorators & Helpers)
# ============================================================================
#
# Description
# -----------
# @seal_function decorator and track_execution context manager for
# automatic passport sealing. Handles timing, hashing, logging,
# and error tracking automatically to reduce boilerplate.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

"""
Helper utilities for passport sealing and tracking.

Usage:
    from src.utils.passport_utils import seal_function
    
    @seal_function(
        function_name='compute_cvar',
        phase='Phase 1',
        transformation_type='risk_aggregation'
    )
    def compute_cvar(passport, returns, weights, alpha=0.05):
        # ... your computation ...
        return cvar, {
            'execution_time_ms': 234.5,
            'validation_status': 'valid',
            'warnings': [],
            'errors': [],
            'parameters': {'alpha': alpha},
            'data_shape': returns.shape,
        }
"""

import time
import hashlib
import json
from functools import wraps
from typing import Callable, Any, Dict, Optional, Tuple
from contextlib import contextmanager

try:
    from .passport_orchestrator import get_orchestrator
    from .logger import get_logger
except ImportError:
    from passport_orchestrator import get_orchestrator
    from logger import get_logger

def seal_function(
    function_name: Optional[str] = None,
    phase: str = 'Phase 1',
    transformation_type: str = 'unknown'
):
    """
    Decorator that automatically seals passport after function execution.
    
    Parameters
    ----------
    function_name : str, optional
        Function name for logging (defaults to actual function name)
    phase : str, default='Phase 1'
        Phase name ('Phase 1' or 'Phase 2')
    transformation_type : str, default='unknown'
        Type of transformation (e.g., 'risk_aggregation', 'data_loading')
    
    Returns
    -------
    Callable
        Decorated function
    
    Examples
    --------
    >>> @seal_function('compute_cvar', phase='Phase 1', transformation_type='risk_aggregation')
    ... def compute_cvar(passport, returns, weights, alpha=0.05):
    ...     portfolio_returns = returns @ weights
    ...     var_est = np.percentile(portfolio_returns, alpha * 100)
    ...     tail = portfolio_returns[portfolio_returns <= var_est]
    ...     cvar = np.mean(tail) if len(tail) > 0 else var_est
    ...     return cvar, {
    ...         'execution_time_ms': 234.5,
    ...         'validation_status': 'valid',
    ...         'warnings': [],
    ...     }
    
    >>> passport = get_orchestrator().assign_passport(returns, 'returns', 'yahoo')
    >>> cvar, _ = compute_cvar(passport, returns, weights)
    """
    
    def decorator(func: Callable) -> Callable:
        # Use function name if not provided
        _function_name = function_name or func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger
            logger = get_logger(_function_name)
            
            # Get orchestrator
            orchestrator = get_orchestrator()
            
            # Extract passport (first positional arg if present)
            passport = None
            if args and hasattr(args[0], 'passport_id'):
                passport = args[0]
            elif 'passport' in kwargs:
                passport = kwargs['passport']
            
            # Track execution
            start_time = time.time()
            _start_timestamp = start_time
            
            logger.debug(f"Starting {_function_name}...")
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Handle result tuple (value, metadata) or single value
                if isinstance(result, tuple) and len(result) == 2:
                    value, metadata = result
                else:
                    value = result
                    metadata = {}
                
                # Calculate execution time
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                
                # Ensure metadata has required fields
                if not isinstance(metadata, dict):
                    metadata = {}
                
                if 'execution_time_ms' not in metadata:
                    metadata['execution_time_ms'] = execution_time_ms
                
                if 'validation_status' not in metadata:
                    metadata['validation_status'] = 'valid'
                
                if 'warnings' not in metadata:
                    metadata['warnings'] = []
                
                if 'errors' not in metadata:
                    metadata['errors'] = []
                
                if 'transformation_type' not in metadata:
                    metadata['transformation_type'] = transformation_type
                
                if 'parameters' not in metadata:
                    metadata['parameters'] = {}
                
                # Compute input/output hashes
                try:
                    input_hash = _compute_hash(args, kwargs)
                    output_hash = _compute_hash(value)
                except Exception as e:
                    input_hash = f"error_{str(e)[:20]}"
                    output_hash = f"error_{str(e)[:20]}"
                
                metadata['input_hash'] = input_hash
                metadata['output_hash'] = output_hash
                
                # Get data shape if available
                if hasattr(value, 'shape'):
                    metadata['data_shape'] = value.shape
                
                # Seal passport
                if passport is not None and orchestrator.enable_tracking:
                    passport = orchestrator.seal(
                        passport,
                        function_name=_function_name,
                        phase=phase,
                        result=metadata
                    )
                
                logger.info(
                    f"OK {_function_name} completed ({execution_time_ms:.2f}ms) - "
                    f"status={metadata.get('validation_status', 'unknown')}"
                )
                
                # Return original result format
                if isinstance(result, tuple) and len(result) == 2:
                    return value, passport
                else:
                    return value
            
            except Exception as e:
                # Calculate execution time
                end_time = time.time()
                execution_time_ms = (end_time - start_time) * 1000
                
                # Log error
                logger.error(
                    f"FAIL {_function_name} failed ({execution_time_ms:.2f}ms): {str(e)}"
                )
                
                # Seal with error status
                if passport is not None and orchestrator.enable_tracking:
                    error_metadata = {
                        'execution_time_ms': execution_time_ms,
                        'validation_status': 'error',
                        'errors': [str(e)],
                        'warnings': [],
                        'transformation_type': transformation_type,
                        'parameters': {},
                    }
                    
                    passport = orchestrator.seal(
                        passport,
                        function_name=_function_name,
                        phase=phase,
                        result=error_metadata
                    )
                
                # Re-raise exception
                raise
        
        return wrapper
    
    return decorator

@contextmanager
def track_execution(
    function_name: str,
    phase: str = 'Phase 1',
    logger_name: Optional[str] = None
):
    """
    Context manager for tracking execution time and logging.
    
    Parameters
    ----------
    function_name : str
        Function name for logging
    phase : str, default='Phase 1'
        Phase name
    logger_name : str, optional
        Logger name (defaults to function_name)
    
    Yields
    ------
    dict
        Metadata dict with 'execution_time_ms' after context exits
    
    Examples
    --------
    >>> with track_execution('my_computation') as metadata:
    ...     result = expensive_computation()
    >>> print(f"Took {metadata['execution_time_ms']:.2f}ms")
    """
    
    logger = get_logger(logger_name or function_name)
    logger.debug(f"Starting {function_name}...")
    
    metadata = {}
    start_time = time.time()
    
    try:
        yield metadata
        
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        metadata['execution_time_ms'] = execution_time_ms
        metadata['validation_status'] = 'valid'
        
        logger.info(
            f"OK {function_name} completed ({execution_time_ms:.2f}ms)"
        )
    
    except Exception as e:
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        metadata['execution_time_ms'] = execution_time_ms
        metadata['validation_status'] = 'error'
        metadata['errors'] = [str(e)]
        
        logger.error(
            f"FAIL {function_name} failed ({execution_time_ms:.2f}ms): {str(e)}"
        )
        
        raise

# ============================================================================
# PRIVATE HELPERS
# ============================================================================

def _compute_hash(data: Any, additional_data: Optional[Dict] = None) -> str:
    """
    Compute hash of data for tracking.
    
    Parameters
    ----------
    data : Any
        Data to hash
    additional_data : dict, optional
        Additional data to include in hash
    
    Returns
    -------
    str
        MD5 hash of data
    """
    try:
        # Combine data
        combined = data
        if additional_data:
            combined = (data, additional_data)
        
        # Convert to JSON
        if hasattr(data, 'tobytes'):  # numpy array
            data_bytes = data.tobytes()
        else:
            try:
                data_bytes = json.dumps(combined, default=str, sort_keys=True).encode()
            except (TypeError, ValueError):
                data_bytes = str(combined).encode()
        
        return hashlib.md5(data_bytes).hexdigest()
    
    except Exception as e:
        return f"hash_error_{str(e)[:20]}"

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example of decorated function:

    @seal_function(
        function_name='load_data',
        phase='Phase 1',
        transformation_type='data_loading'
    )
    def load_data(passport, tickers: list):
        orchestrator = get_orchestrator()
        returns = get_returns_from_yahoo(tickers)
        
        # Create checkpoint
        passport = orchestrator.checkpoint(
            passport,
            name='after_load',
            data=returns,
            metadata={'tickers': len(tickers), 'shape': returns.shape}
        )
        
        return returns, {
            'validation_status': 'valid',
            'transformation_type': 'data_loading',
            'parameters': {'tickers': tickers},
            'data_shape': returns.shape,
        }

    # Usage:
    orchestrator = get_orchestrator()
    passport = orchestrator.assign_passport(
        data=None,
        data_type='returns',
        source='yahoo_finance'
    )
    returns, _ = load_data(passport, ['AAPL', 'MSFT', 'GOOGL'])

Example with context manager:

    with track_execution('my_function') as metadata:
        result = expensive_operation()
        
    print(f"Execution time: {metadata['execution_time_ms']:.2f}ms")
"""
