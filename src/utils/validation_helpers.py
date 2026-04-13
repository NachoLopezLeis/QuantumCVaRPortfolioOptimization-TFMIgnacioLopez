#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : validation_helpers.py  (Utils - Validation Helpers)
# ============================================================================
#
# Description
# -----------
# Centralized validation functions across four levels:
#   L1: Type validation (correct types).
#   L2: Shape validation (correct dimensions).
#   L3: Value validation (correct ranges).
#   L4: Data quality (NaN, Inf detection).
# Plus convenient combinations: validate_returns_matrix,
# validate_weights_vector, validate_alpha.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

"""
Centralized validation system for all TFM modules.

Provides reusable validators across 4 levels:
1. Type validation
2. Shape validation
3. Value validation
4. Data quality validation

Plus convenient combinations for portfolios, weights, and risk parameters.
"""

import numpy as np
from typing import Union, Tuple, Any, Optional
from numpy.typing import NDArray

# ============================================================================
# CUSTOM EXCEPTION
# ============================================================================

class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass

# ============================================================================
# LEVEL 1: TYPE VALIDATION
# ============================================================================

def validate_type(
    value: Any,
    expected_types: Union[type, Tuple[type, ...]],
    paramname: str
) -> None:
    """
    Validate that value is of expected type(s).
    
    Parameters
    ----------
    value : Any
        Value to validate
    expected_types : type or tuple of types
        Expected type(s)
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    TypeError
        If value is not of expected type
    
    Examples
    --------
    >>> validate_type(5, int, "n_iterations")  # OK
    >>> validate_type([1, 2, 3], np.ndarray, "data")  # Raises TypeError
    """
    if not isinstance(value, expected_types):
        if isinstance(expected_types, tuple):
            type_names = " or ".join(t.__name__ for t in expected_types)
        else:
            type_names = expected_types.__name__
        raise TypeError(
            f"{paramname} must be {type_names}, got {type(value).__name__}"
        )

def validate_numpy_array(
    array: Any,
    paramname: str,
    dtype: Union[type, Tuple[type, ...]] = (np.floating, np.integer)
) -> None:
    """
    Validate that value is numpy array with expected dtype.
    
    Parameters
    ----------
    array : Any
        Value to validate
    paramname : str
        Parameter name for error messages
    dtype : type or tuple, default=(np.floating, np.integer)
        Expected dtype(s)
    
    Raises
    ------
    TypeError
        If not numpy array
    ValueError
        If dtype doesn't match
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> validate_numpy_array(data, "returns")  # OK
    >>> validate_numpy_array([1, 2, 3], "returns")  # Raises TypeError
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"{paramname} must be numpy array, got {type(array).__name__}"
        )
    
    if not np.issubdtype(array.dtype, dtype):
        raise ValueError(
            f"{paramname} dtype must be numeric, got {array.dtype}"
        )

# ============================================================================
# LEVEL 2: SHAPE VALIDATION
# ============================================================================

def validate_ndim(
    array: NDArray,
    expected_ndim: int,
    paramname: str
) -> None:
    """
    Validate that array has exactly expected number of dimensions.
    
    Parameters
    ----------
    array : NDArray
        Array to validate
    expected_ndim : int
        Expected number of dimensions
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    ValueError
        If dimensions don't match
    
    Examples
    --------
    >>> data = np.array([[1, 2], [3, 4]])
    >>> validate_ndim(data, 2, "returns")  # OK
    >>> validate_ndim(data, 1, "returns")  # Raises ValueError
    """
    if array.ndim != expected_ndim:
        raise ValueError(
            f"{paramname} must be {expected_ndim}D array, "
            f"got {array.ndim}D with shape {array.shape}"
        )

def validate_shape(
    array: NDArray,
    expected_shape: Tuple[int, ...],
    paramname: str,
    allow_broadcast: bool = False
) -> None:
    """
    Validate that array has expected shape.
    
    Use -1 for any size in that dimension.
    
    Parameters
    ----------
    array : NDArray
        Array to validate
    expected_shape : tuple of int
        Expected shape. Use -1 for flexible dimension.
        Example: (-1, 50) means any rows, exactly 50 columns
    paramname : str
        Parameter name for error messages
    allow_broadcast : bool, default=False
        If True, allow shapes that can broadcast
    
    Raises
    ------
    ValueError
        If shape doesn't match
    
    Examples
    --------
    >>> data = np.array([[1, 2, 3], [4, 5, 6]])
    >>> validate_shape(data, (-1, 3), "returns")  # OK
    >>> validate_shape(data, (2, -1), "returns")  # OK
    >>> validate_shape(data, (2, 4), "returns")  # Raises ValueError
    """
    if array.ndim != len(expected_shape):
        raise ValueError(
            f"{paramname} must be {len(expected_shape)}D, "
            f"got {array.ndim}D with shape {array.shape}"
        )
    
    for i, (actual, expected) in enumerate(zip(array.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{paramname} dimension {i} must be {expected}, "
                f"got {actual}. Full shape {array.shape}, expected {expected_shape}"
            )

def validate_shape_match(
    array1: NDArray,
    array2: NDArray,
    dim: int,
    paramname1: str,
    paramname2: str
) -> None:
    """
    Validate that two arrays have same size in specified dimension.
    
    Parameters
    ----------
    array1, array2 : NDArray
        Arrays to compare
    dim : int
        Dimension to compare (0, 1, 2, ...)
    paramname1, paramname2 : str
        Parameter names for error messages
    
    Raises
    ------
    ValueError
        If sizes don't match
    
    Examples
    --------
    >>> returns = np.random.randn(252, 5)
    >>> weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> validate_shape_match(returns, weights, 1, "returns", "weights")  # OK
    """
    if array1.shape[dim] != array2.shape[dim]:
        raise ValueError(
            f"{paramname1} dimension {dim} ({array1.shape[dim]}) "
            f"doesn't match {paramname2} dimension {dim} ({array2.shape[dim]})"
        )

def validate_min_size(
    array: NDArray,
    minsize: int,
    paramname: str,
    dim: int = 0
) -> None:
    """
    Validate that array has at least minsize elements in dimension.
    
    Parameters
    ----------
    array : NDArray
        Array to validate
    minsize : int
        Minimum size required
    paramname : str
        Parameter name for error messages
    dim : int, default=0
        Dimension to check
    
    Raises
    ------
    ValueError
        If size is too small
    
    Examples
    --------
    >>> data = np.random.randn(252, 50)
    >>> validate_min_size(data, 2, "returns", dim=0)  # OK (252 >= 2)
    >>> validate_min_size(data, 100, "returns", dim=0)  # Raises ValueError
    """
    if array.shape[dim] < minsize:
        raise ValueError(
            f"{paramname} dimension {dim} must have at least {minsize} elements, "
            f"got {array.shape[dim]}. Shape: {array.shape}"
        )

# ============================================================================
# LEVEL 3: VALUE VALIDATION
# ============================================================================

def validate_range(
    value: Union[int, float],
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
    paramname: str = "value",
    inclusive_min: bool = True,
    inclusive_max: bool = True
) -> None:
    """
    Validate that scalar value is within range.
    
    Parameters
    ----------
    value : int or float
        Value to validate
    minval : float, optional
        Minimum value (None = no lower bound)
    maxval : float, optional
        Maximum value (None = no upper bound)
    paramname : str
        Parameter name for error messages
    inclusive_min : bool, default=True
        If True, minval is inclusive (>=); if False, exclusive (>)
    inclusive_max : bool, default=True
        If True, maxval is inclusive (<=); if False, exclusive (<)
    
    Raises
    ------
    ValueError
        If value is outside range
    
    Examples
    --------
    >>> validate_range(0.05, 0, 0.5, "alpha")  # OK
    >>> validate_range(0.6, 0, 0.5, "alpha")  # Raises ValueError
    >>> validate_range(5, 1, 10, "count", inclusive_min=False)  # OK (5 > 1)
    """
    error_parts = []
    
    if minval is not None:
        if inclusive_min:
            if value < minval:
                error_parts.append(f">= {minval}")
        else:
            if value <= minval:
                error_parts.append(f"> {minval}")
    
    if maxval is not None:
        if inclusive_max:
            if value > maxval:
                error_parts.append(f"<= {maxval}")
        else:
            if value >= maxval:
                error_parts.append(f"< {maxval}")
    
    if error_parts:
        range_desc = " and ".join(error_parts)
        raise ValueError(
            f"{paramname} must be {range_desc}, got {value}"
        )

def validate_array_range(
    array: NDArray,
    minval: Optional[float] = None,
    maxval: Optional[float] = None,
    paramname: str = "array",
    allow_equal: bool = True
) -> None:
    """
    Validate that ALL elements of array are within range.
    
    Parameters
    ----------
    array : NDArray
        Array to validate
    minval, maxval : float, optional
        Range bounds
    paramname : str
        Parameter name for error messages
    allow_equal : bool, default=True
        If True, allow values equal to bounds
    
    Raises
    ------
    ValueError
        If any elements are outside range
    
    Examples
    --------
    >>> weights = np.array([0.1, 0.2, 0.3, 0.4])
    >>> validate_array_range(weights, 0, 1, "weights")  # OK
    >>> weights_bad = np.array([0.1, -0.2, 0.3, 0.8])
    >>> validate_array_range(weights_bad, 0, 1, "weights")  # Raises ValueError
    """
    if minval is not None:
        if allow_equal:
            violations = np.where(array < minval)
        else:
            violations = np.where(array <= minval)
        
        if len(violations[0]) > 0:
            raise ValueError(
                f"{paramname} {len(violations[0])} values are < {minval}. "
                f"Min value: {np.min(array):.6e}. "
                f"Examples: {array[violations][:3]}"
            )
    
    if maxval is not None:
        if allow_equal:
            violations = np.where(array > maxval)
        else:
            violations = np.where(array >= maxval)
        
        if len(violations[0]) > 0:
            raise ValueError(
                f"{paramname} {len(violations[0])} values are > {maxval}. "
                f"Max value: {np.max(array):.6e}. "
                f"Examples: {array[violations][:3]}"
            )

def validate_positive(
    value: Union[int, float],
    paramname: str,
    strict: bool = False
) -> None:
    """
    Validate that value is positive.
    
    Parameters
    ----------
    value : int or float
        Value to validate
    paramname : str
        Parameter name for error messages
    strict : bool, default=False
        If True, require > 0; if False, require >= 0
    
    Raises
    ------
    ValueError
        If value is not positive
    
    Examples
    --------
    >>> validate_positive(5, "count")  # OK
    >>> validate_positive(0, "count", strict=True)  # Raises ValueError
    >>> validate_positive(0, "count", strict=False)  # OK
    """
    if strict:
        if value <= 0:
            raise ValueError(f"{paramname} must be > 0, got {value}")
    else:
        if value < 0:
            raise ValueError(f"{paramname} must be >= 0, got {value}")

def validate_sum_to_one(
    array: NDArray,
    paramname: str,
    atol: float = 1e-6
) -> None:
    """
    Validate that array sums to 1 (useful for weights/probabilities).
    
    Parameters
    ----------
    array : NDArray
        Array to validate (typically 1D)
    paramname : str
        Parameter name for error messages
    atol : float, default=1e-6
        Absolute tolerance for comparison
    
    Raises
    ------
    ValueError
        If sum is not approximately 1
    
    Examples
    --------
    >>> weights = np.array([0.2, 0.3, 0.5])
    >>> validate_sum_to_one(weights, "weights")  # OK
    >>> weights_bad = np.array([0.2, 0.3, 0.4])
    >>> validate_sum_to_one(weights_bad, "weights")  # Raises ValueError
    """
    total = np.sum(array)
    if not np.isclose(total, 1.0, atol=atol):
        raise ValueError(
            f"{paramname} must sum to 1.0, got {total:.10f}. "
            f"Error: {abs(total - 1.0):.2e}. "
            f"Min: {np.min(array):.6e}, Max: {np.max(array):.6e}"
        )

# ============================================================================
# LEVEL 4: DATA QUALITY VALIDATION
# ============================================================================

def validate_no_nan(
    array: NDArray,
    paramname: str
) -> None:
    """
    Validate that array contains NO NaN values.
    
    Parameters
    ----------
    array : NDArray
        Array to validate
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    ValueError
        If NaN values found
    
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> validate_no_nan(data, "returns")  # OK
    >>> data_bad = np.array([1.0, np.nan, 3.0])
    >>> validate_no_nan(data_bad, "returns")  # Raises ValueError
    """
    nan_count = np.sum(np.isnan(array))
    if nan_count > 0:
        nan_pct = 100 * nan_count / array.size
        raise ValueError(
            f"{paramname} contains {nan_count} NaN values "
            f"({nan_pct:.2f}%). Indices: {np.where(np.isnan(array))[0][:10]}"
        )

def validate_no_inf(
    array: NDArray,
    paramname: str
) -> None:
    """
    Validate that array contains NO infinite values.
    
    Parameters
    ----------
    array : NDArray
        Array to validate
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    ValueError
        If infinite values found
    
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> validate_no_inf(data, "returns")  # OK
    >>> data_bad = np.array([1.0, np.inf, 3.0])
    >>> validate_no_inf(data_bad, "returns")  # Raises ValueError
    """
    inf_count = np.sum(np.isinf(array))
    if inf_count > 0:
        pos_inf = np.sum(np.isposinf(array))
        neg_inf = np.sum(np.isneginf(array))
        raise ValueError(
            f"{paramname} contains {inf_count} infinite values: "
            f"+Inf: {pos_inf}, -Inf: {neg_inf}"
        )

def validate_finite(
    array: NDArray,
    paramname: str
) -> None:
    """
    Validate that ALL values are finite (not NaN, not Inf).
    
    Parameters
    ----------
    array : NDArray
        Array to validate
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    ValueError
        If non-finite values found
    
    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> validate_finite(data, "returns")  # OK
    >>> data_bad = np.array([1.0, np.nan, np.inf])
    >>> validate_finite(data_bad, "returns")  # Raises ValueError
    """
    non_finite = ~np.isfinite(array)
    if np.any(non_finite):
        nan_count = np.sum(np.isnan(array))
        inf_count = np.sum(np.isinf(array))
        raise ValueError(
            f"{paramname} contains non-finite values: "
            f"{nan_count} NaN, {inf_count} Inf. "
            f"Total: {np.sum(non_finite)} of {array.size} elements"
        )

# ============================================================================
# CONVENIENT COMBINATIONS
# ============================================================================

def validate_returns_matrix(
    returns: NDArray,
    paramname: str = "returns"
) -> None:
    """
    Comprehensive validation for returns matrix.
    
    Validates:
    - Must be 2D numpy array
    - Must be float dtype
    - No NaN or Inf
    - At least 2 periods
    - Typical range [-1, 1]
    
    Parameters
    ----------
    returns : NDArray
        Returns matrix to validate (shape: n_periods, n_assets)
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    TypeError, ValueError
        If validation fails
    
    Examples
    --------
    >>> returns = np.random.randn(252, 50) * 0.02
    >>> validate_returns_matrix(returns, "returns")  # OK
    """
    # Type check
    validate_numpy_array(returns, paramname, dtype=np.floating)
    
    # Shape checks
    validate_ndim(returns, 2, paramname)
    validate_min_size(returns, 2, paramname, dim=0)
    
    # Data quality
    validate_finite(returns, paramname)
    
    # Range check (warning bounds)
    validate_array_range(returns, -1.0, 1.0, paramname)

def validate_weights_vector(
    weights: NDArray,
    expected_size: Optional[int] = None,
    paramname: str = "weights"
) -> None:
    """
    Comprehensive validation for portfolio weights.
    
    Validates:
    - Must be 1D numpy array
    - Must sum to 1
    - All weights >= 0
    - All weights <= 1
    - No NaN values
    - Optional: size check
    
    Parameters
    ----------
    weights : NDArray
        Weight vector to validate
    expected_size : int, optional
        Expected number of weights (None = don't check)
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    TypeError, ValueError
        If validation fails
    
    Examples
    --------
    >>> weights = np.array([0.2, 0.3, 0.5])
    >>> validate_weights_vector(weights, expected_size=3, paramname="weights")  # OK
    """
    # Type check
    validate_numpy_array(weights, paramname, dtype=np.floating)
    
    # Shape checks
    validate_ndim(weights, 1, paramname)
    if expected_size is not None:
        validate_shape(weights, (expected_size,), paramname)
    
    # Sum to 1
    validate_sum_to_one(weights, paramname)
    
    # Range checks
    validate_array_range(weights, 0.0, 1.0, paramname)
    
    # Data quality
    validate_no_nan(weights, paramname)

def validate_alpha(
    alpha: float,
    paramname: str = "alpha"
) -> None:
    """
    Comprehensive validation for CVaR confidence level alpha.
    
    Validates:
    - Must be float
    - Must be in range (0, 0.5)
    
    Parameters
    ----------
    alpha : float
        Confidence level to validate
    paramname : str
        Parameter name for error messages
    
    Raises
    ------
    TypeError, ValueError
        If validation fails
    
    Examples
    --------
    >>> validate_alpha(0.05, "alpha")  # OK
    >>> validate_alpha(0.6, "alpha")  # Raises ValueError
    """
    # Type check
    validate_type(alpha, (int, float), paramname)
    
    # Range check (exclusive bounds)
    validate_range(
        alpha, 0.0, 0.5, paramname,
        inclusive_min=False, inclusive_max=False
    )
