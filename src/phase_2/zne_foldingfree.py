# ============================================================================
# MODULE: zne_foldingfree.py
# PURPOSE: Noise-scaling primitives for the ZNE comparative analysis (Sec. 7.2)
# PHASE:   Phase 2
# ============================================================================
#
# Implements three digital noise-scaling strategies plus four extrapolators,
# with NO external dependency beyond Qiskit (no mitiq). Each strategy maps a
# circuit U and a scale factor lambda (odd integer >= 1) to a logically
# equivalent circuit whose physical gate count grows ~ lambda * depth_0.
#
# Strategies
# ----------
#   global_fold        Giurgica-Tiron 2020, Eq. (1)/(3):  U -> U (U^dag U)^n
#                      (the BASELINE that fails on IQM at opt_level=0; kept here
#                       so the comparison script is self-contained).
#   gate_fold          Giurgica-Tiron 2020, Eq. (5): local/layer folding,
#                      L_j -> L_j (L_j^dag L_j)^n applied in place. Uses the
#                      *inverse* of each gate (works for any gate).
#   identity_insertion He, Nachman, de Jong, Bauer, Phys. Rev. A 102, 012426
#                      (2020). FIIM/RIIM: amplify ONLY self-inverse two-qubit
#                      gates (cx, cz, ...) by repetition G -> G (G G)^n. Truly
#                      inversion-free: G^-1 = G, so no inverse is ever built.
#
# For every strategy with lambda = 1 + 2n the physical gate count of the
# targeted gates is multiplied by (1 + 2n); on a noiseless backend the logical
# action is unchanged (U^dag U = I, or G G = I for self-inverse G).
#
# Extrapolators (estimate the lambda -> 0 / noise-free value)
# -----------------------------------------------------------
#   richardson      exact polynomial interpolation at lambda = 0 (Vandermonde).
#   linear          first-order fit a + b*lambda.
#   poly2           second-order fit a + b*lambda + c*lambda^2.
#   exponential     a + b*p^lambda  (Giurgica-Tiron 2020, Eq. (9); the
#                   depolarising-justified ansatz used by the aer noise model).
#
# Author: Ignacio Lopez Leis (TFM)
# ============================================================================

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction
    QISKIT_AVAILABLE = True
except ImportError:                                            # pragma: no cover
    QISKIT_AVAILABLE = False
    QuantumCircuit = object  # type: ignore

try:
    from scipy.optimize import curve_fit
    SCIPY_AVAILABLE = True
except ImportError:                                            # pragma: no cover
    SCIPY_AVAILABLE = False


# Two-qubit gates that are their own inverse (G G = I). FIIM/RIIM target these.
SELF_INVERSE_2Q = {"cx", "cz", "cy", "swap", "ecr", "dcx", "iswap"}
# Instructions that must never be folded.
NON_FOLDABLE = {"measure", "barrier", "reset", "delay", "snapshot", "initialize"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _check_scale(scale_factor: float) -> int:
    """Validate an odd-integer scale factor and return the fold count n.

    lambda = 1 + 2n  =>  n = (lambda - 1) / 2.
    Only odd integers (1, 3, 5, ...) are supported here, matching the
    {1, 3, 5} schedule of Sec. 7.2. Non-integer lambda would require the
    partial folding of Giurgica-Tiron 2020, Eq. (4), which is out of scope.
    """
    if scale_factor < 1.0:
        raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
    n_float = (scale_factor - 1.0) / 2.0
    n = int(round(n_float))
    if abs(n_float - n) > 1e-9:
        raise ValueError(
            f"scale_factor must be an odd integer (1, 3, 5, ...); got {scale_factor}. "
            "Non-integer factors need partial folding (Giurgica-Tiron 2020, Eq. 4)."
        )
    return n


def _require_qiskit() -> None:
    if not QISKIT_AVAILABLE:
        raise RuntimeError("Qiskit is required for ZNE folding but is not installed.")


# ----------------------------------------------------------------------------
# Noise-scaling strategies
# ----------------------------------------------------------------------------

def global_fold(circuit: "QuantumCircuit", scale_factor: float) -> "QuantumCircuit":
    """Global unitary folding (BASELINE), Giurgica-Tiron 2020, Eq. (1)/(3).

        U -> U (U^dag U)^n,   lambda = 1 + 2n.

    The returned circuit is NOT transpiled, so the U^dag U structure is
    preserved for the comparison; the caller controls optimization_level.
    """
    _require_qiskit()
    n = _check_scale(scale_factor)
    if n == 0:
        return circuit.copy()

    body, meas = _split_measurements(circuit)
    folded = body.copy()
    inv = body.inverse()
    for _ in range(n):
        folded.compose(inv,  inplace=True)
        folded.compose(body, inplace=True)
    return _reattach_measurements(folded, meas, circuit)


def gate_fold(
    circuit: "QuantumCircuit",
    scale_factor: float,
    method: str = "random",
    subset_fraction: float = 1.0,
    seed: Optional[int] = None,
) -> "QuantumCircuit":
    """Local / layer folding, Giurgica-Tiron 2020, Eq. (5).

        L_j -> L_j (L_j^dag L_j)^n   (in place, distributed along the circuit)

    Parameters
    ----------
    method : {'random', 'left', 'right'}
        How to choose which gates receive the fold (Table I of the paper).
    subset_fraction : float in [0, 1]
        Fraction of foldable gates to fold. 1.0 reproduces the exact odd-lambda
        scaling where every gate is folded n times.
    seed : int, optional
        Seed for the 'random' subset selection (reproducibility).
    """
    _require_qiskit()
    n = _check_scale(scale_factor)
    if n == 0:
        return circuit.copy()

    body, meas = _split_measurements(circuit)
    foldable_idx = [
        i for i, ci in enumerate(body.data)
        if ci.operation.name not in NON_FOLDABLE
    ]
    n_fold = int(round(subset_fraction * len(foldable_idx)))
    chosen = _select_subset(foldable_idx, n_fold, method, seed)

    out = QuantumCircuit(*body.qregs, *body.cregs, name=f"{body.name}_gatefold_l{int(scale_factor)}")
    for i, ci in enumerate(body.data):
        out.append(ci.operation, ci.qubits, ci.clbits)
        if i in chosen:
            inv = ci.operation.inverse()
            for _ in range(n):
                out.append(inv,           ci.qubits, ci.clbits)
                out.append(ci.operation,  ci.qubits, ci.clbits)
    return _reattach_measurements(out, meas, circuit)


def identity_insertion(
    circuit: "QuantumCircuit",
    scale_factor: float,
    method: str = "fixed",
    seed: Optional[int] = None,
) -> "QuantumCircuit":
    """Identity insertion on self-inverse 2q gates, He et al. PRA 102, 012426.

        G -> G (G G)^n   for self-inverse G (G^-1 = G)  =>  inversion-free.

    Parameters
    ----------
    method : {'fixed', 'random'}
        'fixed'  -> FIIM: every targeted 2q gate folded n times (cost 3*n_cx
                    at lambda=3); deterministic.
        'random' -> RIIM: a random subset of targeted gates folded, sized so
                    the *expected* amplification matches lambda (fewer gates,
                    higher shot variance). Uses `seed`.

    Only self-inverse two-qubit gates (cx, cz, ...) are amplified, mirroring
    the paper's focus on two-qubit-gate noise (the dominant error channel).
    """
    _require_qiskit()
    n = _check_scale(scale_factor)
    if n == 0:
        return circuit.copy()

    body, meas = _split_measurements(circuit)
    target_idx = [
        i for i, ci in enumerate(body.data)
        if ci.operation.name in SELF_INVERSE_2Q
    ]
    if not target_idx:
        raise ValueError(
            "identity_insertion found no self-inverse 2q gates (cx, cz, ...). "
            "Transpile to a basis that contains cx/cz, or use gate_fold instead."
        )

    if method == "fixed":                       # FIIM
        chosen = set(target_idx)
        reps = n                                # each chosen gate -> 2n extra copies
    elif method == "random":                    # RIIM
        # expected extra copies = 2 * reps * |chosen|; size the subset so the
        # average gate is amplified by lambda across the circuit.
        n_choose = max(1, int(round(len(target_idx) * (scale_factor - 1) /
                                    (2.0 * max(1, n)))))
        chosen = set(_select_subset(target_idx, n_choose, "random", seed))
        reps = n
    else:
        raise ValueError(f"method must be 'fixed' or 'random', got {method!r}")

    out = QuantumCircuit(*body.qregs, *body.cregs,
                         name=f"{body.name}_iim_{method}_l{int(scale_factor)}")
    for i, ci in enumerate(body.data):
        out.append(ci.operation, ci.qubits, ci.clbits)
        if i in chosen:
            for _ in range(reps):
                # G G = I for self-inverse G: insert the gate twice, no inverse.
                out.append(ci.operation, ci.qubits, ci.clbits)
                out.append(ci.operation, ci.qubits, ci.clbits)
    return _reattach_measurements(out, meas, circuit)


# ----------------------------------------------------------------------------
# Circuit-surgery helpers
# ----------------------------------------------------------------------------

def _split_measurements(circuit: "QuantumCircuit") -> Tuple["QuantumCircuit", list]:
    """Return (body_without_terminal_measurements, list_of_measure_instructions)."""
    meas = [ci for ci in circuit.data if ci.operation.name in ("measure", "barrier")]
    body = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
    for ci in circuit.data:
        if ci.operation.name not in ("measure", "barrier"):
            body.append(ci.operation, ci.qubits, ci.clbits)
    return body, meas


def _reattach_measurements(folded: "QuantumCircuit", meas: list,
                           original: "QuantumCircuit") -> "QuantumCircuit":
    for ci in meas:
        folded.append(ci.operation, ci.qubits, ci.clbits)
    return folded


def _select_subset(indices: Sequence[int], k: int, method: str,
                   seed: Optional[int]) -> set:
    k = max(0, min(k, len(indices)))
    if method == "left":
        return set(indices[:k])
    if method == "right":
        return set(indices[-k:]) if k else set()
    if method == "random":
        rng = np.random.default_rng(seed)
        return set(rng.choice(indices, size=k, replace=False).tolist()) if k else set()
    raise ValueError(f"method must be 'random', 'left' or 'right', got {method!r}")


# ----------------------------------------------------------------------------
# Extrapolators  (x = scale factors lambda; y = measured observable, e.g. p_hw)
# ----------------------------------------------------------------------------

def richardson_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    """Exact polynomial interpolation evaluated at lambda = 0 (Vandermonde).

    Solves sum_j c_j lambda_j^k = delta_{k,0} for k = 0..n-1 and returns
    sum_j c_j y_j. For lambda = {1, 3, 5} the coefficients are
    c = (1.875, -1.25, 0.375).
    """
    s = np.asarray(scales, float)
    y = np.asarray(values, float)
    if len(s) < 2:
        return float(y[0])
    A = np.vander(s, increasing=True).T          # rows k = 0..n-1
    b = np.zeros(len(s)); b[0] = 1.0
    try:
        coeffs = np.linalg.solve(A, b)
        return float(coeffs @ y)
    except np.linalg.LinAlgError:
        return linear_extrapolate(s, y)


def linear_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    """First-order fit a + b*lambda, evaluated at lambda = 0."""
    s = np.asarray(scales, float); y = np.asarray(values, float)
    if len(s) < 2:
        return float(y[0])
    return float(np.polyval(np.polyfit(s, y, 1), 0.0))


def poly2_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    """Second-order fit a + b*lambda + c*lambda^2, evaluated at lambda = 0."""
    s = np.asarray(scales, float); y = np.asarray(values, float)
    if len(s) < 3:
        return linear_extrapolate(s, y)
    return float(np.polyval(np.polyfit(s, y, 2), 0.0))


def exponential_extrapolate(scales: Sequence[float], values: Sequence[float]) -> float:
    """Depolarising-justified fit a + b*p^lambda (Giurgica-Tiron 2020, Eq. 9).

    Falls back to Richardson if SciPy is unavailable or the fit fails.
    """
    s = np.asarray(scales, float); y = np.asarray(values, float)
    if len(s) < 3 or not SCIPY_AVAILABLE:
        return richardson_extrapolate(s, y)

    def model(lam, a, b, p):
        return a + b * np.power(p, lam)

    a0 = float(y[-1])
    b0 = float(y[0] - y[-1])
    try:
        popt, _ = curve_fit(
            model, s, y, p0=[a0, b0, 0.5],
            bounds=([-1.0, -2.0, 1e-3], [1.0, 2.0, 1.0 - 1e-3]),
            maxfev=10000,
        )
        return float(model(0.0, *popt))
    except Exception:
        return richardson_extrapolate(s, y)


EXTRAPOLATORS = {
    "richardson":  richardson_extrapolate,
    "linear":      linear_extrapolate,
    "poly2":       poly2_extrapolate,
    "exponential": exponential_extrapolate,
}


def extrapolate(scales: Sequence[float], values: Sequence[float],
                method: str = "richardson") -> float:
    """Dispatch to a named extrapolator with a physical clip to [0, 1]."""
    if method not in EXTRAPOLATORS:
        raise ValueError(f"Unknown extrapolation method {method!r}; "
                         f"valid: {list(EXTRAPOLATORS)}")
    val = EXTRAPOLATORS[method](scales, values)
    return float(np.clip(val, 0.0, 1.0))
