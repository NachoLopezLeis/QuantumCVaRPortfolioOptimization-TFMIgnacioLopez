#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 Module: qae_circuits.py
 QUANTUM AMPLITUDE ESTIMATION CIRCUITS FOR CVaR PORTFOLIO OPTIMIZATION
================================================================================

 Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
           Master's Thesis (TFM) - MSc in Quantum Computing
 Author  : Ignacio Lopez Leis
 Affil.  : Universidad Autonoma de Madrid (UAM)
 Date    : February 2026
 Version : 7.19

 Description
 -----------
 Implements the quantum circuits and QAE estimation pipeline for CVaR
 computation within the hybrid optimizer (Phase 2). Provides two QAE
 methods:
   - IQAE: Iterative Quantum Amplitude Estimation (optimal O(1/eps) queries).
   - Woerner: Canonical QPE-based QAE with adaptive evaluation qubits.

 Sampler assignment:
   - Woerner always runs on StatevectorSampler (ideal/noiseless), since
     AmplitudeEstimation builds QPE as a composite gate incompatible with
     AerSamplerV2. Woerner serves as a noiseless cross-validation reference.
   - IQAE runs on TranspilingAerSamplerV2 (noisy) when noise is enabled,
     or StatevectorSampler (ideal) otherwise.

 Multi-tier error mitigation stack:
   - TIER 1: Measurement Error Mitigation (MEM) via readout correction.
   - TIER 2: ZNE on state preparation distribution (noise-rate scaling).
   - TIER 3: ZNE on QAE amplitude via unitary folding (U -> U*U_dag*U).

 The IQAE three-way noise comparison (ideal / noisy-raw / noisy+ZNE+MEM)
 quantifies the NISQ gap and mitigation effectiveness.

 References
 ----------
 [1] Woerner, S. & Egger, D.J. (2019). "Quantum Risk Analysis."
     npj Quantum Information, 5, 15.
 [2] Brassard, G. et al. (2002). "Quantum Amplitude Amplification and
     Estimation." Contemporary Mathematics, 305, 53-74.
 [3] LaRose, R. et al. (2022). "Mitiq: A Software Package for Error
     Mitigation on Noisy Quantum Computers." Quantum, 6, 774.
 [4] Preskill, J. (2018). "Quantum Computing in the NISQ Era and Beyond."
     Quantum, 2, 79.
 [5] Rockafellar, R.T. & Uryasev, S. (2000). "Optimization of Conditional
     Value-at-Risk." Journal of Risk, 2(3), 21-41.
 [6] Grinko, D. et al. (2021). "Iterative Quantum Amplitude Estimation."
     npj Quantum Information, 7, 52.

 License
 -------
 Academic use only. This code is part of a Master's Thesis and is provided
 for academic and research purposes. For any non-academic or commercial use,
 please contact: nacholopezleis@gmail.com

================================================================================
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import time
import warnings
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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

# Optional scipy import (not used in v6.8 but kept for compatibility)
try:
    from scipy.optimize import minimize_scalar, brentq
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ==============================================================================
# CONFIGURATION CONSTANTS - v6.9
# ==============================================================================

DEFAULT_QAE_TIMEOUT = 120
MAX_WOERNER_EVAL_QUBITS = 6      # v7.10: Reduced from 8 â€” QPE with 2^8=256 Grover
                                  # iterations on StatevectorSampler is too slow.
                                  # 2^6=64 iterations gives Îµ â‰ˆ 0.025, sufficient for
                                  # cross-validation against IQAE.
MIN_WOERNER_EVAL_QUBITS = 4      # v7.10: Reduced from 5 â€” 2^4=16 iterations minimum
WOERNER_SAFETY_FACTOR = 1.0      # v7.10: Reduced from 2.0 â€” no safety inflation.
                                  # Woerner serves as noiseless reference; high
                                  # precision is not critical.
WOERNER_MAX_RETRIES = 2          # v6.9: retry with more qubits if error > threshold
WOERNER_RETRY_ERROR_THRESHOLD = 0.15  # v6.9: 15% relative error triggers retry
MAX_IQAE_TIMEOUT = 60
MAX_TOTAL_QUBITS = 16            # v6.9: INCREASED from 14 to support 8 eval qubits
ZNE_DEFAULT_SCALE_FACTORS = [1.0, 3.0, 5.0]


# ==============================================================================
# CUSTOM EXCEPTIONS
# ==============================================================================

class CircuitTooLargeError(Exception):
    """Raised when circuit exceeds maximum allowed qubits."""
    pass


class QAETimeoutError(Exception):
    """Raised when QAE estimation times out."""
    pass


class QAEExecutionError(Exception):
    """Raised when QAE execution fails (no fallback available)."""
    pass


# ==============================================================================
# LOGGING SYSTEM
# ==============================================================================

class QAELogger:
    """Enhanced logger with circuit tracking."""
    
    def __init__(self, log_dir: Union[str, Path] = None, log_name: str = 'qae_circuits.txt'):
        self.log_dir = Path(log_dir or './logs')
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_name
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.start_time = time.time()
        self._initialize_log()
    
    def _initialize_log(self):
        header = [
            "=" * 80,
            "QAE CIRCUITS - Quantum Amplitude Estimation for CVaR",
            "=" * 80,
            f"Session: {self.session_id}",
            f"Started: {datetime.now().isoformat()}",
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
        except Exception as _e:
            logger.warning(f'Non-critical error (suppressed): {_e}')
    
    def info(self, msg): self._write("INFO", msg)
    def warning(self, msg): self._write("WARNING", msg)
    def error(self, msg): self._write("ERROR", msg)
    def debug(self, msg): self._write("DEBUG", msg)
    
    def section(self, title: str):
        self._write("", "")
        self._write("", "â”€" * 70)
        self._write("", f"  {title}")
        self._write("", "â”€" * 70)
    
    def metric(self, name: str, value):
        if isinstance(value, float):
            self._write("METRIC", f"{name}: {value:.6f}")
        else:
            self._write("METRIC", f"{name}: {value}")


# Global logger
logger = QAELogger()


# ==============================================================================
# QISKIT IMPORTS
# ==============================================================================

QISKIT_AVAILABLE = False
QISKIT_ALGORITHMS_AVAILABLE = False
MATPLOTLIB_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
    from qiskit.circuit.library import GroverOperator, MCMT, MCXGate
    from qiskit_aer import AerSimulator
    # Qiskit >= 1.0 removed legacy Sampler from qiskit.primitives.
    # Try V2 StatevectorSampler first, then legacy Sampler.
    try:
        from qiskit.primitives import StatevectorSampler as Sampler
        logger.info("Qiskit V2 StatevectorSampler imported as Sampler")
    except ImportError:
        try:
            from qiskit.primitives import Sampler
            logger.info("Qiskit legacy Sampler imported")
        except ImportError:
            # Last resort: use AerSampler as ideal sampler (loaded later)
            Sampler = None
            logger.warning("No Sampler available from qiskit.primitives; "
                           "will use AerSampler as fallback")
    QISKIT_AVAILABLE = True
    logger.info("Qiskit core imported")
except ImportError as e:
    Sampler = None
    logger.error(f"Qiskit not available: {e}")

# v6.9: Noise model imports
NOISE_MODEL_AVAILABLE = False
AER_SAMPLER_AVAILABLE = False
try:
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    NOISE_MODEL_AVAILABLE = True
    logger.info("Qiskit Aer noise model imported")
except ImportError as e:
    logger.warning(f"Noise model not available: {e}")

try:
    from qiskit_aer.primitives import Sampler as AerSampler
    AER_SAMPLER_AVAILABLE = True
    logger.info("Qiskit Aer primitives (AerSampler) imported")
except ImportError as e:
    logger.warning(f"AerSampler not available: {e}")

# v7.1: BackendSampler wraps AerSimulator(noise_model=...) as a V2 primitive
# that is compatible with qiskit_algorithms.IterativeAmplitudeEstimation.
# AerSampler (V1) causes "job not completed successfully" errors with IQAE.
BACKEND_SAMPLER_AVAILABLE = False
BackendSampler = None
try:
    from qiskit.primitives import BackendSampler as _BackendSampler
    BackendSampler = _BackendSampler
    BACKEND_SAMPLER_AVAILABLE = True
    logger.info("BackendSampler imported for noisy V2 sampling")
except ImportError:
    logger.info("BackendSampler not available")

# v7.3: AerSamplerV2 â€” native Aer V2 sampler compatible with IQAE + noise.
# Accepts noise_model via options={"backend_options": {"noise_model": nm}}.
# Confirmed working with qiskit 2.2.3 + qiskit-aer.
AER_SAMPLER_V2_AVAILABLE = False
AerSamplerV2 = None
try:
    from qiskit_aer.primitives import SamplerV2 as _AerSamplerV2
    AerSamplerV2 = _AerSamplerV2
    AER_SAMPLER_V2_AVAILABLE = True
    logger.info("AerSamplerV2 imported â€” IQAE can run with real noise")
except ImportError:
    logger.info("AerSamplerV2 not available; IQAE will use ideal sampler only")

# v6.9: ZNE imports
ZNE_AVAILABLE = False
try:
    from mitiq import zne
    from mitiq.zne.scaling import fold_global
    from mitiq.zne.inference import RichardsonFactory, LinearFactory
    ZNE_AVAILABLE = True
    logger.info("Mitiq ZNE imported")
except ImportError:
    logger.warning("Mitiq not available - ZNE will use manual multi-scale approach")

try:
    from qiskit_algorithms import AmplitudeEstimation, EstimationProblem, IterativeAmplitudeEstimation
    QISKIT_ALGORITHMS_AVAILABLE = True
    logger.info("Qiskit algorithms imported")
except ImportError as e:
    logger.warning(f"Qiskit algorithms not available: {e}")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib imported for circuit visualization")
except ImportError:
    logger.error("Matplotlib not available - PNG export will fail!")


# ==============================================================================
# v7.8: TRANSPILING SAMPLER WRAPPER
#
# IterativeAmplitudeEstimation internally constructs circuits containing a
# GroverOperator as an opaque composite gate, then passes them directly to
# the sampler's run() method.  AerSamplerV2 uses AerSimulator, whose
# compiler cannot decompose GroverOperator â€” causing:
#   "The job was not completed successfully."
# This is the SAME root cause as Woerner's QPE incompatibility with Aer.
#
# StatevectorSampler works because it simulates via matrix multiplication
# without circuit compilation.
#
# SOLUTION: Wrap AerSamplerV2 in a shim that intercepts every run() call,
# transpiles all circuits to basis gates BEFORE forwarding them to Aer.
# This is transparent to IQAE â€” it sees a normal V2 Sampler interface.
# ==============================================================================

if AER_SAMPLER_V2_AVAILABLE and QISKIT_AVAILABLE:
    
    # ---- Probe the AerSamplerV2 constructor at import time ----
    import inspect as _inspect
    _aer_sv2_sig = _inspect.signature(_AerSamplerV2.__init__)
    _aer_sv2_params = set(_aer_sv2_sig.parameters.keys()) - {'self'}
    _sv2_accepts_backend = 'backend' in _aer_sv2_params
    logger.info(f"AerSamplerV2.__init__ accepts: {sorted(_aer_sv2_params)}")
    logger.info(f"  accepts 'backend' kwarg: {_sv2_accepts_backend}")
    logger.info(f"  AerSamplerV2 bases: {[b.__name__ for b in _AerSamplerV2.__mro__]}")
    
    class TranspilingAerSamplerV2(_AerSamplerV2):
        """
        Subclass of AerSamplerV2 that:
          1. Injects a noise model into the internal AerSimulator backend
          2. Transpiles all circuits to basis gates before execution
        
        v7.16: By inheriting from _AerSamplerV2, isinstance() checks pass
        and qiskit_algorithms treats this as a proper V2 sampler.  Previous
        versions used composition (self._inner) which caused IQAE to bypass
        our transpilation layer or reject the sampler type.
        
        The noise model is injected by replacing self._backend after
        super().__init__() with an AerSimulator(noise_model=nm).
        """
        
        _BASIS_GATES = [
            'u1', 'u2', 'u3', 'cx', 'id', 'rz', 'sx', 'x',
            'h', 'ry', 'rx', 'cz', 's', 't', 'sdg', 'tdg',
            'swap', 'ccx',
        ]
        
        def __init__(
            self,
            noise_model=None,
            default_shots: int = 8192,
            seed: int = 42,
        ):
            # Construct parent with the kwargs it accepts
            parent_kwargs = {}
            if 'default_shots' in _aer_sv2_params:
                parent_kwargs['default_shots'] = default_shots
            if 'seed' in _aer_sv2_params:
                parent_kwargs['seed'] = seed
            if _sv2_accepts_backend and noise_model is not None:
                parent_kwargs['backend'] = AerSimulator(
                    noise_model=noise_model,
                    **({"seed_simulator": seed} if seed else {})
                )
            
            super().__init__(**parent_kwargs)
            
            self._noise_model_ref = noise_model
            
            # If backend= was not accepted, patch _backend after init
            if not _sv2_accepts_backend and noise_model is not None:
                noisy_backend = AerSimulator(
                    noise_model=noise_model,
                    **({"seed_simulator": seed} if seed else {})
                )
                
                # Find and replace the backend
                patched = False
                instance_attrs = vars(self)
                logger.info(
                    f"  SamplerV2 instance attrs: "
                    f"{list(instance_attrs.keys())}")
                
                # Try _backend first (most common)
                for attr_name in ('_backend', 'backend', '_simulator',
                                  '_aer', '_sim'):
                    if attr_name in instance_attrs:
                        old_val = instance_attrs[attr_name]
                        if (isinstance(old_val, AerSimulator) or
                            (old_val is not None and 
                             'Simulator' in type(old_val).__name__)):
                            setattr(self, attr_name, noisy_backend)
                            patched = True
                            logger.info(
                                f"  Patched self.{attr_name} -> "
                                f"AerSimulator(noise_model=...)")
                            break
                
                if not patched:
                    logger.error(
                        f"  FAILED to patch noise model. "
                        f"Instance attrs: {list(instance_attrs.keys())}")
                    logger.error(
                        "  IQAE will run WITHOUT noise despite config.")
                else:
                    logger.info(
                        f"TranspilingAerSamplerV2: noise model injected "
                        f"via _backend patch, shots={default_shots}")
            else:
                if noise_model is not None:
                    logger.info(
                        f"TranspilingAerSamplerV2: noise model injected "
                        f"via backend= kwarg, shots={default_shots}")
                else:
                    logger.info(
                        f"TranspilingAerSamplerV2: ideal (no noise), "
                        f"shots={default_shots}")
            
            # ---- SMOKE TEST: verify run() + noise injection ----
            try:
                from qiskit import QuantumCircuit as _QC
                import numpy as _np
                
                # Run X|0> = |1> many times.  Without noise, result is
                # always '1'.  With noise, some '0' counts appear.
                _smoke = _QC(1, 1)
                _smoke.x(0)
                _smoke.measure(0, 0)
                _smoke_job = self.run([_smoke])
                _smoke_result = _smoke_job.result()
                
                # Read counts from V2 PrimitiveResult
                _counts = {}
                _pub_result = _smoke_result[0]
                # V2 DataBin stores bitstring arrays, not named
                # classical registers.  Iterate data fields.
                for _field_name in dir(_pub_result.data):
                    if _field_name.startswith('_'):
                        continue
                    _field = getattr(_pub_result.data, _field_name)
                    if hasattr(_field, 'get_counts'):
                        _counts = _field.get_counts()
                        break
                    elif hasattr(_field, 'array'):
                        # BitArray: convert to counts manually
                        _arr = _field.array
                        if hasattr(_arr, 'flatten'):
                            _arr = _arr.flatten()
                        _unique, _cnts = _np.unique(
                            _arr, return_counts=True)
                        _counts = {str(v): int(c) 
                                   for v, c in zip(_unique, _cnts)}
                        break
                
                _noise_active = len(_counts) > 1
                _total = sum(_counts.values()) if _counts else 0
                logger.info(
                    f"Smoke test: counts={_counts}, "
                    f"total={_total}, noise_active={_noise_active}")
                
                if noise_model is not None and not _noise_active:
                    logger.warning(
                        "NOISE VALIDATION FAILED: noise_model provided "
                        "but X|0> always gives |1>.  The _backend patch "
                        "may not be effective.  Noise will NOT affect "
                        "IQAE results.")
                
            except Exception as _smoke_err:
                logger.warning(
                    f"Smoke test failed (non-critical): {_smoke_err}")
        
        @staticmethod
        def _transpile_to_basis(circuit):
            """Transpile a single circuit to basis gates."""
            return transpile(
                circuit,
                basis_gates=TranspilingAerSamplerV2._BASIS_GATES,
                optimization_level=0
            )
        
        def run(self, pubs, **kwargs):
            """
            Override run() to transpile all circuits to basis gates
            before forwarding to the parent AerSamplerV2.run().
            """
            transpiled_pubs = []
            for pub in pubs:
                if isinstance(pub, QuantumCircuit):
                    transpiled_pubs.append(
                        self._transpile_to_basis(pub))
                    
                elif isinstance(pub, (list, tuple)):
                    circ = pub[0]
                    tc = self._transpile_to_basis(circ)
                    transpiled_pubs.append((tc,) + tuple(pub[1:]))
                    
                elif hasattr(pub, 'circuit'):
                    circ = pub.circuit
                    tc = self._transpile_to_basis(circ)
                    pub_cls = type(pub)
                    if hasattr(pub_cls, 'coerce'):
                        shots = getattr(pub, 'shots', None)
                        transpiled_pubs.append(pub_cls.coerce(
                            (tc, shots) if shots else tc
                        ))
                    else:
                        try:
                            pub.circuit = tc
                            transpiled_pubs.append(pub)
                        except (AttributeError, TypeError):
                            transpiled_pubs.append(tc)
                else:
                    transpiled_pubs.append(pub)
            
            return super().run(transpiled_pubs, **kwargs)
    
    logger.info("TranspilingAerSamplerV2 defined as subclass of AerSamplerV2 (v7.16)")


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class QAEResult:
    """Result from quantum amplitude estimation."""
    method: str
    amplitude: float
    probability: float
    confidence_interval: Tuple[float, float]
    num_oracle_queries: int
    num_grover_iterations: int
    circuit_depth: int
    num_qubits: int
    execution_time_ms: float
    converged: bool
    epsilon_achieved: float
    circuit_id: str = ""
    png_path: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'method': self.method,
            'amplitude': float(self.amplitude),
            'probability': float(self.probability),
            'confidence_interval': [float(x) for x in self.confidence_interval],
            'num_oracle_queries': int(self.num_oracle_queries),
            'num_grover_iterations': int(self.num_grover_iterations),
            'circuit_depth': int(self.circuit_depth),
            'num_qubits': int(self.num_qubits),
            'execution_time_ms': float(self.execution_time_ms),
            'converged': bool(self.converged),
            'epsilon_achieved': float(self.epsilon_achieved),
            'circuit_id': self.circuit_id,
            'png_path': self.png_path
        }


QAEMethodResult = QAEResult


@dataclass
class QuantumCVaRResult:
    """Complete result from TRUE quantum CVaR estimation."""
    var_estimate: float
    cvar_estimate: float
    tail_probability: float
    alpha: float
    
    total_oracle_queries_iqae: int = 0
    total_oracle_queries_woerner: int = 0
    total_circuits_executed: int = 0
    execution_time_ms: float = 0.0
    
    num_distribution_qubits: int = 4
    num_scenarios: int = 0
    
    tail_prob_result: Optional[QAEResult] = None
    conditional_exp_results: List[QAEResult] = field(default_factory=list)
    
    tail_weighted_loss: float = 0.0
    tail_weighted_loss_result: Optional[QAEResult] = None
    
    classical_var: float = 0.0
    classical_cvar: float = 0.0
    classical_tail_prob: float = 0.0
    
    var_error_vs_classical: float = 0.0
    cvar_error_vs_classical: float = 0.0
    tail_prob_error: float = 0.0
    
    method_results: Dict[str, Any] = field(default_factory=dict)
    
    zne_improvement: float = 0.0
    zne_applied: bool = False
    
    # v7.7: IQAE noise impact comparison
    # Three CVaR values from IQAE to quantify the NISQ gap:
    #   iqae_cvar_ideal:     IQAE on ideal (noiseless) sampler
    #   iqae_cvar_noisy_raw: IQAE on noisy sampler, no error mitigation
    #   iqae_cvar_noisy_zne: IQAE on noisy sampler + ZNE + MEM (= cvar_estimate)
    iqae_cvar_ideal: Optional[float] = None
    iqae_cvar_noisy_raw: Optional[float] = None
    iqae_cvar_noisy_zne: Optional[float] = None
    # Supporting amplitudes for the comparison
    iqae_tail_prob_ideal: Optional[float] = None
    iqae_tail_prob_noisy_raw: Optional[float] = None
    iqae_tail_prob_noisy_zne: Optional[float] = None
    iqae_twl_amplitude_ideal: Optional[float] = None
    iqae_twl_amplitude_noisy_raw: Optional[float] = None
    iqae_twl_amplitude_noisy_zne: Optional[float] = None
    
    circuits_exported: List[str] = field(default_factory=list)
    png_files: List[str] = field(default_factory=list)
    
    # v7.9: Per-method CVaR values (e.g. {'iqae': 0.021, 'woerner': 0.020})
    per_method_cvar: Dict[str, float] = field(default_factory=dict)
    
    loss_range: Tuple[float, float] = (0.0, 1.0)
    
    def to_dict(self) -> Dict:
        return {
            'var_estimate': float(self.var_estimate),
            'cvar_estimate': float(self.cvar_estimate),
            'tail_probability': float(self.tail_probability),
            'tail_weighted_loss': float(self.tail_weighted_loss),
            'alpha': float(self.alpha),
            'total_oracle_queries_iqae': int(self.total_oracle_queries_iqae),
            'total_oracle_queries_woerner': int(self.total_oracle_queries_woerner),
            'total_circuits_executed': int(self.total_circuits_executed),
            'execution_time_ms': float(self.execution_time_ms),
            'num_distribution_qubits': int(self.num_distribution_qubits),
            'num_scenarios': int(self.num_scenarios),
            'classical_var': float(self.classical_var),
            'classical_cvar': float(self.classical_cvar),
            'classical_tail_prob': float(self.classical_tail_prob),
            'var_error_vs_classical': float(self.var_error_vs_classical),
            'cvar_error_vs_classical': float(self.cvar_error_vs_classical),
            'tail_prob_error': float(self.tail_prob_error),
            'circuits_exported': self.circuits_exported,
            'png_files': self.png_files,
            'loss_range': [float(x) for x in self.loss_range],
            'zne_improvement': float(self.zne_improvement),
            'zne_applied': bool(self.zne_applied),
            # v7.7: IQAE noise impact comparison
            'iqae_cvar_ideal': float(self.iqae_cvar_ideal) if self.iqae_cvar_ideal is not None else None,
            'iqae_cvar_noisy_raw': float(self.iqae_cvar_noisy_raw) if self.iqae_cvar_noisy_raw is not None else None,
            'iqae_cvar_noisy_zne': float(self.iqae_cvar_noisy_zne) if self.iqae_cvar_noisy_zne is not None else None,
            'iqae_tail_prob_ideal': float(self.iqae_tail_prob_ideal) if self.iqae_tail_prob_ideal is not None else None,
            'iqae_tail_prob_noisy_raw': float(self.iqae_tail_prob_noisy_raw) if self.iqae_tail_prob_noisy_raw is not None else None,
            'iqae_tail_prob_noisy_zne': float(self.iqae_tail_prob_noisy_zne) if self.iqae_tail_prob_noisy_zne is not None else None,
            'per_method_cvar': {k: float(v) for k, v in self.per_method_cvar.items()},
        }


CVaREstimationResult = QuantumCVaRResult


# ==============================================================================
# ============================================================================
# TIER 1: MEASUREMENT ERROR MITIGATION (v7.2)
# ============================================================================

class ReadoutMitigator:
    """
    Measurement error mitigation via tensored calibration matrix inversion.
    
    Corrects readout errors by constructing M = M_0 (x) M_1 (x) ... (x) M_{n-1}
    where M_k = [[1-eps, eps], [eps, 1-eps]], then applying M^{-1} to raw counts.
    No calibration circuits needed (analytical construction from known ro_error).
    """
    
    def __init__(self, n_qubits: int, readout_error: float = 0.02):
        self.n_qubits = n_qubits
        self.n_states = 2 ** n_qubits
        self.readout_error = readout_error
        self.cal_matrix_inv = None
        self._calibrated = False
        
        if readout_error > 0:
            self._build_tensored_calibration(readout_error)
    
    def _build_tensored_calibration(self, ro_error: float):
        """Build M^{-1} from per-qubit symmetric readout error via Kronecker product."""
        m_single = np.array([[1 - ro_error, ro_error], [ro_error, 1 - ro_error]])
        cal_matrix = m_single.copy()
        for _ in range(self.n_qubits - 1):
            cal_matrix = np.kron(cal_matrix, m_single)
        try:
            self.cal_matrix_inv = np.linalg.inv(cal_matrix)
        except np.linalg.LinAlgError:
            self.cal_matrix_inv = np.linalg.pinv(cal_matrix)
        self._calibrated = True
        logger.info(f"ReadoutMitigator: {self.n_qubits}q, ro={ro_error:.4f}, "
                    f"matrix {self.n_states}x{self.n_states}")
    
    def correct_counts(self, counts: Dict[str, int]) -> Dict[str, float]:
        """Apply M^{-1} to raw counts. Returns corrected counts (float values)."""
        if not self._calibrated or self.readout_error <= 0:
            return {k: float(v) for k, v in counts.items()}
        
        total_shots = sum(counts.values())
        p_noisy = np.zeros(self.n_states)
        for bitstring, count in counts.items():
            bits = bitstring.replace(' ', '')
            idx = int(bits, 2)
            if idx < self.n_states:
                p_noisy[idx] = count / total_shots
        
        p_corrected = self.cal_matrix_inv @ p_noisy
        p_corrected = np.maximum(p_corrected, 0.0)
        p_sum = p_corrected.sum()
        if p_sum > 0:
            p_corrected /= p_sum
        
        corrected_counts = {}
        for idx in range(self.n_states):
            if p_corrected[idx] > 1e-10:
                bitstring = format(idx, f'0{self.n_qubits}b')
                corrected_counts[bitstring] = p_corrected[idx] * total_shots
        return corrected_counts
    
    def correct_amplitude(self, raw_amplitude: float, n_objective_qubits: int = 1) -> float:
        """
        Correct a single amplitude P(objective=|1...1>) analytically.
        For k objective qubits: P_corrected = P_raw / (1 - 2*eps)^k
        """
        if not self._calibrated or self.readout_error <= 0:
            return raw_amplitude
        eps = self.readout_error
        correction_factor = (1 - 2 * eps) ** n_objective_qubits
        if abs(correction_factor) < 1e-10:
            return raw_amplitude
        corrected = raw_amplitude / correction_factor
        return float(np.clip(corrected, 0.0, 1.0))


# CIRCUIT EXPORTER
# ==============================================================================

class CircuitExporter:
    """Export quantum circuits to PNG images."""
    
    def __init__(self, output_dir: Union[str, Path] = None, enabled: bool = True):
        self.enabled = enabled
        self.output_dir = Path(output_dir or './results/circuits')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.png_dir = self.output_dir / 'png'
        self.metadata_dir = self.output_dir / 'metadata'
        self.png_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        
        self.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
        self.circuits_exported = 0
        self.export_log = []
        self.png_files = []
        
        logger.info(f"CircuitExporter initialized: {self.output_dir}")
    
    def export_png(
        self,
        circuit: 'QuantumCircuit',
        name: str,
        category: str = 'general',
        context: Dict = None
    ) -> str:
        """Export circuit to PNG using qc.draw('mpl')."""
        if not self.enabled:
            return ""
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for PNG export")
        
        self.circuits_exported += 1
        circuit_id = f"{self.session_id}_{self.circuits_exported:04d}"
        base_name = f"{category}_{name}_{circuit_id}"
        png_path = self.png_dir / f"{base_name}.png"
        
        try:
            fig = circuit.draw("mpl", style="iqp", fold=80, scale=0.7)
            fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            
            logger.info(f"PNG exported: {png_path.name}")
            self.png_files.append(str(png_path))
            
            metadata = {
                'circuit_id': circuit_id,
                'name': name,
                'category': category,
                'num_qubits': circuit.num_qubits,
                'depth': circuit.depth(),
                'num_gates': len(circuit.data),
                'png_path': str(png_path),
                'timestamp': datetime.now().isoformat(),
                'context': context or {}
            }
            
            meta_path = self.metadata_dir / f"{base_name}.json"
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.export_log.append(metadata)
            return str(png_path)
            
        except Exception as e:
            logger.error(f"PNG export failed for {name}: {e}")
            raise RuntimeError(f"Failed to export circuit PNG: {e}")
    
    def get_summary(self) -> Dict:
        return {
            'total_exported': self.circuits_exported,
            'session_id': self.session_id,
            'output_dir': str(self.output_dir),
            'png_files': self.png_files,
            'exports': self.export_log
        }


_circuit_exporter: Optional[CircuitExporter] = None


def get_circuit_exporter(output_dir: str = None, enabled: bool = True) -> CircuitExporter:
    global _circuit_exporter
    if _circuit_exporter is None:
        _circuit_exporter = CircuitExporter(output_dir=output_dir, enabled=enabled)
    return _circuit_exporter


def reset_circuit_exporter(output_dir: str = None, enabled: bool = True) -> CircuitExporter:
    global _circuit_exporter
    _circuit_exporter = CircuitExporter(output_dir=output_dir, enabled=enabled)
    return _circuit_exporter


# ==============================================================================
# QUANTUM CIRCUITS FOR CVaR
# ==============================================================================

class QuantumCVaRCircuits:
    """Build quantum circuits for CVaR estimation."""
    
    def __init__(self, num_qubits: int = 4, exporter: CircuitExporter = None):
        self.num_qubits = num_qubits
        self.num_levels = 2 ** num_qubits
        self.exporter = exporter or get_circuit_exporter()
        logger.info(f"QuantumCVaRCircuits: {num_qubits} qubits, {self.num_levels} levels")
    
    def build_distribution_loading_circuit(
        self,
        probabilities: np.ndarray,
        name: str = "dist_load"
    ) -> 'QuantumCircuit':
        """Build circuit to load probability distribution."""
        n = self.num_qubits
        
        probs = np.abs(probabilities)
        probs = probs / np.sum(probs)
        
        if len(probs) < self.num_levels:
            probs = np.pad(probs, (0, self.num_levels - len(probs)))
        probs = probs[:self.num_levels]
        
        amplitudes = np.sqrt(probs)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        qc = QuantumCircuit(n, name=name)
        qc.initialize(amplitudes, range(n))
        qc_decomposed = qc.decompose().decompose().decompose()
        
        qc_clean = QuantumCircuit(n, name=name)
        for inst in qc_decomposed.data:
            if inst.operation.name != 'reset':
                qc_clean.append(inst.operation, inst.qubits, inst.clbits)
        
        self.exporter.export_png(qc_clean, name, 'distribution',
            context={'num_levels': self.num_levels})
        
        return qc_clean
    
    def build_threshold_oracle(
        self,
        num_data_qubits: int,
        threshold_level: int,
        name: str = "threshold_oracle"
    ) -> 'QuantumCircuit':
        """Build oracle that marks states |iâŸ© where i >= threshold."""
        n = num_data_qubits
        qc = QuantumCircuit(n + 1, name=name)
        ancilla = n
        
        if threshold_level == 0:
            qc.x(ancilla)
        elif threshold_level < self.num_levels:
            for idx in range(threshold_level, self.num_levels):
                # v7.19 FIX: reverse bit string for Qiskit little-endian convention
                # format() produces MSB-first, but Qiskit qubit 0 = LSB
                idx_bits = format(idx, f'0{n}b')[::-1]
                for i, bit in enumerate(idx_bits):
                    if bit == '0':
                        qc.x(i)
                if n == 1:
                    qc.cx(0, ancilla)
                elif n == 2:
                    qc.ccx(0, 1, ancilla)
                else:
                    qc.mcx(list(range(n)), ancilla)
                for i, bit in enumerate(idx_bits):
                    if bit == '0':
                        qc.x(i)
        
        self.exporter.export_png(qc, name, 'oracle',
            context={'threshold_level': threshold_level, 'data_qubits': n})
        
        return qc
    
    def build_tail_probability_circuit(
        self,
        loss_distribution: np.ndarray,
        threshold: float,
        name: str = "tail_prob"
    ) -> Tuple['QuantumCircuit', List[int], Dict]:
        """Build complete circuit for estimating P(Loss >= threshold)."""
        probs, levels, edges, l_min, l_max, threshold_level = \
            self._discretize_with_threshold(loss_distribution, threshold)
        
        classical_tail_prob = float(np.mean(loss_distribution >= threshold))
        quantum_expected_prob = float(np.sum(probs[threshold_level:]))
        discretization_error = abs(quantum_expected_prob - classical_tail_prob)
        
        logger.info(f"Tail probability circuit: threshold={threshold:.6f}, "
                    f"level={threshold_level}/{self.num_levels}")
        logger.debug(f"Expected quantum P(tail): {quantum_expected_prob:.6f}, "
                    f"classical: {classical_tail_prob:.6f}")
        
        dist_qc = self.build_distribution_loading_circuit(probs, f"{name}_dist")
        # [T-001 Hypothesis B] Log VaR threshold bin mapping diagnostic.
        # If the VaR threshold falls on a bin boundary, the quantization
        # error can be larger or smaller depending on n_qubits (bin width).
        # bin_width = (l_max - l_min) / 2^n_qubits
        # threshold_quantization_error = |threshold - edges[threshold_level]|
        _bin_width = (l_max - l_min) / self.num_levels if self.num_levels > 0 else float('nan')
        _threshold_edge = edges[threshold_level] if threshold_level < len(edges) else float('nan')
        _threshold_quant_error = abs(float(threshold) - float(_threshold_edge)) if not (
            isinstance(_threshold_edge, float) and _threshold_edge != _threshold_edge
        ) else float('nan')
        logger.info(
            f"[T-001] VaR threshold mapping [{name}]: "
            f"n_qubits={self.num_qubits}, "
            f"threshold={threshold:.6f}, "
            f"threshold_level={threshold_level}/{self.num_levels}, "
            f"bin_width={_bin_width:.6f}, "
            f"threshold_quantization_error={_threshold_quant_error:.6f}, "
            f"relative_quant_error={(_threshold_quant_error / max(abs(float(threshold)), 1e-10)):.4f}"
        )
        # Store on self for downstream JSON export (T-001)
        self._t001_last_threshold_bin_index      = int(threshold_level)
        self._t001_last_threshold_quantization_error = float(_threshold_quant_error)
        self._t001_last_bin_width                = float(_bin_width)
        self._t001_last_n_qubits                 = int(self.num_qubits)
        oracle_qc = self.build_threshold_oracle(self.num_qubits, threshold_level, f"{name}_oracle")
        
        n = self.num_qubits
        combined = QuantumCircuit(n + 1, name=name)
        combined.compose(dist_qc, qubits=range(n), inplace=True)
        combined.compose(oracle_qc, qubits=range(n + 1), inplace=True)
        
        objective_qubits = [n]
        
        metadata = {
            'threshold': float(threshold),
            'threshold_level': threshold_level,
            'num_levels': self.num_levels,
            'l_min': float(l_min),
            'l_max': float(l_max),
            'classical_tail_prob': classical_tail_prob,
            'quantum_expected_prob': quantum_expected_prob,
            'discretization_error': discretization_error,
            'threshold_on_edge': bool(np.any(np.abs(edges - threshold) < 1e-9))
        }
        
        self.exporter.export_png(combined, name, 'tail_probability', context=metadata)
        
        return combined, objective_qubits, metadata
    
    def build_grover_operator(
        self,
        state_prep: 'QuantumCircuit',
        oracle: 'QuantumCircuit',
        objective_qubits: List[int]
    ) -> 'QuantumCircuit':
        """Build Grover operator Q = S_0 * A^dag * S_f * A."""
        n_qubits = state_prep.num_qubits
        
        try:
            grover_op = GroverOperator(
                oracle=oracle,
                state_preparation=state_prep,
                reflection_qubits=objective_qubits
            )
            return grover_op
        except Exception as e:
            logger.warning(f"GroverOperator failed: {e}, building manually")
            
            qc = QuantumCircuit(n_qubits, name="grover_op")
            qc.compose(oracle, inplace=True)
            qc.compose(state_prep.inverse(), inplace=True)
            qc.x(range(n_qubits))
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
            qc.x(range(n_qubits))
            qc.compose(state_prep, inplace=True)
            
            return qc
    
    def build_conditional_expectation_circuit(
        self,
        asset_returns: np.ndarray,
        portfolio_losses: np.ndarray,
        threshold: float,
        asset_idx: int,
        name: str = "cond_exp"
    ) -> Tuple[Optional['QuantumCircuit'], List[int], Dict]:
        """Build circuit for estimating E[r_j | L >= threshold]."""
        tail_mask = portfolio_losses >= threshold
        tail_returns = asset_returns[tail_mask]
        
        if len(tail_returns) == 0:
            logger.warning(f"No tail observations for asset {asset_idx}")
            return None, [], {'error': 'no_tail_data'}
        
        probs, levels, r_min, r_max = self._discretize(tail_returns)
        classical_exp = float(np.mean(tail_returns))
        
        n = self.num_qubits
        
        if r_max > r_min:
            normalized_levels = (levels - r_min) / (r_max - r_min)
        else:
            normalized_levels = np.ones_like(levels) * 0.5
        
        qc = QuantumCircuit(n + 1, name=f"{name}_asset{asset_idx}")
        
        amplitudes = np.sqrt(probs)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        init_qc = QuantumCircuit(n)
        init_qc.initialize(amplitudes, range(n))
        init_qc = init_qc.decompose().decompose().decompose()
        
        for inst in init_qc.data:
            if inst.operation.name != 'reset':
                qc.append(inst.operation, [q._index for q in inst.qubits], [])
        
        ancilla = n
        
        for idx in range(min(len(normalized_levels), self.num_levels)):
            value = normalized_levels[idx] if idx < len(normalized_levels) else 0.0
            
            if value > 0 and value < 1:
                angle = 2 * np.arcsin(np.sqrt(value))
            elif value >= 1:
                angle = np.pi
            else:
                angle = 0
            
            if abs(angle) > 1e-10:
                # v7.19 FIX: reverse bit string for Qiskit little-endian convention
                idx_bits = format(idx, f'0{n}b')[::-1]
                for i, bit in enumerate(idx_bits):
                    if bit == '0':
                        qc.x(i)
                
                if n == 1:
                    qc.cry(angle, 0, ancilla)
                elif n == 2:
                    qc.cry(angle/2, 1, ancilla)
                    qc.cx(0, 1)
                    qc.cry(-angle/2, 1, ancilla)
                    qc.cx(0, 1)
                    qc.cry(angle/2, 0, ancilla)
                else:
                    from qiskit.circuit.library import RYGate
                    mcry = RYGate(angle).control(n)
                    qc.append(mcry, list(range(n)) + [ancilla])
                
                for i, bit in enumerate(idx_bits):
                    if bit == '0':
                        qc.x(i)
        
        metadata = {
            'asset_idx': asset_idx,
            'threshold': float(threshold),
            'r_min': float(r_min),
            'r_max': float(r_max),
            'tail_count': int(np.sum(tail_mask)),
            'classical_exp': classical_exp
        }
        
        self.exporter.export_png(qc, f"{name}_asset{asset_idx}", 'conditional_expectation', context=metadata)
        
        return qc, [ancilla], metadata
    
    def build_tail_weighted_loss_circuit(
        self,
        loss_distribution: np.ndarray,
        threshold: float,
        name: str = "tail_weighted_loss"
    ) -> Tuple['QuantumCircuit', List[int], Dict]:
        """Build circuit for estimating E[L Â· 1_{L >= threshold}] via QAE."""
        probs, levels, edges, l_min, l_max, threshold_level = \
            self._discretize_with_threshold(loss_distribution, threshold)
        
        n = self.num_qubits
        
        if l_max > l_min:
            normalized_levels = (levels - l_min) / (l_max - l_min)
        else:
            normalized_levels = np.ones_like(levels) * 0.5
        
        normalized_levels = np.clip(normalized_levels, 0.0, 1.0)
        
        qc = QuantumCircuit(n + 1, name=name)
        ancilla = n
        
        amplitudes = np.sqrt(probs)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        init_qc = QuantumCircuit(n)
        init_qc.initialize(amplitudes, range(n))
        init_qc = init_qc.decompose().decompose().decompose()
        
        for inst in init_qc.data:
            if inst.operation.name != 'reset':
                qc.append(inst.operation, [q._index if hasattr(q, '_index') else q.index for q in inst.qubits], [])
        
        for idx in range(threshold_level, self.num_levels):
            if idx >= len(normalized_levels):
                continue
                
            value = normalized_levels[idx]
            
            if value > 1e-10 and value < (1 - 1e-10):
                angle = 2 * np.arcsin(np.sqrt(value))
            elif value >= (1 - 1e-10):
                angle = np.pi
            else:
                angle = 0
            
            if abs(angle) < 1e-10:
                continue
            
            # v7.19 FIX: reverse bit string for Qiskit little-endian convention
            idx_bits = format(idx, f'0{n}b')[::-1]
            
            for i, bit in enumerate(idx_bits):
                if bit == '0':
                    qc.x(i)
            
            if n == 1:
                qc.cry(angle, 0, ancilla)
            elif n == 2:
                qc.cry(angle/2, 1, ancilla)
                qc.cx(0, 1)
                qc.cry(-angle/2, 1, ancilla)
                qc.cx(0, 1)
                qc.cry(angle/2, 0, ancilla)
            else:
                from qiskit.circuit.library import RYGate
                mcry = RYGate(angle).control(n)
                qc.append(mcry, list(range(n)) + [ancilla])
            
            for i, bit in enumerate(idx_bits):
                if bit == '0':
                    qc.x(i)
        
        tail_mask = loss_distribution >= threshold
        classical_tail_prob = float(np.mean(tail_mask))
        classical_tail_weighted_loss = float(np.mean(loss_distribution[tail_mask])) * classical_tail_prob if classical_tail_prob > 0 else 0.0
        
        metadata = {
            'threshold': float(threshold),
            'threshold_level': threshold_level,
            'num_levels': self.num_levels,
            'l_min': float(l_min),
            'l_max': float(l_max),
            'classical_tail_prob': classical_tail_prob,
            'classical_tail_weighted_loss': classical_tail_weighted_loss,
            'levels': levels.tolist(),
            'normalized_levels': normalized_levels.tolist(),
            'tail_levels_count': self.num_levels - threshold_level
        }
        
        self.exporter.export_png(qc, name, 'tail_weighted_loss', context=metadata)
        
        return qc, [ancilla], metadata

    def _discretize_with_threshold(
        self, 
        values: np.ndarray, 
        threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
        """Discretize values ensuring threshold falls exactly on a bin edge."""
        v_min, v_max = float(np.min(values)), float(np.max(values))
        n_total = len(values)
        
        if v_max <= v_min:
            v_max = v_min + 1e-6
        
        threshold = np.clip(threshold, v_min + 1e-10, v_max - 1e-10)
        
        below_mask = values < threshold
        above_mask = values >= threshold
        n_below = np.sum(below_mask)
        n_above = np.sum(above_mask)
        
        if n_above > 0 and n_below > 0:
            # v7.19 FIX: ensure minimum tail bins for adequate CVaR resolution
            # With too few tail bins, discretized midpoints overshoot true CVaR.
            # Error bound: |CVaR_disc - CVaR_true| <= R_tail / (2 * N_tail)
            # FIX (qae_discretization_fix.md Bug 1):
            # Allocate all available resolution to the tail.
            # One body bin is sufficient: it captures P(L < VaR) = 1-alpha
            # for a1 estimation without consuming tail resolution.
            tail_bins = self.num_levels - 1
            body_bins = 1
            
            if body_bins == 0 and n_below > 0:
                body_bins = 1
                tail_bins = self.num_levels - 1
        elif n_above > 0:
            tail_bins = self.num_levels
            body_bins = 0
        else:
            body_bins = self.num_levels
            tail_bins = 0
        
        if body_bins > 0 and n_below > 0:
            body_values = values[below_mask]
            body_quantiles = np.linspace(0, 100, body_bins + 1)
            body_edges = np.percentile(body_values, body_quantiles)
            body_edges[0] = v_min
            body_edges[-1] = threshold
        elif body_bins > 0:
            body_edges = np.linspace(v_min, threshold, body_bins + 1)
        else:
            body_edges = np.array([v_min])
        
        if tail_bins > 0 and n_above > 0:
            tail_values = values[above_mask]
            tail_quantiles = np.linspace(0, 100, tail_bins + 1)
            tail_edges = np.percentile(tail_values, tail_quantiles)
            tail_edges[0] = threshold
            tail_edges[-1] = v_max
        elif tail_bins > 0:
            tail_edges = np.linspace(threshold, v_max, tail_bins + 1)
        else:
            tail_edges = np.array([v_max])
        
        if body_bins > 0 and tail_bins > 0:
            edges = np.concatenate([body_edges, tail_edges[1:]])
        elif body_bins > 0:
            edges = np.concatenate([body_edges, [v_max]])
        else:
            edges = np.concatenate([[v_min], tail_edges])
        
        edges = np.unique(edges)
        
        if len(edges) > self.num_levels + 1:
            target_edges = np.linspace(v_min, v_max, self.num_levels + 1)
            threshold_idx = np.searchsorted(target_edges, threshold)
            target_edges = np.insert(target_edges, threshold_idx, threshold)
            target_edges = np.unique(target_edges)[:self.num_levels + 1]
            edges = target_edges
        
        while len(edges) < self.num_levels + 1:
            widths = np.diff(edges)
            max_width_idx = np.argmax(widths)
            new_edge = (edges[max_width_idx] + edges[max_width_idx + 1]) / 2
            edges = np.sort(np.append(edges, new_edge))
        
        edges = edges[:self.num_levels + 1]
        
        hist, _ = np.histogram(values, bins=edges)
        probs = hist / len(values)
        probs = probs / np.sum(probs)
        
        # FIX (qae_discretization_fix.md Bug 2):
        # Empirical bin mean instead of geometric midpoint.
        # Eliminates systematic bias for non-uniform within-bin distributions.
        levels = np.array([
            np.mean(values[(values >= edges[i]) & (values < edges[i + 1])])
            if np.any((values >= edges[i]) & (values < edges[i + 1]))
            else (edges[i] + edges[i + 1]) / 2
            for i in range(len(edges) - 1)
        ])
        
        threshold_level = self.num_levels
        for i, edge in enumerate(edges[:-1]):
            if edge >= threshold - 1e-10:
                threshold_level = i
                break
        
        return probs, levels, edges, v_min, v_max, threshold_level
    
    def _discretize(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Standard discretization without threshold constraint."""
        v_min, v_max = float(np.min(values)), float(np.max(values))
        
        if v_max <= v_min:
            v_max = v_min + 1e-6
        
        try:
            quantiles = np.linspace(0, 100, self.num_levels + 1)
            edges = np.percentile(values, quantiles)
            edges = np.unique(edges)
            
            if len(edges) < self.num_levels + 1:
                edges = np.linspace(v_min, v_max, self.num_levels + 1)
        except Exception:
            edges = np.linspace(v_min, v_max, self.num_levels + 1)
        
        hist, _ = np.histogram(values, bins=edges)
        probs = hist / len(values)
        
        if len(probs) < self.num_levels:
            probs = np.pad(probs, (0, self.num_levels - len(probs)))
        probs = probs[:self.num_levels]
        probs = probs / np.sum(probs)
        
        # FIX (qae_discretization_fix.md Bug 2):
        # Empirical bin mean instead of geometric midpoint.
        # Eliminates systematic bias for non-uniform within-bin distributions.
        levels = np.array([
            np.mean(values[(values >= edges[i]) & (values < edges[i + 1])])
            if np.any((values >= edges[i]) & (values < edges[i + 1]))
            else (edges[i] + edges[i + 1]) / 2
            for i in range(len(edges) - 1)
        ])
        if len(levels) < self.num_levels:
            levels = np.pad(levels, (0, self.num_levels - len(levels)), mode='edge')
        levels = levels[:self.num_levels]
        
        return probs, levels, v_min, v_max


# ==============================================================================
# QUANTUM AMPLITUDE ESTIMATOR - v6.9 NOISE + ZNE + WOERNER VALIDATION
# ==============================================================================

class QuantumAmplitudeEstimator:
    """
    Unified QAE with IQAE and Woerner (FIXED) methods.
    NO CLASSICAL FALLBACK.
    
    v6.9 CHANGES:
    - ADDED: Noise model support for NISQ-realistic simulation
    - ADDED: ZNE via multi-noise-scale runs + Richardson extrapolation
    - FIXED: Shots propagated to AerSampler (was 1024 default)
    - FIXED: Woerner 2x safety factor + validation retry
    
    v6.8 CHANGES:
    - REMOVED: QSP/MLAE (not needed, IQAE provides optimal complexity)
    - CLEAN: Only two proven methods remain
    """
    
    def __init__(
        self,
        num_distribution_qubits: int = 4,
        epsilon: float = 0.05,
        alpha_confidence: float = 0.05,
        shots: int = 8192,
        backend: str = "aer_simulator",
        export_circuits: bool = True,
        circuit_export_dir: str = None,
        use_zne: bool = False,
        zne_scale_factors: List[float] = None,
        zne_extrapolation: str = "richardson",
        noise_model_config: Dict = None,
        iqae_timeout: float = MAX_IQAE_TIMEOUT,
        woerner_timeout: float = DEFAULT_QAE_TIMEOUT,
        max_woerner_eval_qubits: int = MAX_WOERNER_EVAL_QUBITS,
        max_total_qubits: int = MAX_TOTAL_QUBITS,
        allow_fallback: bool = False,
        # v7.2: Error mitigation tiers
        use_readout_mitigation: bool = False,
        use_zne_state_prep: bool = False
    ):
        logger.section("INITIALIZING QUANTUM AMPLITUDE ESTIMATOR v7.16")
        
        if not QISKIT_AVAILABLE:
            raise ImportError("Qiskit is required")
        
        if not QISKIT_ALGORITHMS_AVAILABLE:
            raise ImportError("Qiskit algorithms is required")
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for PNG export")
        
        self.num_distribution_qubits = num_distribution_qubits
        self.num_levels = 2 ** num_distribution_qubits
        self.epsilon = epsilon
        self.alpha_confidence = alpha_confidence
        self.shots = shots
        self.use_zne = use_zne
        self.zne_scale_factors = zne_scale_factors or ZNE_DEFAULT_SCALE_FACTORS
        self.zne_extrapolation = zne_extrapolation
        
        self.iqae_timeout = iqae_timeout
        self.woerner_timeout = woerner_timeout
        self.max_woerner_eval_qubits = max_woerner_eval_qubits
        self.max_total_qubits = max_total_qubits
        self.allow_fallback = allow_fallback
        
        # v6.9: Create noise model and sampler
        self.noise_model = None
        self.noise_enabled = False
        self._setup_noise_and_sampler(noise_model_config)
        
        # v7.2: TIER 1 â€” Measurement Error Mitigation
        self.readout_mitigator = None
        self.use_readout_mitigation = use_readout_mitigation
        if use_readout_mitigation and self.noise_enabled:
            ro_error = (noise_model_config or {}).get('readout_error', 0.02)
            if ro_error > 0:
                self.readout_mitigator = ReadoutMitigator(
                    n_qubits=num_distribution_qubits,
                    readout_error=ro_error
                )
            else:
                logger.info("ReadoutMitigator: readout_error=0, MEM disabled")
        elif use_readout_mitigation and not self.noise_enabled:
            logger.warning("ReadoutMitigator requested but noise disabled â€” MEM is a no-op")
        
        # v7.2: TIER 2 â€” ZNE on State Preparation
        self.use_zne_state_prep = use_zne_state_prep and self.noise_enabled
        if use_zne_state_prep and not self.noise_enabled:
            logger.warning("ZNE-StatePrep requested but noise disabled â€” skipping")
            self.use_zne_state_prep = False
        
        self.exporter = reset_circuit_exporter(
            output_dir=circuit_export_dir or './results/circuits',
            enabled=True
        )
        
        self.circuit_builder = QuantumCVaRCircuits(
            num_qubits=num_distribution_qubits,
            exporter=self.exporter
        )
        
        self.stats = {
            'total_circuits': 0,
            'total_queries': 0,
            'total_queries_iqae': 0,
            'total_queries_woerner': 0,
            'total_time_ms': 0,
            'png_files': [],
            'zne_corrections': 0,
            'zne_total_improvement': 0.0,
            'woerner_retries': 0,
            # v7.2: Error mitigation stats
            'mem_corrections': 0,
            'zne_state_prep_corrections': 0,
        }
        
        self._loss_range = (0.0, 1.0)
        
        logger.metric("Qubits", num_distribution_qubits)
        logger.metric("Levels", self.num_levels)
        logger.metric("Epsilon", epsilon)
        logger.metric("Shots", shots)
        logger.metric("Max Woerner eval qubits", max_woerner_eval_qubits)
        logger.metric("Max total qubits", max_total_qubits)
        logger.metric("Noise enabled", self.noise_enabled)
        logger.metric("ZNE enabled", use_zne)
        if use_zne:
            logger.metric("ZNE scale factors", self.zne_scale_factors)
            logger.metric("ZNE extrapolation", zne_extrapolation)
        # v7.2: Log error mitigation tiers
        logger.metric("TIER1 MEM", self.use_readout_mitigation and self.readout_mitigator is not None)
        logger.metric("TIER2 ZNE-StatePrep", self.use_zne_state_prep)
        logger.info("v7.2: Error mitigation tiers (MEM + ZNE-StatePrep + ZNE-CVaR)")
    
    def _setup_noise_and_sampler(self, noise_config: Optional[Dict]):
        """
        Create sampler with optional noise model.
        
        v7.3 FIX:
        When noise is enabled AND AerSamplerV2 is available:
          self.sampler = AerSamplerV2(noise_model=...) â€” a V2 primitive
          that feeds real noise into IQAE/Woerner natively.
        
        When noise is disabled OR AerSamplerV2 not available:
          self.sampler = StatevectorSampler() â€” ideal, exact.
        
        AerSampler (V1) is NOT used for IQAE â€” it causes
        'The job was not completed successfully' errors.
        """
        def _get_ideal_sampler():
            """Return best available ideal (noiseless) sampler."""
            if Sampler is not None:
                return Sampler()
            if AER_SAMPLER_AVAILABLE:
                logger.info("Using AerSampler (no noise) as ideal sampler fallback")
                return AerSampler(
                    run_options={'shots': self.shots}
                )
            raise ImportError(
                "No sampler available. Install qiskit-aer or a Qiskit version "
                "with qiskit.primitives.Sampler / StatevectorSampler."
            )
        
        # Also store ideal sampler â€” needed for ZNE baseline comparison
        self.ideal_sampler = _get_ideal_sampler()
        
        if noise_config and noise_config.get('enabled', False):
            if not NOISE_MODEL_AVAILABLE:
                logger.warning("Noise config enabled but qiskit-aer noise not available. "
                               "Falling back to ideal simulation.")
                self.sampler = self.ideal_sampler
                self.backend = AerSimulator()
                self._base_sq_error = 0.0
                self._base_tq_error = 0.0
                self._base_ro_error = 0.0
                return
            
            # Build noise model from config
            self.noise_model = NoiseModel()
            
            sq_error = noise_config.get('single_qubit_gate_error', 0.001)
            tq_error = noise_config.get('two_qubit_gate_error', 0.01)
            ro_error = noise_config.get('readout_error', 0.02)
            
            # Store base rates for ZNE/TIER3 scaling
            self._base_sq_error = sq_error
            self._base_tq_error = tq_error
            self._base_ro_error = ro_error
            
            # Depolarizing errors on gates
            if sq_error > 0:
                error_1q = depolarizing_error(sq_error, 1)
                self.noise_model.add_all_qubit_quantum_error(
                    error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'h', 's', 't']
                )
            
            if tq_error > 0:
                error_2q = depolarizing_error(tq_error, 2)
                self.noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
            
            # Readout error
            if ro_error > 0:
                ro = ReadoutError([[1 - ro_error, ro_error], [ro_error, 1 - ro_error]])
                self.noise_model.add_all_qubit_readout_error(ro)
            
            self.noise_enabled = True
            
            # Noisy backend (kept for _measure_amplitude_noisy and TIER2)
            self.backend = AerSimulator(noise_model=self.noise_model)
            
            # v7.9: Use TranspilingAerSamplerV2 wrapper as the PRIMARY sampler
            # when noise is enabled.  The wrapper constructs AerSimulator with
            # the noise model baked in, then wraps it in SamplerV2.  It also
            # transpiles circuits to basis gates before forwarding to Aer,
            # preventing crashes from opaque composite gates (GroverOperator,
            # QPE) that qiskit_algorithms constructs internally.
            if AER_SAMPLER_V2_AVAILABLE:
                self.sampler = TranspilingAerSamplerV2(
                    noise_model=self.noise_model,
                    default_shots=self.shots,
                    seed=42,
                )
                logger.info(f"Noise model created: sq={sq_error}, tq={tq_error}, ro={ro_error}")
                logger.info(f"Sampler: TranspilingAerSamplerV2 (noisy, {self.shots} shots) â€” "
                           f"IQAE sees real noise")
            else:
                # Fallback: ideal sampler (noise only via _measure_amplitude_noisy)
                self.sampler = self.ideal_sampler
                logger.info(f"Noise model created: sq={sq_error}, tq={tq_error}, ro={ro_error}")
                logger.warning("AerSamplerV2 not available â€” IQAE runs ideal, "
                              "noise only via _measure_amplitude_noisy()")
            
            if self.use_zne:
                logger.info("ZNE enabled with active noise model")
        else:
            # Ideal simulation
            self.sampler = self.ideal_sampler
            self.backend = AerSimulator()
            self._base_sq_error = 0.0
            self._base_tq_error = 0.0
            self._base_ro_error = 0.0
            
            if self.use_zne:
                logger.warning("ZNE requested but noise model disabled. "
                               "ZNE is a no-op on ideal simulator. "
                               "Set noise_model.enabled=true in config to activate ZNE.")
                self.use_zne = False
            
            logger.info("Ideal sampler (statevector, no noise)")
    
    def _create_scaled_sampler(self, noise_scale: float) -> 'Sampler':
        """
        Create a sampler with scaled noise for ZNE / TIER3.
        
        noise_scale=1.0 returns self.sampler (base noise level).
        noise_scale=2.0 doubles all error rates, etc.
        noise_scale=0.0 returns ideal sampler (no noise).
        """
        if abs(noise_scale) < 1e-6:
            return self.ideal_sampler
        
        if not self.noise_enabled or not NOISE_MODEL_AVAILABLE:
            return self.sampler
        
        if abs(noise_scale - 1.0) < 1e-6:
            return self.sampler
        
        # Rebuild noise model with scaled rates
        scaled_model = NoiseModel()
        
        sq_scaled = min(self._base_sq_error * noise_scale, 0.3)
        tq_scaled = min(self._base_tq_error * noise_scale, 0.3)
        ro_scaled = min(self._base_ro_error * noise_scale, 0.45)
        
        if sq_scaled > 0:
            error_1q = depolarizing_error(sq_scaled, 1)
            scaled_model.add_all_qubit_quantum_error(
                error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'h', 's', 't']
            )
        
        if tq_scaled > 0:
            error_2q = depolarizing_error(tq_scaled, 2)
            scaled_model.add_all_qubit_quantum_error(error_2q, ['cx', 'cz', 'swap'])
        
        if ro_scaled > 0:
            ro = ReadoutError([[1 - ro_scaled, ro_scaled], [ro_scaled, 1 - ro_scaled]])
            scaled_model.add_all_qubit_readout_error(ro)
        
        # v7.9: Use TranspilingAerSamplerV2 (wrapper with auto-transpilation)
        if AER_SAMPLER_V2_AVAILABLE:
            return TranspilingAerSamplerV2(
                noise_model=scaled_model,
                default_shots=self.shots,
                seed=42,
            )
        
        # Fallback: BackendSampler
        if BACKEND_SAMPLER_AVAILABLE:
            scaled_backend = AerSimulator(noise_model=scaled_model)
            return BackendSampler(
                backend=scaled_backend,
                options={'shots': self.shots}
            )
        
        # Last resort: AerSampler (V1) â€” may fail with IQAE
        if AER_SAMPLER_AVAILABLE:
            return AerSampler(
                backend_options={'noise_model': scaled_model, 'shots': self.shots},
                run_options={'shots': self.shots}
            )
        
        return self.sampler
    
    def _apply_zne_to_amplitude(
        self,
        run_fn,
        name: str = "zne",
        state_prep: 'QuantumCircuit' = None,
        objective_qubits: List[int] = None
    ) -> Tuple[float, float, Dict]:
        """
        Apply Zero-Noise Extrapolation to a QAE amplitude estimation.
        
        v7.4 FIX:  ZNE now uses **unitary folding** on the same circuit with
        the same noisy sampler, instead of creating different samplers with
        different noise models.  This is the correct ZNE procedure as
        described in Mitiq / Temme et al.:
        
        For integer scale factors s = 1, 3, 5, ... :
            U_folded = U Â· (Uâ€  Â· U)^((s-1)/2)
        
        For non-integer scale factors (e.g. 1.5), we use partial folding:
        fold as many layers as needed from the back of the circuit.
        
        The key property is that all folded circuits implement the SAME
        unitary (on a noiseless machine) but with different circuit depths,
        hence different noise levels.  Richardson / linear extrapolation
        to depth=0 (or equivalently scale=0) is then well-defined.
        
        When state_prep and objective_qubits are provided, we fold the
        state_prep circuit and re-run IQAE.  Otherwise, we fall back to
        the legacy noise-scaling approach (but with a warning).
        
        Args:
            run_fn: Callable(sampler) -> QAEResult.  Used as fallback.
            name: Label for logging.
            state_prep: The state preparation circuit to fold.
            objective_qubits: Objective qubit indices for IQAE.
            
        Returns:
            (zne_amplitude, zne_improvement, zne_details)
        """
        if not self.noise_enabled or not self.use_zne:
            result = run_fn(self.sampler)
            return result.amplitude, 0.0, {'zne_applied': False, 'result': result}
        
        # =================================================================
        # PREFERRED PATH: Unitary folding on the same circuit + sampler
        # =================================================================
        if state_prep is not None and objective_qubits is not None:
            logger.info(f"ZNE (unitary folding): {len(self.zne_scale_factors)} "
                        f"scale factors for {name}")
            
            scale_amplitudes = []
            scale_results = []
            
            for scale in self.zne_scale_factors:
                try:
                    folded_circuit = self._fold_circuit_global(state_prep, scale)
                    result = self._run_iqae_single(
                        folded_circuit, objective_qubits, f"{name}_s{scale}", self.sampler
                    )
                    scale_amplitudes.append(result.amplitude)
                    scale_results.append(result)
                    logger.info(f"  ZNE fold scale={scale:.2f}: "
                                f"amplitude={result.amplitude:.6f}, "
                                f"depth={folded_circuit.depth()}")
                except Exception as e:
                    logger.warning(f"  ZNE fold scale={scale:.2f} failed: {e}")
                    if abs(scale - self.zne_scale_factors[0]) < 1e-6:
                        raise  # base scale must succeed
                    continue
            
            return self._extrapolate_zne(
                scale_amplitudes, scale_results, name
            )
        
        # =================================================================
        # FALLBACK: Noise-rate scaling (legacy, less principled but usable
        # when the circuit object is not available).
        # =================================================================
        logger.warning(f"ZNE fallback (noise-rate scaling) for {name} â€” "
                       f"prefer unitary folding for rigorous ZNE")
        logger.info(f"ZNE: Running at {len(self.zne_scale_factors)} noise scales for {name}")
        
        scale_amplitudes = []
        scale_results = []
        
        for scale in self.zne_scale_factors:
            try:
                scaled_sampler = self._create_scaled_sampler(scale)
                result = run_fn(scaled_sampler)
                scale_amplitudes.append(result.amplitude)
                scale_results.append(result)
                logger.info(f"  ZNE noise scale={scale:.1f}: amplitude={result.amplitude:.6f}")
            except Exception as e:
                logger.warning(f"  ZNE noise scale={scale:.1f} failed: {e}")
                if abs(scale - 1.0) < 1e-6:
                    raise
                continue
        
        return self._extrapolate_zne(
            scale_amplitudes, scale_results, name
        )
    
    def _extrapolate_zne(
        self,
        scale_amplitudes: List[float],
        scale_results: List,
        name: str
    ) -> Tuple[float, float, Dict]:
        """Common ZNE extrapolation logic used by both folding and scaling."""
        if len(scale_amplitudes) < 2:
            logger.warning("ZNE: Not enough scale points, using base amplitude")
            base_result = scale_results[0] if scale_results else None
            amp = base_result.amplitude if base_result else 0.0
            return amp, 0.0, {
                'zne_applied': False,
                'reason': 'insufficient_scale_points',
                'result': base_result
            }
        
        scales = np.array(self.zne_scale_factors[:len(scale_amplitudes)])
        amplitudes = np.array(scale_amplitudes)
        
        if self.zne_extrapolation == 'richardson' and len(scales) >= 3:
            zne_amplitude = self._richardson_extrapolation(scales, amplitudes)
            if zne_amplitude < 0 or zne_amplitude > 1.5 * amplitudes[0]:
                logger.warning(
                    f"Richardson non-physical ({zne_amplitude:.6f}), "
                    f"falling back to linear"
                )
                zne_amplitude = self._linear_extrapolation(scales[:2], amplitudes[:2])
        else:
            zne_amplitude = self._linear_extrapolation(scales[:2], amplitudes[:2])
        
        zne_amplitude = float(np.clip(zne_amplitude, 0.0, 1.0))
        
        base_amplitude = scale_amplitudes[0]
        improvement = abs(base_amplitude - zne_amplitude)
        
        logger.metric(f"ZNE base amplitude ({name})", base_amplitude)
        logger.metric(f"ZNE extrapolated amplitude ({name})", zne_amplitude)
        logger.metric(f"ZNE improvement ({name})", improvement)
        
        self.stats['zne_corrections'] += 1
        self.stats['zne_total_improvement'] += improvement
        
        details = {
            'zne_applied': True,
            'scales': scales.tolist(),
            'amplitudes': amplitudes.tolist(),
            'base_amplitude': base_amplitude,
            'zne_amplitude': zne_amplitude,
            'improvement': improvement,
            'extrapolation': self.zne_extrapolation,
            'result': scale_results[0]
        }
        
        return zne_amplitude, improvement, details
    
    @staticmethod
    def _fold_circuit_global(circuit: 'QuantumCircuit', scale_factor: float) -> 'QuantumCircuit':
        """
        Global unitary folding: U -> U Â· (Uâ€  Â· U)^n  [Â· partial]
        
        For integer scale factors s = 2k+1: apply k full folds.
        For non-integer factors: apply floor folds + partial fold on
        a tail fraction of the circuit's gates.
        
        scale_factor=1.0  -> original circuit  (depth Ã— 1)
        scale_factor=3.0  -> U Â· Uâ€  Â· U        (depth Ã— 3)
        scale_factor=5.0  -> U Â· Uâ€  Â· U Â· Uâ€  Â· U  (depth Ã— 5)
        
        This preserves the unitary: on a noiseless machine, all folded
        circuits produce the same output state.
        """
        if scale_factor < 1.0:
            raise ValueError(f"scale_factor must be >= 1.0, got {scale_factor}")
        
        if abs(scale_factor - 1.0) < 1e-6:
            return circuit.copy()
        
        # Number of full Uâ€ Â·U insertions
        n_full_folds = int((scale_factor - 1) / 2)
        fractional = scale_factor - 1 - 2 * n_full_folds
        
        folded = circuit.copy()
        
        # Full folds
        for _ in range(n_full_folds):
            folded.compose(circuit.inverse(), inplace=True)
            folded.compose(circuit, inplace=True)
        
        # Partial fold: fold a fraction of the circuit from the back
        if fractional > 0.01:
            n_gates = len(circuit.data)
            n_partial = max(1, int(round(fractional / 2.0 * n_gates)))
            
            # Build sub-circuit from the last n_partial gates
            partial_qc = QuantumCircuit(circuit.num_qubits, name="partial")
            for inst in circuit.data[-n_partial:]:
                partial_qc.append(inst.operation, inst.qubits, inst.clbits)
            
            folded.compose(partial_qc.inverse(), inplace=True)
            folded.compose(partial_qc, inplace=True)
        
        return folded
    
    @staticmethod
    def _richardson_extrapolation(scales: np.ndarray, values: np.ndarray) -> float:
        """Richardson extrapolation to zero noise (scale=0)."""
        n = len(scales)
        # Build Richardson coefficient matrix: sum_j c_j * f(lambda_j) â‰ˆ f(0)
        # Vandermonde-like system: sum_j c_j * lambda_j^k = delta_{k,0} for k=0..n-1
        A = np.vander(scales, increasing=True).T
        b = np.zeros(n)
        b[0] = 1.0  # f(0) coefficient
        try:
            coeffs = np.linalg.solve(A, b)
            return float(np.dot(coeffs, values))
        except np.linalg.LinAlgError:
            # Fallback to linear
            return QuantumAmplitudeEstimator._linear_extrapolation(scales, values)
    
    @staticmethod
    def _linear_extrapolation(scales: np.ndarray, values: np.ndarray) -> float:
        """Linear extrapolation to zero noise."""
        if len(scales) < 2:
            return float(values[0])
        coeffs = np.polyfit(scales, values, 1)
        return float(np.polyval(coeffs, 0.0))
    
    def _check_circuit_size(self, total_qubits: int, method: str):
        """Check if circuit size is within limits."""
        if total_qubits > self.max_total_qubits:
            msg = (f"Circuit too large for {method}: {total_qubits} qubits "
                   f"exceeds limit of {self.max_total_qubits}")
            logger.error(msg)
            raise CircuitTooLargeError(msg)
    
    # ==========================================================================
    # IQAE - v6.9: ZNE SUPPORT ADDED
    # ==========================================================================
    
    def _measure_amplitude_noisy(
        self,
        state_prep: 'QuantumCircuit',
        objective_qubits: List[int],
        noise_model,
        shots: int
    ) -> float:
        """
        Measure amplitude by running circuit on AerSimulator with noise model.
        
        This bypasses IterativeAmplitudeEstimation (which is incompatible with
        AerSampler) and directly measures P(objective_qubits=1) from counts.
        
        Used by ZNE to get noisy amplitude estimates at each scale factor.
        """
        # Build measurement circuit
        meas_circ = state_prep.copy()
        if not meas_circ.cregs:
            meas_circ.measure_all()
        
        # Transpile and run on noisy backend
        noisy_backend = AerSimulator(noise_model=noise_model)
        transpiled = transpile(meas_circ, backend=noisy_backend, optimization_level=1)
        job = noisy_backend.run(transpiled, shots=shots)
        counts = job.result().get_counts()
        
        # Extract probability that ALL objective qubits are |1âŸ©
        # In Qiskit, bit ordering is reversed: qubit 0 is rightmost in bitstring
        total_shots = sum(counts.values())
        
        # v7.2 TIER 1: Apply readout mitigation if available
        if self.readout_mitigator is not None and self.use_readout_mitigation:
            counts = self.readout_mitigator.correct_counts(counts)
            total_shots = sum(counts.values())
            self.stats['mem_corrections'] = self.stats.get('mem_corrections', 0) + 1
        
        good_counts = 0
        for bitstring, count in counts.items():
            bits = bitstring.replace(' ', '')
            all_obj_one = True
            for oq in objective_qubits:
                # Qiskit uses little-endian: qubit i is at position -(i+1)
                bit_pos = len(bits) - 1 - oq
                if bit_pos < 0 or bits[bit_pos] != '1':
                    all_obj_one = False
                    break
            if all_obj_one:
                good_counts += count
        
        return good_counts / total_shots
    
    # ========================================================================
    # TIER 2: ZNE ON STATE PREPARATION (v7.2)
    # ========================================================================
    
    def _zne_correct_state_prep_distribution(
        self,
        state_prep: 'QuantumCircuit',
        n_qubits: int
    ) -> Optional[np.ndarray]:
        """
        Apply ZNE to the state preparation circuit to get a corrected
        probability distribution.
        
        Runs the state prep circuit at multiple noise scales (via unitary
        folding or noise scaling), measures the output distribution, and
        extrapolates each probability p_i to zero noise.
        
        Args:
            state_prep: The MÃ¶ttÃ¶nen state preparation circuit.
            n_qubits: Number of distribution qubits.
            
        Returns:
            Corrected probability array of shape (2^n_qubits,), or None if failed.
        """
        if not self.use_zne_state_prep or not self.noise_enabled:
            return None
        if not NOISE_MODEL_AVAILABLE:
            return None
        
        n_states = 2 ** n_qubits
        logger.info(f"TIER2 ZNE-StatePrep: correcting distribution at "
                    f"{len(self.zne_scale_factors)} noise scales")
        
        # Collect distributions at each noise scale
        distributions_at_scale = []  # shape: (n_scales, n_states)
        valid_scales = []
        
        for scale in self.zne_scale_factors:
            try:
                # Build noise model at this scale
                scaled_model = NoiseModel()
                sq_s = min(self._base_sq_error * scale, 0.3)
                tq_s = min(self._base_tq_error * scale, 0.3)
                ro_s = min(self._base_ro_error * scale, 0.45)
                
                if sq_s > 0:
                    scaled_model.add_all_qubit_quantum_error(
                        depolarizing_error(sq_s, 1),
                        ['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'x', 'y', 'z', 'h', 's', 't']
                    )
                if tq_s > 0:
                    scaled_model.add_all_qubit_quantum_error(
                        depolarizing_error(tq_s, 2), ['cx', 'cz', 'swap']
                    )
                if ro_s > 0:
                    ro = ReadoutError([[1 - ro_s, ro_s], [ro_s, 1 - ro_s]])
                    scaled_model.add_all_qubit_readout_error(ro)
                
                # Run state prep + measurement
                meas_circ = state_prep.copy()
                meas_circ.measure_all()
                
                noisy_backend = AerSimulator(noise_model=scaled_model)
                transpiled = transpile(meas_circ, backend=noisy_backend, optimization_level=1)
                job = noisy_backend.run(transpiled, shots=self.shots)
                counts = job.result().get_counts()
                
                # Apply TIER 1 readout correction if available
                if self.readout_mitigator is not None and self.use_readout_mitigation:
                    counts = self.readout_mitigator.correct_counts(counts)
                
                # Convert to probability distribution
                total = sum(counts.values())
                dist = np.zeros(n_states)
                for bitstring, count in counts.items():
                    bits = bitstring.replace(' ', '')
                    idx = int(bits, 2)
                    if idx < n_states:
                        dist[idx] = count / total
                
                distributions_at_scale.append(dist)
                valid_scales.append(scale)
                logger.info(f"  ZNE-SP scale={scale:.1f}: measured {n_states} probabilities, "
                           f"max_p={dist.max():.4f}")
                
            except Exception as e:
                logger.warning(f"  ZNE-SP scale={scale:.1f} failed: {e}")
                continue
        
        if len(distributions_at_scale) < 2:
            logger.warning("TIER2 ZNE-StatePrep: not enough scale points, skipping correction")
            return None
        
        # Extrapolate each probability p_i(scale) -> p_i(0)
        scales_arr = np.array(valid_scales)
        dist_matrix = np.array(distributions_at_scale)  # (n_scales, n_states)
        
        corrected_dist = np.zeros(n_states)
        for i in range(n_states):
            p_values = dist_matrix[:, i]
            if self.zne_extrapolation == 'richardson' and len(scales_arr) >= 3:
                corrected_dist[i] = self._richardson_extrapolation(scales_arr, p_values)
            else:
                corrected_dist[i] = self._linear_extrapolation(scales_arr[:2], p_values[:2])
        
        # Clip negatives and renormalize to valid probability distribution
        corrected_dist = np.maximum(corrected_dist, 0.0)
        dist_sum = corrected_dist.sum()
        if dist_sum > 0:
            corrected_dist /= dist_sum
        else:
            logger.warning("TIER2 ZNE-StatePrep: all probabilities zero after correction")
            return None
        
        self.stats['zne_state_prep_corrections'] = self.stats.get('zne_state_prep_corrections', 0) + 1
        logger.info(f"TIER2 ZNE-StatePrep: corrected distribution, "
                   f"max_p={corrected_dist.max():.4f}, entropy_ratio="
                   f"{(-np.sum(corrected_dist[corrected_dist>0] * np.log2(corrected_dist[corrected_dist>0]))) / np.log2(n_states):.3f}")
        
        return corrected_dist
    
    def estimate_amplitude_iqae(
        self,
        state_prep: 'QuantumCircuit',
        objective_qubits: List[int],
        name: str = "iqae"
    ) -> QAEResult:
        """
        Estimate amplitude using Iterative QAE.
        
        v7.3: self.sampler is AerSamplerV2 (noisy) when noise_enabled=True,
        or StatevectorSampler (ideal) otherwise. IQAE runs directly on
        whichever sampler is configured â€” no separate ZNE path needed here.
        Inner ZNE (use_zne) still works via _apply_zne_to_amplitude for
        multi-scale extrapolation if enabled.
        """
        start_time = time.time()
        logger.section(f"IQAE ESTIMATION: {name}")
        
        self._check_circuit_size(state_prep.num_qubits, 'IQAE')
        
        png_path = self.exporter.export_png(state_prep, f"iqae_stateprep_{name}", 'qae_circuit',
            context={'method': 'iqae', 'num_qubits': state_prep.num_qubits})
        
        if self.use_zne and self.noise_enabled:
            # v7.4: ZNE via unitary folding on the same circuit + same sampler
            def _run_at_sampler(sampler):
                return self._run_iqae_single(state_prep, objective_qubits, name, sampler)
            
            zne_amplitude, zne_improvement, zne_details = self._apply_zne_to_amplitude(
                _run_at_sampler, name=f"iqae_{name}",
                state_prep=state_prep,
                objective_qubits=objective_qubits
            )
            
            result = zne_details.get('result', _run_at_sampler(self.sampler))
            result.amplitude = zne_amplitude
            result.probability = zne_amplitude
            result.png_path = png_path
            result.execution_time_ms = (time.time() - start_time) * 1000
            self.stats['png_files'].append(png_path)
        else:
            # Standard: single run with self.sampler (noisy or ideal)
            result = self._run_iqae_single(state_prep, objective_qubits, name, self.sampler)
            result.execution_time_ms = (time.time() - start_time) * 1000
            result.png_path = png_path
            self.stats['png_files'].append(png_path)
        
        # =====================================================================
        # TIER 1 FIX: Apply MEM post-correction to IQAE amplitude.
        #
        # IQAE internally processes raw (noisy) counts and returns
        # result.estimation which is P(objective=|1>) measured under readout
        # noise.  The analytical correction is:
        #     P_corrected = P_raw / (1 - 2*eps)^k
        # where eps = readout_error and k = number of objective qubits.
        # This inverts the symmetric readout channel: if the true probability
        # is p, the measured probability is p*(1-eps) + (1-p)*eps
        #   = p*(1-2*eps) + eps, so p = (P_measured - eps)/(1-2*eps).
        # For small eps, the simpler 1/(1-2*eps)^k form works well.
        # =====================================================================
        if (self.readout_mitigator is not None
                and self.use_readout_mitigation
                and self.noise_enabled):
            raw_amp = result.amplitude
            corrected_amp = self.readout_mitigator.correct_amplitude(
                raw_amp, n_objective_qubits=len(objective_qubits)
            )
            result.amplitude = corrected_amp
            result.probability = corrected_amp
            # Also correct confidence interval bounds
            ci_lo = self.readout_mitigator.correct_amplitude(
                result.confidence_interval[0], n_objective_qubits=len(objective_qubits)
            )
            ci_hi = self.readout_mitigator.correct_amplitude(
                result.confidence_interval[1], n_objective_qubits=len(objective_qubits)
            )
            result.confidence_interval = (ci_lo, ci_hi)
            self.stats['mem_corrections'] = self.stats.get('mem_corrections', 0) + 1
            logger.info(f"TIER1 MEM correction: {raw_amp:.6f} -> {corrected_amp:.6f} "
                        f"(ro={self.readout_mitigator.readout_error:.4f}, "
                        f"k={len(objective_qubits)})")
        
        return result
    
    # ==================================================================
    # IQAE QUERY EXTRACTION â€” v7.5
    # ==================================================================
    
    def _extract_iqae_queries(self, result, name: str = "") -> int:
        """
        Extract the true oracle query count from a Qiskit IQAE result.
        
        Qiskit's IterativeAmplitudeEstimationResult.num_oracle_queries
        returns 0 when using V2 Sampler primitives (StatevectorSampler,
        AerSamplerV2) because the internal tracking hook
        (_sampler._circuit_counter) is absent from the V2 API.
        
        We recover the actual count via three fallback strategies:
        
        1. DIRECT: result.num_oracle_queries (if non-zero, trust it).
        2. CIRCUIT_RESULTS: len(result.circuit_results) gives the number
           of IQAE rounds.  Each round k uses Grover power k_i, calling
           the oracle (2*k_i + 1) times.  The powers follow the IQAE
           schedule from Grinko et al. (2021), Â§Algorithm 2.
        3. THEORETICAL: If neither works, compute from the IQAE
           convergence bound:
             T_max = ceil(log2(pi / (8 * epsilon)))
             Total queries ~ sum_{i=0}^{T_max} shots * (2*K_i + 1)
           where K_i is the Grover power at round i.
           
        All counts are per-call (not multiplied by shots), consistent
        with the QAE literature convention where "oracle query" means
        one application of the Grover operator Q = A S_0 Aâ€  S_chi.
        
        Returns:
            int: Total number of oracle calls (Grover operator applications).
        """
        # Strategy 1: Direct attribute
        direct = 0
        try:
            direct = int(result.num_oracle_queries)
        except (AttributeError, TypeError):
            pass
        
        if direct > 0:
            return direct
        
        # Strategy 2: Reconstruct from circuit_results
        #
        # result.circuit_results is a list with one entry per IQAE round.
        # The Grover power schedule for IQAE follows:
        #   Round 0: k=0 (just measure A|0>)
        #   Round i: k_i chosen adaptively, but bounded by
        #            K_max = ceil(pi / (8 * epsilon_target)) 
        # 
        # Since we don't have k_i per round, we use the number of rounds
        # to compute the total from the IQAE theoretical schedule.
        # Grinko et al. Theorem 3: total queries <= 
        #   (32/epsilon) * log(2*log2(pi/(4*epsilon)) / alpha)
        #
        # Practical formula from Qiskit source (qiskit-algorithms):
        #   num_rounds = ceil(log2(pi / (8 * epsilon)))
        #   powers = [0, 1, 2, 4, 8, ..., 2^(num_rounds-1)]  (approx.)
        #   total_queries = sum(2*k + 1 for k in powers)
        
        num_rounds = 0
        try:
            circuit_results = getattr(result, 'circuit_results', None)
            if circuit_results is not None and hasattr(circuit_results, '__len__'):
                num_rounds = len(circuit_results)
        except Exception:
            pass
        
        if num_rounds > 0:
            # Reconstruct total queries from IQAE doubling schedule.
            # IQAE typically uses powers: k_0=0, then k_i roughly
            # doubling each round.  The exact sequence depends on the
            # adaptive CI narrowing, but a tight upper bound is:
            #   k_i <= 2^(i-1) for i >= 1, k_0 = 0
            # Total = sum_{i=0}^{R-1} (2*k_i + 1)
            #       = R + 2 * sum_{i=1}^{R-1} 2^(i-1)
            #       = R + 2 * (2^(R-1) - 1)
            #       = R + 2^R - 2
            #
            # This gives the total Grover oracle applications (not shots).
            R = num_rounds
            total_queries = R + (1 << R) - 2  # R + 2^R - 2
            total_queries = max(total_queries, R)  # at least R
            logger.info(f"IQAE queries from circuit_results: "
                        f"{num_rounds} rounds -> {total_queries} oracle calls"
                        f"{f' ({name})' if name else ''}")
            return total_queries
        
        # Strategy 3: Theoretical estimate from epsilon
        # T_max rounds, each roughly doubling the Grover power.
        # Total ~ pi/(4*epsilon) by IQAE's O(1/epsilon) guarantee.
        import math
        T_max = max(1, math.ceil(math.log2(math.pi / (8.0 * self.epsilon))))
        total_theoretical = T_max + (1 << T_max) - 2
        total_theoretical = max(total_theoretical, 1)
        logger.warning(f"IQAE queries: theoretical estimate = {total_theoretical} "
                       f"(epsilon={self.epsilon:.4f}, T_max={T_max} rounds)"
                       f"{f' ({name})' if name else ''}")
        return total_theoretical
    
    def _run_iqae_single(
        self,
        state_prep: 'QuantumCircuit',
        objective_qubits: List[int],
        name: str,
        sampler
    ) -> QAEResult:
        """
        Run a single IQAE estimation with given sampler.

        [T-001] Diagnostic logging added:
        - logs shots_per_round schedule so undersampling of high-k rounds
          (Hypothesis A) is visible in qae_circuits.txt.
        - epsilon_target and estimated n_rounds are logged at INFO level
          so the shot budget distribution is traceable per n_qubits value.
        """
        
        # v7.8: TranspilingAerSamplerV2 wrapper handles the GroverOperator
        # transpilation automatically via its run() interceptor.  We still
        # transpile state_prep here to ensure the state preparation itself
        # (which may contain Mottonen/initialize gates) is also in basis
        # gates before IQAE uses it to build the Grover operator.
        _is_aer = AER_SAMPLER_V2_AVAILABLE and (
            isinstance(sampler, _AerSamplerV2)
            or isinstance(sampler, TranspilingAerSamplerV2)
        )
        if _is_aer:
            state_prep_exec = transpile(
                state_prep,
                basis_gates=['u1', 'u2', 'u3', 'cx', 'id', 'rz', 'sx', 'x'],
                optimization_level=1
            )
        else:
            state_prep_exec = state_prep
        
        def run_iqae():
            problem = EstimationProblem(
                state_preparation=state_prep_exec,
                objective_qubits=objective_qubits
            )
            # FIX (qae_discretization_fix.md Bug 3):
            # Compute T_max from Grinko et al. (2021) Theorem 1:
            # K_min = ceil(log2(pi / (8 * epsilon)))
            # Previously capped at 5 in config; now computed automatically.
            import math as _math_iqae
            _T_max = max(1, _math_iqae.ceil(
                _math_iqae.log2(_math_iqae.pi / (8.0 * self.epsilon))
            ))
            iqae = IterativeAmplitudeEstimation(
                epsilon_target=self.epsilon,
                alpha=self.alpha_confidence,
                sampler=sampler
            )
            iqae.max_iterations = _T_max  # enforce theoretical minimum
            # [T-001] Log shots-per-round diagnostic before estimation.
            # IQAE schedule: round r uses k_r = 2^r Grover amplifications.
            # With fixed shot budget S, shots_per_round = S / n_rounds.
            # For n_qubits=6: K_max = ceil(pi/(8*eps)) ~ 196 => n_rounds >= 7.
            # For n_qubits=3: K_max ~ 25               => n_rounds >= 4.
            # High-k rounds (large r) become undersampled first (Hypothesis A).
            import math
            k_max_est = math.ceil(math.pi / (8 * self.epsilon))
            n_rounds_est = max(1, math.ceil(math.log2(k_max_est))) if k_max_est > 1 else 1
            # shots per IQAE invocation are set via the sampler's default_shots
            sampler_shots = getattr(sampler, '_default_shots',
                            getattr(sampler, 'default_shots',
                            getattr(getattr(sampler, '_inner', None),
                                    '_default_shots', None)))
            shots_per_round_est = (
                int(sampler_shots) // max(n_rounds_est, 1)
                if sampler_shots is not None else None
            )
            logger.info(
                f"[T-001] IQAE diagnostic [{name}]: "
                f"epsilon={self.epsilon:.4f}, "
                f"n_qubits={state_prep.num_qubits}, "
                f"k_max_est={k_max_est}, "
                f"n_rounds_est={n_rounds_est}, "
                f"sampler_shots={sampler_shots}, "
                f"shots_per_round_est={shots_per_round_est}"
            )
            if shots_per_round_est is not None and shots_per_round_est < 50:
                logger.warning(
                    f"[T-001 Hypothesis A] shots_per_round_est={shots_per_round_est} < 50 "
                    f"for n_qubits={state_prep.num_qubits}: "
                    f"high-k rounds may be UNDERSAMPLED. "
                    f"Consider increasing shots to {n_rounds_est * 200}+ for n={state_prep.num_qubits}."
                )
            return iqae.estimate(problem)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_iqae)
                try:
                    result = future.result(timeout=self.iqae_timeout)
                except FuturesTimeoutError:
                    msg = f"IQAE timeout after {self.iqae_timeout}s"
                    logger.error(msg)
                    raise QAETimeoutError(msg)
            
            amplitude = float(result.estimation)
            ci = (float(result.confidence_interval[0]), float(result.confidence_interval[1]))
            
            queries = self._extract_iqae_queries(result, name)
            
            logger.metric("Amplitude", amplitude)
            logger.metric("CI", ci)
            logger.metric("Queries", queries)
            
            self.stats['total_circuits'] += 1
            self.stats['total_queries'] += queries
            self.stats['total_queries_iqae'] += queries
            
            # [T-001] Attach diagnostic fields to the result for JSON export.
            # These enable post-hoc analysis of Hypothesis A/B/C.
            import math as _math
            _k_max = _math.ceil(_math.pi / (8 * self.epsilon))
            _n_rds = max(1, _math.ceil(_math.log2(_k_max))) if _k_max > 1 else 1
            _s_shots = getattr(sampler, '_default_shots',
                       getattr(sampler, 'default_shots',
                       getattr(getattr(sampler, '_inner', None),
                               '_default_shots', None)))
            _spr = int(_s_shots) // max(_n_rds, 1) if _s_shots is not None else None

            qae_res = QAEResult(
                method='iqae',
                amplitude=amplitude,
                probability=amplitude,
                confidence_interval=ci,
                num_oracle_queries=queries,
                num_grover_iterations=queries // 2,
                circuit_depth=state_prep.depth(),
                num_qubits=state_prep.num_qubits,
                execution_time_ms=0,
                converged=True,
                epsilon_achieved=ci[1] - ci[0] if ci[1] > ci[0] else self.epsilon,
                circuit_id=f"iqae_{name}"
            )
            # Attach T-001 diagnostics as extra attributes (not in dataclass schema)
            qae_res._t001_shots_per_round_est = _spr
            qae_res._t001_n_rounds_est        = _n_rds
            qae_res._t001_k_max_est           = _k_max
            qae_res._t001_sampler_shots       = _s_shots
            return qae_res
            
        except (QAETimeoutError, CircuitTooLargeError):
            raise
        except Exception as e:
            msg = f"IQAE execution failed: {e}"
            logger.error(msg)
            raise QAEExecutionError(msg)
    
    # ==========================================================================
    # WOERNER - v6.9 FIXED: SAFETY FACTOR + VALIDATION RETRY + ZNE
    # ==========================================================================
    
    def estimate_amplitude_woerner(
        self,
        state_prep: 'QuantumCircuit',
        objective_qubits: List[int],
        name: str = "woerner",
        reference_amplitude: Optional[float] = None
    ) -> QAEResult:
        """
        Estimate amplitude using Woerner-Egger canonical QAE.
        
        v6.9 FIXES:
        1. 2x safety factor: Îµ_target = Îµ / SAFETY_FACTOR â†’ starts at 6 qubits
           instead of 5 for Îµ=0.05
        2. Validation retry: if reference_amplitude is provided and relative error
           exceeds WOERNER_RETRY_ERROR_THRESHOLD (15%), retry with +1 eval qubit
           up to WOERNER_MAX_RETRIES times
        3. ZNE: when enabled, runs at multiple noise scales and extrapolates
        
        Precision at different eval qubit counts:
        - 5 qubits: M=32, Îµ â‰ˆ 0.049 (too coarse for a=0.05)
        - 6 qubits: M=64, Îµ â‰ˆ 0.025 (good)
        - 7 qubits: M=128, Îµ â‰ˆ 0.012 (very good)
        - 8 qubits: M=256, Îµ â‰ˆ 0.006 (excellent)
        """
        start_time = time.time()
        logger.section(f"WOERNER QAE (v6.9 FIXED): {name}")
        
        # ======================================================================
        # ADAPTIVE EVALUATION QUBIT CALCULATION (v6.9: 2x safety factor)
        # ======================================================================
        
        # v6.9: Apply safety factor to epsilon for initial qubit calculation
        effective_target_epsilon = self.epsilon / WOERNER_SAFETY_FACTOR
        
        # Required eval qubits: Îµ = Ï€ / (2 Â· 2^num_eval) < effective_target_epsilon
        num_eval_raw = int(np.ceil(np.log2(np.pi / (2 * effective_target_epsilon))))
        
        # Apply bounds with INCREASED minimum
        num_eval_initial = max(MIN_WOERNER_EVAL_QUBITS, num_eval_raw)
        num_eval_initial = min(num_eval_initial, self.max_woerner_eval_qubits)
        
        # Check total qubit budget
        total_qubits = state_prep.num_qubits + num_eval_initial
        if total_qubits > self.max_total_qubits:
            num_eval_initial = self.max_total_qubits - state_prep.num_qubits
            if num_eval_initial < 4:
                raise CircuitTooLargeError(
                    f"Cannot fit Woerner QAE: need at least 4 eval qubits but only "
                    f"{num_eval_initial} available"
                )
            logger.warning(f"Woerner eval qubits reduced to {num_eval_initial} due to qubit limit")
        
        logger.info(f"v6.9 Adaptive: initial {num_eval_initial} eval qubits "
                     f"(safety_factor={WOERNER_SAFETY_FACTOR}x, was 5 in v6.7)")
        if reference_amplitude is not None:
            logger.info(f"Reference amplitude for validation: {reference_amplitude:.6f}")
        
        # ======================================================================
        # RUN WITH RETRY LOOP (v6.9)
        # ======================================================================
        
        best_result = None
        best_error = float('inf')
        num_eval = num_eval_initial
        
        for attempt in range(1 + WOERNER_MAX_RETRIES):
            M = 2 ** num_eval
            effective_epsilon = np.pi / (2 * M)
            total_qubits = state_prep.num_qubits + num_eval
            
            logger.info(f"Woerner attempt {attempt + 1}: {num_eval} eval qubits, "
                        f"M={M}, Îµ={effective_epsilon:.6f}, total_qubits={total_qubits}")
            
            self._check_circuit_size(total_qubits, 'Woerner')
            
            if attempt == 0:
                png_path = self.exporter.export_png(
                    state_prep, f"woerner_stateprep_{name}", 'qae_circuit',
                    context={
                        'method': 'woerner_v69',
                        'num_eval': num_eval,
                        'M': M,
                        'effective_epsilon': effective_epsilon,
                        'safety_factor': WOERNER_SAFETY_FACTOR
                    }
                )
            
            try:
                result = self._run_woerner_single(
                    state_prep, objective_qubits, num_eval, name
                )
                
                # Validate against reference if available
                if reference_amplitude is not None and reference_amplitude > 1e-10:
                    rel_error = abs(result.amplitude - reference_amplitude) / reference_amplitude
                    logger.info(f"Woerner validation: amplitude={result.amplitude:.6f}, "
                               f"reference={reference_amplitude:.6f}, "
                               f"rel_error={rel_error:.4f} ({rel_error*100:.1f}%)")
                    
                    if rel_error < best_error:
                        best_error = rel_error
                        best_result = result
                    
                    if rel_error <= WOERNER_RETRY_ERROR_THRESHOLD:
                        logger.info(f"Woerner PASSED validation at {num_eval} eval qubits "
                                    f"(error {rel_error*100:.1f}% <= {WOERNER_RETRY_ERROR_THRESHOLD*100}%)")
                        break
                    else:
                        # Try with more qubits
                        if attempt < WOERNER_MAX_RETRIES:
                            next_eval = num_eval + 1
                            next_total = state_prep.num_qubits + next_eval
                            if (next_eval <= self.max_woerner_eval_qubits and 
                                next_total <= self.max_total_qubits):
                                logger.warning(
                                    f"Woerner error {rel_error*100:.1f}% > "
                                    f"{WOERNER_RETRY_ERROR_THRESHOLD*100}% threshold. "
                                    f"Retrying with {next_eval} eval qubits..."
                                )
                                num_eval = next_eval
                                self.stats['woerner_retries'] += 1
                                continue
                            else:
                                logger.warning(
                                    f"Cannot increase eval qubits beyond {num_eval} "
                                    f"(max_eval={self.max_woerner_eval_qubits}, "
                                    f"max_total={self.max_total_qubits}). "
                                    f"Using best result with error {best_error*100:.1f}%"
                                )
                                break
                        else:
                            logger.warning(
                                f"Max retries ({WOERNER_MAX_RETRIES}) reached. "
                                f"Best error: {best_error*100:.1f}%"
                            )
                            break
                else:
                    # No reference: accept first result
                    best_result = result
                    break
                    
            except (QAETimeoutError, CircuitTooLargeError):
                raise
            except Exception as e:
                if attempt < WOERNER_MAX_RETRIES:
                    logger.warning(f"Woerner attempt {attempt + 1} failed: {e}. Retrying...")
                    num_eval = min(num_eval + 1, self.max_woerner_eval_qubits)
                    continue
                raise
        
        if best_result is None:
            raise QAEExecutionError("Woerner: all attempts failed")
        
        # Update PNG path on best result
        best_result.png_path = png_path
        
        execution_time = (time.time() - start_time) * 1000
        best_result.execution_time_ms = execution_time
        
        # ==================================================================
        # TIER 1 MEM: post-correction for Woerner (v7.5 / v7.8 FIX)
        # Same analytical correction as IQAE: P_corr = P_raw / (1-2*eps)^k
        #
        # v7.8 FIX: When noise_enabled=True AND use_zne=True, Woerner runs
        # on ideal_sampler (noiseless) because QPE is incompatible with Aer.
        # Applying readout correction to noiseless results DISTORTS them.
        # Only apply MEM when Woerner actually ran under noise.
        # ==================================================================
        woerner_ran_noisy = self.noise_enabled and not (
            self.use_zne and self.noise_enabled  # Woerner falls back to ideal
        )
        if (self.readout_mitigator is not None
                and self.use_readout_mitigation
                and woerner_ran_noisy):
            raw_amp = best_result.amplitude
            corrected_amp = self.readout_mitigator.correct_amplitude(
                raw_amp, n_objective_qubits=len(objective_qubits)
            )
            best_result.amplitude = corrected_amp
            best_result.probability = corrected_amp
            ci_lo = self.readout_mitigator.correct_amplitude(
                best_result.confidence_interval[0],
                n_objective_qubits=len(objective_qubits)
            )
            ci_hi = self.readout_mitigator.correct_amplitude(
                best_result.confidence_interval[1],
                n_objective_qubits=len(objective_qubits)
            )
            best_result.confidence_interval = (ci_lo, ci_hi)
            self.stats['mem_corrections'] = self.stats.get(
                'mem_corrections', 0) + 1
            logger.info(f"TIER1 MEM correction (Woerner): "
                        f"{raw_amp:.6f} -> {corrected_amp:.6f} "
                        f"(ro={self.readout_mitigator.readout_error:.4f}, "
                        f"k={len(objective_qubits)})")
        
        return best_result
    
    def _run_woerner_single(
        self,
        state_prep: 'QuantumCircuit',
        objective_qubits: List[int],
        num_eval: int,
        name: str
    ) -> QAEResult:
        """Run a single Woerner QAE estimation on ideal (statevector) sampler.
        
        v7.16: Woerner ALWAYS runs on StatevectorSampler because
        AmplitudeEstimation internally builds QPE as a composite gate
        that AerSamplerV2 cannot decompose.  No transpilation needed
        for StatevectorSampler (it simulates the full unitary).
        """
        
        M = 2 ** num_eval
        total_qubits = state_prep.num_qubits + num_eval
        
        def run_woerner():
            problem = EstimationProblem(
                state_preparation=state_prep,
                objective_qubits=objective_qubits
            )
            ae = AmplitudeEstimation(
                num_eval_qubits=num_eval, 
                sampler=self.ideal_sampler
            )
            return ae.estimate(problem)
        
        logger.info(f"Woerner: running on StatevectorSampler (noiseless), "
                    f"{num_eval} eval qubits, M={M}")
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_woerner)
                try:
                    result = future.result(timeout=self.woerner_timeout)
                except FuturesTimeoutError:
                    msg = f"Woerner timeout after {self.woerner_timeout}s"
                    logger.error(msg)
                    raise QAETimeoutError(msg)
            
            amplitude = float(result.estimation)
            err = np.pi / M + (np.pi ** 2) / (M ** 2)
            ci = (max(0, amplitude - err), min(1, amplitude + err))
            
            logger.metric("Amplitude", amplitude)
            logger.metric("Error bound", err)
            logger.metric("CI", ci)
            
            self.stats['total_circuits'] += 1
            self.stats['total_queries'] += M
            self.stats['total_queries_woerner'] += M
            
            return QAEResult(
                method='woerner',
                amplitude=amplitude,
                probability=amplitude,
                confidence_interval=ci,
                num_oracle_queries=M,
                num_grover_iterations=M,
                circuit_depth=state_prep.depth() * M,
                num_qubits=total_qubits,
                execution_time_ms=0,
                converged=True,
                epsilon_achieved=err,
                circuit_id=f"woerner_{name}"
            )
            
        except (QAETimeoutError, CircuitTooLargeError):
            raise
        except Exception as e:
            msg = f"Woerner execution failed: {e}"
            logger.error(msg)
            raise QAEExecutionError(msg)
    
    # ==========================================================================
    # TAIL PROBABILITY AND CVaR ESTIMATION METHODS
    # ==========================================================================
    
    def estimate_tail_probability(
        self,
        losses: np.ndarray,
        threshold: float,
        methods: List[str] = ['iqae']
    ) -> Dict[str, QAEResult]:
        """Estimate P(Loss >= threshold) using quantum amplitude estimation."""
        logger.section("QUANTUM TAIL PROBABILITY ESTIMATION")
        logger.info(f"Estimating P(L >= {threshold:.6f})")
        logger.metric("Scenarios", len(losses))
        logger.metric("Classical P(tail)", np.mean(losses >= threshold))
        
        self._loss_range = (float(np.min(losses)), float(np.max(losses)))
        
        circuit, obj_qubits, metadata = self.circuit_builder.build_tail_probability_circuit(
            losses, threshold, "tail_prob"
        )
        
        results = {}
        errors = {}
        
        # v6.9: Get classical tail prob as reference for Woerner validation
        classical_tail_prob = float(np.mean(losses >= threshold))
        
        for method in methods:
            try:
                if method.lower() == 'iqae':
                    results['iqae'] = self.estimate_amplitude_iqae(circuit, obj_qubits, "tail_prob")
                elif method.lower() == 'woerner':
                    # v6.9: Pass IQAE result or classical as reference for validation
                    reference = None
                    if 'iqae' in results and results['iqae'].converged:
                        reference = results['iqae'].amplitude
                    elif classical_tail_prob > 1e-10:
                        reference = classical_tail_prob
                    results['woerner'] = self.estimate_amplitude_woerner(
                        circuit, obj_qubits, "tail_prob",
                        reference_amplitude=reference
                    )
                else:
                    logger.warning(f"Unknown method '{method}', using IQAE")
                    results[method] = self.estimate_amplitude_iqae(circuit, obj_qubits, "tail_prob")
            except (CircuitTooLargeError, QAETimeoutError, QAEExecutionError) as e:
                errors[method] = str(e)
                logger.error(f"Method {method} failed: {e}")
        
        if not results and errors:
            error_msg = "; ".join([f"{m}: {e}" for m, e in errors.items()])
            raise QAEExecutionError(f"All QAE methods failed: {error_msg}")
        
        return results
    
    def estimate_conditional_expectation(
        self,
        asset_returns: np.ndarray,
        portfolio_losses: np.ndarray,
        threshold: float,
        asset_idx: int,
        method: str = 'iqae'
    ) -> Tuple[float, Optional[QAEResult]]:
        """Estimate E[r_j | L >= threshold]."""
        logger.info(f"Conditional expectation for asset {asset_idx}")
        
        circuit, obj_qubits, metadata = self.circuit_builder.build_conditional_expectation_circuit(
            asset_returns, portfolio_losses, threshold, asset_idx, "cond_exp"
        )
        
        if circuit is None:
            tail_mask = portfolio_losses >= threshold
            if not np.any(tail_mask):
                raise ValueError(f"No tail observations for asset {asset_idx}")
            classical_exp = float(np.mean(asset_returns[tail_mask]))
            logger.warning(f"No circuit for asset {asset_idx}, classical: {classical_exp}")
            return classical_exp, None
        
        if method.lower() == 'iqae':
            result = self.estimate_amplitude_iqae(circuit, obj_qubits, f"cond_exp_asset{asset_idx}")
        elif method.lower() == 'woerner':
            result = self.estimate_amplitude_woerner(circuit, obj_qubits, f"cond_exp_asset{asset_idx}")
        else:
            logger.warning(f"Unknown method '{method}', using IQAE")
            result = self.estimate_amplitude_iqae(circuit, obj_qubits, f"cond_exp_asset{asset_idx}")
        
        r_min = metadata['r_min']
        r_max = metadata['r_max']
        
        if r_max > r_min:
            expectation = result.amplitude * (r_max - r_min) + r_min
        else:
            expectation = r_min
        
        logger.metric(f"Asset {asset_idx} E[r|tail]", expectation)
        logger.metric(f"Asset {asset_idx} classical", metadata['classical_exp'])
        
        return expectation, result
    
    def estimate_tail_weighted_loss(
        self,
        losses: np.ndarray,
        threshold: float,
        methods: List[str] = ['iqae']
    ) -> Tuple[Dict[str, QAEResult], Dict]:
        """Estimate E[L Â· 1_{L >= threshold}] using quantum amplitude estimation."""
        logger.section("QUANTUM TAIL-WEIGHTED LOSS ESTIMATION")
        logger.info(f"Estimating E[L Â· 1_{{L >= {threshold:.6f}}}]")
        
        tail_mask = losses >= threshold
        classical_tail_prob = float(np.mean(tail_mask))
        if classical_tail_prob > 0:
            classical_E_L_tail = float(np.mean(losses[tail_mask])) * classical_tail_prob
        else:
            classical_E_L_tail = 0.0
        logger.metric("Classical E[LÂ·1_tail]", classical_E_L_tail)
        
        circuit, obj_qubits, metadata = self.circuit_builder.build_tail_weighted_loss_circuit(
            losses, threshold, "tail_weighted_loss"
        )
        
        l_min = metadata['l_min']
        l_max = metadata['l_max']
        
        results = {}
        errors = {}
        
        for method in methods:
            try:
                if method.lower() == 'iqae':
                    results['iqae'] = self.estimate_amplitude_iqae(circuit, obj_qubits, "tail_weighted_loss")
                elif method.lower() == 'woerner':
                    # v6.9: Use IQAE result as reference for Woerner validation
                    reference = results['iqae'].amplitude if 'iqae' in results else None
                    results['woerner'] = self.estimate_amplitude_woerner(
                        circuit, obj_qubits, "tail_weighted_loss",
                        reference_amplitude=reference
                    )
                else:
                    logger.warning(f"Unknown method '{method}', using IQAE")
                    results[method] = self.estimate_amplitude_iqae(circuit, obj_qubits, "tail_weighted_loss")
            except (CircuitTooLargeError, QAETimeoutError, QAEExecutionError) as e:
                errors[method] = str(e)
                logger.error(f"Method {method} failed for tail-weighted loss: {e}")
        
        if not results and errors:
            error_msg = "; ".join([f"{m}: {e}" for m, e in errors.items()])
            raise QAEExecutionError(f"All QAE methods failed for tail-weighted loss: {error_msg}")
        
        scaling = {
            'l_min': l_min,
            'l_max': l_max,
            'classical_tail_prob': classical_tail_prob,
            'classical_E_L_tail': classical_E_L_tail,
            'metadata': metadata
        }
        
        return results, scaling
    
    # ==========================================================================
    # TIER 2 HELPERS: Build circuits from corrected distribution (v7.4)
    # ==========================================================================
    
    def _build_circuit_from_corrected_probs(
        self,
        losses: np.ndarray,
        threshold: float,
        corrected_probs: np.ndarray,
        circuit_type: str = 'tail_prob'
    ) -> Tuple['QuantumCircuit', List[int], Dict]:
        """
        Build a QAE circuit using a ZNE-corrected probability distribution
        instead of the raw histogram.
        
        This replaces the state-preparation amplitudes with sqrt(corrected_probs),
        keeping the oracle / value-loading logic identical to the uncorrected path.
        
        Args:
            losses: Original loss array (for threshold computation).
            threshold: VaR threshold.
            corrected_probs: ZNE-corrected probability vector, shape (num_levels,).
            circuit_type: 'tail_prob' or 'tail_weighted_loss'.
        
        Returns:
            (circuit, objective_qubits, metadata)
        """
        n = self.num_distribution_qubits
        num_levels = 2 ** n
        
        # Get discretization edges (we reuse the same edges, only probs change)
        _, levels, edges, l_min, l_max, threshold_level = \
            self.circuit_builder._discretize_with_threshold(losses, threshold)
        
        # Ensure corrected_probs is valid
        probs = np.maximum(corrected_probs[:num_levels], 0.0)
        probs = probs / np.sum(probs)
        
        amplitudes = np.sqrt(probs)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        if circuit_type == 'tail_prob':
            # Distribution loading + threshold oracle
            dist_qc = QuantumCircuit(n, name="tier2_dist_load")
            dist_qc.initialize(amplitudes, range(n))
            dist_qc = dist_qc.decompose().decompose().decompose()
            
            clean_dist = QuantumCircuit(n, name="tier2_dist_clean")
            for inst in dist_qc.data:
                if inst.operation.name != 'reset':
                    clean_dist.append(inst.operation, inst.qubits, inst.clbits)
            
            oracle_qc = self.circuit_builder.build_threshold_oracle(
                n, threshold_level, "tier2_tail_oracle"
            )
            
            combined = QuantumCircuit(n + 1, name="tier2_tail_prob")
            combined.compose(clean_dist, qubits=range(n), inplace=True)
            combined.compose(oracle_qc, qubits=range(n + 1), inplace=True)
            
            metadata = {
                'threshold_level': threshold_level,
                'corrected': True,
                'quantum_expected_prob': float(np.sum(probs[threshold_level:])),
            }
            
            self.exporter.export_png(combined, "tier2_tail_prob", 'tail_probability',
                                     context=metadata)
            
            return combined, [n], metadata
        
        elif circuit_type == 'tail_weighted_loss':
            # Distribution loading + value-controlled rotations (tail only)
            if l_max > l_min:
                normalized_levels = (levels - l_min) / (l_max - l_min)
            else:
                normalized_levels = np.ones_like(levels) * 0.5
            normalized_levels = np.clip(normalized_levels, 0.0, 1.0)
            
            qc = QuantumCircuit(n + 1, name="tier2_tail_weighted_loss")
            ancilla = n
            
            init_qc = QuantumCircuit(n)
            init_qc.initialize(amplitudes, range(n))
            init_qc = init_qc.decompose().decompose().decompose()
            
            for inst in init_qc.data:
                if inst.operation.name != 'reset':
                    qc.append(inst.operation,
                              [q._index if hasattr(q, '_index') else q.index
                               for q in inst.qubits], [])
            
            for idx in range(threshold_level, num_levels):
                if idx >= len(normalized_levels):
                    continue
                value = normalized_levels[idx]
                if value > 1e-10 and value < (1 - 1e-10):
                    angle = 2 * np.arcsin(np.sqrt(value))
                elif value >= (1 - 1e-10):
                    angle = np.pi
                else:
                    angle = 0
                if abs(angle) < 1e-10:
                    continue
                
                # v7.19 FIX: reverse bit string for Qiskit little-endian convention
                idx_bits = format(idx, f'0{n}b')[::-1]
                for i, bit in enumerate(idx_bits):
                    if bit == '0':
                        qc.x(i)
                if n == 1:
                    qc.cry(angle, 0, ancilla)
                elif n == 2:
                    qc.cry(angle/2, 1, ancilla)
                    qc.cx(0, 1)
                    qc.cry(-angle/2, 1, ancilla)
                    qc.cx(0, 1)
                    qc.cry(angle/2, 0, ancilla)
                else:
                    from qiskit.circuit.library import RYGate
                    mcry = RYGate(angle).control(n)
                    qc.append(mcry, list(range(n)) + [ancilla])
                for i, bit in enumerate(idx_bits):
                    if bit == '0':
                        qc.x(i)
            
            metadata = {
                'threshold_level': threshold_level,
                'l_min': float(l_min),
                'l_max': float(l_max),
                'corrected': True,
            }
            
            self.exporter.export_png(qc, "tier2_tail_weighted_loss",
                                     'tail_weighted_loss', context=metadata)
            
            return qc, [ancilla], metadata
        
        else:
            raise ValueError(f"Unknown circuit_type: {circuit_type}")
    
    def _run_methods_on_circuit(
        self,
        circuit: 'QuantumCircuit',
        objective_qubits: List[int],
        methods: List[str],
        reference_amplitude: Optional[float],
        estimation_name: str,
        original_circuit: 'QuantumCircuit' = None,
        original_objective_qubits: List[int] = None
    ) -> Dict[str, QAEResult]:
        """
        Run one or more QAE methods on an already-built circuit.
        
        Args:
            circuit: The (possibly TIER2-corrected) circuit for IQAE.
            objective_qubits: Qubits to measure.
            methods: QAE methods to use.
            reference_amplitude: Reference for Woerner validation.
            estimation_name: Label for logging.
            original_circuit: If provided, Woerner uses this uncorrected
                circuit (since it runs noiseless, TIER2 correction would
                introduce bias).
            original_objective_qubits: Objective qubits for original_circuit.
        """
        results = {}
        errors = {}
        
        for method in methods:
            try:
                if method.lower() == 'iqae':
                    results['iqae'] = self.estimate_amplitude_iqae(
                        circuit, objective_qubits, estimation_name
                    )
                elif method.lower() == 'woerner':
                    ref = None
                    if 'iqae' in results and results['iqae'].converged:
                        ref = results['iqae'].amplitude
                    elif reference_amplitude is not None and reference_amplitude > 1e-10:
                        ref = reference_amplitude
                    
                    # v7.16: Woerner runs noiseless (ideal_sampler), so it
                    # should use the original (uncorrected) circuit when
                    # TIER2 is active.  TIER2 corrections compensate for
                    # noise â€” applying them to a noiseless simulation
                    # introduces systematic bias.
                    woerner_circuit = circuit
                    woerner_qubits = objective_qubits
                    if original_circuit is not None:
                        woerner_circuit = original_circuit
                        woerner_qubits = (original_objective_qubits 
                                          if original_objective_qubits is not None
                                          else objective_qubits)
                        logger.info(
                            f"Woerner using original (uncorrected) circuit "
                            f"for noiseless estimation")
                    
                    results['woerner'] = self.estimate_amplitude_woerner(
                        woerner_circuit, woerner_qubits, estimation_name,
                        reference_amplitude=ref
                    )
                else:
                    logger.warning(f"Unknown method '{method}', using IQAE")
                    results[method] = self.estimate_amplitude_iqae(
                        circuit, objective_qubits, estimation_name
                    )
            except (CircuitTooLargeError, QAETimeoutError, QAEExecutionError) as e:
                errors[method] = str(e)
                logger.error(f"Method {method} failed: {e}")
            except Exception as e:
                errors[method] = str(e)
                logger.error(f"Method {method} failed with unexpected error: {e}")
        
        if not results and errors:
            error_msg = "; ".join([f"{m}: {e}" for m, e in errors.items()])
            raise QAEExecutionError(f"All QAE methods failed: {error_msg}")
        
        return results
    
    # ==================================================================
    # v7.7: IQAE THREE-WAY COMPARISON (ideal / noisy-raw / noisy+ZNE)
    # ==================================================================
    
    def _run_iqae_comparison(
        self,
        circuit: 'QuantumCircuit',
        objective_qubits: List[int],
        name: str,
        original_circuit: 'QuantumCircuit' = None
    ) -> Dict[str, Optional[QAEResult]]:
        """
        Run IQAE on the same circuit under three noise regimes to quantify
        the NISQ gap and the effectiveness of error mitigation.
        
        Configurations:
            1. ideal:     StatevectorSampler â€” no noise, no ZNE, no MEM
            2. noisy_raw: AerSamplerV2       â€” noise ON, NO ZNE, NO MEM
            3. noisy_zne: AerSamplerV2       â€” noise ON + ZNE + MEM
        
        Args:
            circuit: The (possibly TIER2-corrected) circuit for noisy runs.
            objective_qubits: Qubits to measure.
            name: Label for logging.
            original_circuit: If provided, the ideal run uses this circuit
                (uncorrected) instead of ``circuit``.  This ensures a fair
                comparison: ideal measures the true amplitude while noisy
                runs measure what the corrected circuit produces.
        
        Returns dict with keys 'ideal', 'noisy_raw', 'noisy_zne'.
        Any mode that fails returns None for that key.
        """
        comp_results: Dict[str, Optional[QAEResult]] = {
            'ideal': None,
            'noisy_raw': None,
            'noisy_zne': None,
        }
        
        # The ideal run should use the original (uncorrected) circuit
        # when TIER2 is active, so it measures the true distribution
        # amplitude â€” not the one corrected for noise.
        ideal_circuit = original_circuit if original_circuit is not None else circuit
        
        # --- 1. IDEAL: run on ideal sampler, no corrections ---
        try:
            ideal_result = self._run_iqae_single(
                ideal_circuit, objective_qubits, f"{name}_ideal", self.ideal_sampler
            )
            comp_results['ideal'] = ideal_result
            logger.info(f"IQAE ideal ({name}): amplitude={ideal_result.amplitude:.6f}, "
                        f"queries={ideal_result.num_oracle_queries}")
        except Exception as e:
            logger.error(f"IQAE ideal ({name}) failed: {e}")
        
        # --- 2 & 3 only make sense when noise is active ---
        if not self.noise_enabled:
            comp_results['noisy_raw'] = comp_results['ideal']
            comp_results['noisy_zne'] = comp_results['ideal']
            return comp_results
        
        # --- 2. NOISY RAW: noisy sampler, no ZNE, no MEM ---
        try:
            raw_result = self._run_iqae_single(
                circuit, objective_qubits, f"{name}_noisy_raw", self.sampler
            )
            comp_results['noisy_raw'] = raw_result
            logger.info(f"IQAE noisy_raw ({name}): amplitude={raw_result.amplitude:.6f}, "
                        f"queries={raw_result.num_oracle_queries}")
        except Exception as e:
            logger.error(f"IQAE noisy_raw ({name}) failed: {e}")
        
        # --- 3. NOISY + ZNE + MEM: full existing mitigation pipeline ---
        try:
            zne_result = self.estimate_amplitude_iqae(circuit, objective_qubits, name)
            comp_results['noisy_zne'] = zne_result
            logger.info(f"IQAE noisy+ZNE+MEM ({name}): amplitude={zne_result.amplitude:.6f}, "
                        f"queries={zne_result.num_oracle_queries}")
        except Exception as e:
            logger.error(f"IQAE noisy+ZNE+MEM ({name}) failed: {e}")
        
        return comp_results
    
    def _compute_cvar_from_amplitudes(
        self,
        tail_prob_amplitude: float,
        twl_amplitude: float,
        l_min: float,
        l_max: float
    ) -> float:
        """
        Compute CVaR from tail probability and tail-weighted-loss amplitudes.
        
        CVaR = E[L | L >= VaR]
             = E[L Â· 1_{L>=VaR}] / P(L >= VaR)
             = (a_twl * (l_max - l_min) + l_min * a_tail) / a_tail
        
        where a_tail = tail_prob_amplitude, a_twl = twl_amplitude.
        """
        safe_tail = max(tail_prob_amplitude, 1e-8)
        E_L_tail = twl_amplitude * (l_max - l_min) + l_min * safe_tail
        return E_L_tail / safe_tail
    
    def estimate_cvar_complete(
        self,
        losses: np.ndarray,
        alpha: float = 0.05,
        methods: List[str] = ['iqae'],
        returns: np.ndarray = None,
        weights: np.ndarray = None
    ) -> QuantumCVaRResult:
        """Complete TRUE quantum CVaR estimation.
        
        v7.7: Includes IQAE three-way noise comparison (ideal/noisy_raw/noisy_zne).
        """
        logger.section("TRUE QUANTUM CVaR ESTIMATION (v7.7)")
        start_time = time.time()
        
        T = len(losses)
        self._loss_range = (float(np.min(losses)), float(np.max(losses)))
        l_min, l_max = self._loss_range
        
        # Classical reference
        sorted_losses = np.sort(losses)
        var_idx = int((1 - alpha) * T)
        classical_var = float(sorted_losses[var_idx])
        classical_cvar = float(np.mean(sorted_losses[var_idx:]))
        classical_tail_prob = float(np.mean(losses >= classical_var))
        
        logger.metric("Classical VaR", classical_var)
        logger.metric("Classical CVaR", classical_cvar)
        logger.metric("Classical P(tail)", classical_tail_prob)
        
        # =================================================================
        # TIER 2 FIX: Correct the state preparation distribution via ZNE
        # before building the QAE circuits.
        #
        # If TIER 2 is enabled, we:
        #   1. Build a preliminary state-prep circuit from the raw histogram.
        #   2. Run that circuit at multiple noise scales, measure the output
        #      distribution, and extrapolate each p_i to zero noise.
        #   3. Rebuild the tail-prob and tail-weighted-loss circuits using
        #      the corrected distribution.  This ensures that the amplitude
        #      IQAE estimates is based on noise-corrected state preparation.
        # =================================================================
        corrected_probs = None
        if self.use_zne_state_prep and self.noise_enabled:
            # Build a preliminary state-prep circuit from the raw discretization
            raw_probs, raw_levels, raw_edges, raw_l_min, raw_l_max, raw_thr_level = \
                self.circuit_builder._discretize_with_threshold(losses, classical_var)
            
            n = self.num_distribution_qubits
            amplitudes_prelim = np.sqrt(raw_probs)
            amplitudes_prelim = amplitudes_prelim / np.linalg.norm(amplitudes_prelim)
            
            prelim_qc = QuantumCircuit(n, name="tier2_state_prep")
            prelim_qc.initialize(amplitudes_prelim, range(n))
            prelim_qc = prelim_qc.decompose().decompose().decompose()
            
            # Remove reset gates
            clean_qc = QuantumCircuit(n, name="tier2_state_prep_clean")
            for inst in prelim_qc.data:
                if inst.operation.name != 'reset':
                    clean_qc.append(inst.operation, inst.qubits, inst.clbits)
            
            corrected_probs = self._zne_correct_state_prep_distribution(clean_qc, n)
            
            if corrected_probs is not None:
                logger.info(f"TIER2: Corrected distribution applied. "
                            f"P(tail) raw={float(np.sum(raw_probs[raw_thr_level:])):.6f}, "
                            f"corrected={float(np.sum(corrected_probs[raw_thr_level:])):.6f}")
        
        # Step 1: Quantum tail probability
        #
        # v7.16: When TIER2 is active, also build original (uncorrected)
        # circuits.  These are needed for:
        #   - Woerner: runs noiseless, so TIER2 correction would add bias
        #   - Noise Impact Comparison: ideal baseline uses true distribution
        orig_tail_circuit = None
        orig_tail_qubits = None
        orig_twl_circuit = None
        orig_twl_qubits = None
        
        if corrected_probs is not None:
            # Build original circuits from raw distribution
            orig_tail_circuit, orig_tail_qubits, _ = \
                self.circuit_builder.build_tail_probability_circuit(
                    losses, classical_var, "orig_tail_prob"
                )
            orig_twl_circuit, orig_twl_qubits, _ = \
                self.circuit_builder.build_tail_weighted_loss_circuit(
                    losses, classical_var, "orig_twl"
                )
            
            # Rebuild circuit with corrected probabilities
            tail_circuit, tail_obj_qubits, tail_metadata = \
                self._build_circuit_from_corrected_probs(
                    losses, classical_var, corrected_probs, circuit_type='tail_prob'
                )
            tail_results = self._run_methods_on_circuit(
                tail_circuit, tail_obj_qubits, methods, classical_tail_prob,
                estimation_name="tail_prob",
                original_circuit=orig_tail_circuit,
                original_objective_qubits=orig_tail_qubits
            )
        else:
            tail_results = self.estimate_tail_probability(losses, classical_var, methods)
        
        tail_prob_result = None
        quantum_tail_prob = None
        
        for method in methods:
            if method in tail_results and tail_results[method].converged:
                quantum_tail_prob = tail_results[method].amplitude
                tail_prob_result = tail_results[method]
                break
        
        if quantum_tail_prob is None:
            raise QAEExecutionError("Failed to estimate tail probability via QAE")
        
        quantum_tail_prob_safe = max(quantum_tail_prob, 1e-8)
        tail_prob_error = abs(quantum_tail_prob - classical_tail_prob) / max(classical_tail_prob, 1e-10)
        
        # Step 2: Quantum tail-weighted loss
        if corrected_probs is not None:
            twl_circuit, twl_obj_qubits, twl_metadata = \
                self._build_circuit_from_corrected_probs(
                    losses, classical_var, corrected_probs, circuit_type='tail_weighted_loss'
                )
            tail_weighted_results_raw = self._run_methods_on_circuit(
                twl_circuit, twl_obj_qubits, methods, None,
                estimation_name="tail_weighted_loss",
                original_circuit=orig_twl_circuit,
                original_objective_qubits=orig_twl_qubits
            )
            # Package in the format expected downstream
            tail_weighted_results = tail_weighted_results_raw
            # Recover scaling info from the corrected discretization
            _, corr_levels, corr_edges, corr_l_min, corr_l_max, corr_thr = \
                self.circuit_builder._discretize_with_threshold(losses, classical_var)
            scaling = {
                'l_min': corr_l_min,
                'l_max': corr_l_max,
                'classical_tail_prob': classical_tail_prob,
                'classical_E_L_tail': float(np.mean(losses[losses >= classical_var])) * classical_tail_prob if classical_tail_prob > 0 else 0.0,
                'metadata': {'l_min': corr_l_min, 'l_max': corr_l_max}
            }
        else:
            tail_weighted_results, scaling = self.estimate_tail_weighted_loss(losses, classical_var, methods)
        
        quantum_amplitude_E_L_tail = None
        E_L_tail_result = None
        
        for method in methods:
            if method in tail_weighted_results and tail_weighted_results[method].converged:
                quantum_amplitude_E_L_tail = tail_weighted_results[method].amplitude
                E_L_tail_result = tail_weighted_results[method]
                break
        
        if quantum_amplitude_E_L_tail is None:
            raise QAEExecutionError("Failed to estimate E[LÂ·1_tail] via QAE")
        
        E_L_norm_tail = quantum_amplitude_E_L_tail
        quantum_E_L_tail = E_L_norm_tail * (l_max - l_min) + l_min * quantum_tail_prob_safe
        
        # Step 3: Compute quantum CVaR
        quantum_cvar = quantum_E_L_tail / quantum_tail_prob_safe
        quantum_var = classical_var
        
        cvar_error = abs(quantum_cvar - classical_cvar) / max(abs(classical_cvar), 1e-10)
        
        logger.metric("Quantum CVaR", quantum_cvar)
        logger.metric("CVaR relative error", cvar_error)
        
        # =================================================================
        # v7.9: PER-METHOD CVaR â€” compute CVaR for each method individually
        # so that the caller can compare IQAE vs Woerner directly.
        # =================================================================
        per_method_cvar = {}
        for method in methods:
            m_tail = tail_results.get(method)
            m_twl = tail_weighted_results.get(method)
            if (m_tail is not None and m_tail.converged
                    and m_twl is not None and m_twl.converged):
                m_tail_prob = max(m_tail.amplitude, 1e-8)
                m_E_L_tail = m_twl.amplitude * (l_max - l_min) + l_min * m_tail_prob
                m_cvar = m_E_L_tail / m_tail_prob
                per_method_cvar[method] = float(m_cvar)
                m_err = abs(m_cvar - classical_cvar) / max(abs(classical_cvar), 1e-10)
                logger.metric(f"CVaR ({method})", m_cvar)
                logger.metric(f"CVaR error ({method})", m_err)
            else:
                logger.warning(f"Cannot compute per-method CVaR for '{method}': "
                               f"tail_prob={'OK' if m_tail and m_tail.converged else 'FAIL'}, "
                               f"twl={'OK' if m_twl and m_twl.converged else 'FAIL'}")
        
        # =================================================================
        # v7.7: IQAE THREE-WAY COMPARISON
        #
        # Run IQAE on the SAME circuits under three noise regimes:
        #   ideal:     StatevectorSampler â€” ground truth for IQAE
        #   noisy_raw: AerSamplerV2       â€” NISQ without mitigation
        #   noisy_zne: AerSamplerV2+ZNE+MEM â€” NISQ with mitigation
        #
        # v7.16: When TIER2 is active, the ideal run uses the ORIGINAL
        # (uncorrected) circuit so it measures the true distribution
        # amplitude.  Noisy runs use the TIER2-corrected circuit because
        # that's what the pipeline actually deploys on NISQ.  This makes
        # the comparison fair:
        #   - ideal shows what IQAE *should* measure (ground truth)
        #   - noisy_raw shows how noise degrades the corrected circuit
        #   - noisy_zne shows how ZNE+MEM recover from noise
        # =================================================================
        
        iqae_cvar_ideal = None
        iqae_cvar_noisy_raw = None
        iqae_cvar_noisy_zne = None
        iqae_tail_prob_ideal = None
        iqae_tail_prob_noisy_raw = None
        iqae_tail_prob_noisy_zne = None
        iqae_twl_amp_ideal = None
        iqae_twl_amp_noisy_raw = None
        iqae_twl_amp_noisy_zne = None
        
        if 'iqae' in methods:
            logger.section("IQAE NOISE IMPACT COMPARISON (v7.16)")
            
            # v7.16: When TIER2 is active, orig circuits were already built
            # in Step 1 above.  When TIER2 is off, build them now.
            if orig_tail_circuit is None:
                orig_tail_circuit, orig_tail_qubits, _ = \
                    self.circuit_builder.build_tail_probability_circuit(
                        losses, classical_var, "comp_tail_prob"
                    )
            if orig_twl_circuit is None:
                orig_twl_circuit, orig_twl_qubits, _ = \
                    self.circuit_builder.build_tail_weighted_loss_circuit(
                        losses, classical_var, "comp_twl"
                    )
            
            # Determine which circuits the noisy runs use.
            # When TIER2 is active, noisy runs use the corrected circuits.
            # When TIER2 is off, noisy and ideal use the same circuits.
            if corrected_probs is not None:
                noisy_tail_circuit = tail_circuit
                noisy_tail_qubits = tail_obj_qubits
                noisy_twl_circuit = twl_circuit
                noisy_twl_qubits = twl_obj_qubits
            else:
                noisy_tail_circuit = orig_tail_circuit
                noisy_tail_qubits = orig_tail_qubits
                noisy_twl_circuit = orig_twl_circuit
                noisy_twl_qubits = orig_twl_qubits
            
            # Run tail_prob comparison
            tp_comp = self._run_iqae_comparison(
                noisy_tail_circuit, noisy_tail_qubits, "tail_prob",
                original_circuit=orig_tail_circuit
            )
            
            # Run tail_weighted_loss comparison
            twl_comp = self._run_iqae_comparison(
                noisy_twl_circuit, noisy_twl_qubits, "tail_weighted_loss",
                original_circuit=orig_twl_circuit
            )
            
            # Extract amplitudes
            if tp_comp['ideal'] is not None:
                iqae_tail_prob_ideal = tp_comp['ideal'].amplitude
            if tp_comp['noisy_raw'] is not None:
                iqae_tail_prob_noisy_raw = tp_comp['noisy_raw'].amplitude
            if tp_comp['noisy_zne'] is not None:
                iqae_tail_prob_noisy_zne = tp_comp['noisy_zne'].amplitude
            
            if twl_comp['ideal'] is not None:
                iqae_twl_amp_ideal = twl_comp['ideal'].amplitude
            if twl_comp['noisy_raw'] is not None:
                iqae_twl_amp_noisy_raw = twl_comp['noisy_raw'].amplitude
            if twl_comp['noisy_zne'] is not None:
                iqae_twl_amp_noisy_zne = twl_comp['noisy_zne'].amplitude
            
            # Compute CVaR for each mode
            if iqae_tail_prob_ideal is not None and iqae_twl_amp_ideal is not None:
                iqae_cvar_ideal = self._compute_cvar_from_amplitudes(
                    iqae_tail_prob_ideal, iqae_twl_amp_ideal, l_min, l_max
                )
            if iqae_tail_prob_noisy_raw is not None and iqae_twl_amp_noisy_raw is not None:
                iqae_cvar_noisy_raw = self._compute_cvar_from_amplitudes(
                    iqae_tail_prob_noisy_raw, iqae_twl_amp_noisy_raw, l_min, l_max
                )
            if iqae_tail_prob_noisy_zne is not None and iqae_twl_amp_noisy_zne is not None:
                iqae_cvar_noisy_zne = self._compute_cvar_from_amplitudes(
                    iqae_tail_prob_noisy_zne, iqae_twl_amp_noisy_zne, l_min, l_max
                )
            
            # Log the comparison
            logger.info("=" * 60)
            logger.info("IQAE NOISE IMPACT SUMMARY")
            logger.info("=" * 60)
            logger.metric("Classical CVaR", classical_cvar)
            
            for label, cvar_val, tp_val, twl_val in [
                ("ideal",     iqae_cvar_ideal,     iqae_tail_prob_ideal,     iqae_twl_amp_ideal),
                ("noisy_raw", iqae_cvar_noisy_raw, iqae_tail_prob_noisy_raw, iqae_twl_amp_noisy_raw),
                ("noisy_zne", iqae_cvar_noisy_zne, iqae_tail_prob_noisy_zne, iqae_twl_amp_noisy_zne),
            ]:
                if cvar_val is not None:
                    err = abs(cvar_val - classical_cvar) / max(abs(classical_cvar), 1e-10)
                    logger.info(f"  IQAE {label:10s}: CVaR={cvar_val:.6f}, "
                                f"err={err:.2%}, tail_p={tp_val:.6f}, twl_a={twl_val:.6f}")
                else:
                    logger.info(f"  IQAE {label:10s}: FAILED")
            logger.info("=" * 60)
        
        # Collect results
        execution_time = (time.time() - start_time) * 1000
        
        method_results = {}
        for method_name, result in tail_results.items():
            method_results[f"{method_name}_tail_prob"] = {
                'amplitude': result.amplitude,
                'queries': result.num_oracle_queries,
                'converged': result.converged,
                'png_path': result.png_path
            }
        for method_name, result in tail_weighted_results.items():
            method_results[f"{method_name}_E_L_tail"] = {
                'amplitude': result.amplitude,
                'queries': result.num_oracle_queries,
                'converged': result.converged,
                'png_path': result.png_path
            }
        
        return QuantumCVaRResult(
            var_estimate=quantum_var,
            cvar_estimate=quantum_cvar,
            tail_probability=quantum_tail_prob,
            alpha=alpha,
            total_oracle_queries_iqae=self.stats['total_queries_iqae'],
            total_oracle_queries_woerner=self.stats['total_queries_woerner'],
            total_circuits_executed=self.stats['total_circuits'],
            execution_time_ms=execution_time,
            num_distribution_qubits=self.num_distribution_qubits,
            num_scenarios=T,
            tail_prob_result=tail_prob_result,
            tail_weighted_loss=quantum_E_L_tail,
            tail_weighted_loss_result=E_L_tail_result,
            classical_var=classical_var,
            classical_cvar=classical_cvar,
            classical_tail_prob=classical_tail_prob,
            var_error_vs_classical=0.0,
            cvar_error_vs_classical=cvar_error,
            tail_prob_error=tail_prob_error,
            method_results=method_results,
            zne_improvement=self.stats['zne_total_improvement'] / max(self.stats['zne_corrections'], 1),
            zne_applied=self.use_zne and self.noise_enabled and self.stats['zne_corrections'] > 0,
            circuits_exported=[e.get('circuit_id', '') for e in self.exporter.export_log],
            png_files=self.stats['png_files'],
            per_method_cvar=per_method_cvar,
            loss_range=self._loss_range,
            # v7.7: IQAE noise impact comparison
            iqae_cvar_ideal=iqae_cvar_ideal,
            iqae_cvar_noisy_raw=iqae_cvar_noisy_raw,
            iqae_cvar_noisy_zne=iqae_cvar_noisy_zne,
            iqae_tail_prob_ideal=iqae_tail_prob_ideal,
            iqae_tail_prob_noisy_raw=iqae_tail_prob_noisy_raw,
            iqae_tail_prob_noisy_zne=iqae_tail_prob_noisy_zne,
            iqae_twl_amplitude_ideal=iqae_twl_amp_ideal,
            iqae_twl_amplitude_noisy_raw=iqae_twl_amp_noisy_raw,
            iqae_twl_amplitude_noisy_zne=iqae_twl_amp_noisy_zne,
        )
    
    def compute_quantum_subgradient(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        alpha: float = 0.05,
        method: str = 'iqae'
    ) -> Tuple[np.ndarray, Dict]:
        """Compute CVaR subgradient using quantum amplitude estimation."""
        logger.section("QUANTUM SUBGRADIENT COMPUTATION")
        
        T, N = returns.shape
        portfolio_returns = returns @ weights
        losses = -portfolio_returns
        
        sorted_losses = np.sort(losses)
        var_idx = int((1 - alpha) * T)
        var_threshold = float(sorted_losses[var_idx])
        
        logger.metric("VaR threshold", var_threshold)
        
        tail_results = self.estimate_tail_probability(losses, var_threshold, [method])
        
        if method not in tail_results or not tail_results[method].converged:
            raise QAEExecutionError(f"Failed to estimate tail probability with {method}")
        
        quantum_tail_prob = tail_results[method].amplitude
        tail_queries = tail_results[method].num_oracle_queries
        
        quantum_tail_prob = max(quantum_tail_prob, 1e-8)
        
        conditional_exps = np.zeros(N)
        total_cond_queries = 0
        
        for j in range(N):
            exp_j, result_j = self.estimate_conditional_expectation(
                returns[:, j], losses, var_threshold, j, method
            )
            conditional_exps[j] = exp_j
            if result_j:
                total_cond_queries += result_j.num_oracle_queries
        
        subgradient = -conditional_exps / quantum_tail_prob
        
        # Classical reference
        tail_mask = losses >= var_threshold
        classical_tail_prob = max(np.mean(tail_mask), 1e-8)
        classical_cond_exp = np.mean(returns[tail_mask, :], axis=0) if np.any(tail_mask) else np.zeros(N)
        classical_subgradient = -classical_cond_exp / classical_tail_prob
        
        l2_error = float(np.linalg.norm(subgradient - classical_subgradient))
        cosine_sim = float(np.dot(subgradient, classical_subgradient) / 
                         (np.linalg.norm(subgradient) * np.linalg.norm(classical_subgradient) + 1e-10))
        
        logger.metric("Subgradient norm", np.linalg.norm(subgradient))
        logger.metric("L2 error vs classical", l2_error)
        logger.metric("Cosine similarity", cosine_sim)
        
        metrics = {
            'subgradient': subgradient.tolist(),
            'quantum_tail_prob': quantum_tail_prob,
            'classical_tail_prob': classical_tail_prob,
            'conditional_expectations': conditional_exps.tolist(),
            'l2_error': l2_error,
            'cosine_similarity': cosine_sim,
            'total_queries': tail_queries + total_cond_queries,
            'var_threshold': var_threshold,
            'png_files': self.stats['png_files']
        }
        
        return subgradient, metrics
    
    def get_statistics(self) -> Dict:
        """Get estimator statistics."""
        return {
            **self.stats,
            'circuits_exported': self.exporter.get_summary() if self.exporter else {}
        }


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

    def export_qasm(self, circuit, name: str):
        """
        Export circuit as OpenQASM 3.0 file.

        Fix [T-003]: Replaces PNG export as the canonical circuit format.
        QASM is required for:
          - Exact reproducibility of the circuit
          - Hardware execution on IBM Quantum / IQM
          - Paper appendices (circuits can be included as code)

        Parameters
        ----------
        circuit : QuantumCircuit
            Qiskit circuit to export.
        name : str
            Base filename (without extension).

        Returns
        -------
        Path
            Path to the written .qasm file.
        """
        try:
            from qiskit import qasm3
            qasm_str = qasm3.dumps(circuit)
        except ImportError:
            # Fallback to QASM 2.0 if qasm3 not available
            try:
                qasm_str = circuit.qasm()
            except Exception as e:
                self.logger.error(f"QASM export failed: {e}")
                return None

        import os
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = Path(self.output_dir) / f"{name}.qasm"
        out_path.write_text(qasm_str, encoding="utf-8")
        self.logger.info(f"Circuit exported as QASM: {out_path}")
        return out_path

    def export(self, circuit, name: str, config: dict = None):
        """
        Export circuit in the configured format(s).
        Reads phase2.qae.circuit_export from config to decide format.

        Parameters
        ----------
        circuit : QuantumCircuit
        name : str
        config : dict, optional
            Should contain circuit_export.export_qasm / export_png booleans.
        """
        cfg = (config or {}).get("circuit_export", {})
        export_qasm = cfg.get("export_qasm", True)
        export_png  = cfg.get("export_png",  False)

        exported = []
        if export_qasm:
            path = self.export_qasm(circuit, name)
            if path:
                exported.append(str(path))
        if export_png:
            try:
                path = self.export_png(circuit, name)
                exported.append(str(path))
            except Exception as e:
                self.logger.warning(f"PNG export skipped: {e}")

        return exported

__all__ = [
    'QuantumAmplitudeEstimator',
    'QuantumCVaRCircuits',
    'CircuitExporter',
    'ReadoutMitigator',
    'QAEResult',
    'QAEMethodResult',
    'QuantumCVaRResult',
    'CVaREstimationResult',
    'CircuitTooLargeError',
    'QAETimeoutError',
    'QAEExecutionError',
    'get_circuit_exporter',
    'reset_circuit_exporter',
    'QISKIT_AVAILABLE',
    'QISKIT_ALGORITHMS_AVAILABLE',
    'MATPLOTLIB_AVAILABLE',
    'NOISE_MODEL_AVAILABLE',
    'AER_SAMPLER_AVAILABLE',
    'AER_SAMPLER_V2_AVAILABLE',
    'ZNE_AVAILABLE',
    'SCIPY_AVAILABLE',
    'DEFAULT_QAE_TIMEOUT',
    'MAX_WOERNER_EVAL_QUBITS',
    'MIN_WOERNER_EVAL_QUBITS',
    'WOERNER_SAFETY_FACTOR',
    'WOERNER_MAX_RETRIES',
    'WOERNER_RETRY_ERROR_THRESHOLD',
    'MAX_IQAE_TIMEOUT',
    'MAX_TOTAL_QUBITS',
    'ZNE_DEFAULT_SCALE_FACTORS'
]


