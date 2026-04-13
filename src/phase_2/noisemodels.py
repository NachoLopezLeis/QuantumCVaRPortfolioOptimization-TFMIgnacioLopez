# ============================================================================
# MODULE: noisemodels.py
# PURPOSE: Quantum noise model management for Phase 2 circuit simulation
# PHASE: Phase 2
# ============================================================================
#
# Provides depolarizing, amplitude damping, phase damping, thermal
# relaxation, and IBM-like noise models compatible with Qiskit Aer.
#
# Mathematical Framework:
#   Depolarizing:       rho -> (1-p)*rho + p*(X*rho*X + Y*rho*Y + Z*rho*Z)/3
#   Amplitude damping:  gamma = 1 - exp(-t_gate / T1)
#   Phase damping:      lambda = 1 - exp(-t_gate / T2)
#   Thermal relaxation: combined T1 + T2 + thermal population
#     Constraint: T2 <= 2*T1 (physical coherence bound)
#
# Dependencies:
#   External: numpy, qiskit_aer.noise (optional)
#   Internal: logger, passport_orchestrator
#
# Author: Ignacio Lopez Leis (TFM)
# ============================================================================

import numpy as np
import logging
from typing import Dict, List, Optional, Any as AnyType
from pathlib import Path
from datetime import datetime

from src.utils.logger import get_logger
from src.utils.passport_orchestrator import get_orchestrator

logger = get_logger('noise_models')

# Qiskit imports (optional with graceful degradation)
try:
    from qiskit_aer.noise import (
        NoiseModel,
        depolarizing_error,
        amplitude_damping_error,
        phase_damping_error,
        thermal_relaxation_error,
    )
    QISKIT_AVAILABLE = True
    logger.debug("Qiskit Aer noise imports OK")
except ImportError as e:
    logger.warning(f"Qiskit Aer not available: {e}")
    QISKIT_AVAILABLE = False
    NoiseModel = None


# ============================================================================
# CLASS: NoiseModelManager
# ============================================================================

class NoiseModelManager:
    """
    Quantum noise model manager for Qiskit Aer simulation.

    Supports depolarizing, amplitude damping, phase damping, thermal
    relaxation, and IBM-like backend noise models. Each model creation
    is tracked via Passport for audit trail.

    Parameters
    ----------
    backend_name : str
        Target backend name (default: 'aer_simulator').
    log_dir : Path, optional
        Directory for log files.
    """

    def __init__(
        self,
        backend_name: str = "aer_simulator",
        log_dir: Optional[Path] = None,
    ):
        if not isinstance(backend_name, str):
            raise TypeError(f"backend_name must be str, got {type(backend_name).__name__}")
        if log_dir is not None and not isinstance(log_dir, Path):
            raise TypeError(f"log_dir must be Path or None, got {type(log_dir).__name__}")
        if not backend_name:
            raise ValueError("backend_name cannot be empty")

        self.backend_name = str(backend_name)
        self.noise_models: Dict[str, Dict] = {}
        self.orchestrator = get_orchestrator()
        self.log_dir = log_dir

        logger.info(f"NoiseModelManager initialized: backend={backend_name}, "
                     f"qiskit_available={QISKIT_AVAILABLE}")

    # ────────────────────────────────────────────────────────────────────
    # Internal: passport helpers (adapted to cleaned PassportOrchestrator)
    # ────────────────────────────────────────────────────────────────────

    def _passport_seal(self, passport, method_name: str, result: dict):
        """Seal passport after successful model creation."""
        try:
            return self.orchestrator.seal(
                passport=passport,
                function_name=method_name,
                phase='Phase 2',
                result={
                    'execution_time_ms': 0,
                    'validation_status': 'valid',
                    'parameters': result,
                    'transformation_type': 'noise_model_creation',
                    'warnings': [],
                    'errors': [],
                },
            )
        except Exception:
            return passport

    def _passport_error(self, passport, method_name: str, error_msg: str):
        """Checkpoint passport on error."""
        try:
            return self.orchestrator.checkpoint(
                passport=passport,
                name=f'{method_name}_failed',
                data=None,
                metadata={'error': error_msg},
            )
        except Exception:
            return passport

    def _passport_create(self, noise_type: str, params: dict):
        """Create a new passport for a noise model operation."""
        try:
            return self.orchestrator.assign_passport(
                data=params,
                data_type='noise_model_creation',
                source=f'NoiseModelManager.{noise_type}',
            )
        except Exception:
            return None

    # ────────────────────────────────────────────────────────────────────
    # Depolarizing noise
    # ────────────────────────────────────────────────────────────────────

    def get_depolarizing_noise(
        self,
        error_rate: float = 0.01,
        passport=None,
    ) -> Optional[AnyType]:
        """
        Create depolarizing noise model.

        rho -> (1-p)*rho + p*(X*rho*X + Y*rho*Y + Z*rho*Z)/3

        Parameters
        ----------
        error_rate : float
            Probability of error per gate. Range: [0.0, 1.0].
        passport : optional
            Passport object for audit trail.

        Returns
        -------
        Optional[NoiseModel]
            Qiskit NoiseModel, or None if Qiskit unavailable.
        """
        if not isinstance(error_rate, (int, float)):
            raise TypeError(f"error_rate must be numeric, got {type(error_rate).__name__}")
        if not (0.0 <= error_rate <= 1.0):
            raise ValueError(f"error_rate must be in [0.0, 1.0], got {error_rate}")
        if np.isnan(error_rate) or np.isinf(error_rate):
            raise ValueError(f"error_rate contains NaN/Inf: {error_rate}")

        if passport is None:
            passport = self._passport_create('get_depolarizing_noise', {
                'backend_name': self.backend_name,
                'error_rate': float(error_rate),
                'noise_type': 'depolarizing',
            })

        try:
            if not QISKIT_AVAILABLE:
                logger.warning("Qiskit Aer not available - returning None")
                return None

            noise_model = NoiseModel()
            error_1q = depolarizing_error(error_rate, 1)
            noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3'])
            error_2q = depolarizing_error(error_rate, 2)
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

            self._passport_seal(passport, 'get_depolarizing_noise', {
                'noise_type': 'depolarizing',
                'error_rate': float(error_rate),
                'gates_1q': ['u1', 'u2', 'u3'],
                'gates_2q': ['cx'],
            })

            logger.info(f"Depolarizing noise created: error_rate={error_rate:.4f}")
            return noise_model

        except Exception as e:
            error_msg = f"Failed to create depolarizing noise: {e}"
            logger.error(error_msg)
            self._passport_error(passport, 'get_depolarizing_noise', str(e))
            raise RuntimeError(error_msg) from e

    # ────────────────────────────────────────────────────────────────────
    # Amplitude damping noise
    # ────────────────────────────────────────────────────────────────────

    def get_amplitude_damping_noise(
        self,
        t1: float,
        t_gate: float,
        passport=None,
    ) -> Optional[AnyType]:
        """
        Create amplitude damping noise model (T1 relaxation).

        gamma = 1 - exp(-t_gate / T1)

        Parameters
        ----------
        t1 : float
            T1 relaxation time in seconds. Must be > 0.
        t_gate : float
            Gate execution time in seconds. Must be > 0.
        passport : optional
            Passport for audit trail.

        Returns
        -------
        Optional[NoiseModel]
            Qiskit NoiseModel, or None if Qiskit unavailable.
        """
        if not isinstance(t1, (int, float)):
            raise TypeError(f"t1 must be numeric, got {type(t1).__name__}")
        if not isinstance(t_gate, (int, float)):
            raise TypeError(f"t_gate must be numeric, got {type(t_gate).__name__}")
        if t1 <= 0:
            raise ValueError(f"t1 must be > 0, got {t1}")
        if t_gate <= 0:
            raise ValueError(f"t_gate must be > 0, got {t_gate}")
        for val, name in [(t1, 't1'), (t_gate, 't_gate')]:
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"{name} contains NaN/Inf: {val}")
        if t_gate > t1:
            logger.warning(f"Gate time ({t_gate:.2e}s) exceeds T1 ({t1:.2e}s)")

        if passport is None:
            passport = self._passport_create('get_amplitude_damping_noise', {
                'backend_name': self.backend_name,
                't1': float(t1), 't_gate': float(t_gate),
                'noise_type': 'amplitude_damping',
            })

        try:
            if not QISKIT_AVAILABLE:
                logger.warning("Qiskit Aer not available - returning None")
                return None

            noise_model = NoiseModel()
            gamma = 1.0 - np.exp(-t_gate / t1) if t1 > 0 else 0.0
            error = amplitude_damping_error(gamma)
            noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

            self._passport_seal(passport, 'get_amplitude_damping_noise', {
                'noise_type': 'amplitude_damping',
                't1': float(t1), 't_gate': float(t_gate), 'gamma': float(gamma),
            })

            logger.info(f"Amplitude damping noise created: T1={t1:.2e}s, "
                         f"t_gate={t_gate:.2e}s, gamma={gamma:.6f}")
            return noise_model

        except Exception as e:
            error_msg = f"Failed to create amplitude damping noise: {e}"
            logger.error(error_msg)
            self._passport_error(passport, 'get_amplitude_damping_noise', str(e))
            raise RuntimeError(error_msg) from e

    # ────────────────────────────────────────────────────────────────────
    # Phase damping noise
    # ────────────────────────────────────────────────────────────────────

    def get_phase_damping_noise(
        self,
        t2: float,
        t_gate: float,
        passport=None,
    ) -> Optional[AnyType]:
        """
        Create phase damping noise model (T2 dephasing).

        lambda = 1 - exp(-t_gate / T2)

        Parameters
        ----------
        t2 : float
            T2 dephasing time in seconds. Must be > 0.
        t_gate : float
            Gate execution time in seconds. Must be > 0.
        passport : optional
            Passport for audit trail.

        Returns
        -------
        Optional[NoiseModel]
            Qiskit NoiseModel, or None if Qiskit unavailable.
        """
        if not isinstance(t2, (int, float)):
            raise TypeError(f"t2 must be numeric, got {type(t2).__name__}")
        if not isinstance(t_gate, (int, float)):
            raise TypeError(f"t_gate must be numeric, got {type(t_gate).__name__}")
        if t2 <= 0:
            raise ValueError(f"t2 must be > 0, got {t2}")
        if t_gate <= 0:
            raise ValueError(f"t_gate must be > 0, got {t_gate}")
        for val, name in [(t2, 't2'), (t_gate, 't_gate')]:
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"{name} contains NaN/Inf: {val}")

        if passport is None:
            passport = self._passport_create('get_phase_damping_noise', {
                'backend_name': self.backend_name,
                't2': float(t2), 't_gate': float(t_gate),
                'noise_type': 'phase_damping',
            })

        try:
            if not QISKIT_AVAILABLE:
                logger.warning("Qiskit Aer not available - returning None")
                return None

            noise_model = NoiseModel()
            lambda_param = 1.0 - np.exp(-t_gate / t2) if t2 > 0 else 0.0
            error = phase_damping_error(lambda_param)
            noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

            self._passport_seal(passport, 'get_phase_damping_noise', {
                'noise_type': 'phase_damping',
                't2': float(t2), 't_gate': float(t_gate), 'lambda': float(lambda_param),
            })

            logger.info(f"Phase damping noise created: T2={t2:.2e}s, "
                         f"t_gate={t_gate:.2e}s, lambda={lambda_param:.6f}")
            return noise_model

        except Exception as e:
            error_msg = f"Failed to create phase damping noise: {e}"
            logger.error(error_msg)
            self._passport_error(passport, 'get_phase_damping_noise', str(e))
            raise RuntimeError(error_msg) from e

    # ────────────────────────────────────────────────────────────────────
    # Thermal relaxation noise
    # ────────────────────────────────────────────────────────────────────

    def get_thermal_relaxation_noise(
        self,
        t1: float,
        t2: float,
        t_gate: float,
        excited_pop: float = 0.0,
        passport=None,
    ) -> Optional[AnyType]:
        """
        Create thermal relaxation noise model (combined T1 + T2).

        Parameters
        ----------
        t1 : float
            T1 relaxation time in seconds. Must be > 0.
        t2 : float
            T2 dephasing time in seconds. Must be > 0, ideally T2 <= 2*T1.
        t_gate : float
            Gate execution time in seconds. Must be > 0.
        excited_pop : float
            Thermal population of |1> state. Range: [0.0, 1.0].
        passport : optional
            Passport for audit trail.

        Returns
        -------
        Optional[NoiseModel]
            Qiskit NoiseModel, or None if Qiskit unavailable.
        """
        if not all(isinstance(x, (int, float)) for x in [t1, t2, t_gate, excited_pop]):
            raise TypeError("All parameters must be numeric")
        if t1 <= 0 or t2 <= 0 or t_gate <= 0:
            raise ValueError(f"t1, t2, t_gate must be > 0: t1={t1}, t2={t2}, t_gate={t_gate}")
        if not (0.0 <= excited_pop <= 1.0):
            raise ValueError(f"excited_pop must be in [0, 1], got {excited_pop}")
        for val, name in [(t1, 't1'), (t2, 't2'), (t_gate, 't_gate'), (excited_pop, 'excited_pop')]:
            if np.isnan(val) or np.isinf(val):
                raise ValueError(f"{name} contains NaN/Inf: {val}")

        if passport is None:
            passport = self._passport_create('get_thermal_relaxation_noise', {
                'backend_name': self.backend_name,
                't1': float(t1), 't2': float(t2), 't_gate': float(t_gate),
                'excited_pop': float(excited_pop), 'noise_type': 'thermal_relaxation',
            })

        try:
            if not QISKIT_AVAILABLE:
                logger.warning("Qiskit Aer not available - returning None")
                return None

            noise_model = NoiseModel()
            for gate_name in ['u1', 'u2', 'u3']:
                error = thermal_relaxation_error(t1, t2, t_gate, excited_pop)
                noise_model.add_all_qubit_quantum_error(error, [gate_name])

            self._passport_seal(passport, 'get_thermal_relaxation_noise', {
                'noise_type': 'thermal_relaxation',
                't1': float(t1), 't2': float(t2), 't_gate': float(t_gate),
                'excited_pop': float(excited_pop),
            })

            logger.info(f"Thermal relaxation noise created: T1={t1:.2e}s, "
                         f"T2={t2:.2e}s, t_gate={t_gate:.2e}s")
            return noise_model

        except Exception as e:
            error_msg = f"Failed to create thermal relaxation noise: {e}"
            logger.error(error_msg)
            self._passport_error(passport, 'get_thermal_relaxation_noise', str(e))
            raise RuntimeError(error_msg) from e

    # ────────────────────────────────────────────────────────────────────
    # IBM-like backend noise
    # ────────────────────────────────────────────────────────────────────

    IBM_PARAMS = {
        'default':     {'t1': 1e-4,   't2': 5e-5,  't_gate': 1e-8,   'error_rate': 0.001},
        'standard':    {'t1': 8e-5,   't2': 4e-5,  't_gate': 1.2e-8, 'error_rate': 0.002},
        'hummingbird': {'t1': 1.2e-4, 't2': 6e-5,  't_gate': 9e-9,   'error_rate': 0.0008},
        'falcon':      {'t1': 1.5e-4, 't2': 7e-5,  't_gate': 8e-9,   'error_rate': 0.0005},
    }

    def get_ibm_backend_noise(
        self,
        backend_type: str = "default",
        passport=None,
    ) -> Optional[AnyType]:
        """
        Create IBM-like backend noise model with realistic parameters.

        Parameters
        ----------
        backend_type : str
            One of 'default', 'standard', 'hummingbird', 'falcon'.
        passport : optional
            Passport for audit trail.

        Returns
        -------
        Optional[NoiseModel]
            Qiskit NoiseModel, or None if Qiskit unavailable.
        """
        if not isinstance(backend_type, str):
            raise TypeError(f"backend_type must be str, got {type(backend_type).__name__}")
        if backend_type not in self.IBM_PARAMS:
            raise ValueError(f"backend_type must be in {list(self.IBM_PARAMS)}, got {backend_type}")

        if passport is None:
            passport = self._passport_create('get_ibm_backend_noise', {
                'backend_name': self.backend_name,
                'backend_type': backend_type, 'noise_type': 'ibm_like',
            })

        try:
            if not QISKIT_AVAILABLE:
                logger.warning("Qiskit Aer not available - returning None")
                return None

            p = self.IBM_PARAMS[backend_type]
            noise_model = NoiseModel()

            for gate_name in ['u1', 'u2', 'u3']:
                error = thermal_relaxation_error(p['t1'], p['t2'], p['t_gate'], 0.0)
                noise_model.add_all_qubit_quantum_error(error, [gate_name])

            error_2q = depolarizing_error(p['error_rate'], 2)
            noise_model.add_all_qubit_quantum_error(error_2q, ['cx'])

            self._passport_seal(passport, 'get_ibm_backend_noise', {
                'noise_type': 'ibm_like', 'backend_type': backend_type,
                'parameters': p,
            })

            logger.info(f"IBM-like noise created: type={backend_type}, "
                         f"T1={p['t1']:.2e}s, T2={p['t2']:.2e}s")
            return noise_model

        except Exception as e:
            error_msg = f"Failed to create IBM-like noise: {e}"
            logger.error(error_msg)
            self._passport_error(passport, 'get_ibm_backend_noise', str(e))
            raise RuntimeError(error_msg) from e


# ============================================================================
# Standalone API
# ============================================================================

def create_noise_model(
    noise_type: str,
    backend_name: str = "aer_simulator",
    **kwargs,
) -> Optional[AnyType]:
    """
    Create a noise model with minimal configuration.

    Parameters
    ----------
    noise_type : str
        One of 'depolarizing', 'amplitude_damping', 'phase_damping',
        'thermal_relaxation', 'ibm_like'.
    backend_name : str
        Target backend (default: 'aer_simulator').
    **kwargs
        Parameters for the specific noise type.

    Returns
    -------
    Optional[NoiseModel]
        Qiskit NoiseModel, or None if Qiskit unavailable.

    Examples
    --------
    >>> noise = create_noise_model('depolarizing', error_rate=0.01)
    >>> noise = create_noise_model('amplitude_damping', t1=1e-4, t_gate=1e-8)
    >>> noise = create_noise_model('ibm_like', backend_type='falcon')
    """
    manager = NoiseModelManager(backend_name=backend_name)

    dispatch = {
        'depolarizing':       manager.get_depolarizing_noise,
        'amplitude_damping':  manager.get_amplitude_damping_noise,
        'phase_damping':      manager.get_phase_damping_noise,
        'thermal_relaxation': manager.get_thermal_relaxation_noise,
        'ibm_like':           manager.get_ibm_backend_noise,
    }

    if noise_type not in dispatch:
        raise ValueError(f"Unknown noise_type: {noise_type}. Valid: {list(dispatch)}")

    return dispatch[noise_type](**kwargs)
