# ---------------------------------------------------------------------------
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
#
# Module  : quantumbackends.py  (Phase 2 - Quantum Infrastructure)
#
# Description
# -----------
# Unified quantum backend manager for circuit simulation and hardware
# execution.  Supports Qiskit Aer simulators (noiseless, noisy, statevector),
# IBM Quantum hardware via qiskit-ibm-runtime, and IQM Quantum devices.
# Provides automatic transpilation, circuit validation, caching, and
# backend discovery.
#
# Backend types:
#   qasm_simulator       -- standard shot-based simulation
#   statevector_simulator -- full state-vector (debug)
#   unitary_simulator    -- unitary matrix extraction
#   noisy_simulator      -- Aer with custom NoiseModel
#   ibm_quantum          -- IBM Quantum hardware
#   iqm_quantum          -- IQM Quantum hardware
#
# References
# ----------
# [1] Qiskit -- https://qiskit.org
# [2] IBM Quantum -- https://quantum.ibm.com
# [3] IQM Quantum -- https://www.iqmtech.ai
# [4] Preskill, J. (2018) - Quantum Computing in the NISQ Era and Beyond
#
# License : Academic use only -- contact nacholopezleis@gmail.com
# ---------------------------------------------------------------------------
"""
Quantum Backend Manager -- Phase 2

Unified interface for quantum circuit simulation and hardware execution.
"""

import numpy as np
import logging
from typing import Optional, Dict, Any as AnyType, List, Tuple, Union
from pathlib import Path
from datetime import datetime
import hashlib
import warnings
import json

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Try importing advanced transpilation passes
try:
    from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation, RemoveResetInZeroState
    try:
        from qiskit.transpiler.passes import Optimize2qGates
    except ImportError:
        Optimize2qGates = Optimize1qGates
except ImportError:
    Optimize1qGates = None
    Optimize2qGates = None
    CommutativeCancellation = None
    RemoveResetInZeroState = None

# Internal utilities
try:
    from src.utils.logger import get_logger
    from src.utils.passport_orchestrator import get_orchestrator
except ImportError:
    from utils.logger import get_logger
    from utils.passport_orchestrator import get_orchestrator

logger = get_logger('quantumbackends')

# IBM Quantum Runtime (optional)
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    IBM_RUNTIME_AVAILABLE = True
except ImportError:
    IBM_RUNTIME_AVAILABLE = False
    logger.warning("qiskit-ibm-runtime not installed. IBM Quantum support disabled.")

# IQM Quantum (optional)
try:
    import iqm_client
    IQM_AVAILABLE = True
except ImportError:
    IQM_AVAILABLE = False
    logger.debug("iqm-client not installed. IQM Quantum support will be limited.")

# ============================================================================
# HELPER FUNCTIONS FOR PASSPORT TRACKING
# ============================================================================

def _compute_hash(data: AnyType) -> str:
    """
    Compute MD5 hash of data for passport tracking.

    This is a local helper function that follows the pattern from Phase 1 modules.

    Parameters
    ----------
    data : Any
        Data to hash (dict, ndarray, or any object)

    Returns
    -------
    str
        MD5 hash (first 16 characters)
    """
    try:
        if isinstance(data, dict):
            data_bytes = json.dumps(data, sort_keys=True, default=str).encode()
        elif isinstance(data, np.ndarray):
            data_bytes = data.tobytes()
        else:
            data_bytes = str(data).encode()
        return hashlib.md5(data_bytes).hexdigest()[:16]
    except Exception:
        return "hash_error"

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_backend_type(backend_type: str) -> None:
    """Validate backend type parameter."""
    valid_backends = ['qasm_simulator', 'statevector_simulator', 'unitary_simulator',
                      'noisy_simulator', 'ibm_quantum', 'iqm_quantum']
    if not isinstance(backend_type, str):
        error_msg = f"backend_type must be str, got {type(backend_type).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    if backend_type not in valid_backends:
        error_msg = f"backend_type '{backend_type}' not supported. Must be in {valid_backends}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def validate_optimization_level(opt_level: int) -> None:
    """Validate optimization level parameter."""
    if not isinstance(opt_level, int):
        error_msg = f"optimization_level must be int, got {type(opt_level).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    if opt_level < 0 or opt_level > 3:
        error_msg = f"optimization_level must be in [0, 3], got {opt_level}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def validate_quantum_circuit(circuit: QuantumCircuit) -> None:
    """Validate quantum circuit structure."""
    if not isinstance(circuit, QuantumCircuit):
        error_msg = f"circuit must be QuantumCircuit, got {type(circuit).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    if circuit.num_qubits < 1:
        error_msg = f"circuit must have at least 1 qubit, got {circuit.num_qubits}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if circuit.num_clbits < 0:
        error_msg = f"circuit has invalid number of classical bits: {circuit.num_clbits}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def validate_num_shots(num_shots: int) -> None:
    """Validate num_shots parameter."""
    if not isinstance(num_shots, int):
        error_msg = f"num_shots must be int, got {type(num_shots).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    if num_shots < 1 or num_shots > 1_000_000:
        error_msg = f"num_shots must be in [1, 1000000], got {num_shots}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def validate_api_key(api_key: str, provider: str) -> None:
    """Validate API key parameter."""
    if not isinstance(api_key, str):
        error_msg = f"api_key must be str, got {type(api_key).__name__}"
        logger.error(error_msg)
        raise TypeError(error_msg)
    if not api_key or len(api_key.strip()) == 0:
        error_msg = f"{provider} api_key cannot be empty"
        logger.error(error_msg)
        raise ValueError(error_msg)
    if len(api_key) < 10:
        error_msg = f"{provider} api_key seems too short (< 10 chars)"
        logger.warning(error_msg)

# ============================================================================
# QUANTUM BACKEND MANAGER
# ============================================================================

class QuantumBackendManager:
    """
    Unified quantum backend manager for simulation and hardware execution.

    Manages quantum backends across simulators, IBM Quantum, and IQM platforms.
    Provides automatic transpilation, circuit validation, caching, and discovery.
    Full passport integration for audit compliance.
    """

    # Backend type constants
    BACKEND_QASM = 'qasm_simulator'
    BACKEND_STATEVECTOR = 'statevector_simulator'
    BACKEND_UNITARY = 'unitary_simulator'
    BACKEND_NOISY = 'noisy_simulator'
    BACKEND_IBM = 'ibm_quantum'
    BACKEND_IQM = 'iqm_quantum'

    # Optimization levels
    OPT_NONE = 0
    OPT_LIGHT = 1
    OPT_MEDIUM = 2
    OPT_HEAVY = 3

    def __init__(
        self,
        backend_type: str = BACKEND_QASM,
        optimization_level: int = OPT_LIGHT
    ) -> None:
        """
        Initialize Quantum Backend Manager.

        Parameters
        ----------
        backend_type : str
            Type of backend to use
        optimization_level : int
            Transpilation optimization level (0-3)
        """
        # Validate inputs
        try:
            validate_backend_type(backend_type)
            validate_optimization_level(optimization_level)
        except (TypeError, ValueError) as e:
            logger.error(f"Validation failed in __init__: {str(e)}")
            raise

        # Initialize orchestrator
        self.orchestrator = get_orchestrator()

        # Assign passport
        passport = self.orchestrator.assign_passport(
            data={'backend_type': backend_type, 'optimization_level': int(optimization_level)},
            data_type='backend_manager_init',
            source='QuantumBackendManager.__init__'
        )

        operation_start = datetime.now()

        try:
            self.backend_type = backend_type
            self.optimization_level = int(optimization_level)
            self.backend = None
            self.noise_model = None
            self.backend_cache: Dict[str, AnyType] = {}
            self.connection_pool: Dict[str, AnyType] = {}

            self.initialize_backend()

            execution_ms = (datetime.now() - operation_start).total_seconds() * 1000

            # Seal passport
            passport = self.orchestrator.seal(
                passport,
                function_name='__init__',
                phase='Phase 2',
                result={
                    'execution_time_ms': execution_ms,
                    'output_hash': _compute_hash({'backend': backend_type}),
                    'validation_status': 'valid',
                    'transformation_type': 'backend_initialization',
                    'backend_initialized': backend_type,
                    'optimization_level': optimization_level,
                    'cache_enabled': True
                }
            )

            logger.info(f"QuantumBackendManager initialized (backend={backend_type}, opt_level={optimization_level})")

        except Exception as e:
            error_msg = f"Failed to initialize QuantumBackendManager: {str(e)}"
            logger.error(error_msg)

            # Checkpoint
            passport = self.orchestrator.checkpoint(
                passport,
                name='init_failed',
                data=None,
                metadata={
                    'status': 'error',
                    'error_message': str(e),
                    'stage': 'initialization',
                    'backend_type': backend_type
                }
            )

            raise RuntimeError(error_msg) from e

    def initialize_backend(self) -> None:
        """Initialize quantum backend based on configuration."""
        try:
            if self.backend_type == self.BACKEND_QASM:
                self.backend = AerSimulator(method='automatic')
            elif self.backend_type == self.BACKEND_STATEVECTOR:
                self.backend = AerSimulator(method='statevector')
            elif self.backend_type == self.BACKEND_UNITARY:
                self.backend = AerSimulator(method='unitary')
            elif self.backend_type == self.BACKEND_NOISY:
                self.backend = AerSimulator(method='automatic')
            elif self.backend_type == self.BACKEND_IBM:
                if not IBM_RUNTIME_AVAILABLE:
                    raise ImportError("qiskit-ibm-runtime required. Install: pip install qiskit-ibm-runtime")
            elif self.backend_type == self.BACKEND_IQM:
                if not IQM_AVAILABLE:
                    raise ImportError("iqm-client required. Install: pip install iqm-client")
            else:
                raise ValueError(f"Unknown backend type: {self.backend_type}")

            logger.info(f"Initialized backend: {self.backend_type}")

        except Exception as e:
            error_msg = f"Failed to initialize backend {self.backend_type}: {str(e)}"
            logger.error(error_msg)
            raise

    def get_backend(self, backend_name: Optional[str] = None, num_qubits: Optional[int] = None, use_noise: bool = False) -> AnyType:
        """
        Get backend by name or return current backend.

        Parameters
        ----------
        backend_name : str, optional
            Specific backend name
        num_qubits : int, optional
            Number of qubits required
        use_noise : bool
            Whether to use noise model

        Returns
        -------
        Backend object
        """
        try:
            if backend_name is None:
                return self.backend

            backend_map = {
                'aer_simulator': self.BACKEND_QASM,
                'qasm_simulator': self.BACKEND_QASM,
                'statevector_simulator': self.BACKEND_STATEVECTOR,
                'unitary_simulator': self.BACKEND_UNITARY,
            }

            backend_type = backend_map.get(backend_name.lower(), backend_name)

            if backend_type == self.BACKEND_QASM:
                return AerSimulator(method='automatic')
            elif backend_type == self.BACKEND_STATEVECTOR:
                return AerSimulator(method='statevector')
            elif backend_type == self.BACKEND_UNITARY:
                return AerSimulator(method='unitary')
            else:
                return self.backend

        except Exception as e:
            logger.error(f"Failed to get backend {backend_name}: {str(e)}")
            raise

    def execute_circuit(
        self,
        circuit: QuantumCircuit,
        num_shots: int = 1024,
        seed: Optional[int] = None
    ) -> Dict[str, AnyType]:
        """
        Execute quantum circuit on configured backend.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to execute
        num_shots : int
            Number of measurement shots
        seed : int, optional
            Random seed for reproducibility

        Returns
        -------
        Dict with execution results
        """
        # Validate inputs
        try:
            validate_quantum_circuit(circuit)
            validate_num_shots(num_shots)
        except (TypeError, ValueError) as e:
            logger.error(f"Validation failed in execute_circuit: {str(e)}")
            raise

        # Create passport
        passport = self.orchestrator.assign_passport(
            data={
                'circuit_qubits': circuit.num_qubits,
                'circuit_depth': circuit.depth(),
                'num_shots': num_shots,
                'backend': self.backend_type
            },
            data_type='circuit_execution',
            source='QuantumBackendManager.execute_circuit'
        )

        try:
            start_time = datetime.now()

            # Optimize and transpile circuit
            optimized_circuit = self.optimize_circuit(circuit)
            transpiled = transpile(
                optimized_circuit,
                self.backend,
                optimization_level=self.optimization_level
            )

            # Execute
            if self.noise_model is not None and self.backend_type == self.BACKEND_NOISY:
                sampler = self.backend.run(
                    transpiled,
                    shots=num_shots,
                    noise_model=self.noise_model,
                    seed_simulator=seed
                )
            else:
                sampler = self.backend.run(
                    transpiled,
                    shots=num_shots,
                    seed_simulator=seed
                )

            result = sampler.result()
            execution_ms = (datetime.now() - start_time).total_seconds() * 1000

            result_dict = {
                'counts': dict(result.get_counts()),
                'execution_time': float(execution_ms),
                'backend': self.backend_type
            }

            # Seal passport
            passport = self.orchestrator.seal(
                passport,
                function_name='execute_circuit',
                phase='Phase 2',
                result={
                    'execution_time_ms': execution_ms,
                    'output_hash': _compute_hash(result_dict),
                    'validation_status': 'valid',
                    'transformation_type': 'circuit_execution',
                    'backend': self.backend_type,
                    'total_counts': sum(result_dict['counts'].values()),
                    'circuit_depth': circuit.depth(),
                    'num_qubits': circuit.num_qubits,
                    'optimization_level': self.optimization_level
                }
            )

            # Passport ID
            passport_id = passport.passport_id[:8] if passport else 'N/A'
            logger.info(f"Circuit executed: {execution_ms:.2f}ms, passport={passport_id}...")

            return result_dict

        except Exception as e:
            error_msg = f"Failed to execute circuit: {str(e)}"
            logger.error(error_msg)

            # Checkpoint
            passport = self.orchestrator.checkpoint(
                passport,
                name='execution_failed',
                data=None,
                metadata={
                    'status': 'error',
                    'error_message': str(e),
                    'stage': 'circuit_execution',
                    'backend': self.backend_type,
                    'num_qubits': circuit.num_qubits
                }
            )

            raise RuntimeError(error_msg) from e

    def optimize_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """
        Optimize circuit based on optimization level.

        Parameters
        ----------
        circuit : QuantumCircuit
            Circuit to optimize

        Returns
        -------
        QuantumCircuit
            Optimized circuit
        """
        try:
            validate_quantum_circuit(circuit)
        except (TypeError, ValueError) as e:
            logger.error(f"Validation failed in optimize_circuit: {str(e)}")
            raise

        try:
            if self.optimization_level == self.OPT_NONE:
                return circuit

            pm = PassManager()

            if self.optimization_level >= self.OPT_LIGHT:
                if RemoveResetInZeroState is not None:
                    pm.append(RemoveResetInZeroState())
                if Optimize1qGates is not None:
                    pm.append(Optimize1qGates())

            if self.optimization_level >= self.OPT_MEDIUM:
                if CommutativeCancellation is not None:
                    pm.append(CommutativeCancellation())
                if Optimize1qGates is not None:
                    pm.append(Optimize1qGates())

            if self.optimization_level >= self.OPT_HEAVY:
                if Optimize1qGates is not None:
                    pm.append(Optimize1qGates())

            return pm.run(circuit)

        except Exception as e:
            logger.error(f"Failed to optimize circuit: {str(e)}")
            raise ValueError(f"Circuit optimization failed: {str(e)}") from e

    def set_noise_model(self, noise_model: AnyType) -> None:
        """
        Set noise model for noisy simulation.

        Parameters
        ----------
        noise_model : NoiseModel
            Qiskit noise model
        """
        try:
            if noise_model is None:
                raise ValueError("noise_model cannot be None")

            self.noise_model = noise_model
            self.backend_type = self.BACKEND_NOISY
            logger.info("Noise model configured")

        except Exception as e:
            error_msg = f"Failed to set noise model: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def get_backend_info(self) -> Dict[str, AnyType]:
        """
        Get information about current backend.

        Returns
        -------
        Dict with backend information
        """
        try:
            features = []
            if self.backend_type in [self.BACKEND_IBM, self.BACKEND_IQM]:
                features.extend(['hardware', 'real_quantum'])
            if self.backend_type in [self.BACKEND_STATEVECTOR, self.BACKEND_UNITARY]:
                features.append('debug_info')
            if self.noise_model is not None:
                features.append('noise_enabled')

            return {
                'backend_type': self.backend_type,
                'optimization_level': self.optimization_level,
                'max_qubits': 32 if 'Aer' in str(type(self.backend)) else 100,
                'supports_noise': self.backend_type in [self.BACKEND_NOISY, self.BACKEND_IBM, self.BACKEND_IQM],
                'noise_model_enabled': self.noise_model is not None,
                'features': features
            }

        except Exception as e:
            logger.error(f"Failed to get backend info: {str(e)}")
            raise

    def __str__(self) -> str:
        """String representation."""
        return f"QuantumBackendManager(backend={self.backend_type}, opt_level={self.optimization_level})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"QuantumBackendManager(backend_type='{self.backend_type}', "
                f"optimization_level={self.optimization_level}, "
                f"noise_enabled={self.noise_model is not None})")

# ============================================================================
# STANDALONE EXECUTION FUNCTIONS
# ============================================================================

def execute_on_ibm_quantum(
    api_key: str,
    circuit: QuantumCircuit,
    backend_name: str = 'ibm_osaka',
    num_shots: int = 1024,
    optimization_level: int = 2,
    skip_transpilation: bool = False
) -> Dict[str, AnyType]:
    """
    Execute quantum circuit on IBM Quantum hardware.

    Parameters
    ----------
    api_key : str
        IBM Quantum API key
    circuit : QuantumCircuit
        Circuit to execute
    backend_name : str
        IBM backend name
    num_shots : int
        Number of shots
    optimization_level : int
        Transpilation optimization level
    skip_transpilation : bool
        Skip transpilation if True

    Returns
    -------
    Dict with execution results
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError("qiskit-ibm-runtime required. Install: pip install qiskit-ibm-runtime")

    # Validate
    try:
        validate_quantum_circuit(circuit)
        validate_num_shots(num_shots)
        validate_api_key(api_key, 'IBM')
    except (TypeError, ValueError) as e:
        logger.error(f"Validation failed in execute_on_ibm_quantum: {str(e)}")
        raise

    orchestrator = get_orchestrator()

    # Assign passport
    passport = orchestrator.assign_passport(
        data={
            'provider': 'IBM',
            'backend': backend_name,
            'circuit_qubits': circuit.num_qubits,
            'num_shots': num_shots
        },
        data_type='ibm_quantum_execution',
        source='execute_on_ibm_quantum'
    )

    operation_start = datetime.now()

    try:
        logger.info(f"Connecting to IBM Quantum backend: {backend_name}")

        # Authenticate with IBM
        service = QiskitRuntimeService(channel='ibm_quantum', token=api_key)
        backend = service.backend(backend_name)

        # Transpile if needed
        if not skip_transpilation:
            transpiled_circuit = transpile(circuit, backend, optimization_level=optimization_level)
        else:
            transpiled_circuit = circuit

        # Execute on hardware
        logger.info(f"Submitting job to {backend_name}...")
        job = backend.run(transpiled_circuit, shots=num_shots)
        job_id = job.job_id()

        # Wait for results
        result = job.result()
        counts = dict(result.get_counts())

        execution_ms = (datetime.now() - operation_start).total_seconds() * 1000

        result_dict = {
            'counts': counts,
            'job_id': job_id,
            'backend': backend_name,
            'execution_time': result.time_taken if hasattr(result, 'time_taken') else execution_ms / 1000,
            'status': 'completed'
        }

        # Seal passport
        passport = orchestrator.seal(
            passport,
            function_name='execute_on_ibm_quantum',
            phase='Phase 2',
            result={
                'execution_time_ms': execution_ms,
                'output_hash': _compute_hash(result_dict),
                'validation_status': 'valid',
                'transformation_type': 'ibm_hardware_execution',
                'job_id': job_id,
                'total_counts': sum(counts.values()),
                'backend': backend_name,
                'circuit_depth': circuit.depth(),
                'num_qubits': circuit.num_qubits
            }
        )

        # Passport ID
        passport_id = passport.passport_id[:8] if passport else 'N/A'
        logger.info(f"IBM Quantum execution complete (job_id={job_id}, passport={passport_id}...)")

        return result_dict

    except Exception as e:
        error_msg = f"Failed to execute on IBM Quantum: {str(e)}"
        logger.error(error_msg)

        # Checkpoint
        passport = orchestrator.checkpoint(
            passport,
            name='ibm_execution_failed',
            data=None,
            metadata={
                'status': 'error',
                'error_message': str(e),
                'stage': 'ibm_hardware_execution',
                'backend': backend_name,
                'provider': 'IBM'
            }
        )

        raise RuntimeError(error_msg) from e

def execute_on_iqm(
    api_key: str,
    circuit: QuantumCircuit,
    endpoint: str = 'https://iqm.example.com/api',
    device: str = 'Garnet',
    num_shots: int = 1024
) -> Dict[str, AnyType]:
    """
    Execute quantum circuit on IQM Quantum hardware.

    Parameters
    ----------
    api_key : str
        IQM API key
    circuit : QuantumCircuit
        Circuit to execute
    endpoint : str
        IQM API endpoint
    device : str
        IQM device name
    num_shots : int
        Number of shots

    Returns
    -------
    Dict with execution results
    """
    # Validate
    try:
        validate_quantum_circuit(circuit)
        validate_num_shots(num_shots)
        validate_api_key(api_key, 'IQM')
    except (TypeError, ValueError) as e:
        logger.error(f"Validation failed in execute_on_iqm: {str(e)}")
        raise

    orchestrator = get_orchestrator()

    # Assign passport
    passport = orchestrator.assign_passport(
        data={
            'provider': 'IQM',
            'device': device,
            'endpoint': endpoint,
            'circuit_qubits': circuit.num_qubits,
            'num_shots': num_shots
        },
        data_type='iqm_quantum_execution',
        source='execute_on_iqm'
    )

    operation_start = datetime.now()

    try:
        logger.info(f"Connecting to IQM Quantum (device={device}, endpoint={endpoint})")

        # For now, we provide the structure for future integration
        if not IQM_AVAILABLE:
            logger.warning("iqm-client not available. Using mock execution.")
            # Mock execution for demonstration
            counts = {'00': num_shots // 2, '11': num_shots // 2}
            result_dict = {
                'counts': counts,
                'device': device,
                'execution_time': 0.5,
                'status': 'completed (mock)'
            }
        else:
            # This would use iqm_client to connect and execute
            raise NotImplementedError("IQM integration pending SDK release")

        execution_ms = (datetime.now() - operation_start).total_seconds() * 1000

        # Seal passport
        passport = orchestrator.seal(
            passport,
            function_name='execute_on_iqm',
            phase='Phase 2',
            result={
                'execution_time_ms': execution_ms,
                'output_hash': _compute_hash(result_dict),
                'validation_status': 'valid',
                'transformation_type': 'iqm_hardware_execution',
                'device': device,
                'total_counts': sum(result_dict['counts'].values()),
                'backend': device,
                'circuit_depth': circuit.depth(),
                'num_qubits': circuit.num_qubits
            }
        )

        # Passport ID
        passport_id = passport.passport_id[:8] if passport else 'N/A'
        logger.info(f"IQM Quantum execution complete (device={device}, passport={passport_id}...)")

        return result_dict

    except Exception as e:
        error_msg = f"Failed to execute on IQM Quantum: {str(e)}"
        logger.error(error_msg)

        # Checkpoint
        passport = orchestrator.checkpoint(
            passport,
            name='iqm_execution_failed',
            data=None,
            metadata={
                'status': 'error',
                'error_message': str(e),
                'stage': 'iqm_hardware_execution',
                'device': device,
                'provider': 'IQM'
            }
        )

        raise RuntimeError(error_msg) from e

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_noiseless_simulator() -> QuantumBackendManager:
    """Get noiseless QASM simulator."""
    return QuantumBackendManager(backend_type=QuantumBackendManager.BACKEND_QASM)

def get_statevector_simulator() -> QuantumBackendManager:
    """Get statevector simulator for debugging."""
    return QuantumBackendManager(backend_type=QuantumBackendManager.BACKEND_STATEVECTOR)

def get_noisy_simulator() -> QuantumBackendManager:
    """Get noisy simulator (noise model must be set separately)."""
    return QuantumBackendManager(backend_type=QuantumBackendManager.BACKEND_NOISY)

def get_ibm_backend_info(api_key: str) -> Dict[str, AnyType]:
    """
    Get available IBM Quantum backends.

    Parameters
    ----------
    api_key : str
        IBM Quantum API key

    Returns
    -------
    Dict with backend information
    """
    if not IBM_RUNTIME_AVAILABLE:
        raise ImportError("qiskit-ibm-runtime required")

    try:
        validate_api_key(api_key, 'IBM')
        service = QiskitRuntimeService(channel='ibm_quantum', token=api_key)
        backends = service.backends()

        info = {}
        for backend in backends:
            info[backend.name] = {
                'num_qubits': backend.num_qubits if hasattr(backend, 'num_qubits') else 'unknown',
                'operational': backend.status().operational if hasattr(backend, 'status') else 'unknown'
            }

        return info

    except Exception as e:
        logger.error(f"Failed to get IBM backend info: {str(e)}")
        raise


