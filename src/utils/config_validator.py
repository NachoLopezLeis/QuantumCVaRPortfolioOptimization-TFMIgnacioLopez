#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : config_validator.py  (Utils - Configuration Validation)
# ============================================================================
#
# Description
# -----------
# Validates, normalizes, and fills defaults for YAML configuration.
# Detects range violations, type errors, and cross-section
# inconsistencies. Returns (validated_config, errors, warnings).
#
# License : Academic use only - TFM Project
# ============================================================================
"""

from typing import Dict, Any, Tuple, List, Optional
from pathlib import Path
import os

class ConfigValidator:
    """
    Validates and normalizes configuration.
    
    Responsibilities:
    - Check config structure
    - Validate value types
    - Check value ranges
    - Detect inconsistencies
    - Fill missing defaults
    - Normalize values
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Parameters
        ----------
        strict_mode : bool, default=False
            If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
        """
        Validate and normalize configuration.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary to validate
        
        Returns
        -------
        tuple
            (validated_config, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Create copy to avoid modifying original
        validated_config = self._deep_copy(config)
        
        # Normalize values (convert strings to numbers where appropriate)
        self._normalize_values(validated_config)
        
        # Validate each section
        self._validate_global(validated_config)
        self._validate_phase0(validated_config)
        self._validate_phase1(validated_config)
        self._validate_phase2(validated_config)
        self._validate_phase3(validated_config)
        self._validate_export(validated_config)
        
        # Check cross-section inconsistencies
        self._check_inconsistencies(validated_config)
        
        # Fill defaults if not in strict mode
        if not self.strict_mode:
            self._fill_defaults(validated_config)
        
        return validated_config, self.errors, self.warnings

    # =================================================================
    # NORMALIZATION
    # =================================================================

    def _normalize_values(self, config: Dict[str, Any]) -> None:
        """Convert string values to appropriate types (e.g., "0.05" -> 0.05)."""
        try:
            # Normalize global section
            if 'global' in config:
                if 'alpha' in config['global'] and isinstance(config['global']['alpha'], str):
                    config['global']['alpha'] = float(config['global']['alpha'])
            
            # Normalize phase1
            if 'phase1' in config and 'qubo' in config['phase1']:
                if 'num_bits' in config['phase1']['qubo'] and isinstance(config['phase1']['qubo']['num_bits'], str):
                    config['phase1']['qubo']['num_bits'] = int(config['phase1']['qubo']['num_bits'])
            
            # Normalize phase2
            if 'phase2' in config:
                # Backend
                if 'backend' in config['phase2']:
                    if 'use_optimization_level' in config['phase2']['backend'] and isinstance(config['phase2']['backend']['use_optimization_level'], str):
                        config['phase2']['backend']['use_optimization_level'] = int(config['phase2']['backend']['use_optimization_level'])
                
                # Execution
                if 'execution' in config['phase2']:
                    if 'shots' in config['phase2']['execution'] and isinstance(config['phase2']['execution']['shots'], str):
                        config['phase2']['execution']['shots'] = int(config['phase2']['execution']['shots'])
                
                # QAE
                if 'qae' in config['phase2']:
                    if 'bisection_tolerance' in config['phase2']['qae'] and isinstance(config['phase2']['qae']['bisection_tolerance'], str):
                        try:
                            config['phase2']['qae']['bisection_tolerance'] = float(config['phase2']['qae']['bisection_tolerance'])
                        except ValueError:
                            pass
                    
                    if 'max_bisection_iterations' in config['phase2']['qae'] and isinstance(config['phase2']['qae']['max_bisection_iterations'], str):
                        config['phase2']['qae']['max_bisection_iterations'] = int(config['phase2']['qae']['max_bisection_iterations'])
                
                # Optimizer
                if 'optimizer' in config['phase2']:
                    if 'max_iterations' in config['phase2']['optimizer'] and isinstance(config['phase2']['optimizer']['max_iterations'], str):
                        config['phase2']['optimizer']['max_iterations'] = int(config['phase2']['optimizer']['max_iterations'])
                    
                    if 'learning_rate' in config['phase2']['optimizer'] and isinstance(config['phase2']['optimizer']['learning_rate'], str):
                        try:
                            config['phase2']['optimizer']['learning_rate'] = float(config['phase2']['optimizer']['learning_rate'])
                        except ValueError:
                            pass
                    
                    if 'convergence_threshold' in config['phase2']['optimizer'] and isinstance(config['phase2']['optimizer']['convergence_threshold'], str):
                        try:
                            config['phase2']['optimizer']['convergence_threshold'] = float(config['phase2']['optimizer']['convergence_threshold'])
                        except ValueError:
                            pass
                    
                    if 'patience' in config['phase2']['optimizer'] and isinstance(config['phase2']['optimizer']['patience'], str):
                        config['phase2']['optimizer']['patience'] = int(config['phase2']['optimizer']['patience'])
                
                # Hardware
                if 'hardware' in config['phase2']:
                    if 'gate_error' in config['phase2']['hardware'] and isinstance(config['phase2']['hardware']['gate_error'], str):
                        try:
                            config['phase2']['hardware']['gate_error'] = float(config['phase2']['hardware']['gate_error'])
                        except ValueError:
                            pass
                    
                    if 'readout_error' in config['phase2']['hardware'] and isinstance(config['phase2']['hardware']['readout_error'], str):
                        try:
                            config['phase2']['hardware']['readout_error'] = float(config['phase2']['hardware']['readout_error'])
                        except ValueError:
                            pass
                
                # Clustering
                if 'clustering' in config['phase2']:
                    if 'n_clusters' in config['phase2']['clustering'] and isinstance(config['phase2']['clustering']['n_clusters'], str):
                        config['phase2']['clustering']['n_clusters'] = int(config['phase2']['clustering']['n_clusters'])
            
            # Normalize phase3
            if 'phase3' in config:
                if 'qae_validation' in config['phase3']:
                    if 'convergence_tolerance' in config['phase3']['qae_validation'] and isinstance(config['phase3']['qae_validation']['convergence_tolerance'], str):
                        try:
                            config['phase3']['qae_validation']['convergence_tolerance'] = float(config['phase3']['qae_validation']['convergence_tolerance'])
                        except ValueError:
                            pass
                    
                    if 'accuracy_tolerance' in config['phase3']['qae_validation'] and isinstance(config['phase3']['qae_validation']['accuracy_tolerance'], str):
                        try:
                            config['phase3']['qae_validation']['accuracy_tolerance'] = float(config['phase3']['qae_validation']['accuracy_tolerance'])
                        except ValueError:
                            pass
        except Exception as e:
            self.warnings.append(f"Error during normalization: {str(e)[:100]}")

    # =================================================================
    # VALIDATION METHODS BY SECTION
    # =================================================================

    def _validate_global(self, config: Dict[str, Any]) -> None:
        """Validate global configuration section."""
        if 'global' not in config:
            self.errors.append("Missing 'global' section in config")
            return
        
        global_config = config['global']
        
        if 'alpha' not in global_config:
            self.errors.append("Missing 'global.alpha' parameter")
        else:
            alpha = global_config['alpha']
            if not isinstance(alpha, (int, float)):
                self.errors.append(f"global.alpha must be numeric, got {type(alpha)}")
            elif not (0.01 <= alpha <= 0.50):
                self.errors.append(f"global.alpha must be in [0.01, 0.50], got {alpha}")
        
        if 'project_name' in global_config:
            if not isinstance(global_config['project_name'], str):
                self.errors.append("global.project_name must be string")
        
        if 'project_version' in global_config:
            if not isinstance(global_config['project_version'], str):
                self.errors.append("global.project_version must be string")
        
        if 'constraints' in global_config:
            constraints = global_config['constraints']
            
            if 'long_only' in constraints:
                if not isinstance(constraints['long_only'], bool):
                    self.errors.append("global.constraints.long_only must be boolean")
            
            if 'max_position_size' in constraints:
                max_pos = constraints['max_position_size']
                if not isinstance(max_pos, (int, float)):
                    self.errors.append("global.constraints.max_position_size must be numeric")
                elif not (0.0 < max_pos <= 1.0):
                    self.errors.append(f"global.constraints.max_position_size must be in (0, 1], got {max_pos}")
            
            if 'sum_weights' in constraints:
                sum_w = constraints['sum_weights']
                if not isinstance(sum_w, (int, float)):
                    self.errors.append("global.constraints.sum_weights must be numeric")
                elif sum_w != 1.0:
                    self.warnings.append(f"global.constraints.sum_weights should be 1.0, got {sum_w}")

    def _validate_phase0(self, config: Dict[str, Any]) -> None:
        """Validate phase0 (initialization) configuration."""
        if 'phase0' not in config:
            self.errors.append("Missing 'phase0' section in config")
            return
        
        phase0 = config['phase0']
        
        if 'data' in phase0:
            data = phase0['data']
            
            if 'source' in data:
                valid_sources = ['yahoo_finance', 'csv', 'bloomberg', 'quandl']
                if data['source'] not in valid_sources:
                    self.warnings.append(f"phase0.data.source='{data['source']}' not in standard list: {valid_sources}")
            
            if 'start_date' in data and 'end_date' in data:
                start = data['start_date']
                end = data['end_date']
                if isinstance(start, str) and isinstance(end, str):
                    if start > end:
                        self.errors.append(f"phase0.data.start_date ({start}) > end_date ({end})")
            
            if 'frequency' in data:
                valid_freq = ['daily', 'weekly', 'monthly']
                if data['frequency'] not in valid_freq:
                    self.warnings.append(f"phase0.data.frequency='{data['frequency']}' not in {valid_freq}")
            
            if 'tickers' in data and data['tickers'] is not None:
                tickers = data['tickers']
                if isinstance(tickers, list):
                    if len(tickers) == 0:
                        self.errors.append("phase0.data.tickers is empty list")
                    elif len(tickers) > 500:
                        self.warnings.append(f"phase0.data.tickers has {len(tickers)} assets (very large portfolio)")
                elif not isinstance(tickers, str):
                    self.errors.append(f"phase0.data.tickers must be list or string, got {type(tickers)}")
        
        if 'logging' in phase0:
            logging = phase0['logging']
            
            if 'log_level' in logging:
                valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
                if logging['log_level'] not in valid_levels:
                    self.warnings.append(f"phase0.logging.log_level='{logging['log_level']}' not in {valid_levels}")
            
            if 'overwrite_logs' in logging:
                if not isinstance(logging['overwrite_logs'], bool):
                    self.errors.append("phase0.logging.overwrite_logs must be boolean")
        
        if 'passport' in phase0:
            passport = phase0['passport']
            
            if 'enable_passport_tracking' in passport:
                if not isinstance(passport['enable_passport_tracking'], bool):
                    self.errors.append("phase0.passport.enable_passport_tracking must be boolean")
            
            if 'passport_format' in passport:
                valid_formats = ['json', 'yaml', 'pickle']
                if passport['passport_format'] not in valid_formats:
                    self.warnings.append(f"phase0.passport.passport_format not in {valid_formats}")

    def _validate_phase1(self, config: Dict[str, Any]) -> None:
        """Validate phase1 (classical formulation) configuration."""
        if 'phase1' not in config:
            self.errors.append("Missing 'phase1' section in config")
            return
        
        phase1 = config['phase1']
        
        if 'cvar' in phase1:
            cvar = phase1['cvar']
            if 'method' in cvar:
                valid_methods = ['historical', 'monte_carlo', 'parametric']
                if cvar['method'] not in valid_methods:
                    self.warnings.append(f"phase1.cvar.method not in {valid_methods}")
        
        if 'penalty_calibration' in phase1:
            penalty = phase1['penalty_calibration']
            if 'method' in penalty:
                valid_methods = ['scale_based', 'ratio_based', 'adaptive']
                if penalty['method'] not in valid_methods:
                    self.warnings.append(f"phase1.penalty_calibration.method not in {valid_methods}")
        
        if 'qubo' in phase1:
            qubo = phase1['qubo']
            
            if 'num_bits' not in qubo:
                self.errors.append("Missing 'phase1.qubo.num_bits'")
            else:
                num_bits = qubo['num_bits']
                if not isinstance(num_bits, int):
                    self.errors.append(f"phase1.qubo.num_bits must be int, got {type(num_bits)}")
                elif not (4 <= num_bits <= 16):
                    self.errors.append(f"phase1.qubo.num_bits must be in [4, 16], got {num_bits}")
            
            if 'discretization_method' in qubo:
                valid_methods = ['linear', 'logarithmic', 'custom']
                if qubo['discretization_method'] not in valid_methods:
                    self.warnings.append(f"phase1.qubo.discretization_method not in {valid_methods}")
        
        if 'spectral_analysis' in phase1:
            spectral = phase1['spectral_analysis']
            if 'compute_eigenvalues' in spectral:
                if not isinstance(spectral['compute_eigenvalues'], bool):
                    self.errors.append("phase1.spectral_analysis.compute_eigenvalues must be boolean")
        
        if 'error_propagation' in phase1:
            error = phase1['error_propagation']
            if 'confidence_level' in error:
                conf = error['confidence_level']
                if not isinstance(conf, (int, float)):
                    self.errors.append(f"phase1.error_propagation.confidence_level must be numeric, got {type(conf)}")
                elif not (0.0 < conf < 1.0):
                    self.errors.append(f"phase1.error_propagation.confidence_level must be in (0, 1), got {conf}")

    def _validate_phase2(self, config: Dict[str, Any]) -> None:
        """Validate phase2 (quantum-hybrid) configuration."""
        if 'phase2' not in config:
            self.errors.append("Missing 'phase2' section in config")
            return
        
        phase2 = config['phase2']
        
        if 'backend' in phase2:
            backend = phase2['backend']
            
            if 'backend_type' not in backend:
                self.errors.append("Missing 'phase2.backend.backend_type'")
            else:
                valid_backends = ['aer_simulator', 'ibm_manila', 'ibm_quito', 'rigetti', 'cirq','qasm_simulator']
                if backend['backend_type'] not in valid_backends:
                    self.warnings.append(f"phase2.backend.backend_type='{backend['backend_type']}' not in {valid_backends}")
            
            if 'use_optimization_level' in backend:
                opt_level = backend['use_optimization_level']
                if not isinstance(opt_level, int):
                    self.errors.append(f"phase2.backend.use_optimization_level must be int, got {type(opt_level)}")
                elif not (0 <= opt_level <= 3):
                    self.errors.append(f"phase2.backend.use_optimization_level must be in [0, 3], got {opt_level}")
        
        if 'hardware' in phase2:
            hardware = phase2['hardware']
            
            if 'gate_error' in hardware:
                gate_error = hardware['gate_error']
                if not isinstance(gate_error, (int, float)):
                    self.errors.append(f"phase2.hardware.gate_error must be numeric, got {type(gate_error)}")
                elif not (0.0 <= gate_error <= 0.1):
                    self.warnings.append(f"phase2.hardware.gate_error should be in [0, 0.1], got {gate_error}")
            
            if 'readout_error' in hardware:
                readout = hardware['readout_error']
                if not isinstance(readout, (int, float)):
                    self.errors.append(f"phase2.hardware.readout_error must be numeric, got {type(readout)}")
                elif not (0.0 <= readout <= 0.1):
                    self.warnings.append(f"phase2.hardware.readout_error should be in [0, 0.1], got {readout}")
        
        if 'execution' in phase2:
            execution = phase2['execution']
            
            if 'shots' not in execution:
                self.errors.append("Missing 'phase2.execution.shots'")
            else:
                shots = execution['shots']
                if not isinstance(shots, int):
                    self.errors.append(f"phase2.execution.shots must be int, got {type(shots)}")
                elif shots < 512:
                    self.warnings.append(f"phase2.execution.shots={shots} is very low (< 512)")
                elif shots > 4096:
                    self.warnings.append(f"phase2.execution.shots={shots} is very high (> 4096)")
        
        if 'qae' in phase2:
            qae = phase2['qae']
            
            if 'use_bisection' in qae:
                if not isinstance(qae['use_bisection'], bool):
                    self.errors.append("phase2.qae.use_bisection must be boolean")
            
            if 'bisection_tolerance' in qae:
                tol = qae['bisection_tolerance']
                if not isinstance(tol, (int, float)):
                    self.errors.append(f"phase2.qae.bisection_tolerance must be numeric, got {type(tol)}")
                elif not (1e-5 < tol < 1e-1):
                    self.warnings.append(f"phase2.qae.bisection_tolerance should be in (1e-5, 1e-1), got {tol}")
            
            if 'max_bisection_iterations' in qae:
                max_iter = qae['max_bisection_iterations']
                if not isinstance(max_iter, int):
                    self.errors.append(f"phase2.qae.max_bisection_iterations must be int, got {type(max_iter)}")
                elif max_iter < 5:
                    self.warnings.append(f"phase2.qae.max_bisection_iterations should be >= 5, got {max_iter}")
        
        if 'optimizer' in phase2:
            optimizer = phase2['optimizer']
            
            if 'max_iterations' not in optimizer:
                self.errors.append("Missing 'phase2.optimizer.max_iterations'")
            else:
                max_iter = optimizer['max_iterations']
                if not isinstance(max_iter, int):
                    self.errors.append(f"phase2.optimizer.max_iterations must be int, got {type(max_iter)}")
                elif max_iter < 10:
                    self.errors.append(f"phase2.optimizer.max_iterations should be >= 10, got {max_iter}")
                elif max_iter > 1000:
                    self.warnings.append(f"phase2.optimizer.max_iterations is very high ({max_iter})")
            
            if 'learning_rate' not in optimizer:
                self.errors.append("Missing 'phase2.optimizer.learning_rate'")
            else:
                lr = optimizer['learning_rate']
                if not isinstance(lr, (int, float)):
                    self.errors.append(f"phase2.optimizer.learning_rate must be numeric, got {type(lr)}")
                elif not (0.0001 <= lr <= 0.1):
                    self.errors.append(f"phase2.optimizer.learning_rate must be in [0.0001, 0.1], got {lr}")
            
            if 'convergence_threshold' in optimizer:
                threshold = optimizer['convergence_threshold']
                if not isinstance(threshold, (int, float)):
                    self.errors.append(f"phase2.optimizer.convergence_threshold must be numeric, got {type(threshold)}")
                elif not (1e-8 < threshold < 1e-3):
                    self.warnings.append(f"phase2.optimizer.convergence_threshold should be in (1e-8, 1e-3), got {threshold}")
            
            if 'patience' in optimizer:
                patience = optimizer['patience']
                if not isinstance(patience, int):
                    self.errors.append(f"phase2.optimizer.patience must be int, got {type(patience)}")
                elif patience < 5:
                    self.warnings.append(f"phase2.optimizer.patience should be >= 5, got {patience}")
        
        if 'zne' in phase2:
            zne = phase2['zne']
            if 'enable_zne' in zne:
                if not isinstance(zne['enable_zne'], bool):
                    self.errors.append("phase2.zne.enable_zne must be boolean")
        
        if 'clustering' in phase2:
            clustering = phase2['clustering']
            
            if 'enable_clustering' in clustering:
                if not isinstance(clustering['enable_clustering'], bool):
                    self.errors.append("phase2.clustering.enable_clustering must be boolean")
            
            if 'n_clusters' in clustering:
                n_clusters = clustering['n_clusters']
                if not isinstance(n_clusters, int):
                    self.errors.append(f"phase2.clustering.n_clusters must be int, got {type(n_clusters)}")
                elif not (2 <= n_clusters <= 20):
                    self.warnings.append(f"phase2.clustering.n_clusters should be in [2, 20], got {n_clusters}")

    def _validate_phase3(self, config: Dict[str, Any]) -> None:
        """Validate phase3 (validation & metrics) configuration."""
        if 'phase3' not in config:
            self.errors.append("Missing 'phase3' section in config")
            return
        
        phase3 = config['phase3']
        
        if 'qae_validation' in phase3:
            qae_val = phase3['qae_validation']
            
            if 'convergence_tolerance' in qae_val:
                tol = qae_val['convergence_tolerance']
                if not isinstance(tol, (int, float)):
                    self.errors.append(f"phase3.qae_validation.convergence_tolerance must be numeric, got {type(tol)}")
                elif not (0.0 < tol <= 1.0):
                    self.errors.append(f"phase3.qae_validation.convergence_tolerance must be in (0, 1], got {tol}")
            
            if 'accuracy_tolerance' in qae_val:
                tol = qae_val['accuracy_tolerance']
                if not isinstance(tol, (int, float)):
                    self.errors.append(f"phase3.qae_validation.accuracy_tolerance must be numeric, got {type(tol)}")
                elif not (0.0 < tol <= 1.0):
                    self.errors.append(f"phase3.qae_validation.accuracy_tolerance must be in (0, 1], got {tol}")
        
        if 'risk_analysis' in phase3:
            risk = phase3['risk_analysis']
            
            if 'compute_regulatory_reporting' in risk:
                if not isinstance(risk['compute_regulatory_reporting'], bool):
                    self.errors.append("phase3.risk_analysis.compute_regulatory_reporting must be boolean")
            
            if 'regulatory_framework' in risk:
                valid_frameworks = ['basel_iii', 'ucits', 'mifid', 'solvency_ii']
                if risk['regulatory_framework'] not in valid_frameworks:
                    self.warnings.append(f"phase3.risk_analysis.regulatory_framework not in {valid_frameworks}")
        
        if 'benchmarking' in phase3:
            bench = phase3['benchmarking']
            
            if 'run_classical_benchmark' in bench:
                if not isinstance(bench['run_classical_benchmark'], bool):
                    self.errors.append("phase3.benchmarking.run_classical_benchmark must be boolean")
            
            if 'classical_method' in bench:
                valid_methods = ['SLSQP', 'L-BFGS-B', 'trust-constr']
                if bench['classical_method'] not in valid_methods:
                    self.warnings.append(f"phase3.benchmarking.classical_method not in {valid_methods}")

    def _validate_export(self, config: Dict[str, Any]) -> None:
        """Validate export configuration."""
        if 'export' not in config:
            self.errors.append("Missing 'export' section in config")
            return
        
        export = config['export']
        
        if 'export_formats' in export:
            formats = export['export_formats']
            valid_formats = ['csv', 'json', 'excel', 'html', 'pdf']
            for fmt in formats:
                if fmt not in valid_formats:
                    self.warnings.append(f"export.export_formats contains unknown format: {fmt}")

    # =================================================================
    # CROSS-SECTION VALIDATION
    # =================================================================

    def _check_inconsistencies(self, config: Dict[str, Any]) -> None:
        """Check for inconsistencies across sections."""
        num_bits = config.get('phase1', {}).get('qubo', {}).get('num_bits')
        shots = config.get('phase2', {}).get('execution', {}).get('shots')
        
        if num_bits and shots:
            if num_bits > 12 and shots < 2048:
                self.warnings.append(
                    f"High num_bits ({num_bits}) with low shots ({shots}) "
                    f"may reduce QAE accuracy. Consider increasing shots."
                )
        
        lr = config.get('phase2', {}).get('optimizer', {}).get('learning_rate')
        max_iter = config.get('phase2', {}).get('optimizer', {}).get('max_iterations')
        
        if lr and max_iter:
            if lr > 0.05 and max_iter < 50:
                self.warnings.append(
                    f"High learning_rate ({lr}) with low max_iterations ({max_iter}) "
                    f"may cause divergence."
                )
        
        alpha = config.get('global', {}).get('alpha')
        if alpha and alpha > 0.20:
            self.warnings.append(
                f"CVaR alpha={alpha} is high (> 0.20). "
                f"Typical range is 0.01-0.10."
            )

    # =================================================================
    # FILL DEFAULTS
    # =================================================================

    def _fill_defaults(self, config: Dict[str, Any]) -> None:
        """Fill missing parameters with sensible defaults."""
        if 'global' not in config:
            config['global'] = {}
        
        if 'alpha' not in config['global']:
            config['global']['alpha'] = 0.05
        
        if 'phase1' not in config:
            config['phase1'] = {}
        
        if 'qubo' not in config['phase1']:
            config['phase1']['qubo'] = {}
        
        if 'num_bits' not in config['phase1']['qubo']:
            config['phase1']['qubo']['num_bits'] = 8
        
        if 'phase2' not in config:
            config['phase2'] = {}
        
        if 'backend' not in config['phase2']:
            config['phase2']['backend'] = {}
        
        if 'backend_type' not in config['phase2']['backend']:
            config['phase2']['backend']['backend_type'] = 'aer_simulator'
        
        if 'execution' not in config['phase2']:
            config['phase2']['execution'] = {}
        
        if 'shots' not in config['phase2']['execution']:
            config['phase2']['execution']['shots'] = 1024
        
        if 'optimizer' not in config['phase2']:
            config['phase2']['optimizer'] = {}
        
        if 'max_iterations' not in config['phase2']['optimizer']:
            config['phase2']['optimizer']['max_iterations'] = 100
        
        if 'learning_rate' not in config['phase2']['optimizer']:
            config['phase2']['optimizer']['learning_rate'] = 0.02

    # =================================================================
    # UTILITY METHODS
    # =================================================================

    @staticmethod
    def _deep_copy(obj: Any) -> Any:
        """Deep copy an object (dict, list, or primitive)."""
        if isinstance(obj, dict):
            return {k: ConfigValidator._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ConfigValidator._deep_copy(item) for item in obj]
        else:
            return obj

    def get_summary(self) -> str:
        """Get summary of validation results."""
        lines = []
        lines.append("=" * 80)
        lines.append("CONFIGURATION VALIDATION SUMMARY")
        lines.append("=" * 80)
        
        lines.append(f"\nOK Errors: {len(self.errors)}")
        if self.errors:
            for error in self.errors:
                lines.append(f"  [ERR] {error}")
        
        lines.append(f"\nOK Warnings: {len(self.warnings)}")
        if self.warnings:
            for warning in self.warnings:
                lines.append(f"  [WARN]  {warning}")
        
        if not self.errors and not self.warnings:
            lines.append("\n[OK] Configuration is valid!")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
