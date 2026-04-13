#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : February 2026
# Module  : circuit_logger.py  (Utils - Quantum Circuit Tracking)
# ============================================================================
#
# Description
# -----------
# Centralizes logging and visualization of all quantum circuits executed
# in the project. Automatic metadata extraction (qubits, depth, gates),
# diagram generation via qc.draw('mpl'), and statistics export.
# 
# Integration: qae_circuits, qsp_estimation, quantumsubgradient, zne.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union
import logging

# Qiskit imports with error handling
try:
    from qiskit import QuantumCircuit
    from qiskit.visualization import circuit_drawer
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    QuantumCircuit = None

# Project logger
try:
    from src.utils.logger import get_logger
    logger = get_logger("circuit_logger")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("circuit_logger")

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_LOG_FILE = Path("logs/circuits_executed.txt")
DEFAULT_DIAGRAM_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "circuits"  # Changed from results/figures/circuits
DEFAULT_STATISTICS_FILE =  Path(__file__).resolve().parent.parent.parent / "results" / "circuits" / "statistics.json"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _extract_circuit_metadata(circuit: Any) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from quantum circuit.

    Parameters
    ----------
    circuit : QuantumCircuit or compatible
        Quantum circuit to analyze

    Returns
    -------
    Dict[str, Any]
        Circuit metadata including qubits, depth, gate counts
    """
    metadata = {
        "num_qubits": 0,
        "depth": 0,
        "num_gates": 0,
        "gate_types": {},
        "valid": False
    }

    try:
        if not QISKIT_AVAILABLE:
            metadata["error"] = "Qiskit not available"
            return metadata

        if not isinstance(circuit, QuantumCircuit):
            metadata["error"] = f"Invalid circuit type: {type(circuit).__name__}"
            return metadata

        # Extract basic properties
        metadata["num_qubits"] = circuit.num_qubits
        metadata["depth"] = circuit.depth()
        metadata["num_gates"] = len(circuit.data)

        # Count gate types
        gate_counts = {}
        for instruction in circuit.data:
            gate_name = instruction.operation.name
            gate_counts[gate_name] = gate_counts.get(gate_name, 0) + 1

        metadata["gate_types"] = gate_counts
        metadata["valid"] = True

    except Exception as e:
        metadata["error"] = str(e)
        logger.warning(f"Failed to extract circuit metadata: {e}")

    return metadata

def _format_log_entry(
    circuit_type: str,
    operation: str,
    module: str,
    metadata: Dict[str, Any],
    timestamp: str,
    diagram_path: Optional[Path] = None
) -> str:
    """
    Format circuit log entry for file writing.

    Parameters
    ----------
    circuit_type : str
        Type of circuit (QAE, QSP, Subgradient, ZNE)
    operation : str
        Operation being performed
    module : str
        Module name where circuit was created
    metadata : Dict[str, Any]
        Circuit metadata dictionary
    timestamp : str
        ISO format timestamp
    diagram_path : Optional[Path], optional
        Path to saved diagram (if any)

    Returns
    -------
    str
        Formatted log entry
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"TIMESTAMP: {timestamp}")
    lines.append(f"MODULE:    {module}")
    lines.append(f"TYPE:      {circuit_type}")
    lines.append(f"OPERATION: {operation}")
    lines.append("-" * 80)

    if metadata.get("valid", False):
        lines.append(f"Qubits:    {metadata['num_qubits']}")
        lines.append(f"Depth:     {metadata['depth']}")
        lines.append(f"Gates:     {metadata['num_gates']}")

        if metadata.get("gate_types"):
            lines.append("Gate breakdown:")
            for gate, count in sorted(metadata["gate_types"].items()):
                lines.append(f"  {gate:12s}: {count:4d}")

        if diagram_path:
            lines.append(f"Diagram:   {diagram_path}")
    else:
        error_msg = metadata.get("error", "Unknown error")
        lines.append(f"ERROR: {error_msg}")

    lines.append("=" * 80)
    lines.append("")  # Blank line separator

    return "\n".join(lines)

def _generate_diagram_filename(circuit_type: str, operation: str, timestamp: str) -> str:
    """
    Generate standardized filename for circuit diagram.

    Parameters
    ----------
    circuit_type : str
        Type of circuit (QAE, QSP, etc.)
    operation : str
        Operation name
    timestamp : str
        ISO format timestamp

    Returns
    -------
    str
        Filename (without extension)
    """
    # Clean timestamp for filename
    clean_timestamp = timestamp.replace(":", "").replace("-", "").replace(".", "")[:15]

    # Clean operation name
    clean_operation = operation.replace(" ", "_").replace("/", "_").lower()

    # Format: circuittype_operation_timestamp
    filename = f"{circuit_type.lower()}_{clean_operation}_{clean_timestamp}"

    return filename

# ============================================================================
# MAIN LOGGING FUNCTIONS
# ============================================================================

def log_circuit(
    circuit: Any,
    circuit_type: str,
    operation: str,
    module: str,
    log_file: Optional[Union[str, Path]] = None,
    metadata_override: Optional[Dict[str, Any]] = None,
    save_diagram: bool = True,
    diagram_format: str = "png"
) -> bool:
    """
    Log quantum circuit execution to unified log file and save diagram.

    Appends circuit metadata to a centralized log file with timestamp,
    module information, and circuit properties. Automatically saves
    circuit diagram to results/circuits/ using qc.draw("mpl").

    Parameters
    ----------
    circuit : QuantumCircuit
        Qiskit QuantumCircuit object to log
    circuit_type : str
        Type of circuit: "QAE", "QSP", "Subgradient", "ZNE", "Custom"
    operation : str
        Operation being performed (e.g., "amplitude_estimation", "folding")
    module : str
        Module name where circuit was created (e.g., "qae_circuits")
    log_file : Optional[Union[str, Path]], optional
        Custom log file path. Defaults to logs/circuits_executed.txt
    metadata_override : Optional[Dict[str, Any]], optional
        Override auto-extracted metadata (for custom circuit types)
    save_diagram : bool, optional
        Whether to automatically save circuit diagram. Default is True
    diagram_format : str, optional
        Diagram format: "png" or "svg". Default is "png"

    Returns
    -------
    bool
        True if logging succeeded, False otherwise

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(2)
    >>> qc.h(0)
    >>> qc.cx(0, 1)
    >>> log_circuit(qc, "Custom", "bell_state", "test_module")
    True

    Notes
    -----
    - Log file is opened in append mode ('a')
    - Diagrams are saved automatically to results/circuits/
    - Timestamps use ISO 8601 format
    - Failures are logged but don't raise exceptions
    """
    try:
        # Determine log file path
        if log_file is None:
            log_file = DEFAULT_LOG_FILE
        log_path = Path(log_file)

        # Create parent directory if needed
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract or use override metadata
        if metadata_override is not None:
            metadata = metadata_override
        else:
            metadata = _extract_circuit_metadata(circuit)

        # Generate timestamp
        timestamp = datetime.now().isoformat()

        # Save diagram if requested
        diagram_path = None
        if save_diagram and metadata.get("valid", False):
            diagram_filename = _generate_diagram_filename(circuit_type, operation, timestamp)
            diagram_path = save_circuit_diagram(
                circuit=circuit,
                filename=diagram_filename,
                format=diagram_format
            )

        # Format log entry
        log_entry = _format_log_entry(
            circuit_type=circuit_type,
            operation=operation,
            module=module,
            metadata=metadata,
            timestamp=timestamp,
            diagram_path=diagram_path
        )

        # Write to log file (append mode)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(log_entry)

        logger.debug(
            f"Circuit logged: {circuit_type} | {operation} | "
            f"{metadata.get('num_qubits', '?')} qubits | "
            f"{metadata.get('depth', '?')} depth"
        )

        return True

    except Exception as e:
        logger.error(f"Failed to log circuit: {e}")
        return False

def save_circuit_diagram(
    circuit: Any,
    filename: str,
    output_dir: Optional[Union[str, Path]] = None,
    format: str = "png"
) -> Optional[Path]:
    """
    Save quantum circuit diagram using qc.draw("mpl").

    Generates visual representation of circuit using Qiskit's matplotlib
    drawer and saves to results/circuits/.

    Parameters
    ----------
    circuit : QuantumCircuit
        Qiskit QuantumCircuit to visualize
    filename : str
        Output filename (without extension)
    output_dir : Optional[Union[str, Path]], optional
        Output directory. Defaults to results/circuits/
    format : str, optional
        Output format: "png" or "svg". Default is "png"

    Returns
    -------
    Optional[Path]
        Path to saved file, or None if failed

    Examples
    --------
    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(3)
    >>> qc.h(range(3))
    >>> path = save_circuit_diagram(qc, "hadamard_3qubits")
    >>> print(f"Saved to: {path}")

    Notes
    -----
    - Uses qc.draw("mpl") for high-quality matplotlib rendering
    - PNG format at 300 DPI for publication quality
    - Failures return None but don't raise exceptions
    """
    try:
        if not QISKIT_AVAILABLE:
            logger.warning("Qiskit not available, cannot save circuit diagram")
            return None

        if not isinstance(circuit, QuantumCircuit):
            logger.warning(f"Invalid circuit type: {type(circuit).__name__}")
            return None

        # Determine output directory
        if output_dir is None:
            output_dir = DEFAULT_DIAGRAM_DIR
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Build full output path
        file_path = out_path / f"{filename}.{format}"

        # Generate circuit diagram using qc.draw("mpl")
        try:
            fig = circuit.draw(
                output="mpl",
                style="default",
                plot_barriers=True,
                fold=None
            )

            # Save figure
            fig.savefig(
                str(file_path),
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none'
            )

            # Close figure to free memory
            import matplotlib.pyplot as plt
            plt.close(fig)

            logger.debug(f"Circuit diagram saved: {file_path}")
            return file_path

        except Exception as draw_error:
            logger.warning(f"Failed to draw circuit with matplotlib: {draw_error}")
            return None

    except Exception as e:
        logger.error(f"Failed to save circuit diagram: {e}")
        return None

def clear_circuit_log(log_file: Optional[Union[str, Path]] = None) -> bool:
    """
    Clear (overwrite) the circuit execution log file.

    Creates a new empty log file with header. Use at the start of
    a new benchmark/experiment run.

    Parameters
    ----------
    log_file : Optional[Union[str, Path]], optional
        Log file path. Defaults to logs/circuits_executed.txt

    Returns
    -------
    bool
        True if successful, False otherwise

    Examples
    --------
    >>> clear_circuit_log()  # Start fresh log
    True
    """
    try:
        if log_file is None:
            log_file = DEFAULT_LOG_FILE
        log_path = Path(log_file)

        log_path.parent.mkdir(parents=True, exist_ok=True)

        header = f"""{"="*80}
QUANTUM CIRCUIT EXECUTION LOG
{"="*80}
Project: TFM - Quantum Portfolio Optimization
Log initialized: {datetime.now().isoformat()}
Diagrams location: results/circuits/
{"="*80}

"""

        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(header)

        logger.info(f"Circuit log cleared: {log_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to clear circuit log: {e}")
        return False

def get_circuit_statistics(
    log_file: Optional[Union[str, Path]] = None,
    save_to_file: bool = True
) -> Dict[str, Any]:
    """
    Analyze circuit execution log and return statistics.

    Optionally saves statistics to results/circuits/statistics.json

    Parameters
    ----------
    log_file : Optional[Union[str, Path]], optional
        Log file path. Defaults to logs/circuits_executed.txt
    save_to_file : bool, optional
        Whether to save statistics to JSON file. Default is True

    Returns
    -------
    Dict[str, Any]
        Statistics dictionary with counts by type, module, etc.

    Examples
    --------
    >>> stats = get_circuit_statistics()
    >>> print(f"Total circuits: {stats['total_circuits']}")
    >>> print(f"QAE circuits: {stats['by_type']['QAE']}")
    """
    stats = {
        "total_circuits": 0,
        "by_type": {},
        "by_module": {},
        "total_qubits": 0,
        "total_gates": 0,
        "average_qubits": 0.0,
        "average_depth": 0.0,
        "total_depth": 0,
        "log_file": str(log_file) if log_file else str(DEFAULT_LOG_FILE),
        "timestamp": datetime.now().isoformat()
    }

    try:
        if log_file is None:
            log_file = DEFAULT_LOG_FILE
        log_path = Path(log_file)

        if not log_path.exists():
            stats["error"] = "Log file does not exist"
            return stats

        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Count circuit entries (each starts with "=" line)
        entries = content.split("="*80)
        stats["total_circuits"] = max(0, len(entries) - 2)  # Subtract header

        qubit_counts = []
        depth_counts = []

        # Parse each entry
        for entry in entries:
            if "TYPE:" in entry and "MODULE:" in entry:
                # Extract type
                type_match = entry.split("TYPE:")[1].split("\n")[0].strip()
                stats["by_type"][type_match] = stats["by_type"].get(type_match, 0) + 1

                # Extract module
                module_match = entry.split("MODULE:")[1].split("\n")[0].strip()
                stats["by_module"][module_match] = stats["by_module"].get(module_match, 0) + 1

                # Extract qubits and gates
                if "Qubits:" in entry:
                    qubits = int(entry.split("Qubits:")[1].split("\n")[0].strip())
                    stats["total_qubits"] += qubits
                    qubit_counts.append(qubits)

                if "Gates:" in entry:
                    gates = int(entry.split("Gates:")[1].split("\n")[0].strip())
                    stats["total_gates"] += gates

                if "Depth:" in entry:
                    depth = int(entry.split("Depth:")[1].split("\n")[0].strip())
                    stats["total_depth"] += depth
                    depth_counts.append(depth)

        # Calculate averages
        if qubit_counts:
            stats["average_qubits"] = sum(qubit_counts) / len(qubit_counts)
        if depth_counts:
            stats["average_depth"] = sum(depth_counts) / len(depth_counts)

        # Save to file if requested
        if save_to_file:
            stats_path = DEFAULT_STATISTICS_FILE
            stats_path.parent.mkdir(parents=True, exist_ok=True)

            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            stats["statistics_file"] = str(stats_path)
            logger.info(f"Circuit statistics saved to: {stats_path}")

    except Exception as e:
        stats["error"] = str(e)
        logger.error(f"Failed to analyze circuit log: {e}")

    return stats

# ============================================================================
# CONVENIENCE FUNCTIONS FOR SPECIFIC CIRCUIT TYPES
# ============================================================================

def log_qae_circuit(circuit: Any, operation: str, **kwargs) -> bool:
    """Log QAE (Quantum Amplitude Estimation) circuit."""
    return log_circuit(circuit, "QAE", operation, "qae_circuits", **kwargs)

def log_qsp_circuit(circuit: Any, operation: str, **kwargs) -> bool:
    """Log QSP (Quantum Signal Processing) circuit."""
    return log_circuit(circuit, "QSP", operation, "qsp_estimation", **kwargs)

def log_subgradient_circuit(circuit: Any, operation: str, **kwargs) -> bool:
    """Log Subgradient estimation circuit."""
    return log_circuit(circuit, "Subgradient", operation, "quantumsubgradient", **kwargs)

def log_zne_circuit(circuit: Any, operation: str, **kwargs) -> bool:
    """Log ZNE (Zero-Noise Extrapolation) folded circuit."""
    return log_circuit(circuit, "ZNE", operation, "zne_error_mitigation", **kwargs)

# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Ensure default directories exist on import
try:
    DEFAULT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_STATISTICS_FILE.parent.mkdir(parents=True, exist_ok=True)
except Exception as e:
    logger.warning(f"Failed to create default directories: {e}")
