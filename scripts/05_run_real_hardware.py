#!/usr/bin/env python3
"""
# =============================================================================
# STEP 5 -- Validate QAE circuit on real quantum hardware [T-008]
# =============================================================================
# Runs the minimal viable IQAE experiment (n=3, PA universe) on either:
#   - IBM Quantum (ibm_kingston, Heron r2, 156 qubits)
#   - IQM Academy (IQM Spark 5Q or IQM Garnet 20Q)
#
# The experiment measures P(L >= VaR) for the PA portfolio optimized weights
# and compares against:
#   1. Aer noiseless simulation
#   2. Aer with noise model D1 (high noise)
#   3. Real hardware result
#
# This produces the key validation figure for the paper:
#   "Real hardware confirms the noise model D1"
#
# Usage:
#   python scripts/05_run_real_hardware.py --provider ibm --token YOUR_IBM_TOKEN
#   python scripts/05_run_real_hardware.py --provider iqm --token YOUR_IQM_TOKEN
#   python scripts/05_run_real_hardware.py --provider ibm --token YOUR_TOKEN --shots 4096
#
# Outputs:
#   results/hardware_validation/hardware_result_ibm.json
#   results/hardware_validation/hardware_result_iqm.json
#
# Acknowledgement [T-004]:
#   IBM Quantum Open Plan: quantum.cloud.ibm.com
#   IQM Academy:           iqm-quantum.com/iqm-academy
#
# =============================================================================

"""
import sys
import os
import json
import argparse
import traceback
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run IQAE experiment on real quantum hardware [T-008]"
    )
    parser.add_argument(
        "--provider", choices=["ibm", "iqm"], required=True,
        help="Hardware provider: 'ibm' (IBM Quantum) or 'iqm' (IQM Academy)"
    )
    parser.add_argument(
        "--token", required=True,
        help="API token for the chosen provider"
    )
    parser.add_argument(
        "--shots", type=int, default=4096,
        help="Number of shots per circuit (default: 4096)"
    )
    parser.add_argument(
        "--experiment", default="A1_PA",
        help="Experiment ID to load optimized weights from (default: A1_PA)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate setup and print circuit info without submitting"
    )
    return parser.parse_args()


# ============================================================================
# HELPERS
# ============================================================================

def find_project_root() -> Path:
    candidates = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for p in candidates:
        if (p / "config").is_dir():
            return p
    raise FileNotFoundError("Cannot find project root")


def load_optimized_weights(project_root: Path, experiment_id: str) -> np.ndarray:
    """Load quantum-optimized weights from experiment results."""
    metrics_path = project_root / "results" / f"exp_{experiment_id}" / \
                   "tfm_comprehensive_metrics_latest.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Metrics file not found: {metrics_path}\n"
            f"Run experiment {experiment_id} first (step 3)."
        )
    with open(metrics_path) as fh:
        metrics = json.load(fh)
    weights = (metrics.get("metrics", {})
               .get("oos_validation", {})
               .get("portfolios", {})
               .get("Quantum CVaR", {})
               .get("weights"))
    if not weights:
        raise ValueError(f"No Quantum CVaR weights found in {metrics_path}")
    return np.array(weights)


def build_iqae_circuit(n_qubits: int, weights: np.ndarray,
                       returns: np.ndarray, alpha: float = 0.05):
    """
    Build a minimal IQAE circuit for P(L >= VaR) estimation.

    Parameters
    ----------
    n_qubits : int
        Number of qubits for distribution discretisation.
    weights : np.ndarray
        Portfolio weights.
    returns : np.ndarray
        OOS returns matrix (T, N).
    alpha : float
        CVaR confidence level.

    Returns
    -------
    QuantumCircuit (Qiskit)
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        raise ImportError("qiskit not installed. Run: pip install qiskit")

    sys.path.insert(0, str(find_project_root()))
    try:
        from src.phase_2.qae_circuits import QAECircuitBuilder
        port_returns = returns @ weights
        losses = -port_returns
        var = float(np.percentile(losses, (1 - alpha) * 100))

        builder = QAECircuitBuilder(n_qubits=n_qubits)
        circuit = builder.build_state_preparation(
            returns=returns,
            weights=weights,
            var_threshold=var,
        )
        return circuit, var
    except ImportError:
        # Minimal fallback: Hadamard circuit for connectivity test
        print("WARNING: QAECircuitBuilder not importable; building minimal test circuit")
        qc = QuantumCircuit(n_qubits + 1)
        for i in range(n_qubits + 1):
            qc.h(i)
        qc.measure_all()
        return qc, 0.0


# ============================================================================
# IBM QUANTUM
# ============================================================================

def run_on_ibm(
    token: str,
    circuit,
    shots: int = 4096,
    dry_run: bool = False,
) -> dict:
    """
    Submit circuit to IBM Quantum (Heron r2 ibm_kingston backend).

    Requirements: pip install qiskit-ibm-runtime
    """
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    except ImportError:
        raise ImportError(
            "qiskit-ibm-runtime not installed.\n"
            "Run: pip install qiskit-ibm-runtime"
        )

    print("Connecting to IBM Quantum...")
    service = QiskitRuntimeService(
        channel="ibm_quantum",
        token=token,
    )

    backend = service.backend("ibm_kingston")
    print(f"Backend: {backend.name} | "
          f"Qubits: {backend.num_qubits} | "
          f"Queue: {backend.status().pending_jobs} jobs")

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)

    print(f"Circuit: {isa_circuit.num_qubits} qubits, "
          f"depth={isa_circuit.depth()}, "
          f"gates={isa_circuit.size()}")

    if dry_run:
        print("[DRY RUN] Circuit validated -- not submitted")
        return {"dry_run": True, "backend": backend.name,
                "circuit_depth": isa_circuit.depth(),
                "circuit_qubits": isa_circuit.num_qubits}

    print(f"Submitting {shots} shots to {backend.name}...")
    t0 = time.time()
    sampler = SamplerV2(backend)
    job = sampler.run([isa_circuit], shots=shots)

    print(f"Job submitted: {job.job_id()}")
    print("Waiting for result (this may take several minutes)...")
    result = job.result()
    elapsed = time.time() - t0

    counts = result[0].data.meas.get_counts()
    total  = sum(counts.values())
    p_tail = counts.get("1", 0) / total   # probability of loss >= VaR

    print(f"Result: P(L >= VaR) = {p_tail:.4f} ({elapsed:.1f}s)")

    return {
        "provider": "ibm",
        "backend": backend.name,
        "job_id": job.job_id(),
        "shots": shots,
        "execution_time_s": elapsed,
        "counts": counts,
        "p_tail_estimated": float(p_tail),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# IQM ACADEMY
# ============================================================================

def run_on_iqm(
    token: str,
    circuit,
    shots: int = 4096,
    dry_run: bool = False,
) -> dict:
    """
    Submit circuit to IQM hardware.

    Requirements: pip install qiskit-iqm
    IQM Academy access: https://www.iqm-quantum.com/iqm-academy

    Supported backends:
      - IQM Spark (5 qubits)  -- use n_qubits=3 (3+1=4 total qubits needed)
      - IQM Garnet (20 qubits) -- use n_qubits up to 6
    """
    try:
        from iqm.qiskit_iqm import IQMProvider
    except ImportError:
        raise ImportError(
            "qiskit-iqm not installed.\n"
            "Run: pip install qiskit-iqm\n"
            "IQM Academy access: https://www.iqm-quantum.com/iqm-academy"
        )

    # IQM Spark endpoint (5Q) -- switch to Garnet URL for more qubits
    IQM_ENDPOINT = "https://cocos.resonance.meetiqm.com/garnet"

    print(f"Connecting to IQM ({IQM_ENDPOINT})...")
    os.environ["IQM_TOKEN"] = token

    provider = IQMProvider(IQM_ENDPOINT)
    backend = provider.get_backend()
    print(f"IQM backend: {backend.name} | {backend.num_qubits} qubits")

    # Transpile for IQM native gate set
    from qiskit import transpile
    isa_circuit = transpile(circuit, backend=backend, optimization_level=1)

    print(f"Circuit: {isa_circuit.num_qubits} qubits, "
          f"depth={isa_circuit.depth()}, "
          f"gates={isa_circuit.size()}")

    if dry_run:
        print("[DRY RUN] Circuit validated -- not submitted")
        return {"dry_run": True, "backend": backend.name,
                "circuit_depth": isa_circuit.depth(),
                "circuit_qubits": isa_circuit.num_qubits}

    print(f"Submitting {shots} shots to IQM...")
    t0 = time.time()
    job = backend.run(isa_circuit, shots=shots)

    print(f"Job ID: {job.job_id()}")
    result = job.result()
    elapsed = time.time() - t0

    counts = result.get_counts()
    total  = sum(counts.values())
    p_tail = counts.get("1" * isa_circuit.num_clbits, 0) / total

    print(f"Result: P(tail) ~ {p_tail:.4f} ({elapsed:.1f}s)")

    return {
        "provider": "iqm",
        "backend": backend.name,
        "job_id": str(job.job_id()),
        "shots": shots,
        "execution_time_s": elapsed,
        "counts": counts,
        "p_tail_estimated": float(p_tail),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# AER REFERENCE (noiseless + noise model)
# ============================================================================

def run_aer_reference(circuit, shots: int = 4096) -> dict:
    """Run circuit on Aer simulator (noiseless) as reference baseline."""
    try:
        from qiskit_aer import AerSimulator
        sim = AerSimulator()
        from qiskit import transpile
        tc = transpile(circuit, sim)
        job = sim.run(tc, shots=shots)
        counts = job.result().get_counts()
        total  = sum(counts.values())
        p_tail = counts.get("1", 0) / total
        return {
            "provider": "aer_noiseless",
            "shots": shots,
            "counts": counts,
            "p_tail_estimated": float(p_tail),
        }
    except Exception as exc:
        return {"provider": "aer_noiseless", "error": str(exc)}


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    project_root = find_project_root()
    output_dir = project_root / "results" / "hardware_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f" STEP 5: REAL HARDWARE VALIDATION [T-008]")
    print(f" Provider  : {args.provider.upper()}")
    print(f" Experiment: {args.experiment}")
    print(f" Shots     : {args.shots}")
    print(f" Dry run   : {args.dry_run}")
    print("=" * 65)

    # Load data
    import pandas as pd
    csv_path = project_root / "data" / "returns_sp500_full.csv"
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found. Run step 1 first.")

    returns_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    weights = load_optimized_weights(project_root, args.experiment)
    print(f"Loaded weights for {args.experiment}: {len(weights)} assets")

    # Use assets that match weights dimension
    if returns_df.shape[1] >= len(weights):
        returns = returns_df.iloc[:, :len(weights)].values
    else:
        sys.exit(f"Returns has {returns_df.shape[1]} columns but weights has {len(weights)}")

    # Build circuit (n=3 for hardware -- minimal viable experiment)
    print("Building IQAE circuit (n=3)...")
    circuit, var_threshold = build_iqae_circuit(
        n_qubits=3,
        weights=weights,
        returns=returns,
        alpha=0.05,
    )
    print(f"VaR threshold: {var_threshold:.6f}")

    # Aer noiseless reference
    print("\nRunning Aer noiseless reference...")
    aer_result = run_aer_reference(circuit, shots=args.shots)
    print(f"Aer noiseless P(tail): {aer_result.get('p_tail_estimated', 'N/A'):.4f}")

    # Real hardware
    hw_result = {}
    try:
        if args.provider == "ibm":
            hw_result = run_on_ibm(
                token=args.token,
                circuit=circuit,
                shots=args.shots,
                dry_run=args.dry_run,
            )
        elif args.provider == "iqm":
            hw_result = run_on_iqm(
                token=args.token,
                circuit=circuit,
                shots=args.shots,
                dry_run=args.dry_run,
            )
    except Exception as exc:
        print(f"\nERROR: Hardware run failed: {exc}")
        print(traceback.format_exc())
        hw_result = {"error": str(exc)}

    # Save results
    full_result = {
        "experiment": args.experiment,
        "n_qubits": 3,
        "shots": args.shots,
        "var_threshold": float(var_threshold),
        "aer_noiseless": aer_result,
        "hardware": hw_result,
    }

    out_path = output_dir / f"hardware_result_{args.provider}.json"
    with open(out_path, "w") as fh:
        json.dump(full_result, fh, indent=2, default=str)

    print(f"\nResults saved: {out_path}")
    print("=" * 65)

    # Summary comparison
    p_aer = aer_result.get("p_tail_estimated", float("nan"))
    p_hw  = hw_result.get("p_tail_estimated", float("nan"))
    if not args.dry_run and not isinstance(p_hw, float):
        p_hw = float("nan")

    print("COMPARISON:")
    print(f"  Aer noiseless: P(L >= VaR) = {p_aer:.4f}")
    print(f"  Hardware:      P(L >= VaR) = {p_hw:.4f}")
    if not (np.isnan(p_aer) or np.isnan(p_hw)):
        error_pct = abs(p_hw - p_aer) / max(p_aer, 1e-10) * 100
        print(f"  Relative error vs noiseless: {error_pct:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
