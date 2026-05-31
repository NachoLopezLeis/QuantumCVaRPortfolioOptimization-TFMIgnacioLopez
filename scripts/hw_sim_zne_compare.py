"""
hw_sim_zne_compare.py — ZNE comparative analysis in the IQM noise simulator
===========================================================================
Self-contained (independent) experiment for Sec. 7.2 of the manuscript.
Compares four mitigation conditions on the m=1 IQAE circuit under the
qiskit-aer depolarising noise model, with NO hardware and NO mitiq:

  (a) raw           no mitigation (lambda = 1 only)
  (b) global_fold   U -> U (U^dag U)^n          [Giurgica-Tiron 2020, baseline]
  (c) gate_fold     local folding, Eq. (5)      [Giurgica-Tiron 2020]
  (d) iim_fiim      identity insertion on cx/cz [He et al., PRA 102, 012426]

Noise levels mirror E1/E2/E3:  p_2q in {1e-3, 5e-3, 1e-2}, p_1q = p_2q/10.
Scale factors:                 lambda in {1, 3, 5}.
Shots:                         4096.   Replicas (for variance): 3.   Seed: 42.

IMPORTANT — optimization level. Folded circuits are transpiled at
optimization_level=0 so the U^dag U / G G structure survives and amplifies
noise (matching scripts/hw_wave1_zne_iqae_v2.py). At opt_level=1 the
transpiler cancels all digital folding back to the base depth, so no noise
amplification occurs; that fact is itself logged as a finding.

Usage
-----
    python scripts/hw_sim_zne_compare.py

No IQM_TOKEN, no data file, no network. Runs in a few seconds.
Output: results/hardware_validation/sim_zne_compare_latest.json
"""

import os
import sys
import json
import math
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
HW_OUT       = PROJECT_ROOT / "results" / "hardware_validation"
HW_OUT.mkdir(parents=True, exist_ok=True)
OUT_JSON     = HW_OUT / "sim_zne_compare_latest.json"
LOG_FILE     = HW_OUT / "sim_zne_compare_latest.log"

# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
THETA        = 0.2265                       # same IQAE angle as Sec. 7.1 / 7.2
ZNE_SCALES   = [1.0, 3.0, 5.0]
NOISE_LEVELS = {"E1": 1e-3, "E2": 5e-3, "E3": 1e-2}     # p_2q, mirrors E1/E2/E3
P1Q_RATIO    = 0.1                          # p_1q = p_2q / 10  (E-family ratio)
SHOTS        = 4096
REPLICAS     = 3
SEED         = 42
BASIS        = ["cx", "u", "p", "x", "h", "rx", "ry", "rz", "sx", "id"]
GATES_1Q     = ["u", "p", "x", "h", "rx", "ry", "rz", "sx"]
P_THEORY_M1  = math.sin(3 * THETA) ** 2     # p_th(m=1) = sin^2(3*theta)

CONDITIONS = ["raw", "global_fold", "gate_fold", "iim_fiim"]


# ---------------------------------------------------------------------------
# Tiny logger (same spirit as the wave scripts)
# ---------------------------------------------------------------------------
class Logger:
    def __init__(self, path):
        self._f = open(path, "w")

    def _w(self, tag, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {msg}"
        print(line)
        self._f.write(line + "\n")
        self._f.flush()

    def info(self, m):  self._w("INFO ", m)
    def ok(self, m):    self._w("OK   ", m)
    def warn(self, m):  self._w("WARN ", m)
    def error(self, m): self._w("ERROR", m)
    def sep(self, t=""): self._w("=====", f"{'='*10} {t} {'='*10}" if t else "=" * 40)


def _jd(o):
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"Not serializable: {type(o).__name__}")


# ---------------------------------------------------------------------------
# Imports that need qiskit
# ---------------------------------------------------------------------------
log = Logger(LOG_FILE)
log.sep("SIMULATOR ZNE COMPARISON  (Sec. 7.2)")

try:
    from qiskit import QuantumCircuit, ClassicalRegister, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
except ImportError as e:
    log.error(f"Qiskit not available: {e}. pip install qiskit==2.1.2 qiskit-aer==0.17.2")
    sys.exit(1)

from src.phase_2.zne_foldingfree import (
    global_fold, gate_fold, identity_insertion, extrapolate,
)

log.ok(f"theta={THETA}  p_theory(m=1)={P_THEORY_M1:.4f}")


# ---------------------------------------------------------------------------
# Circuit construction (self-contained; mirrors hw_wave1_zne_iqae_v2.py)
# ---------------------------------------------------------------------------
def to_basis(qc, opt=1):
    return transpile(qc, basis_gates=BASIS, optimization_level=opt)


def build_A():
    qc = QuantumCircuit(2, name="A")
    qc.ry(2 * THETA, 0)
    qc.cx(0, 1)
    return to_basis(qc)


def build_Q(A):
    S0 = QuantumCircuit(2, name="S0")
    S0.x(0); S0.x(1); S0.cz(0, 1); S0.x(0); S0.x(1)
    Q = QuantumCircuit(2, name="Q")
    Q.z(1)
    Q.compose(A.inverse(), inplace=True)
    Q.compose(to_basis(S0), inplace=True)
    Q.compose(A, inplace=True)
    return to_basis(Q)


def build_base_m1():
    A = build_A()
    Q = build_Q(A)
    qc = QuantumCircuit(2, name="base_m1")
    qc.compose(A, inplace=True)
    qc.compose(Q, inplace=True)
    return qc


def with_meas(base):
    qc = base.copy()
    cr = ClassicalRegister(1, "anc")
    qc.add_register(cr)
    qc.measure(1, 0)
    return qc


def make_noise_model(p_2q, p_1q):
    """Depolarising model matched to the script's basis and the p_1q/p_2q split."""
    m = NoiseModel()
    m.add_all_qubit_quantum_error(depolarizing_error(p_1q, 1), GATES_1Q)
    m.add_all_qubit_quantum_error(depolarizing_error(p_2q, 2), ["cx"])
    return m


def apply_condition(base, condition, scale):
    """Return the scaled circuit for a given mitigation condition."""
    if scale == 1.0 or condition == "raw":
        return base.copy()
    if condition == "global_fold":
        return global_fold(base, scale)
    if condition == "gate_fold":
        return gate_fold(base, scale, method="random", seed=SEED)
    if condition == "iim_fiim":
        return identity_insertion(base, scale, method="fixed")
    raise ValueError(f"Unknown condition {condition!r}")


# ---------------------------------------------------------------------------
# Single measurement (condition, noise, scale, replica)
# ---------------------------------------------------------------------------
def measure_p(sim, base, condition, scale, shots, seed):
    scaled = apply_condition(base, condition, scale)
    # raw / lambda=1 use opt_level=1 (a normal run); folded use opt_level=0
    # so the U^dag U / G G structure survives and amplifies noise.
    opt = 1 if (scale == 1.0 or condition == "raw") else 0
    qc  = to_basis(with_meas(scaled), opt=opt)
    t0  = time.perf_counter()
    counts = sim.run(qc, shots=shots, seed_simulator=seed).result().get_counts()
    wall = time.perf_counter() - t0
    tot = sum(counts.values())
    p = counts.get("1", 0) / tot
    return {
        "p_meas": p,
        "depth_actual": int(qc.depth()),
        "wall_clock_s": wall,
        "shots": shots,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    base = build_base_m1()
    base_depth_opt1 = to_basis(base, opt=1).depth()
    log.info(f"base m=1 depth (opt_level=1) = {base_depth_opt1}")
    log.info(f"conditions={CONDITIONS}  scales={ZNE_SCALES}  "
             f"noise={list(NOISE_LEVELS)}  shots={SHOTS}  replicas={REPLICAS}")

    results = {
        "metadata": {
            "script": "hw_sim_zne_compare.py",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "theta": THETA,
            "p_theory_m1": P_THEORY_M1,
            "scales": ZNE_SCALES,
            "noise_levels": NOISE_LEVELS,
            "p1q_ratio": P1Q_RATIO,
            "shots": SHOTS,
            "replicas": REPLICAS,
            "seed": SEED,
            "basis": BASIS,
            "base_depth_opt1": int(base_depth_opt1),
            "opt_level_note": ("folded circuits transpiled at opt_level=0 to "
                               "preserve folds; opt_level=1 cancels them to base"),
            "qiskit": __import__("qiskit").__version__,
            "aer": __import__("qiskit_aer").__version__,
        },
        "comparison": {},     # comparison[noise][condition] = {...}
        "block_A_zne": {},    # wave1-compatible view (global_fold @ E3) for fig reuse
    }

    for noise_label, p_2q in NOISE_LEVELS.items():
        p_1q = p_2q * P1Q_RATIO
        sim = AerSimulator(noise_model=make_noise_model(p_2q, p_1q))
        results["comparison"][noise_label] = {}
        log.sep(f"NOISE {noise_label}  p_2q={p_2q}  p_1q={p_1q:.5f}")

        for cond in CONDITIONS:
            scale_results = {}
            scales_used = [1.0] if cond == "raw" else ZNE_SCALES
            for scale in scales_used:
                reps = [measure_p(sim, base, cond, scale, SHOTS, SEED + r)
                        for r in range(REPLICAS)]
                p_vals = np.array([r["p_meas"] for r in reps])
                scale_results[str(scale)] = {
                    "p_hw": float(p_vals.mean()),
                    "p_std": float(p_vals.std(ddof=1)) if REPLICAS > 1 else 0.0,
                    "bias_raw": float(abs(p_vals.mean() - P_THEORY_M1)),
                    "depth_intended": int(round(scale * base_depth_opt1)),
                    "depth_actual": reps[0]["depth_actual"],
                    "wall_clock_s": float(np.mean([r["wall_clock_s"] for r in reps])),
                    "replicas": [float(p) for p in p_vals],
                }

            entry = {"scale_results": scale_results}
            if cond != "raw":
                scales = [float(s) for s in ZNE_SCALES]
                p_curve = [scale_results[str(s)]["p_hw"] for s in ZNE_SCALES]
                p_rich = extrapolate(scales, p_curve, "richardson")
                p_exp  = extrapolate(scales, p_curve, "exponential")
                p_lin  = extrapolate(scales, p_curve, "linear")
                entry.update({
                    "p_extrap_richardson": p_rich,
                    "p_extrap_exponential": p_exp,
                    "p_extrap_linear": p_lin,
                    "bias_zne_richardson": abs(p_rich - P_THEORY_M1),
                    "bias_zne_exponential": abs(p_exp - P_THEORY_M1),
                    "raw_better_than_zne": (abs(p_curve[0] - P_THEORY_M1)
                                            < abs(p_rich - P_THEORY_M1)),
                })
            results["comparison"][noise_label][cond] = entry

            # console line
            if cond == "raw":
                sr = scale_results["1.0"]
                log.info(f"  {cond:<12} p={sr['p_hw']:.4f}+-{sr['p_std']:.4f}  "
                         f"bias={sr['bias_raw']:.4f}  depth={sr['depth_actual']}")
            else:
                d = [scale_results[str(s)]["depth_actual"] for s in ZNE_SCALES]
                log.info(f"  {cond:<12} p(1,3,5)="
                         f"[{p_curve[0]:.3f},{p_curve[1]:.3f},{p_curve[2]:.3f}]  "
                         f"depth(1,3,5)={d}  "
                         f"ZNE_rich={entry['p_extrap_richardson']:.3f}"
                         f"(bias={entry['bias_zne_richardson']:.4f})  "
                         f"raw_better={entry['raw_better_than_zne']}")

    # wave1-compatible block (global_fold @ E3) so the legacy figure path works
    e3 = results["comparison"]["E3"]
    g = e3["global_fold"]
    results["block_A_zne"] = {
        "scale_results": {s: {"p_hw": g["scale_results"][s]["p_hw"]} for s in g["scale_results"]},
        "p_theory": P_THEORY_M1,
        "p_extrap_poly3": g.get("p_extrap_richardson", P_THEORY_M1),
        "p_extrap_linear": g.get("p_extrap_linear", P_THEORY_M1),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2, default=_jd))
    log.sep("DONE")
    log.ok(f"written: {OUT_JSON}")

    # headline finding
    any_zne_helps = any(
        not results["comparison"][nl][c].get("raw_better_than_zne", True)
        for nl in NOISE_LEVELS for c in CONDITIONS if c != "raw"
    )
    if any_zne_helps:
        log.ok("Finding: ZNE improves over raw in at least one (condition, noise).")
    else:
        log.ok("Finding: raw beats ZNE in ALL conditions/noise — ZNE counter-"
               "productive for IQAE m=1 below the aliasing threshold (Case B).")


if __name__ == "__main__":
    main()
