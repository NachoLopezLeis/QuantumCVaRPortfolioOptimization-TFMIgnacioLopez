"""
hw_zne_compare_resonance.py - Hardware ZNE comparison on IQM Resonance
=====================================================================
Executes the Section 7.2 ZNE comparison on REAL IQM hardware (Emerald via
IQM Resonance), upgrading the simulator comparison to a hardware result.

Conditions at m=1 (sub-threshold), scale factors lambda in {1,3,5}:
  raw          lambda=1 only
  global_fold  U -> U(U^dag U)^n           [Giurgica-Tiron 2020]
  gate_fold    local folding, Eq. (5)       [Giurgica-Tiron 2020]
  iim_fiim     identity insertion on cz/cx  [He et al., PRA 102, 012426]

Connection pattern reused verbatim from scripts/hw_wave1_zne_iqae_v2.py
(iqm.qiskit_iqm.IQMProvider, Resonance cocos URL). Folding primitives reused
from src/phase_2/zne_foldingfree.py.

SAFETY
------
* Runs in --dry-run mode BY DEFAULT: builds and transpiles every circuit,
  prints the depth and a credit estimate, but submits NOTHING.
* Add --execute to actually submit jobs (spends credits).
* --max-credits N aborts before submitting if the estimate exceeds N.

Usage
-----
    $env:IQM_TOKEN = "your_token"                       # PowerShell
    python scripts/hw_zne_compare_resonance.py                       # dry-run
    python scripts/hw_zne_compare_resonance.py --execute --max-credits 95

Output (only with --execute):
    results/hardware_validation/hw_zne_compare_latest.json
    results/hardware_validation/hw_zne_compare_latest.log
"""

import os
import sys
import json
import math
import time
import argparse
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
OUT_JSON     = HW_OUT / "hw_zne_compare_latest.json"
LOG_FILE     = HW_OUT / "hw_zne_compare_latest.log"

# ---------------------------------------------------------------------------
# Experiment constants (mirror hw_wave1 and hw_sim_zne_compare)
# ---------------------------------------------------------------------------
THETA        = 0.2265
THETA_RY     = 2 * THETA
ZNE_SCALES   = [1.0, 3.0, 5.0]
SHOTS        = 2048                       # budget-aware, as in Wave 2
REPLICAS     = 2                          # 2 independent runs per circuit
EXTRA_REPEAT = ["iim_fiim"]               # +1 extra run for FIIM at lambda=5
BASIS        = ["cz", "r", "rx", "ry", "rz", "x", "h", "p", "cx", "u", "sx", "id"]
P_THEORY_M1  = math.sin(3 * THETA) ** 2
RESONANCE_URL = "https://cocos.resonance.meetiqm.com/emerald"

# Credit calibration: prior work was ~5 credits per circuit-execution
# (110 credits / ~23 jobs). Adjustable; the estimate is printed before any run.
CREDITS_PER_JOB = 5.0

# raw + global_fold + iim_fiim. gate_fold is omitted on hardware: under uniform
# depolarising noise it coincides exactly with global_fold (verified in the
# simulator run, Sec. 7.2), so it would spend credits for a redundant curve.
CONDITIONS = ["raw", "global_fold", "iim_fiim"]


class Logger:
    def __init__(self, path, enabled=True):
        self._f = open(path, "w") if enabled else None
    def _w(self, tag, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {msg}"
        print(line)
        if self._f:
            self._f.write(line + "\n"); self._f.flush()
    def info(self, m):  self._w("INFO ", m)
    def ok(self, m):    self._w("OK   ", m)
    def warn(self, m):  self._w("WARN ", m)
    def error(self, m): self._w("ERROR", m)
    def sep(self, t=""): self._w("=====", f"{'='*8} {t} {'='*8}" if t else "="*36)


def _jd(o):
    if isinstance(o, (np.bool_,)):    return bool(o)
    if isinstance(o, (np.integer,)):  return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray):     return o.tolist()
    raise TypeError(type(o).__name__)


# ---------------------------------------------------------------------------
# Circuit construction (identical to hw_wave1_zne_iqae_v2.py)
# ---------------------------------------------------------------------------
def build_circuits():
    from qiskit import QuantumCircuit, ClassicalRegister, transpile

    def to_basis(qc, opt=1):
        return transpile(qc, basis_gates=BASIS, optimization_level=opt)

    def build_A():
        qc = QuantumCircuit(2, name="A"); qc.ry(THETA_RY, 0); qc.cx(0, 1)
        return to_basis(qc)

    def build_Q(A):
        S0 = QuantumCircuit(2, name="S0")
        S0.x(0); S0.x(1); S0.cz(0, 1); S0.x(0); S0.x(1)
        Q = QuantumCircuit(2, name="Q")
        Q.z(1); Q.compose(A.inverse(), inplace=True)
        Q.compose(to_basis(S0), inplace=True); Q.compose(A, inplace=True)
        return to_basis(Q)

    def build_base(A, Q, m):
        qc = QuantumCircuit(2, name=f"base_m{m}")
        qc.compose(A, inplace=True)
        for _ in range(m): qc.compose(Q, inplace=True)
        return qc

    def with_meas(base):
        qc = base.copy(); cr = ClassicalRegister(1, "anc")
        qc.add_register(cr); qc.measure(1, 0); return qc

    A = build_A(); Q = build_Q(A)
    base_m1 = build_base(A, Q, 1)
    return base_m1, with_meas, to_basis


def scaled_circuit(base, condition, scale):
    """Return the scaled circuit for a condition (reuses zne_foldingfree)."""
    from src.phase_2.zne_foldingfree import global_fold, gate_fold, identity_insertion
    if scale == 1.0 or condition == "raw":
        return base.copy()
    if condition == "global_fold":
        return global_fold(base, scale)
    if condition == "gate_fold":
        return gate_fold(base, scale, method="random", seed=42)
    if condition == "iim_fiim":
        return identity_insertion(base, scale, method="fixed")
    raise ValueError(condition)


def build_job_list(base, with_meas, to_basis):
    """Enumerate (condition, scale, replica) jobs to run."""
    from qiskit import transpile
    jobs = []
    for cond in CONDITIONS:
        scales = [1.0] if cond == "raw" else ZNE_SCALES
        for scale in scales:
            n_rep = REPLICAS + (1 if cond in EXTRA_REPEAT and scale in (1.0, 5.0) else 0)
            scaled = scaled_circuit(base, cond, scale)
            opt = 1 if (scale == 1.0 or cond == "raw") else 0
            qc = to_basis(with_meas(scaled), opt=opt)
            for r in range(n_rep):
                jobs.append({
                    "condition": cond, "scale": scale, "replica": r,
                    "opt_level": opt, "depth_local": int(qc.depth()),
                    "label": f"{cond}_s{int(scale)}_r{r}",
                })
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="ZNE comparison on IQM Resonance")
    ap.add_argument("--execute", action="store_true",
                    help="Actually submit jobs (spends credits). Default: dry-run.")
    ap.add_argument("--max-credits", type=float, default=95.0,
                    help="Abort before submitting if estimate exceeds this.")
    ap.add_argument("--shots", type=int, default=SHOTS)
    args = ap.parse_args()

    dry = not args.execute
    log = Logger(LOG_FILE, enabled=args.execute)
    log.sep("HW ZNE COMPARISON on IQM Resonance (Sec. 7.2)")
    log.info(f"mode={'DRY-RUN (no submission)' if dry else 'EXECUTE'}  "
             f"shots={args.shots}  p_theory(m=1)={P_THEORY_M1:.4f}")

    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError as e:
        log.error(f"qiskit not available: {e}"); sys.exit(1)

    base, with_meas, to_basis = build_circuits()

    # Aer sanity: every condition is logically equivalent (p ~ p_theory)
    sim = AerSimulator()
    log.sep("AER SANITY (logical equivalence)")
    for cond in CONDITIONS:
        for scale in ([1.0] if cond == "raw" else ZNE_SCALES):
            sc = scaled_circuit(base, cond, scale)
            opt = 1 if (scale == 1.0 or cond == "raw") else 0
            qc = to_basis(with_meas(sc), opt=opt)
            cnt = sim.run(qc, shots=4096, seed_simulator=42).result().get_counts()
            p = cnt.get("1", 0) / sum(cnt.values())
            log.info(f"  {cond:<12} lam={int(scale)} depth={qc.depth():>3} "
                     f"p_aer={p:.4f} (theory {P_THEORY_M1:.4f})")

    jobs = build_job_list(base, with_meas, to_basis)
    est_credits = len(jobs) * CREDITS_PER_JOB
    log.sep("JOB PLAN + CREDIT ESTIMATE")
    for j in jobs:
        log.info(f"  {j['label']:<18} depth={j['depth_local']:>3} opt={j['opt_level']}")
    log.info(f"TOTAL JOBS = {len(jobs)}  ->  EST. CREDITS = {est_credits:.0f} "
             f"(at {CREDITS_PER_JOB:.0f}/job)")

    if est_credits > args.max_credits:
        log.error(f"Estimate {est_credits:.0f} exceeds --max-credits "
                  f"{args.max_credits:.0f}. Aborting before any submission.")
        sys.exit(2)

    if dry:
        log.sep("DRY-RUN COMPLETE")
        log.ok("No jobs submitted, no credits spent. Re-run with --execute to submit.")
        return

    # ---- real submission ----
    IQM_TOKEN = os.environ.get("IQM_TOKEN", "")
    if not IQM_TOKEN:
        log.error("IQM_TOKEN not set. Aborting."); sys.exit(1)
    try:
        from iqm.qiskit_iqm import IQMProvider
    except ImportError:
        log.error("pip install 'iqm-client[qiskit]'"); sys.exit(1)
    os.environ["IQM_TOKEN"] = IQM_TOKEN
    backend = IQMProvider(RESONANCE_URL).get_backend()
    log.ok(f"Connected: {backend.name}  {backend.num_qubits}q")

    from src.phase_2.zne_foldingfree import extrapolate

    raw_data = {}     # (cond, scale) -> list of p
    job_meta = []
    for j in jobs:
        sc = scaled_circuit(base, j["condition"], j["scale"])
        qc = transpile(with_meas(sc), backend=backend, optimization_level=j["opt_level"])
        t0 = time.perf_counter()
        job = backend.run(qc, shots=args.shots, use_timeslot=False)
        jid = str(job.job_id())
        cnt = job.result().get_counts()
        wall = time.perf_counter() - t0
        p = cnt.get("1", 0) / sum(cnt.values())
        raw_data.setdefault((j["condition"], j["scale"]), []).append(p)
        job_meta.append({**j, "job_id": jid, "p_hw": p, "depth_hw": int(qc.depth()),
                         "wall_s": wall})
        log.info(f"  [{j['label']}] job={jid} depth={qc.depth()} p={p:.4f}")

    # aggregate
    comparison = {}
    for cond in CONDITIONS:
        entry = {"scale_results": {}}
        for scale in ([1.0] if cond == "raw" else ZNE_SCALES):
            ps = np.array(raw_data[(cond, scale)])
            entry["scale_results"][str(scale)] = {
                "p_hw": float(ps.mean()),
                "p_std": float(ps.std(ddof=1)) if len(ps) > 1 else 0.0,
                "bias_raw": float(abs(ps.mean() - P_THEORY_M1)),
                "replicas": ps.tolist(),
            }
        if cond != "raw":
            curve = [entry["scale_results"][str(s)]["p_hw"] for s in ZNE_SCALES]
            p_rich = extrapolate([1.0, 3.0, 5.0], curve, "richardson")
            entry["p_extrap_richardson"] = p_rich
            entry["bias_zne_richardson"] = abs(p_rich - P_THEORY_M1)
            entry["raw_better_than_zne"] = (abs(curve[0] - P_THEORY_M1)
                                            < abs(p_rich - P_THEORY_M1))
        comparison[cond] = entry

    out = {
        "metadata": {
            "script": "hw_zne_compare_resonance.py",
            "generated": datetime.now().isoformat(timespec="seconds"),
            "device": backend.name, "resonance_url": RESONANCE_URL,
            "theta": THETA, "p_theory_m1": P_THEORY_M1,
            "scales": ZNE_SCALES, "shots": args.shots, "replicas": REPLICAS,
            "n_jobs": len(jobs), "est_credits": est_credits,
        },
        "comparison": comparison,
        "jobs": job_meta,
    }
    OUT_JSON.write_text(json.dumps(out, indent=2, default=_jd))
    log.sep("DONE")
    log.ok(f"written: {OUT_JSON}")


if __name__ == "__main__":
    main()
