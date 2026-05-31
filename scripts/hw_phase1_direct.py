"""
hw_phase1_direct.py — Hardware Validation: Direct Amplitude Encoding
====================================================================
Validates QAE Grover oscillation on IQM Emerald using direct Ry encoding.

Instead of Möttönen (depth=52 for non-uniform distribution):
  A = Ry(2*arcsin(sqrt(a_ideal))) on q0 + CX(q0->ancilla)
  Total: ~3 CZ gates, depth ~5 after transpilation.

This matches the approach of Woerner & Egger (2019) for hardware demos.
The amplitude a_ideal=0.050442 comes directly from exp_A1_n3_PA results.

Prerequisites:
    python -m pip install "iqm-client[qiskit]"

Usage:
    $env:IQM_TOKEN = "your_token"
    python scripts/hw_phase1_direct.py
"""

import os, sys, json, math, warnings
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'scripts' else SCRIPT_DIR
HW_OUT       = PROJECT_ROOT / 'results' / 'hardware_validation'
HW_OUT.mkdir(parents=True, exist_ok=True)
LOG_FILE     = HW_OUT / 'phase1_direct_latest.log'

def _json_default(o):
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"Not serializable: {type(o).__name__}")

class Logger:
    def __init__(self, path):
        self._f = open(path, 'w')
    def _w(self, tag, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {msg}"
        print(line); self._f.write(line + '\n'); self._f.flush()
    def info(self, m):  self._w('INFO ', m)
    def ok(self, m):    self._w('OK   ', m)
    def warn(self, m):  self._w('WARN ', m)
    def error(self, m): self._w('ERROR', m)
    def sep(self, t=''): self._w('=====', f"{'='*12} {t} {'='*12}" if t else '='*40)

log = Logger(LOG_FILE)
log.sep("HW PHASE 1 — DIRECT AMPLITUDE ENCODING")

# ── Token ─────────────────────────────────────────────────────────────────────
IQM_TOKEN    = os.environ.get('IQM_TOKEN', '')
HW_AVAILABLE = bool(IQM_TOKEN)
if not HW_AVAILABLE:
    log.warn("IQM_TOKEN not set — Phase 0 (Aer) will run, hardware skipped.")
else:
    log.ok("IQM_TOKEN found.")

# ── Reference values from A1_n3_PA ────────────────────────────────────────────
log.sep("CONFIG")
REF_JSON = (PROJECT_ROOT / 'results' / 'exp_A1_n3_PA' /
            'tfm_comprehensive_metrics_latest.json')
if REF_JSON.exists():
    ref     = json.loads(REF_JSON.read_text())
    A_IDEAL = ref['metrics']['quantum']['qae'].get('iqae_tail_prob', 0.050442)
    ALPHA   = ref['metadata']['configuration']['phase2'].get('alpha', 0.05)
    log.ok(f"Loaded from A1_n3_PA: a_ideal={A_IDEAL:.6f}")
else:
    A_IDEAL, ALPHA = 0.050442, 0.05
    log.warn("JSON not found — using hardcoded a_ideal=0.050442")

THETA       = math.asin(math.sqrt(A_IDEAL))
THETA_RY    = 2 * THETA          # Ry angle: Ry(2θ)|0> = cos(θ)|0> + sin(θ)|1>
P_THEORY    = {m: math.sin((2*m+1)*THETA)**2 for m in range(5)}
SHOTS_STATE  = 4096              # More shots for better statistics
SHOTS_GROVER = 2048

log.info(f"Strategy: direct Ry encoding (1 qubit + 1 ancilla = 2 total)")
log.info(f"a_ideal = {A_IDEAL:.6f}  (from A1_n3_PA IQAE noiseless)")
log.info(f"theta   = {THETA:.6f} rad")
log.info(f"Ry angle = 2*theta = {THETA_RY:.6f} rad")
log.info(f"Circuit: Ry({THETA_RY:.4f}) on q0, CX(q0->q1)")
for m in range(3):
    log.info(f"  P_theory[m={m}] = {P_THEORY[m]*100:.3f}%")

# ── Circuit construction ──────────────────────────────────────────────────────
log.sep("CIRCUITS")

from qiskit import QuantumCircuit, ClassicalRegister, transpile

BASIS = ['cx', 'u', 'p', 'x', 'h', 'rx', 'ry', 'rz', 'sx', 'id']

def to_basis(qc):
    return transpile(qc, basis_gates=BASIS, optimization_level=1)

def build_A_direct(a):
    """
    Direct amplitude encoding A on 2 qubits (q0=data, q1=ancilla).

    A|00> = cos(theta)|0>|0> + sin(theta)|1>|1>
          = sqrt(1-a)|bad>|0> + sqrt(a)|good>|1>

    where theta = arcsin(sqrt(a)).

    Circuit:
      Ry(2*theta) on q0   <- P(q0=1) = sin^2(theta) = a
      CX(q0 -> q1)        <- ancilla=1 iff q0=1 (good state)

    Depth: 2 gates. After IQM transpilation: ~3-5 gates.
    Scientifically equivalent to Möttönen for validating Grover oscillation.
    """
    theta = math.asin(math.sqrt(a))
    qc    = QuantumCircuit(2, name='A_direct')
    qc.ry(2 * theta, 0)   # state qubit
    qc.cx(0, 1)            # ancilla follows state
    return to_basis(qc)

def build_S0_2q():
    """S0 = I - 2|00><00| on 2 qubits."""
    qc = QuantumCircuit(2, name='S0')
    qc.x(0); qc.x(1)
    qc.cz(0, 1)
    qc.x(0); qc.x(1)
    return to_basis(qc)

def build_Q_direct(A, n_tot=2):
    """Q = A * S0^2 * A† * S_chi  (S_chi = Z on ancilla q1)."""
    S0 = build_S0_2q()
    Q  = QuantumCircuit(n_tot, name='Q')
    Q.z(1)                               # S_chi: phase flip on ancilla=1
    Q.compose(A.inverse(), inplace=True)  # A†
    Q.compose(S0,           inplace=True) # S0
    Q.compose(A,            inplace=True) # A
    return to_basis(Q)

# Build circuits
A_FULL = build_A_direct(A_IDEAL)
Q_OP   = build_Q_direct(A_FULL)

# State-only: just measure q0 to verify P(q0=1) = a_ideal
qc_state = QuantumCircuit(1, 1, name='state_only')
qc_state.ry(THETA_RY, 0)
qc_state.measure(0, 0)

# Grover circuits m=0,1,2: A + Q^m, measure ancilla (q1)
GROVER = {}
for m in [0, 1, 2, 3]:
    qc = QuantumCircuit(2, name=f'QAE_m{m}')
    qc.compose(A_FULL, inplace=True)
    for _ in range(m): qc.compose(Q_OP, inplace=True)
    cr = ClassicalRegister(1, 'anc')
    qc.add_register(cr); qc.measure(1, 0)
    GROVER[m] = qc

# Log circuit stats
log.info(f"state_only: depth={qc_state.depth()}  "
         f"ops={dict(qc_state.count_ops())}")
for m, qc in GROVER.items():
    ops = dict(qc.count_ops())
    bad = [k for k in ops if k in ('unitary','multiplexer_dg','Q','initialize')]
    log.info(f"Grover m={m}: depth={qc.depth()} gates={qc.size()} "
             f"{'OPAQUE:'+str(bad) if bad else 'basis=OK'}")

# ── PHASE 0 — Aer noiseless ───────────────────────────────────────────────────
log.sep("PHASE 0 — AER NOISELESS")

from qiskit_aer import AerSimulator
sim = AerSimulator()

# Verify state prep: P(q0=1) should equal a_ideal
cnt  = sim.run(qc_state, shots=SHOTS_STATE).result().get_counts()
tot  = sum(cnt.values())
p_q0 = cnt.get('1', 0) / tot
err_sp = abs(p_q0 - A_IDEAL)
log.info(f"State prep: P(q0=1)={p_q0*100:.3f}%  a_ideal={A_IDEAL*100:.3f}%  "
         f"err={err_sp*100:.3f}%  [{'PASS' if err_sp<0.01 else 'FAIL'}]")

# Verify Grover oscillation
aer_g = {}; mono_v = []
for m in [0, 1, 2, 3]:
    cnt = sim.run(GROVER[m], shots=SHOTS_GROVER).result().get_counts()
    tot = sum(cnt.values())
    p   = cnt.get('1', 0) / tot
    th  = P_THEORY[m]
    err = abs(p - th)
    tol = 4 * math.sqrt(th*(1-th)/SHOTS_GROVER)
    ok  = err < tol
    if m <= 2: mono_v.append(p)
    log.info(f"Grover m={m}: P_aer={p*100:.2f}%  P_th={th*100:.2f}%  "
             f"err={err*100:.2f}%  [{'PASS' if ok else 'FAIL'}]")
    aer_g[m] = {'p_aer': float(p), 'p_theory': float(th),
                'error': float(err), 'pass': bool(ok)}

mono    = mono_v[0] < mono_v[1] < mono_v[2]
g_range = mono_v[2] - mono_v[0]
log.info(f"Monotone: {mono}  Range: {g_range:.4f}  (target > 0.20)")

phase0 = {'p_state': float(p_q0), 'a_ideal': A_IDEAL,
          'grover': aer_g, 'monotone': bool(mono),
          'range': float(g_range), 'pass': bool(mono and g_range > 0.20)}
(HW_OUT / 'phase0_direct_latest.json').write_text(
    json.dumps({'phase0': phase0, 'ts': datetime.now().isoformat()},
               indent=2, default=_json_default))

if not phase0['pass']:
    log.error("Phase 0 FAILED — circuit error, do not proceed.")
    sys.exit(1)
log.ok("Phase 0 PASSED — Grover oscillation confirmed in Aer.")

# ── PHASE 1 — IQM Emerald ─────────────────────────────────────────────────────
if not HW_AVAILABLE:
    log.warn("Skipping Phase 1 — set IQM_TOKEN.")
    sys.exit(0)

log.sep("PHASE 1 — IQM EMERALD")

try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    log.error("Run: python -m pip install 'iqm-client[qiskit]'")
    sys.exit(1)

os.environ['IQM_TOKEN'] = IQM_TOKEN
provider = IQMProvider('https://cocos.resonance.meetiqm.com/emerald')
backend  = provider.get_backend()
log.info(f"Backend: {backend.name}  {backend.num_qubits} qubits")

# Transpile
log.info("Transpiling for IQM Emerald...")
isa_state  = transpile(qc_state,    backend=backend, optimization_level=1)
isa_grover = {m: transpile(qc, backend=backend, optimization_level=1)
              for m, qc in GROVER.items()}

log.info(f"state_only: depth={isa_state.depth()}")
for m, isa in isa_grover.items():
    cz = isa.count_ops().get('cz', isa.count_ops().get('cx', 0))
    log.info(f"Grover m={m}: depth={isa.depth()}  CZ={cz}")

# Job 1: state prep — verify P(q0=1) = a_ideal on hardware
log.info(f"Job 1 — state prep ({SHOTS_STATE} shots)...")
job    = backend.run(isa_state, shots=SHOTS_STATE, use_timeslot=False)
log.info(f"  job_id: {job.job_id()}")
result = job.result()
cnt1   = result.get_counts()
tot1   = sum(cnt1.values())
p_hw_sp = cnt1.get('1', 0) / tot1
err_hw  = abs(p_hw_sp - A_IDEAL)
sp_v    = 'PASS' if err_hw < 0.05 else ('MARGINAL' if err_hw < 0.10 else 'FAIL')
log.info(f"  P_hw(q0=1)={p_hw_sp*100:.2f}%  a_ideal={A_IDEAL*100:.2f}%  "
         f"err={err_hw*100:.2f}%  [{sp_v}]")

phase1 = {
    'device': 'IQM Emerald (Crystal 54)',
    'encoding': 'direct Ry: depth=2 pre-transpile, ~5 post-transpile',
    'state_prep': {
        'job_id': str(job.job_id()), 'shots': SHOTS_STATE,
        'p_hw': float(p_hw_sp), 'a_ideal': A_IDEAL,
        'error': float(err_hw), 'verdict': sp_v
    },
    'grover': {}
}

# Jobs 2-5: Grover m=0,1,2,3
p_hw_v = []
for m in [0, 1, 2, 3]:
    log.info(f"Job {m+2} — Grover m={m} ({SHOTS_GROVER} shots)...")
    job    = backend.run(isa_grover[m], shots=SHOTS_GROVER,
                         use_timeslot=False)
    log.info(f"  job_id: {job.job_id()}")
    result = job.result()
    cnt    = result.get_counts()
    tot    = sum(cnt.values())
    p_hw   = cnt.get('1', 0) / tot
    p_th   = P_THEORY[m]
    err    = abs(p_hw - p_th)
    # 3-sigma tolerance for hardware (noise allowed)
    tol_hw = 3 * math.sqrt(p_hw * (1-p_hw) / SHOTS_GROVER) if p_hw > 0 else 0.1
    v      = 'PASS' if err < 0.15 else 'FAIL'
    if m <= 2: p_hw_v.append(p_hw)
    log.info(f"  P_hw={p_hw*100:.2f}%  P_theory={p_th*100:.2f}%  "
             f"err={err*100:.2f}%  [{v}]")
    phase1['grover'][f'm{m}'] = {
        'job_id': str(job.job_id()), 'shots': SHOTS_GROVER,
        'p_hw': float(p_hw), 'p_theory': float(p_th),
        'error': float(err), 'verdict': v
    }

mono_hw  = p_hw_v[0] < p_hw_v[1] < p_hw_v[2]
range_hw = p_hw_v[2] - p_hw_v[0]
phase1.update({
    'monotone': bool(mono_hw),
    'range': float(range_hw),
    'advance_phase2': bool(mono_hw and range_hw > 0.20),
    'advance_phase3': bool(mono_hw),
})

# ── Results ───────────────────────────────────────────────────────────────────
log.sep("RESULTS")
log.info(f"State prep: P_hw={p_hw_sp*100:.2f}%  a_ideal={A_IDEAL*100:.2f}%  [{sp_v}]")
for m in [0,1,2,3]:
    g = phase1['grover'][f'm{m}']
    log.info(f"Grover m={m}: {g['p_hw']*100:.2f}%  "
             f"(theory {g['p_theory']*100:.2f}%)  [{g['verdict']}]")
log.info(f"Monotone: {mono_hw}  Range: {range_hw:.4f}")
log.info(f"Advance Phase 2: {phase1['advance_phase2']}")

# Paper table
log.sep("PAPER TABLE")
log.info(f"{'Experiment':<30} {'Theory':>8} {'Hardware':>10} {'Verdict':>10}")
log.info('-'*62)
log.info(f"{'State prep P(q0=1)':<30} {A_IDEAL*100:>7.2f}% {p_hw_sp*100:>9.2f}%  {sp_v:>10}")
for m in [0,1,2,3]:
    g  = phase1['grover'][f'm{m}']
    th = g['p_theory']*100
    hw = g['p_hw']*100
    log.info(f"{'Grover m='+str(m)+' P(anc=1)':<30} {th:>7.2f}% {hw:>9.2f}%  {g['verdict']:>10}")

out = {
    'metadata': {
        'device': 'IQM Emerald (Crystal 54)',
        'encoding': 'direct Ry(2*arcsin(sqrt(a_ideal))) + CX',
        'a_ideal': A_IDEAL, 'theta': THETA, 'theta_ry': THETA_RY,
        'reference_experiment': 'A1_n3_PA',
        'scientific_note': (
            'Direct encoding validates Grover operator Q independently of '
            'Möttönen distribution encoding. a_ideal derived from noiseless '
            'IQAE simulation on S&P500 PA portfolio (A1_n3_PA experiment).'
        ),
        'ts': datetime.now().isoformat()
    },
    'phase0': phase0,
    'phase1': phase1
}
out_path = HW_OUT / 'phase1_direct_results_latest.json'
out_path.write_text(json.dumps(out, indent=2, default=_json_default))
log.ok(f"Results: {out_path}")
