"""
hw_phase1_emerald.py — Hardware Validation: IQM Emerald (Crystal 54)
=====================================================================
Same IQM_TOKEN as Garnet. Device: emerald (54 qubits, square lattice,
p2q~0.5%). Uses complementary oracle: 1 MCX instead of 7 → ~6x fewer CZ.

Install:
    python -m pip install "iqm-client[qiskit]"

Usage:
    $env:IQM_TOKEN = "your_token"
    python scripts/hw_phase1_emerald.py
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
LOG_FILE     = HW_OUT / 'phase1_emerald_latest.log'

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
    def sep(self, t=''): self._w('=====', f"{'='*15} {t} {'='*15}" if t else '='*45)

log = Logger(LOG_FILE)
log.sep("HW PHASE 1 — IQM EMERALD (Crystal 54)")

# ── Token ─────────────────────────────────────────────────────────────────────
IQM_TOKEN    = os.environ.get('IQM_TOKEN', '')
HW_AVAILABLE = bool(IQM_TOKEN)
if not HW_AVAILABLE:
    log.warn("IQM_TOKEN not set — hardware skipped, Phase 0 (Aer) will run.")
else:
    log.ok("IQM_TOKEN found.")

# ── Reference values ──────────────────────────────────────────────────────────
log.sep("CONFIG")
REF_JSON = (PROJECT_ROOT / 'results' / 'exp_A1_n3_PA' /
            'tfm_comprehensive_metrics_latest.json')
if REF_JSON.exists():
    ref     = json.loads(REF_JSON.read_text())
    A_IDEAL = ref['metrics']['quantum']['qae'].get('iqae_tail_prob', 0.050442)
    ALPHA   = ref['metadata']['configuration']['phase2'].get('alpha', 0.05)
    log.ok(f"Loaded JSON: a_ideal={A_IDEAL:.6f}")
else:
    A_IDEAL, ALPHA = 0.050442, 0.05
    log.warn("JSON not found — using hardcoded values.")

N_QUBITS     = 3
NUM_LEVELS   = 2 ** N_QUBITS
THETA        = math.asin(math.sqrt(A_IDEAL))
P_THEORY     = {m: math.sin((2*m+1)*THETA)**2 for m in range(5)}
SHOTS_STATE  = 2048
SHOTS_GROVER = 1024
BASIS        = ['cx', 'u', 'p', 'x', 'h', 'rx', 'ry', 'rz', 'sx', 'id']

log.info(f"Device: IQM Emerald (Crystal 54, 54 qubits, square lattice)")
log.info(f"n={N_QUBITS}  alpha={ALPHA}  a_ideal={A_IDEAL:.6f}  theta={THETA:.6f}")
for m in range(3):
    log.info(f"  P_theory[m={m}] = {P_THEORY[m]*100:.3f}%")

# ── Portfolio data ────────────────────────────────────────────────────────────
log.sep("DATA")
import pandas as pd

returns_df = pd.read_csv(
    PROJECT_ROOT / 'data' / 'returns_sp500_full.csv',
    index_col=0, parse_dates=True)
sel     = json.loads((PROJECT_ROOT / 'results' / 'network_selection' /
                      'selection_PA.json').read_text())
TICKERS = sel['selected_tickers']
WEIGHTS = np.ones(len(TICKERS)) / len(TICKERS)
LOSSES  = -(returns_df.loc[:'2023-12-31', TICKERS].dropna().values @ WEIGHTS)
VAR_THR = np.percentile(LOSSES, (1 - ALPHA) * 100)

# Post-fix bin-mean discretization
tail_bins = NUM_LEVELS - 1
ls = np.sort(LOSSES)
tail = ls[ls >= VAR_THR]; body = ls[ls < VAR_THR]
PROBS = np.zeros(NUM_LEVELS)
PROBS[0] = len(body) / len(LOSSES)
splits = np.array_split(tail, tail_bins) if len(tail) else [np.array([])]*tail_bins
for i, s in enumerate(splits): PROBS[i+1] = len(s) / len(LOSSES)
PROBS /= PROBS.sum()

log.info(f"PA ({len(TICKERS)} tickers)  scenarios={len(LOSSES)}  VaR={VAR_THR:.6f}")
log.info(f"Discretization: body={PROBS[0]*100:.2f}%  tail={PROBS[1:].sum()*100:.2f}%")

# ── Circuit construction ──────────────────────────────────────────────────────
log.sep("CIRCUITS")

from qiskit import QuantumCircuit, ClassicalRegister, transpile

def to_basis(qc):
    return transpile(qc, basis_gates=BASIS, optimization_level=1)

def build_dist_prep(probs, n):
    """Möttönen state preparation."""
    amps = np.sqrt(np.abs(probs)); amps /= np.linalg.norm(amps)
    qc   = QuantumCircuit(n)
    qc.initialize(amps.tolist(), range(n))
    qc2  = QuantumCircuit(n, name='dist')
    for inst in qc.decompose().decompose().decompose().data:
        if inst.operation.name != 'reset':
            qc2.append(inst.operation, inst.qubits)
    return to_basis(qc2)

def build_compl_oracle(n):
    """
    Complementary oracle: mark tail states (bins 1..2^n-1).
    Instead of 7 MCX gates (marking bins 1-7), use 1 MCX (marking bin 0)
    then flip ancilla. Equivalent result with ~6x fewer CZ gates.

    Circuit:
      X(0..n-1)          <- turn |000> into |111>
      MCX(0..n-1 -> anc) <- flip ancilla when all qubits are 1 (i.e. was |000>)
      X(0..n-1)          <- restore
      X(anc)             <- invert: ancilla=1 now means tail (NOT bin0)
    """
    qc = QuantumCircuit(n+1, name='oracle')
    anc = n
    for i in range(n): qc.x(i)
    if n == 1:   qc.cx(0, anc)
    elif n == 2: qc.ccx(0, 1, anc)
    else:        qc.mcx(list(range(n)), anc)
    for i in range(n): qc.x(i)
    qc.x(anc)
    return to_basis(qc)

def build_A_full(probs, n):
    """Full QAE state prep: A|0>^{n+1} = sqrt(1-a)|bad>|0> + sqrt(a)|good>|1>"""
    dist   = build_dist_prep(probs, n)
    oracle = build_compl_oracle(n)
    A      = QuantumCircuit(n+1, name='A')
    A.compose(dist,   qubits=range(n),   inplace=True)
    A.compose(oracle, qubits=range(n+1), inplace=True)
    return to_basis(A)

def build_Q(A, n):
    """
    Grover operator Q = A * S0^{n+1} * A† * S_chi
    S_chi = Z on ancilla qubit n (phase flip on good states)
    S0 on all n+1 qubits via MCX sandwich
    """
    n_tot = n + 1
    # S0 = X^{n+1} * H_{n} * MCX * H_{n} * X^{n+1}
    S0 = QuantumCircuit(n_tot, name='S0')
    for i in range(n_tot): S0.x(i)
    S0.h(n_tot-1)
    if n_tot == 2:   S0.cx(0, 1)
    elif n_tot == 3: S0.ccx(0, 1, 2)
    else:            S0.mcx(list(range(n_tot-1)), n_tot-1)
    S0.h(n_tot-1)
    for i in range(n_tot): S0.x(i)
    S0 = to_basis(S0)

    Q = QuantumCircuit(n_tot, name='Q')
    Q.z(n)                              # S_chi
    Q.compose(A.inverse(), inplace=True) # A†
    Q.compose(S0, inplace=True)          # S0
    Q.compose(A, inplace=True)           # A
    return to_basis(Q)

# Build circuits
A_FULL = build_A_full(PROBS, N_QUBITS)
Q_OP   = build_Q(A_FULL, N_QUBITS)

# State-only (for TV distance)
qc_state = QuantumCircuit(N_QUBITS, N_QUBITS, name='state_only')
qc_state.compose(build_dist_prep(PROBS, N_QUBITS), inplace=True)
qc_state.measure(range(N_QUBITS), range(N_QUBITS))

# Grover circuits m=0,1,2
GROVER = {}
for m in [0, 1, 2]:
    qc = QuantumCircuit(N_QUBITS+1, name=f'QAE_m{m}')
    qc.compose(A_FULL, inplace=True)
    for _ in range(m): qc.compose(Q_OP, inplace=True)
    cr = ClassicalRegister(1, 'anc')
    qc.add_register(cr); qc.measure(N_QUBITS, 0)
    GROVER[m] = qc

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

cnt = sim.run(qc_state, shots=SHOTS_STATE).result().get_counts()
tot = sum(cnt.values())
p_aer_s = np.zeros(NUM_LEVELS)
for bs, c in cnt.items():
    idx = int(bs[::-1], 2) if len(bs) == N_QUBITS else int(bs, 2)
    if idx < NUM_LEVELS: p_aer_s[idx] = c / tot
tv = 0.5 * np.sum(np.abs(p_aer_s - PROBS))
log.info(f"State prep TV: {tv:.4f}  [{'PASS' if tv<0.10 else 'FAIL'}]")

aer_g = {}; mono_v = []
for m, qc in GROVER.items():
    cnt = sim.run(qc, shots=SHOTS_GROVER).result().get_counts()
    tot = sum(cnt.values())
    p   = cnt.get('1', 0) / tot
    th  = P_THEORY[m]
    err = abs(p - th)
    tol = 4 * math.sqrt(th*(1-th)/SHOTS_GROVER)
    ok  = err < tol
    mono_v.append(p)
    log.info(f"Grover m={m}: P_aer={p*100:.2f}%  P_th={th*100:.2f}%  "
             f"err={err*100:.2f}%  [{'PASS' if ok else 'FAIL'}]")
    aer_g[m] = {'p_aer': float(p), 'p_theory': float(th),
                'error': float(err), 'pass': bool(ok)}

mono   = mono_v[0] < mono_v[1] < mono_v[2]
g_range = mono_v[2] - mono_v[0]
log.info(f"Monotone: {mono}  Range: {g_range:.4f}  (target > 0.20)")

phase0 = {'tv': float(tv), 'state_prep_pass': bool(tv < 0.10),
          'grover': aer_g, 'monotone': bool(mono),
          'range': float(g_range), 'pass': bool(mono and g_range > 0.20)}
(HW_OUT / 'phase0_emerald_latest.json').write_text(
    json.dumps({'phase0': phase0, 'ts': datetime.now().isoformat()},
               indent=2, default=_json_default))

if not phase0['pass']:
    log.error("Phase 0 FAILED — do not proceed to hardware.")
    sys.exit(1)
log.ok("Phase 0 PASSED.")

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

# Connect — same pattern as user's example
IQM_ENDPOINT = 'https://cocos.resonance.meetiqm.com/emerald'
os.environ['IQM_TOKEN'] = IQM_TOKEN
provider = IQMProvider(IQM_ENDPOINT)
backend  = provider.get_backend()
log.info(f"Backend: {backend.name}  {backend.num_qubits} qubits")
log.info(f"Native gates: {backend.operation_names}")

# Transpile for Emerald — same pattern as user's example
log.info("Transpiling for IQM Emerald (square lattice)...")
qc_transpiled_state = transpile(qc_state, backend=backend, optimization_level=1)
qc_transpiled_grover = {m: transpile(qc, backend=backend, optimization_level=1)
                        for m, qc in GROVER.items()}

log.info(f"state_only: depth={qc_transpiled_state.depth()}")
for m, isa in qc_transpiled_grover.items():
    cz = isa.count_ops().get('cz', isa.count_ops().get('cx', 0))
    log.info(f"Grover m={m}: depth={isa.depth()}  CZ={cz}")

max_depth = max(isa.depth() for isa in qc_transpiled_grover.values())
if max_depth > 200:
    log.warn(f"Max depth {max_depth} > 200 — results may be noise-dominated.")

# Job 1: state preparation
log.info(f"Job 1 — state prep ({SHOTS_STATE} shots)...")
job = backend.run(qc_transpiled_state, shots=SHOTS_STATE, use_timeslot=False)
log.info(f"  job_id: {job.job_id()}")
result = job.result()
cnt1   = result.get_counts()
tot1   = sum(cnt1.values())
p_hw_s = np.zeros(NUM_LEVELS)
for bs, c in cnt1.items():
    idx = int(bs[::-1], 2) if len(bs) == N_QUBITS else int(bs, 2)
    if idx < NUM_LEVELS: p_hw_s[idx] = c / tot1
tv_hw = 0.5 * np.sum(np.abs(p_hw_s - PROBS))
sp_v  = 'PASS' if tv_hw < 0.10 else ('MARGINAL' if tv_hw < 0.15 else 'FAIL')
log.info(f"  TV={tv_hw:.4f}  [{sp_v}]")

phase1 = {
    'device': 'IQM Emerald (Crystal 54)',
    'state_prep': {
        'job_id': str(job.job_id()), 'shots': SHOTS_STATE,
        'tv': float(tv_hw), 'verdict': sp_v,
        'p_ideal': PROBS.tolist(), 'p_hw': p_hw_s.tolist()
    },
    'grover': {}
}

# Jobs 2-4: Grover m=0,1,2 — same pattern as user's example
p_hw_v = []
for m in [0, 1, 2]:
    log.info(f"Job {m+2} — Grover m={m} ({SHOTS_GROVER} shots)...")
    job    = backend.run(qc_transpiled_grover[m], shots=SHOTS_GROVER,
                         use_timeslot=False)
    log.info(f"  job_id: {job.job_id()}")
    result = job.result()
    cnt    = result.get_counts()
    tot    = sum(cnt.values())
    p_hw   = cnt.get('1', 0) / tot
    p_th   = P_THEORY[m]
    err    = abs(p_hw - p_th)
    v      = 'PASS' if err < 0.15 else 'FAIL'
    p_hw_v.append(p_hw)
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
    'range':    float(range_hw),
    'advance_phase2': bool(mono_hw and range_hw > 0.20),
    'advance_phase3': bool(mono_hw),
})

log.sep("PHASE 1 RESULTS")
log.info(f"State prep TV : {tv_hw:.4f}  [{sp_v}]")
for m in [0,1,2]:
    g = phase1['grover'][f'm{m}']
    log.info(f"Grover m={m}   : {g['p_hw']*100:.2f}%  "
             f"(theory {g['p_theory']*100:.2f}%)  [{g['verdict']}]")
log.info(f"Monotone      : {mono_hw}  |  Range: {range_hw:.4f}")
log.info(f"Advance Ph2   : {phase1['advance_phase2']}")
log.info(f"Advance Ph3   : {phase1['advance_phase3']}")

# Save results
out = {
    'metadata': {
        'device': 'IQM Emerald (Crystal 54, 54 qubits)',
        'endpoint': 'https://cocos.resonance.meetiqm.com/emerald',
        'oracle': 'complementary (1 MCX, ~6x fewer CZ vs original)',
        'a_ideal': A_IDEAL, 'theta': THETA,
        'ts': datetime.now().isoformat()
    },
    'phase0': phase0,
    'phase1': phase1
}
out_path = HW_OUT / 'phase1_emerald_results_latest.json'
out_path.write_text(json.dumps(out, indent=2, default=_json_default))
log.ok(f"Results saved: {out_path}")
