"""
hw_iqae_garnet.py — IQAE on IQM Garnet (3 jobs)
=================================================
Rationale: Garnet m=1 gave 39.77% (error 0.27pp) — best result of entire
campaign. IQAE on Garnet may produce tighter CI than Emerald (6.0% error).

Jobs: m=0, m=1, m=3 on Garnet — same direct Ry encoding as all prior runs.
Cost estimate: ~3 jobs x 4 cr = ~12 cr.

Usage:
    $env:IQM_TOKEN = "your_token"
    python scripts/hw_iqae_garnet.py
"""

import os, sys, json, math, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'scripts' else SCRIPT_DIR
HW_OUT       = PROJECT_ROOT / 'results' / 'hardware_validation'
HW_OUT.mkdir(parents=True, exist_ok=True)
LOG_FILE     = HW_OUT / 'iqae_garnet_latest.log'

def _jd(o):
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"Not serializable: {type(o).__name__}")

class Logger:
    def __init__(self, p):
        self._f = open(p, 'w', encoding='utf-8')
    def _w(self, tag, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {msg}"
        print(line); self._f.write(line+'\n'); self._f.flush()
    def info(self, m):  self._w('INFO ', m)
    def ok(self, m):    self._w('OK   ', m)
    def warn(self, m):  self._w('WARN ', m)
    def error(self, m): self._w('ERROR', m)
    def sep(self, t=''): self._w('=====', f"{'='*12} {t} {'='*12}" if t else '='*40)

log = Logger(LOG_FILE)
log.sep("IQAE ON IQM GARNET — direct Ry encoding")

IQM_TOKEN = os.environ.get('IQM_TOKEN', '')
if not IQM_TOKEN:
    log.error("IQM_TOKEN not set."); sys.exit(1)
log.ok("IQM_TOKEN found.")

# ── Reference ─────────────────────────────────────────────────────────────────
ref_path = HW_OUT / 'phase1_direct_results_latest.json'
if not ref_path.exists():
    log.error("phase1_direct_results_latest.json not found."); sys.exit(1)
ref      = json.loads(ref_path.read_text())
A_IDEAL  = ref['metadata']['a_ideal']
THETA    = ref['metadata']['theta']
THETA_RY = ref['metadata']['theta_ry']
P_THEORY = {m: math.sin((2*m+1)*THETA)**2 for m in range(8)}

log.ok(f"a_ideal={A_IDEAL:.6f}  theta={THETA:.6f}")
log.info("Motivation: Garnet m=1 gave 39.77% (error 0.27pp vs theory 39.50%)")
log.info("Emerald IQAE gave CI=[4.88%,5.86%], error 6.0% — Garnet may be tighter")

# Load Emerald IQAE results for comparison
emerald_iqae = {}
try:
    d = json.loads((HW_OUT/'wave1_v2_results_latest.json').read_text())
    for r in d['block_C_iqae']['rounds']:
        emerald_iqae[r['m']] = r
    log.info("Loaded Emerald IQAE results for comparison:")
    for m, r in emerald_iqae.items():
        log.info(f"  Emerald m={m}: a_est={r['a_est']*100:.3f}%  "
                 f"CI=[{r['ci_a_lo']*100:.3f}%,{r['ci_a_hi']*100:.3f}%]")
except Exception:
    log.warn("Could not load Emerald IQAE results.")

SHOTS_IQAE = 8192

# ── Circuits ──────────────────────────────────────────────────────────────────
from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

BASIS = ['cx','u','p','x','h','rx','ry','rz','sx','id']
def to_basis(qc): return transpile(qc, basis_gates=BASIS, optimization_level=1)

def build_A():
    qc = QuantumCircuit(2, name='A')
    qc.ry(THETA_RY, 0); qc.cx(0, 1)
    return to_basis(qc)

def build_Q(A):
    S0 = QuantumCircuit(2, name='S0')
    S0.x(0); S0.x(1); S0.cz(0,1); S0.x(0); S0.x(1)
    Q = QuantumCircuit(2, name='Q')
    Q.z(1)
    Q.compose(A.inverse(), inplace=True)
    Q.compose(to_basis(S0), inplace=True)
    Q.compose(A, inplace=True)
    return to_basis(Q)

def build_grover_meas(A, Q, m):
    qc = QuantumCircuit(2, name=f'QAE_m{m}')
    qc.compose(A, inplace=True)
    for _ in range(m): qc.compose(Q, inplace=True)
    cr = ClassicalRegister(1, 'anc')
    qc.add_register(cr); qc.measure(1, 0)
    return qc

def resolve_theta(p_hw, m):
    """Alias-safe theta recovery using prior THETA as reference."""
    phi = math.asin(math.sqrt(max(0.0, min(1.0, p_hw))))
    k   = 2*m + 1
    t1  = phi / k
    t2  = (math.pi - phi) / k
    theta_est = t1 if abs(t1-THETA) < abs(t2-THETA) else t2
    return float(theta_est), float(math.sin(theta_est)**2)

def wilson_ci_a(p_hw, n, m, alpha=0.05):
    """Wilson CI on p_hw propagated to CI on amplitude a."""
    z    = sp_stats.norm.ppf(1-alpha/2)
    lo_p = (p_hw+z**2/(2*n)-z*math.sqrt(p_hw*(1-p_hw)/n+z**2/(4*n**2)))/(1+z**2/n)
    hi_p = (p_hw+z**2/(2*n)+z*math.sqrt(p_hw*(1-p_hw)/n+z**2/(4*n**2)))/(1+z**2/n)
    lo_p = max(0.0,lo_p); hi_p = min(1.0,hi_p)
    if m == 0:
        return float(lo_p), float(hi_p), float(hi_p-lo_p)
    _, a_lo = resolve_theta(lo_p, m)
    _, a_hi = resolve_theta(hi_p, m)
    a_lo_f, a_hi_f = min(a_lo,a_hi), max(a_lo,a_hi)
    return float(a_lo_f), float(a_hi_f), float(a_hi_f-a_lo_f)

A_circ = build_A()
Q_circ = build_Q(A_circ)

# Aer validation
sim = AerSimulator()
log.sep("AER VALIDATION")
for m in [0, 1, 3]:
    qc  = build_grover_meas(A_circ, Q_circ, m)
    cnt = sim.run(qc, shots=8192).result().get_counts()
    p   = cnt.get('1',0)/sum(cnt.values())
    log.info(f"  m={m}: {p*100:.2f}%  theory={P_THEORY[m]*100:.2f}%")

# ── Connect Garnet ────────────────────────────────────────────────────────────
try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    log.error("pip install 'iqm-client[qiskit]'"); sys.exit(1)

os.environ['IQM_TOKEN'] = IQM_TOKEN
backend = IQMProvider('https://cocos.resonance.meetiqm.com/garnet').get_backend()
log.ok(f"Connected: {backend.name}  {backend.num_qubits}q")

def hw_run(qc_meas, shots, label):
    isa = transpile(qc_meas, backend=backend, optimization_level=1)
    log.info(f"  [{label}] depth={isa.depth()} "
             f"CZ={isa.count_ops().get('cz',0)} shots={shots}")
    job = backend.run(isa, shots=shots, use_timeslot=False)
    log.info(f"  [{label}] job_id={job.job_id()}")
    cnt = job.result().get_counts()
    p   = cnt.get('1',0)/sum(cnt.values())
    log.info(f"  [{label}] P_hw={p*100:.3f}%")
    return float(p), str(job.job_id()), isa.depth()

# ── IQAE on Garnet ────────────────────────────────────────────────────────────
log.sep("IQAE ON GARNET (m=0,1,3 — alias-safe)")

rounds = []
for m_round, m_val in [(1,0),(2,1),(3,3)]:
    label    = f"IQAE_GAR_R{m_round}_m{m_val}"
    qc       = build_grover_meas(A_circ, Q_circ, m_val)
    p_hw, jid, depth = hw_run(qc, SHOTS_IQAE, label)
    t_est, a_est = resolve_theta(p_hw, m_val)
    a_lo, a_hi, a_w = wilson_ci_a(p_hw, SHOTS_IQAE, m_val)
    captured = a_lo <= A_IDEAL <= a_hi
    aliased  = (2*m_val+1)*THETA > math.pi/2
    log.info(f"  m={m_val}: p_hw={p_hw*100:.3f}%  a_est={a_est*100:.3f}%")
    log.info(f"    CI_a=[{a_lo*100:.3f}%,{a_hi*100:.3f}%]  "
             f"width={a_w*100:.3f}%  captures={captured}  aliased={aliased}")
    rounds.append({
        'round': m_round, 'm': m_val, 'p_hw': p_hw,
        'theta_est': t_est, 'a_est': a_est,
        'ci_a_lo': a_lo, 'ci_a_hi': a_hi, 'ci_a_width': a_w,
        'ci_captures_a_ideal': captured,
        'aliased': aliased, 'job_id': jid, 'depth_iqm': depth
    })

# Best estimate from m=0 (no aliasing)
r0     = rounds[0]
a_best = r0['a_est']
ci_lo  = r0['ci_a_lo']
ci_hi  = r0['ci_a_hi']
ci_w   = r0['ci_a_width']
ok     = r0['ci_captures_a_ideal']

log.sep("RESULTS — Garnet vs Emerald IQAE")
log.info(f"a_ideal = {A_IDEAL*100:.3f}%")
log.info("")
log.info("Garnet IQAE:")
for r in rounds:
    log.info(f"  m={r['m']}: a_est={r['a_est']*100:.3f}%  "
             f"CI=[{r['ci_a_lo']*100:.3f}%,{r['ci_a_hi']*100:.3f}%]  "
             f"width={r['ci_a_width']*100:.3f}%  "
             f"captures={r['ci_captures_a_ideal']}")

log.info("")
log.info("Emerald IQAE (reference):")
for m, er in emerald_iqae.items():
    log.info(f"  m={m}: a_est={er['a_est']*100:.3f}%  "
             f"CI=[{er['ci_a_lo']*100:.3f}%,{er['ci_a_hi']*100:.3f}%]  "
             f"width={er['ci_a_width']*100:.3f}%  "
             f"captures={er['ci_captures_a_ideal']}")

log.info("")
log.info("Cross-device comparison (m=0, direct estimate):")
e_r0  = emerald_iqae.get(0, {})
e_err = abs(e_r0.get('a_est',0) - A_IDEAL)/A_IDEAL*100 if e_r0 else float('nan')
g_err = abs(a_best - A_IDEAL)/A_IDEAL*100
log.info(f"  Garnet : a_best={a_best*100:.3f}%  point_error={g_err:.1f}%  "
         f"CI_width={ci_w*100:.3f}%  captures={ok}")
log.info(f"  Emerald: a_best={e_r0.get('a_est',0)*100:.3f}%  "
         f"point_error={e_err:.1f}%  "
         f"CI_width={e_r0.get('ci_a_width',0)*100:.3f}%  "
         f"captures={e_r0.get('ci_captures_a_ideal','?')}")

# Improvement
if e_r0:
    ci_improvement = (e_r0.get('ci_a_width',1) - ci_w) / e_r0.get('ci_a_width',1)
    err_improvement = (e_err - g_err) / e_err if e_err > 0 else 0
    log.info(f"  CI width improvement  : {ci_improvement*100:+.1f}% "
             f"({'Garnet better' if ci_improvement>0 else 'Emerald better'})")
    log.info(f"  Point error improvement: {err_improvement*100:+.1f}% "
             f"({'Garnet better' if err_improvement>0 else 'Emerald better'})")

out = {
    'metadata': {
        'device': 'IQM Garnet (Star 20)',
        'encoding': 'direct Ry + CX',
        'motivation': 'Garnet m=1 gave 0.27pp error vs Emerald 6.01pp — test IQAE precision',
        'a_ideal': A_IDEAL, 'theta': THETA,
        'ts': datetime.now().isoformat()
    },
    'garnet_iqae': {
        'rounds': rounds,
        'a_best_m0': float(a_best),
        'best_ci_a': {'lo': ci_lo,'hi': ci_hi,'width': ci_w},
        'ci_captures_a_ideal': bool(ok),
        'point_error_pct': float(g_err),
    },
    'emerald_iqae_reference': {
        r['m']: {k:v for k,v in r.items()} for r in emerald_iqae.values()
    } if emerald_iqae else {},
}
p = HW_OUT / 'iqae_garnet_results_latest.json'
p.write_text(json.dumps(out, indent=2, default=_jd))
log.ok(f"Results: {p}")
