"""
hw_wave1_zne_iqae_v2.py — Oleada 1 v2: ZNE + IQAE (bugs fixed)
================================================================
Fix 1 — ZNE: use optimization_level=0 for folded circuits to prevent
         transpiler from canceling U†U pairs.
Fix 2 — IQAE: correct aliasing resolution for m>1 rounds using
         (pi - arcsin) branch when (2m+1)*theta > pi/2.
Fix 3 — IQAE: CI stored on 'a' (amplitude), not on 'p_hw'.

Usage:
    $env:IQM_TOKEN = "your_token"
    python scripts/hw_wave1_zne_iqae_v2.py
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
LOG_FILE     = HW_OUT / 'wave1_v2_latest.log'

def _jd(o):
    if isinstance(o, np.bool_):    return bool(o)
    if isinstance(o, np.integer):  return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray):  return o.tolist()
    raise TypeError(f"Not serializable: {type(o).__name__}")

class Logger:
    def __init__(self, p):
        self._f = open(p, 'w')
    def _w(self, tag, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {tag} {msg}"
        print(line); self._f.write(line+'\n'); self._f.flush()
    def info(self, m):  self._w('INFO ', m)
    def ok(self, m):    self._w('OK   ', m)
    def warn(self, m):  self._w('WARN ', m)
    def error(self, m): self._w('ERROR', m)
    def sep(self, t=''): self._w('=====', f"{'='*12} {t} {'='*12}" if t else '='*40)

log = Logger(LOG_FILE)
log.sep("OLEADA 1 v2 — ZNE (fixed) + IQAE (fixed)")

IQM_TOKEN = os.environ.get('IQM_TOKEN', '')
if not IQM_TOKEN:
    log.error("IQM_TOKEN not set."); sys.exit(1)
log.ok("IQM_TOKEN found.")

ref_path = HW_OUT / 'phase1_direct_results_latest.json'
if not ref_path.exists():
    log.error("Run hw_phase1_direct.py first."); sys.exit(1)
ref      = json.loads(ref_path.read_text())
A_IDEAL  = ref['metadata']['a_ideal']
THETA    = ref['metadata']['theta']
THETA_RY = ref['metadata']['theta_ry']
P_THEORY = {m: math.sin((2*m+1)*THETA)**2 for m in range(8)}
log.ok(f"a_ideal={A_IDEAL:.6f}  theta={THETA:.6f}")
for m in range(5):
    log.info(f"  P_theory[m={m}] = {P_THEORY[m]*100:.3f}%  "
             f"(2m+1)*theta={(2*m+1)*THETA:.4f} rad "
             f"{'> pi/2 ALIASED' if (2*m+1)*THETA > math.pi/2 else 'OK'}")

SHOTS_ZNE  = 4096
SHOTS_IQAE = 8192
ZNE_SCALES = [1.0, 3.0, 5.0]

from qiskit import QuantumCircuit, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

BASIS = ['cx','u','p','x','h','rx','ry','rz','sx','id']

def to_basis(qc, opt=1):
    return transpile(qc, basis_gates=BASIS, optimization_level=opt)

def build_A():
    qc = QuantumCircuit(2, name='A')
    qc.ry(THETA_RY, 0); qc.cx(0, 1)
    return to_basis(qc)

def build_Q(A):
    S0 = QuantumCircuit(2, name='S0')
    S0.x(0); S0.x(1); S0.cz(0,1); S0.x(0); S0.x(1)
    Q  = QuantumCircuit(2, name='Q')
    Q.z(1)
    Q.compose(A.inverse(), inplace=True)
    Q.compose(to_basis(S0), inplace=True)
    Q.compose(A, inplace=True)
    return to_basis(Q)

def build_base(A, Q, m):
    qc = QuantumCircuit(2, name=f'base_m{m}')
    qc.compose(A, inplace=True)
    for _ in range(m): qc.compose(Q, inplace=True)
    return qc

def with_meas(base):
    qc = base.copy()
    cr = ClassicalRegister(1,'anc')
    qc.add_register(cr); qc.measure(1, 0)
    return qc

def fold(base, scale):
    """
    Global unitary folding U -> U(U†U)^k.
    IMPORTANT: returns circuit NOT transpiled — preserve the U†U structure
    so the transpiler cannot cancel them at optimization_level=0.
    """
    if abs(scale-1.0) < 1e-9: return base.copy()
    k = int(round((scale-1)/2))
    f = base.copy()
    for _ in range(k):
        f.compose(base.inverse(), inplace=True)
        f.compose(base,           inplace=True)
    return f  # NOT transpiled here intentionally

A_circ = build_A()
Q_circ = build_Q(A_circ)

# Aer sanity
sim = AerSimulator()
log.sep("CIRCUITS + AER VALIDATION")
log.info(f"A: depth={A_circ.depth()}  Q: depth={Q_circ.depth()}")
for m in [0,1,2,3]:
    base = build_base(A_circ, Q_circ, m)
    qc   = with_meas(base)
    cnt  = sim.run(qc, shots=4096).result().get_counts()
    tot  = sum(cnt.values())
    p    = cnt.get('1',0)/tot
    log.info(f"  m={m}: {p*100:.2f}%  theory={P_THEORY[m]*100:.2f}%  "
             f"err={abs(p-P_THEORY[m])*100:.2f}%")

try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    log.error("pip install 'iqm-client[qiskit]'"); sys.exit(1)

os.environ['IQM_TOKEN'] = IQM_TOKEN
backend = IQMProvider('https://cocos.resonance.meetiqm.com/emerald').get_backend()
log.ok(f"Connected: {backend.name}  {backend.num_qubits}q")

def hw_run(qc_meas, shots, label, opt_level=1):
    """
    opt_level=1 for normal circuits (optimization OK).
    opt_level=0 for folded circuits (preserve U†U structure for ZNE).
    """
    isa = transpile(qc_meas, backend=backend, optimization_level=opt_level)
    log.info(f"  [{label}] depth={isa.depth()} "
             f"CZ={isa.count_ops().get('cz',0)} shots={shots} opt={opt_level}")
    job = backend.run(isa, shots=shots, use_timeslot=False)
    log.info(f"  [{label}] job_id={job.job_id()}")
    cnt = job.result().get_counts()
    tot = sum(cnt.values())
    p   = cnt.get('1',0)/tot
    log.info(f"  [{label}] P_hw={p*100:.3f}%")
    return float(p), str(job.job_id()), isa.depth()

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK A — ZNE on m=1 (fixed: opt_level=0 for scale>1)
# ─────────────────────────────────────────────────────────────────────────────
log.sep("BLOCK A — ZNE m=1 (opt_level=0 for folded)")
log.info("Fix: optimization_level=0 prevents transpiler from canceling U†U.")

base_m1  = build_base(A_circ, Q_circ, 1)
p_scales = {}
jobs_zne = {}

for scale in ZNE_SCALES:
    label  = f"ZNE_m1_s{scale}"
    folded = fold(base_m1, scale)
    # KEY FIX: to_basis with opt=0, then hw_run with opt=0
    qc_f   = with_meas(to_basis(folded, opt=0))
    opt    = 0 if scale > 1.0 else 1
    p_hw, jid, depth = hw_run(qc_f, SHOTS_ZNE, label, opt_level=opt)
    p_scales[scale]  = p_hw
    jobs_zne[scale]  = {'job_id': jid, 'depth': depth,
                        'p_hw': p_hw, 'shots': SHOTS_ZNE,
                        'opt_level': opt}
    log.info(f"  scale={scale}: depth={depth}  (should increase with scale)")

# Verify folding worked: depths should grow
d1 = jobs_zne[1.0]['depth']
d3 = jobs_zne[3.0]['depth']
d5 = jobs_zne[5.0]['depth']
folding_ok = d1 < d3 < d5
log.info(f"Depth progression: {d1} -> {d3} -> {d5}  "
         f"[{'OK — folding preserved' if folding_ok else 'FAIL — transpiler still optimizing'}]")

p1, p3, p5 = p_scales[1.0], p_scales[3.0], p_scales[5.0]
lams   = np.array([1.0, 3.0, 5.0])
vals   = np.array([p1, p3, p5])
coeffs = np.polyfit(lams, vals, deg=2)
p_zne  = float(np.polyval(coeffs, 0.0))
p_lin  = float(1.5*p1 - 0.5*p3)
p_th1  = P_THEORY[1]
err_r  = abs(p1    - p_th1)
err_z  = abs(p_zne - p_th1)
impr   = (err_r - err_z) / err_r if err_r > 1e-9 else 0.0

log.sep("BLOCK A RESULTS")
log.info(f"P_theory(m=1)      = {p_th1*100:.3f}%")
log.info(f"P_raw  (scale=1.0) = {p1*100:.3f}%  err={err_r*100:.3f}%")
log.info(f"P_zne  (3-pt poly) = {p_zne*100:.3f}%  err={err_z*100:.3f}%")
log.info(f"P_lin  (2-pt)      = {p_lin*100:.3f}%")
log.info(f"ZNE improvement    = {impr*100:.1f}%  "
         f"[{'EFFECTIVE' if impr>0.20 else 'MARGINAL'}]")
if not folding_ok:
    log.warn("Folding may not be preserved — ZNE result unreliable.")

block_A = {
    'scale_results': {str(k): v for k,v in jobs_zne.items()},
    'depth_progression': {'d1':d1,'d3':d3,'d5':d5,'folding_ok':folding_ok},
    'p_extrap_poly3': p_zne,
    'p_extrap_linear': p_lin,
    'p_theory': p_th1,
    'err_raw': err_r, 'err_zne': err_z,
    'improvement': impr,
    'effective': bool(impr > 0.20),
    'fix_applied': 'optimization_level=0 for folded circuits',
}
(HW_OUT/'wave1_v2_blockA_latest.json').write_text(
    json.dumps({'block_A':block_A,'ts':datetime.now().isoformat()},
               indent=2, default=_jd))
log.ok("Block A saved.")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK C — IQAE adaptive (fixed: alias resolution + CI on 'a')
# ─────────────────────────────────────────────────────────────────────────────
log.sep("BLOCK C — IQAE (alias-safe + CI on amplitude a)")

def resolve_theta(p_hw, m):
    """
    Recover theta from P = sin²((2m+1)*theta).
    Handles aliasing when (2m+1)*theta > pi/2.
    Returns best estimate of theta using prior theta ~ THETA.
    """
    phi = math.asin(math.sqrt(max(0.0, min(1.0, p_hw))))  # in [0, pi/2]
    # Two branches: phi = (2m+1)*theta  or  phi = pi - (2m+1)*theta
    k   = 2*m + 1
    t1  = phi / k
    t2  = (math.pi - phi) / k
    # Choose branch closest to prior THETA
    theta_est = t1 if abs(t1 - THETA) < abs(t2 - THETA) else t2
    a_est     = math.sin(theta_est) ** 2
    return float(theta_est), float(a_est)

def wilson_ci_a(p_hw, n, m, alpha=0.05):
    """
    Wilson CI on p_hw, then propagate to CI on a via Jacobian.
    For m=0: a = p_hw directly.
    For m>0: a = sin²(arcsin(sqrt(p_hw))/(2m+1))² — propagate numerically.
    """
    z  = sp_stats.norm.ppf(1-alpha/2)
    lo_p = (p_hw+z**2/(2*n)-z*math.sqrt(p_hw*(1-p_hw)/n+z**2/(4*n**2)))/(1+z**2/n)
    hi_p = (p_hw+z**2/(2*n)+z*math.sqrt(p_hw*(1-p_hw)/n+z**2/(4*n**2)))/(1+z**2/n)
    lo_p = max(0.0, lo_p); hi_p = min(1.0, hi_p)
    if m == 0:
        return lo_p, hi_p, float(hi_p-lo_p)
    # Propagate via delta method
    _, a_lo = resolve_theta(lo_p, m)
    _, a_hi = resolve_theta(hi_p, m)
    # Order correctly (a may not be monotone with p for aliased rounds)
    a_lo_f, a_hi_f = min(a_lo,a_hi), max(a_lo,a_hi)
    return float(a_lo_f), float(a_hi_f), float(a_hi_f-a_lo_f)

EPSILON    = 0.05
log.info(f"Target epsilon={EPSILON} on amplitude a")
log.info(f"Non-aliased rounds: m=0 only (7*theta={7*THETA:.3f} rad > pi/2 for m=3)")
log.info(f"Alias-aware inversion used for m=1,3")

iqae_rounds = []

for m_round, m_val in [(1, 0), (2, 1), (3, 3)]:
    label = f"IQAE_R{m_round}_m{m_val}"
    base  = build_base(A_circ, Q_circ, m_val)
    p_hw, jid, depth = hw_run(with_meas(base), SHOTS_IQAE, label)
    theta_est, a_est = resolve_theta(p_hw, m_val)
    a_lo, a_hi, a_w  = wilson_ci_a(p_hw, SHOTS_IQAE, m_val)
    captured = a_lo <= A_IDEAL <= a_hi
    aliased  = (2*m_val+1)*THETA > math.pi/2
    log.info(f"  m={m_val}: p_hw={p_hw*100:.3f}%  "
             f"theta_est={theta_est:.4f}  a_est={a_est*100:.3f}%")
    log.info(f"    CI_a=[{a_lo*100:.3f}%,{a_hi*100:.3f}%]  "
             f"width={a_w*100:.3f}%  captures_a={captured}  aliased={aliased}")
    iqae_rounds.append({
        'round': m_round, 'm': m_val, 'p_hw': p_hw,
        'theta_est': theta_est, 'a_est': a_est,
        'ci_a_lo': a_lo, 'ci_a_hi': a_hi, 'ci_a_width': a_w,
        'ci_captures_a_ideal': captured,
        'aliased': aliased, 'job_id': jid
    })

# Best estimate from non-aliased round (m=0) — most reliable
r0     = iqae_rounds[0]
a_best = r0['a_est']
ci_lo  = r0['ci_a_lo']
ci_hi  = r0['ci_a_hi']
ci_w   = r0['ci_a_width']
ok     = r0['ci_captures_a_ideal']

# Weighted combination of all rounds
weights   = [1/max(r['ci_a_width'],1e-6) for r in iqae_rounds]
a_weighted = sum(r['a_est']*w for r,w in zip(iqae_rounds,weights)) / sum(weights)

log.sep("BLOCK C RESULTS")
log.info(f"a_ideal                = {A_IDEAL*100:.3f}%")
for r in iqae_rounds:
    log.info(f"  m={r['m']}: a_est={r['a_est']*100:.3f}%  "
             f"CI=[{r['ci_a_lo']*100:.3f}%,{r['ci_a_hi']*100:.3f}%]  "
             f"captures={r['ci_captures_a_ideal']}  aliased={r['aliased']}")
log.info(f"a_best (m=0, no alias) = {a_best*100:.3f}%")
log.info(f"a_weighted (all rounds)= {a_weighted*100:.3f}%")
log.info(f"Best CI on a (m=0)     = [{ci_lo*100:.3f}%, {ci_hi*100:.3f}%]  "
         f"width={ci_w*100:.3f}%")
log.info(f"CI captures a_ideal    = {ok}")
log.info(f"Point error (m=0)      = {abs(a_best-A_IDEAL)/A_IDEAL*100:.1f}%")

block_C = {
    'epsilon_target': EPSILON,
    'rounds': iqae_rounds,
    'a_ideal': A_IDEAL,
    'a_best_m0': float(a_best),
    'a_weighted': float(a_weighted),
    'best_ci_a': {'lo': ci_lo, 'hi': ci_hi, 'width': ci_w},
    'ci_captures_a_ideal': bool(ok),
    'point_error_pct': float(abs(a_best-A_IDEAL)/A_IDEAL*100),
    'fix_applied': 'alias-safe theta inversion + CI on amplitude a',
}

# ── Summary ───────────────────────────────────────────────────────────────────
log.sep("WAVE 1 v2 SUMMARY")
log.info(f"ZNE folding preserved  : {folding_ok}")
log.info(f"ZNE improvement        : {impr*100:.1f}%  "
         f"[{'EFFECTIVE' if impr>0.20 else 'MARGINAL'}]")
log.info(f"IQAE a_best (m=0)      : {a_best*100:.3f}%  (ideal={A_IDEAL*100:.3f}%)")
log.info(f"IQAE CI captures       : {ok}")
log.info(f"IQAE point error       : {abs(a_best-A_IDEAL)/A_IDEAL*100:.1f}%")

out = {
    'metadata': {
        'device': 'IQM Emerald (Crystal 54)',
        'encoding': 'direct Ry + CX',
        'version': 'v2 — ZNE opt0 fix + IQAE alias fix',
        'a_ideal': A_IDEAL, 'theta': THETA,
        'ts': datetime.now().isoformat()
    },
    'block_A_zne': block_A,
    'block_C_iqae': block_C,
}
p = HW_OUT / 'wave1_v2_results_latest.json'
p.write_text(json.dumps(out, indent=2, default=_jd))
log.ok(f"Results: {p}")
