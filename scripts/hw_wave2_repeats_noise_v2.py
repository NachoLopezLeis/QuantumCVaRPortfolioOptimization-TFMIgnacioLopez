"""
hw_wave2_repeats_noise_v2.py — Oleada 2 revisada (budget-aware)
================================================================
Cambios vs v1:
  - Block F: 10 → 5 repeticiones (ya tenemos 6 medidas previas de m=1)
  - Block B ZNE: solo m=0, scales [1,3] — scale=5 satura en decoherencia
  - ZNE m=2,3 cancelado — depth post-fold >250, resultado garantizado ruido

Cost estimate: ~5 jobs × 4 cr = ~20 cr

Usage:
    $env:IQM_TOKEN = "your_token"
    python scripts/hw_wave2_repeats_noise_v2.py
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
LOG_FILE     = HW_OUT / 'wave2_v2_latest.log'

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
log.sep("OLEADA 2 v2 — REPEATS + ZNE m=0")
log.info("Budget-aware: 5 repeats + ZNE m=0 only (scales 1,3)")

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

# Load existing m=1 measurements to pool with new repeats
existing_m1 = []
for fname in ['phase1_direct_results_latest.json',
              'wave1_results_latest.json',
              'wave1_v2_results_latest.json']:
    fpath = HW_OUT / fname
    if not fpath.exists(): continue
    try:
        d = json.loads(fpath.read_text())
        # phase1_direct
        if 'phase1' in d:
            g = d['phase1'].get('grover',{}).get('m1',{})
            if g: existing_m1.append({'p_hw': g['p_hw'],
                                      'source': fname, 'shots': g.get('shots',2048)})
        # wave1 block A scale=1
        if 'block_A_zne' in d:
            sc = d['block_A_zne']['scale_results'].get('1.0',{})
            if sc: existing_m1.append({'p_hw': sc['p_hw'],
                                       'source': fname+'_zne_s1', 'shots': sc.get('shots',4096)})
    except Exception:
        pass
log.info(f"Existing m=1 measurements loaded: {len(existing_m1)}")
for e in existing_m1:
    log.info(f"  {e['source']}: {e['p_hw']*100:.2f}%")

SHOTS_REP  = 2048
SHOTS_ZNE  = 4096
N_REPEATS  = 5    # reduced from 10 — already have existing measurements
ZNE_SCALES = [1.0, 3.0]  # scale=5 saturates decoherence — skip

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
    S0.x(0); S0.x(1); S0.cz(0, 1); S0.x(0); S0.x(1)
    Q = QuantumCircuit(2, name='Q')
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
    cr = ClassicalRegister(1, 'anc')
    qc.add_register(cr); qc.measure(1, 0)
    return qc

def fold(base, scale):
    if abs(scale-1.0) < 1e-9: return base.copy()
    k = int(round((scale-1)/2))
    f = base.copy()
    for _ in range(k):
        f.compose(base.inverse(), inplace=True)
        f.compose(base,           inplace=True)
    return f

A_circ = build_A()
Q_circ = build_Q(A_circ)

# Aer check
sim = AerSimulator()
log.sep("AER CHECK")
for m in [0, 1]:
    qc  = with_meas(build_base(A_circ, Q_circ, m))
    cnt = sim.run(qc, shots=4096).result().get_counts()
    p   = cnt.get('1', 0) / sum(cnt.values())
    log.info(f"m={m}: {p*100:.2f}%  theory={P_THEORY[m]*100:.2f}%")

try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    log.error("pip install 'iqm-client[qiskit]'"); sys.exit(1)

os.environ['IQM_TOKEN'] = IQM_TOKEN
backend = IQMProvider('https://cocos.resonance.meetiqm.com/emerald').get_backend()
log.ok(f"Connected: {backend.name}  {backend.num_qubits}q")

def hw_run(qc_meas, shots, label, opt_level=1):
    isa = transpile(qc_meas, backend=backend, optimization_level=opt_level)
    log.info(f"  [{label}] depth={isa.depth()} "
             f"CZ={isa.count_ops().get('cz',0)} shots={shots} opt={opt_level}")
    job = backend.run(isa, shots=shots, use_timeslot=False)
    log.info(f"  [{label}] job_id={job.job_id()}")
    cnt = job.result().get_counts()
    p   = cnt.get('1', 0) / sum(cnt.values())
    log.info(f"  [{label}] P_hw={p*100:.3f}%")
    return float(p), str(job.job_id()), isa.depth()

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK F — 5 new repetitions of m=1
# ─────────────────────────────────────────────────────────────────────────────
log.sep(f"BLOCK F — {N_REPEATS} repetitions of m=1 (shots={SHOTS_REP})")

base_m1 = build_base(A_circ, Q_circ, 1)
qc_m1   = with_meas(base_m1)
new_repeats = []

for i in range(N_REPEATS):
    p_hw, jid, _ = hw_run(qc_m1, SHOTS_REP, f"Rep_{i+1:02d}_m1")
    new_repeats.append({'run': i+1, 'p_hw': p_hw, 'job_id': jid,
                        'shots': SHOTS_REP, 'source': 'wave2_v2'})

# Pool all m=1 measurements
all_m1 = existing_m1 + [{'p_hw': r['p_hw'], 'source': 'wave2_v2',
                          'shots': SHOTS_REP} for r in new_repeats]
p_all  = [r['p_hw'] for r in all_m1]
p_mean = float(np.mean(p_all))
p_std  = float(np.std(p_all, ddof=1))
p_sem  = float(p_std / math.sqrt(len(p_all)))
p_th1  = P_THEORY[1]
z_score = (p_mean - p_th1) / p_sem if p_sem > 0 else 0.0

log.sep("BLOCK F RESULTS")
log.info(f"Total m=1 measurements: {len(p_all)} "
         f"({len(existing_m1)} existing + {N_REPEATS} new)")
log.info(f"P_theory(m=1) = {p_th1*100:.3f}%")
log.info(f"P_mean        = {p_mean*100:.3f}%  ± {p_sem*100:.3f}% SEM")
log.info(f"P_std         = {p_std*100:.3f}%")
log.info(f"95% CI        = [{(p_mean-1.96*p_sem)*100:.3f}%, "
         f"{(p_mean+1.96*p_sem)*100:.3f}%]")
log.info(f"z-score       = {z_score:.2f}  "
         f"(|z|<2 consistent with depolarizing noise)")
for r in new_repeats:
    log.info(f"  New run {r['run']:2d}: {r['p_hw']*100:.2f}%")

block_F = {
    'n_new_repeats': N_REPEATS,
    'n_existing': len(existing_m1),
    'n_total': len(p_all),
    'shots_per_run': SHOTS_REP,
    'new_runs': new_repeats,
    'existing_runs': existing_m1,
    'pooled_stats': {
        'p_mean': p_mean, 'p_std': p_std, 'p_sem': p_sem,
        'ci_95_lo': p_mean-1.96*p_sem, 'ci_95_hi': p_mean+1.96*p_sem,
        'z_score': float(z_score), 'p_theory': p_th1,
    }
}

(HW_OUT/'wave2_v2_blockF_latest.json').write_text(
    json.dumps({'block_F': block_F, 'ts': datetime.now().isoformat()},
               indent=2, default=_jd))
log.ok("Block F saved. Proceeding to Block B (ZNE m=0 only)...")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK B — ZNE on m=0 (scales 1,3 only — scale=5 saturates)
# ─────────────────────────────────────────────────────────────────────────────
log.sep("BLOCK B — ZNE on m=0 (scales 1,3 — scale=5 skipped)")
log.info("Rationale: scale=3 on m=1 already reached decoherence floor (50%)")
log.info("m=0 has shallower base circuit — more likely to show ZNE signal")

base_m0 = build_base(A_circ, Q_circ, 0)
p_sc_b  = {}
jobs_b  = {}

for scale in ZNE_SCALES:
    label  = f"ZNE_m0_s{scale}"
    folded = fold(base_m0, scale)
    opt    = 0 if scale > 1.0 else 1
    qc_f   = with_meas(to_basis(folded, opt=0) if scale > 1.0 else to_basis(folded))
    p_hw, jid, depth = hw_run(qc_f, SHOTS_ZNE, label, opt_level=opt)
    p_sc_b[scale] = p_hw
    jobs_b[scale] = {'job_id': jid, 'depth': depth, 'p_hw': p_hw,
                     'shots': SHOTS_ZNE, 'opt_level': opt}

# Check depth progression
d1 = jobs_b[1.0]['depth']; d3 = jobs_b[3.0]['depth']
folding_ok = d3 > d1
log.info(f"Depth progression: {d1} -> {d3}  "
         f"[{'OK' if folding_ok else 'WARN: possible cancellation'}]")

p1_b   = p_sc_b[1.0]; p3_b = p_sc_b[3.0]
p_lin  = float(1.5*p1_b - 0.5*p3_b)   # 2-point Richardson
p_th0  = P_THEORY[0]
err_r  = abs(p1_b  - p_th0)
err_z  = abs(p_lin - p_th0)
impr   = (err_r - err_z) / err_r if err_r > 1e-9 else 0.0

log.sep("BLOCK B RESULTS")
log.info(f"P_theory(m=0)       = {p_th0*100:.3f}%")
log.info(f"P_raw  (scale=1.0)  = {p1_b*100:.3f}%  err={err_r*100:.3f}%")
log.info(f"P_lin  (scale=3.0)  = {p3_b*100:.3f}%")
log.info(f"P_zne  (2-pt)       = {p_lin*100:.3f}%  err={err_z*100:.3f}%")
log.info(f"ZNE improvement     = {impr*100:.1f}%  "
         f"[{'EFFECTIVE' if impr>0.20 else 'MARGINAL'}]")

# Physical interpretation
if p3_b > 0.40:
    log.warn("scale=3 already at decoherence floor — ZNE not applicable at this depth")
elif folding_ok and impr > 0.20:
    log.ok("ZNE effective for m=0 — decoherence not yet saturated at scale=3")
else:
    log.info("ZNE marginal — noise may not be purely depolarizing")

block_B = {
    'm': 0, 'scales_used': ZNE_SCALES,
    'scale_results': {str(k): v for k, v in jobs_b.items()},
    'depth_progression': {'d1': d1, 'd3': d3, 'folding_ok': folding_ok},
    'p_extrap_linear': float(p_lin),
    'p_theory': float(p_th0),
    'err_raw': float(err_r), 'err_zne': float(err_z),
    'improvement': float(impr),
    'effective': bool(impr > 0.20),
    'note': 'scale=5 and m=2,3 ZNE skipped — depth saturates decoherence floor',
}

# ── Consolidated output ───────────────────────────────────────────────────────
log.sep("WAVE 2 v2 SUMMARY")
log.info(f"m=1 pooled mean  : {p_mean*100:.2f}% ± {p_sem*100:.2f}% SEM  "
         f"({len(p_all)} measurements)")
log.info(f"ZNE m=0 raw      : {p1_b*100:.2f}%  (theory {p_th0*100:.2f}%)")
log.info(f"ZNE m=0 extrap   : {p_lin*100:.2f}%  improvement={impr*100:.1f}%")

out = {
    'metadata': {
        'device': 'IQM Emerald (Crystal 54)',
        'version': 'v2 — budget-aware (5 repeats + ZNE m=0 scales 1,3)',
        'a_ideal': A_IDEAL, 'theta': THETA,
        'ts': datetime.now().isoformat()
    },
    'block_F_repeats': block_F,
    'block_B_zne_m0': block_B,
}
p = HW_OUT / 'wave2_v2_results_latest.json'
p.write_text(json.dumps(out, indent=2, default=_jd))
log.ok(f"Results: {p}")
