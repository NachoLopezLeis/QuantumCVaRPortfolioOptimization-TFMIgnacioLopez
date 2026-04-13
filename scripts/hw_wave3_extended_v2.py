"""
hw_wave3_extended_v2.py — Oleada 3 revisada (budget-aware)
===========================================================
Block E: Grover m=4,5,6 on Emerald — first oscillation minimum
Block D: Shot study on m=1 (shots=1024,4096,8192 only — 3 instead of 6)
Block G: Garnet vs Emerald comparison (m=0,1 on Garnet)

Cancelled vs v1:
  - shot levels 512, 2048, 16384 (marginal scientific value)

Cost estimate: ~8 jobs × 4 cr = ~32 cr

Usage:
    $env:IQM_TOKEN = "your_token"
    python scripts/hw_wave3_extended_v2.py
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
LOG_FILE     = HW_OUT / 'wave3_v2_latest.log'

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
log.sep("OLEADA 3 v2 — EXTENDED + SHOT STUDY + GARNET")
log.info("Budget-aware: m=4,5,6 + 3 shot levels + Garnet m=0,1")

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
P_THEORY = {m: math.sin((2*m+1)*THETA)**2 for m in range(10)}
log.ok(f"a_ideal={A_IDEAL:.6f}  theta={THETA:.6f}")
log.info("Theory values m=0..6:")
for m in range(7):
    aliased = (2*m+1)*THETA > math.pi/2
    log.info(f"  m={m}: {P_THEORY[m]*100:.3f}%  "
             f"{'[ALIASED]' if aliased else ''}")

SHOTS_EXT    = 4096
SHOTS_GARNET = 4096
SHOT_LEVELS  = [1024, 4096, 8192]   # reduced from 6 to 3

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

def build_grover_meas(A, Q, m):
    qc = QuantumCircuit(2, name=f'QAE_m{m}')
    qc.compose(A, inplace=True)
    for _ in range(m): qc.compose(Q, inplace=True)
    cr = ClassicalRegister(1, 'anc')
    qc.add_register(cr); qc.measure(1, 0)
    return qc

A_circ = build_A()
Q_circ = build_Q(A_circ)

# Aer check for extended m
sim = AerSimulator()
log.sep("AER CHECK m=0..6")
for m in range(7):
    qc  = build_grover_meas(A_circ, Q_circ, m)
    cnt = sim.run(qc, shots=8192).result().get_counts()
    p   = cnt.get('1', 0) / sum(cnt.values())
    log.info(f"  m={m}: {p*100:.2f}%  theory={P_THEORY[m]*100:.2f}%")

# ── Connect both backends ─────────────────────────────────────────────────────
try:
    from iqm.qiskit_iqm import IQMProvider
except ImportError:
    log.error("pip install 'iqm-client[qiskit]'"); sys.exit(1)

os.environ['IQM_TOKEN'] = IQM_TOKEN
be_emerald = IQMProvider('https://cocos.resonance.meetiqm.com/emerald').get_backend()
be_garnet  = IQMProvider('https://cocos.resonance.meetiqm.com/garnet').get_backend()
log.ok(f"Emerald: {be_emerald.name}  {be_emerald.num_qubits}q")
log.ok(f"Garnet:  {be_garnet.name}  {be_garnet.num_qubits}q")

def hw_run(qc_meas, shots, label, backend):
    isa  = transpile(qc_meas, backend=backend, optimization_level=1)
    name = 'EME' if backend.num_qubits == 54 else 'GAR'
    log.info(f"  [{label}|{name}] depth={isa.depth()} "
             f"CZ={isa.count_ops().get('cz',0)} shots={shots}")
    job = backend.run(isa, shots=shots, use_timeslot=False)
    log.info(f"  [{label}|{name}] job_id={job.job_id()}")
    cnt = job.result().get_counts()
    p   = cnt.get('1', 0) / sum(cnt.values())
    log.info(f"  [{label}|{name}] P_hw={p*100:.3f}%  "
             f"theory={P_THEORY.get(int(''.join(filter(str.isdigit,label.split('m')[-1][:2]))),0)*100:.2f}%")
    return float(p), str(job.job_id()), isa.depth()

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK E — Extended oscillation m=4,5,6
# ─────────────────────────────────────────────────────────────────────────────
log.sep("BLOCK E — Extended Grover m=4,5,6")
log.info("Expected: m=4->79.7%, m=5->48.6% (minimum), m=6->17.5%")
log.info("With noise: oscillation may be attenuated but pattern should be visible")

block_E = {}
for m in [4, 5, 6]:
    qc = build_grover_meas(A_circ, Q_circ, m)
    p_hw, jid, depth = hw_run(qc, SHOTS_EXT, f"EXT_m{m}", be_emerald)
    p_th = P_THEORY[m]
    err  = abs(p_hw - p_th)
    block_E[f'm{m}'] = {
        'job_id': jid, 'shots': SHOTS_EXT,
        'p_hw': p_hw, 'p_theory': p_th,
        'error': err, 'depth_iqm': depth,
        'aliased': (2*m+1)*THETA > math.pi/2,
    }

# Check if oscillation is still visible
p4 = block_E['m4']['p_hw']
p5 = block_E['m5']['p_hw']
p6 = block_E['m6']['p_hw']
min_visible = p4 > p5 < p6 or p4 > p5 or p5 < p6
log.sep("BLOCK E RESULTS")
log.info(f"m=4: {p4*100:.2f}%  (theory {P_THEORY[4]*100:.1f}%)")
log.info(f"m=5: {p5*100:.2f}%  (theory {P_THEORY[5]*100:.1f}%  ← minimum)")
log.info(f"m=6: {p6*100:.2f}%  (theory {P_THEORY[6]*100:.1f}%)")
log.info(f"Oscillation minimum visible: {min_visible}")

(HW_OUT/'wave3_v2_blockE_latest.json').write_text(
    json.dumps({'block_E': block_E, 'ts': datetime.now().isoformat()},
               indent=2, default=_jd))
log.ok("Block E saved.")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK D — Shot budget study on m=1 (3 levels)
# ─────────────────────────────────────────────────────────────────────────────
log.sep(f"BLOCK D — Shot study m=1 (levels={SHOT_LEVELS})")

block_D = {'m': 1, 'p_theory': P_THEORY[1], 'results': []}
qc_m1   = build_grover_meas(A_circ, Q_circ, 1)

for shots in SHOT_LEVELS:
    p_hw, jid, depth = hw_run(qc_m1, shots, f"SHOT_m1_n{shots}", be_emerald)
    shot_noise = math.sqrt(p_hw*(1-p_hw)/shots) if shots > 0 else 0
    block_D['results'].append({
        'shots': shots, 'p_hw': p_hw,
        'shot_noise_1sigma': shot_noise,
        'job_id': jid
    })
    log.info(f"  shots={shots:5d}: P_hw={p_hw*100:.3f}%  "
             f"±{shot_noise*100:.3f}% (1σ shot noise)")

log.sep("BLOCK D RESULTS")
for r in block_D['results']:
    log.info(f"  shots={r['shots']:5d}: {r['p_hw']*100:.2f}% "
             f"± {r['shot_noise_1sigma']*100:.2f}%")

(HW_OUT/'wave3_v2_blockD_latest.json').write_text(
    json.dumps({'block_D': block_D, 'ts': datetime.now().isoformat()},
               indent=2, default=_jd))
log.ok("Block D saved.")

# ─────────────────────────────────────────────────────────────────────────────
# BLOCK G — Garnet vs Emerald (m=0,1)
# ─────────────────────────────────────────────────────────────────────────────
log.sep("BLOCK G — Garnet vs Emerald cross-platform (m=0,1)")

# Load existing Emerald results from phase1_direct
emerald_ref = {}
try:
    d = json.loads((HW_OUT/'phase1_direct_results_latest.json').read_text())
    for m in [0, 1, 2, 3]:
        key = f'm{m}'
        if key in d.get('phase1',{}).get('grover',{}):
            emerald_ref[m] = d['phase1']['grover'][key]
except Exception:
    pass

block_G = {'emerald_from_phase1': {}, 'garnet_new': {}}

for m in [0, 1]:
    # Emerald — from existing phase1_direct results
    if m in emerald_ref:
        block_G['emerald_from_phase1'][f'm{m}'] = {
            'p_hw': emerald_ref[m]['p_hw'],
            'p_theory': emerald_ref[m]['p_theory'],
            'error': emerald_ref[m]['error'],
            'source': 'phase1_direct'
        }
        log.info(f"Emerald m={m}: {emerald_ref[m]['p_hw']*100:.2f}% (existing)")

    # Garnet — new job
    qc = build_grover_meas(A_circ, Q_circ, m)
    p_hw, jid, depth = hw_run(qc, SHOTS_GARNET, f"GAR_m{m}", be_garnet)
    p_th = P_THEORY[m]
    block_G['garnet_new'][f'm{m}'] = {
        'job_id': jid, 'shots': SHOTS_GARNET,
        'p_hw': p_hw, 'p_theory': p_th,
        'error': abs(p_hw-p_th), 'depth_iqm': depth
    }
    log.info(f"Garnet  m={m}: {p_hw*100:.2f}%  (theory {p_th*100:.2f}%)")

log.sep("BLOCK G — Comparison table")
log.info(f"{'m':<3} {'Emerald':>10} {'Garnet':>10} {'Theory':>10} {'Δ (E-G)':>10}")
log.info('-'*47)
for m in [0, 1]:
    p_em = block_G['emerald_from_phase1'].get(f'm{m}',{}).get('p_hw', float('nan'))
    p_ga = block_G['garnet_new'][f'm{m}']['p_hw']
    p_th = P_THEORY[m]
    delta = (p_em-p_ga) if not math.isnan(p_em) else float('nan')
    log.info(f"m={m}  {p_em*100:>9.2f}%  {p_ga*100:>9.2f}%  "
             f"{p_th*100:>9.2f}%  {delta*100:>+9.2f}pp")

# ── Consolidated output ───────────────────────────────────────────────────────
log.sep("WAVE 3 v2 SUMMARY")
for m in [4, 5, 6]:
    b = block_E[f'm{m}']
    log.info(f"  m={m}: {b['p_hw']*100:.2f}%  theory={b['p_theory']*100:.2f}%")
log.info(f"Shot study: " + "  ".join(
    f"{r['shots']}->{r['p_hw']*100:.1f}%" for r in block_D['results']))
for m in [0, 1]:
    g = block_G['garnet_new'][f'm{m}']
    e = block_G['emerald_from_phase1'].get(f'm{m}',{}).get('p_hw', float('nan'))
    log.info(f"  Cross-platform m={m}: Emerald={e*100:.1f}%  Garnet={g['p_hw']*100:.1f}%")

out = {
    'metadata': {
        'device_primary': 'IQM Emerald (Crystal 54)',
        'device_secondary': 'IQM Garnet (Star 20)',
        'version': 'v2 — budget-aware',
        'a_ideal': A_IDEAL, 'theta': THETA,
        'ts': datetime.now().isoformat()
    },
    'block_E_extended': block_E,
    'block_D_shot_study': block_D,
    'block_G_comparison': block_G,
}
p = HW_OUT / 'wave3_v2_results_latest.json'
p.write_text(json.dumps(out, indent=2, default=_jd))
log.ok(f"Results: {p}")
