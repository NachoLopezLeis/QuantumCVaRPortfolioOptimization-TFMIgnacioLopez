#!/usr/bin/env python3
r"""
===============================================================================
TFM Circuit Exporter v2
===============================================================================
Llama a las funciones reales de qae_circuits.py para construir los circuitos
cuánticos que se usan en el pipeline y los exporta como PNG.

CAMBIOS v2:
- Descomposición profunda para eliminar bloques multiplexer_dg opacos
- Función extra que dibuja qué es multiplexer / multiplexer_dg
- Doble exportación: nivel alto (bloques) + nivel bajo (puertas básicas)

USO:
    Coloca este fichero en el mismo directorio que qae_circuits.py y ejecuta:
        python export_tfm_circuits.py

    Los PNG se guardan en: E:\\\\TFM\\\Codigo\results\circuits\
===============================================================================
"""

import sys
import os
from pathlib import Path
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

MODULES_DIR = Path(r"E:\\\TFM\\Codigo\src\phase_2")
OUTPUT_DIR = Path(r"E:\\\TFM\\Codigo\results\circuits")

NUM_QUBITS = 4
ALPHA = 0.05
SEED = 42
DPI = 300

# Cuántos niveles de decompose() aplicar para llegar a puertas básicas
# 3 = lo que hace tu código original (deja multiplexer_dg)
# 6-8 = descompone hasta Ry, CX, X, etc.
DECOMPOSE_DEPTH = 8

# ─────────────────────────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [MODULES_DIR, MODULES_DIR.parent, MODULES_DIR.parent.parent]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))
sys.path.insert(0, str(Path(__file__).parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qae_circuits import QuantumCVaRCircuits, CircuitExporter
from qiskit import QuantumCircuit

print("=" * 70)
print("TFM CIRCUIT EXPORTER v2")
print(f"  Output:  {OUTPUT_DIR}")
print(f"  Qubits:  {NUM_QUBITS} (N={2**NUM_QUBITS})")
print(f"  Decompose depth: {DECOMPOSE_DEPTH}")
print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# DATOS
# ─────────────────────────────────────────────────────────────────────────────

def generate_data():
    rng = np.random.default_rng(SEED)
    n_assets, n_periods = 10, 375
    mean_ret = rng.uniform(-0.0002, 0.001, n_assets)
    vols = rng.uniform(0.008, 0.025, n_assets)
    A = rng.standard_normal((n_assets, n_assets))
    corr = np.corrcoef(A.T)
    cov = np.outer(vols, vols) * corr
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < 0:
        cov += (-eigvals.min() + 1e-8) * np.eye(n_assets)
    returns = rng.multivariate_normal(mean_ret, cov, n_periods)
    weights = np.ones(n_assets) / n_assets
    losses = -(returns @ weights)
    return returns, weights, losses


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────────────────────────────────────

def deep_decompose(qc, depth=DECOMPOSE_DEPTH):
    """Descompone un circuito iterativamente hasta puertas básicas."""
    result = qc
    for i in range(depth):
        prev_ops = set(inst.operation.name for inst in result.data)
        result = result.decompose()
        new_ops = set(inst.operation.name for inst in result.data)
        if new_ops == prev_ops:
            break
    return result


def has_opaque_gates(qc):
    """Comprueba si el circuito tiene bloques opacos."""
    opaque = set()
    for inst in qc.data:
        name = inst.operation.name
        if 'multiplexer' in name or 'initialize' in name or 'isometry' in name:
            opaque.add(name)
    return opaque


def save_png(qc, filename, title=None, fold=40, scale=0.8):
    """Guarda un QuantumCircuit como PNG."""
    path = OUTPUT_DIR / filename
    try:
        fig = qc.draw("mpl", style="iqp", fold=fold, scale=scale)
        if title:
            fig.suptitle(title, fontsize=11, fontfamily="serif",
                         fontweight="bold", y=1.02)
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  [OK] {filename}  ({qc.num_qubits}q, depth={qc.depth()}, "
              f"gates={len(qc.data)})")
        return path
    except Exception as e:
        print(f"  [FAIL] {filename}: {e}")
        import traceback; traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIÓN EXTRA: DIAGRAMA EXPLICATIVO DE MULTIPLEXER_DG
# ─────────────────────────────────────────────────────────────────────────────

def draw_multiplexer_explanation():
    """
    Dibuja un diagrama que explica qué es multiplexer / multiplexer_dg:
    una rotación Ry controlada uniformemente (Möttonen decomposition).

    multiplexer    = aplica Ry(alpha_i) condicionado al estado |i> de los controles
    multiplexer_dg = su adjunto (dagger)

    Se descompone en una secuencia alternada de Ry y CNOT (Gray code ordering).
    """
    from matplotlib.patches import FancyBboxPatch

    fig, (ax_block, ax_decomp) = plt.subplots(
        1, 2, figsize=(16, 5), gridspec_kw={'width_ratios': [1, 1.8]})

    # ── Panel izquierdo: bloque de alto nivel ──
    ax = ax_block
    ax.set_xlim(-1, 7)
    ax.set_ylim(-0.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    C_MUX = "#831843"
    C_MUX_FILL = "#FCE7F3"
    C_WIRE = "#333333"

    for y in [4, 3, 2, 1]:
        ax.plot([-0.5, 6], [y, y], color=C_WIRE, lw=0.8)

    for i, y in enumerate([4, 3, 2]):
        ax.text(-0.8, y, f"$q_{i}$", fontsize=11, fontfamily="serif",
                ha="right", va="center")
    ax.text(-0.8, 1, "$q_3$ (target)", fontsize=11, fontfamily="serif",
            ha="right", va="center")

    rect = FancyBboxPatch((1.5, 0.5), 2.5, 4.0,
        boxstyle="round,pad=0.1", facecolor=C_MUX_FILL,
        edgecolor=C_MUX, linewidth=2, zorder=3)
    ax.add_patch(rect)
    ax.text(2.75, 2.9, "multiplexer_dg", fontsize=11, fontfamily="serif",
            fontweight="bold", ha="center", va="center", color=C_MUX, zorder=4)
    ax.text(2.75, 2.2, r"$= U_{\mathrm{M\ddot{o}ttonen}}^\dagger$", fontsize=12,
            fontfamily="serif", ha="center", va="center", color=C_MUX, zorder=4)

    for y in [4, 3, 2]:
        ax.plot(2.75, y, "o", color=C_MUX, markersize=6, zorder=5)

    ax.plot(2.75, 1, "s", color=C_MUX, markersize=10, markerfacecolor=C_MUX_FILL,
            markeredgewidth=2, zorder=5)
    ax.text(2.75, 1, r"$R_y$", fontsize=8, fontfamily="serif",
            ha="center", va="center", color=C_MUX, fontweight="bold", zorder=6)

    ax.text(2.75, 0.0,
            r"$|i\rangle|0\rangle \;\to\; |i\rangle\, R_y(\alpha_i)|0\rangle$",
            fontsize=12, fontfamily="serif", ha="center", va="center", color=C_MUX)

    ax.set_title("(a) Bloque de alto nivel", fontsize=12,
                 fontfamily="serif", fontweight="bold", pad=10)

    # ── Panel derecho: descomposición en puertas básicas ──
    ax = ax_decomp
    ax.set_xlim(-1, 16)
    ax.set_ylim(-1.5, 4.5)
    ax.set_aspect("equal")
    ax.axis("off")

    C_RY = "#1E40AF"
    C_RY_FILL = "#DBEAFE"

    for y in [4, 2.5, 1]:
        ax.plot([-0.5, 15], [y, y], color=C_WIRE, lw=0.8)

    ax.text(-0.8, 4, r"$q_0$ (ctrl)", fontsize=10, fontfamily="serif",
            ha="right", va="center")
    ax.text(-0.8, 2.5, r"$q_1$ (ctrl)", fontsize=10, fontfamily="serif",
            ha="right", va="center")
    ax.text(-0.8, 1, r"$q_2$ (target)", fontsize=10, fontfamily="serif",
            ha="right", va="center")

    def _ry(ax, x, y, label):
        r = FancyBboxPatch((x-0.4, y-0.3), 0.8, 0.6,
            boxstyle="round,pad=0.04", facecolor=C_RY_FILL,
            edgecolor=C_RY, linewidth=1.2, zorder=3)
        ax.add_patch(r)
        ax.text(x, y, label, fontsize=8, fontfamily="serif",
                ha="center", va="center", color=C_RY, fontweight="bold", zorder=4)

    def _cx(ax, x, cy, ty):
        ax.plot(x, cy, "o", color=C_RY, markersize=6, zorder=5)
        ax.plot([x, x], [cy, ty], color=C_RY, lw=1, zorder=2)
        ax.plot(x, ty, "o", color=C_RY, markersize=10,
                markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        ax.plot([x-0.12, x+0.12], [ty, ty], color=C_RY, lw=1.5, zorder=6)
        ax.plot([x, x], [ty-0.12, ty+0.12], color=C_RY, lw=1.5, zorder=6)

    # Gray code sequence: Ry - CX - Ry - CX - Ry - CX - Ry - CX
    _ry(ax, 0.5, 1, r"$R_y(\theta_0)$")
    _cx(ax, 2.0, 2.5, 1)
    _ry(ax, 3.5, 1, r"$R_y(\theta_1)$")
    _cx(ax, 5.0, 4, 1)
    _ry(ax, 6.5, 1, r"$R_y(\theta_2)$")
    _cx(ax, 8.0, 2.5, 1)
    _ry(ax, 9.5, 1, r"$R_y(\theta_3)$")
    _cx(ax, 11.0, 4, 1)
    ax.text(12.5, 1, r"$\cdots$", fontsize=14, ha="center",
            va="center", color="#6B7280")

    ax.text(7.5, -0.7,
            r"$\vec{\theta} = M_n^{-1} \vec{\alpha}$  "
            "(Gray code, Ref: Möttonen et al. 2004)",
            fontsize=10, fontfamily="serif", ha="center", va="center",
            color="#6B7280", fontstyle="italic")

    ax.set_title("(b) Descomposición en puertas básicas",
                 fontsize=12, fontfamily="serif", fontweight="bold", pad=10)

    fig.suptitle(
        r"¿Qué es multiplexer_dg? — Rotación $R_y$ controlada uniformemente",
        fontsize=14, fontfamily="serif", fontweight="bold", y=1.03)

    fig.tight_layout()
    path = OUTPUT_DIR / "circuit_0_multiplexer_explanation.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK] circuit_0_multiplexer_explanation.png")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n[1] Generando datos...")
    returns, weights, losses = generate_data()
    var_threshold = float(np.percentile(losses, (1 - ALPHA) * 100))
    print(f"    {returns.shape[0]} periodos x {returns.shape[1]} activos")
    print(f"    VaR (percentil 95): {var_threshold:.6f}")

    exporter = CircuitExporter(output_dir=str(OUTPUT_DIR / "_raw"), enabled=False)
    circuits = QuantumCVaRCircuits(num_qubits=NUM_QUBITS, exporter=exporter)
    n = NUM_QUBITS

    probs, levels, edges, l_min, l_max, threshold_level = \
        circuits._discretize_with_threshold(losses, var_threshold)
    print(f"    threshold_level={threshold_level}/{2**n}")

    # ── 0: Explicación multiplexer_dg ──
    print("\n[2] Diagrama multiplexer_dg...")
    draw_multiplexer_explanation()

    # ── 1: State Preparation ──
    print("\n[3] Circuit 1: State Preparation...")
    qc_dist = circuits.build_distribution_loading_circuit(probs, name="A_load")
    save_png(qc_dist, "circuit_1a_state_prep_blocks.png",
             title=r"$A_{\mathrm{load}}$: State Preparation — alto nivel")
    qc_dist_decomp = deep_decompose(qc_dist)
    save_png(qc_dist_decomp, "circuit_1b_state_prep_gates.png",
             title=r"$A_{\mathrm{load}}$: State Preparation — puertas básicas",
             fold=80, scale=0.5)

    # ── 2: Threshold Oracle ──
    print("\n[4] Circuit 2: Threshold Oracle...")
    qc_oracle = circuits.build_threshold_oracle(n, threshold_level, name="U_threshold")
    save_png(qc_oracle, "circuit_2_threshold_oracle.png",
             title=r"$U_{\geq z}$: Threshold Oracle — "
                   f"level={threshold_level}/{2**n}")

    # ── 3: Tail Probability ──
    print("\n[5] Circuit 3: Tail Probability (Operador A)...")
    qc_tail, obj_qubits, meta = circuits.build_tail_probability_circuit(
        losses, var_threshold, name="A_tail")
    save_png(qc_tail, "circuit_3a_tail_prob_blocks.png",
             title=r"$\mathcal{A}$: $P(L \geq z)$ — alto nivel")
    qc_tail_decomp = deep_decompose(qc_tail)
    save_png(qc_tail_decomp, "circuit_3b_tail_prob_gates.png",
             title=r"$\mathcal{A}$: $P(L \geq z)$ — puertas básicas",
             fold=100, scale=0.4)

    # ── 4: Grover Operator ──
    print("\n[6] Circuit 4: Grover Operator Q...")
    state_prep = QuantumCircuit(n + 1, name="state_prep")
    state_prep.compose(qc_dist, qubits=range(n), inplace=True)
    state_prep.compose(qc_oracle, qubits=range(n + 1), inplace=True)
    qc_grover = circuits.build_grover_operator(state_prep, qc_oracle, obj_qubits)
    if isinstance(qc_grover, QuantumCircuit):
        grover_draw = qc_grover
    else:
        grover_draw = QuantumCircuit(n + 1, name="Q_Grover")
        grover_draw.append(qc_grover, range(n + 1))
    save_png(grover_draw.decompose(), "circuit_4a_grover_blocks.png",
             title=r"Grover $Q$ — alto nivel", fold=60, scale=0.6)
    grover_decomp = deep_decompose(grover_draw)
    save_png(grover_decomp, "circuit_4b_grover_gates.png",
             title=r"Grover $Q$ — puertas básicas",
             fold=120, scale=0.35)

    # ── 5: Subgradient Component ──
    print("\n[7] Circuit 5: Subgradient A_j (j=0)...")
    qc_cond, cond_obj, cond_meta = circuits.build_conditional_expectation_circuit(
        returns[:, 0], losses, var_threshold, asset_idx=0, name="A_j")
    if qc_cond is not None:
        save_png(qc_cond, "circuit_5a_subgradient_blocks.png",
                 title=r"$\mathcal{A}_0$: Subgradiente $g_0$ — alto nivel")
        qc_cond_decomp = deep_decompose(qc_cond)
        save_png(qc_cond_decomp, "circuit_5b_subgradient_gates.png",
                 title=r"$\mathcal{A}_0$: Subgradiente — puertas básicas",
                 fold=100, scale=0.4)

    # ── 6: Tail Weighted Loss ──
    print("\n[8] Circuit 6: Tail Weighted Loss...")
    qc_twl, twl_obj, twl_meta = circuits.build_tail_weighted_loss_circuit(
        losses, var_threshold, name="A_twl")
    save_png(qc_twl, "circuit_6a_tail_weighted_blocks.png",
             title=r"$E[L \cdot \mathbb{1}\{L \geq VaR\}]$ — alto nivel")
    qc_twl_decomp = deep_decompose(qc_twl)
    save_png(qc_twl_decomp, "circuit_6b_tail_weighted_gates.png",
             title=r"$E[L \cdot \mathbb{1}\{L \geq VaR\}]$ — puertas básicas",
             fold=100, scale=0.4)

    # ── Resumen ──
    print("\n" + "=" * 70)
    pngs = sorted(OUTPUT_DIR.glob("circuit_*.png"))
    print(f"{len(pngs)} PNG exportados en: {OUTPUT_DIR}\n")
    for p in pngs:
        print(f"  {p.name:50s} ({p.stat().st_size/1024:6.0f} KB)")
    print("\n" + "=" * 70)
    print("\nPARA EL TFM:")
    print("  Cuerpo   -> *_blocks.png  (limpio, bloques legibles)")
    print("  Apéndice -> *_gates.png   (profundidad real, puertas Ry/CX/X)")
    print("  Leyenda  -> circuit_0_multiplexer_explanation.png")


if __name__ == "__main__":
    main()
