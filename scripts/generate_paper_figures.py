#!/usr/bin/env python3
# =============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
# Author  : Ignacio Lopez Leis — Universidad Autonoma de Madrid
# Script  : generate_paper_figures.py
#
# Generates all 12 publication-quality figures for the TFM paper.
# Output  : results/paper_figures/fig{N:02d}_*.png  (300 dpi)
#
# Figures
# -------
#  01  Pipeline diagram
#  02  QUBO network + community selection
#  03  Correlation matrices PA / PB / PC
#  04  IQAE circuit depth diagram
#  05  Convergence curves: ideal vs E1/E2/E3 noise
#  06  Cosine similarity + HHI vs noise level
#  07  IQAE aliasing threshold on IQM hardware
#  08  ZNE depth overhead
#  09  Euler decomposition: four methods (PA)
#  10  Cumulative return OOS
#  11  VaR breach chart + Acerbi-Székely
#  12  Scalability: N vs CVaR improvement, fixed qubits
# =============================================================================

from __future__ import annotations
import json, glob, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "paper_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Colour palette
C_BLUE   = "#2563EB"
C_GREEN  = "#16A34A"
C_ORANGE = "#EA580C"
C_RED    = "#DC2626"
C_PURPLE = "#7C3AED"
C_GOLD   = "#D97706"
C_GREY   = "#6B7280"
C_LIGHT  = "#F3F4F6"

STYLE = {
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
}
plt.rcParams.update(STYLE)

def save(fig, name: str):
    path = OUTDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path.name}")

def load(exp: str) -> dict:
    p = ROOT / "results" / f"exp_{exp}" / "tfm_comprehensive_metrics_latest.json"
    with open(p) as f:
        return json.load(f)

# =============================================================================
# FIG 01 — Pipeline diagram
# =============================================================================
def fig01_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)

    blocks = [
        (1.0,  2.5, "D-Wave\nHybrid\nQUBO",       C_PURPLE, "374 S&P500 assets\n11 communities\nenergy = −54,998"),
        (4.2,  2.5, "IQM\n4-qubit\nIQAE",          C_BLUE,   "n_qubits = 4\ndepth(m) = 4+6m\nε = 0.005"),
        (7.4,  2.5, "Adam\nHybrid\nOptimizer",      C_GREEN,  "38 iterations\ncosine sim > 0.9999\nCVaR −25.6%"),
        (10.6, 2.5, "OOS\nBacktest\n501 days",      C_ORANGE, "Acerbi-Székely PASS\nCVaR = 0.0184\nSharpe = 0.681"),
    ]

    for x, y, title, col, sub in blocks:
        rect = mpatches.FancyBboxPatch((x-0.9, y-1.3), 1.8, 2.6,
                                        boxstyle="round,pad=0.1",
                                        facecolor=col, alpha=0.15,
                                        edgecolor=col, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y+0.55, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=col)
        ax.text(x, y-0.55, sub, ha="center", va="center",
                fontsize=8, color="#374151", linespacing=1.5)

    # Arrows
    for x0, x1 in [(1.9, 3.3), (5.1, 6.5), (8.3, 9.7)]:
        ax.annotate("", xy=(x1, 2.5), xytext=(x0, 2.5),
                    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=2))

    # Labels above arrows
    for x, lbl in [(2.6, "tickers\nselection"), (5.8, "IQAE\nsubgradient"), (9.0, "optimal\nweights")]:
        ax.text(x, 3.6, lbl, ha="center", va="bottom", fontsize=7.5,
                color=C_GREY, style="italic")

    ax.set_title("Hybrid Quantum-Classical CVaR Portfolio Optimization Pipeline",
                 fontsize=14, fontweight="bold", pad=12)
    save(fig, "fig01_pipeline.png")


# =============================================================================
# FIG 02 — QUBO network (schematic: no full graph, show key metrics)
# =============================================================================
def fig02_qubo_network():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: community structure schematic
    ax = axes[0]
    ax.axis("off")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.3)
    ax.set_title("S&P500 Correlation Network\n374 assets, 11 communities", fontsize=11, fontweight="bold")

    np.random.seed(42)
    n_communities = 11
    community_sizes = [57, 49, 44, 39, 36, 33, 33, 28, 23, 16, 16]
    community_colors = plt.cm.tab20(np.linspace(0, 1, n_communities))

    # Draw communities as circles
    angles = np.linspace(0, 2*np.pi, n_communities, endpoint=False)
    r = 0.78
    for i, (ang, sz, col) in enumerate(zip(angles, community_sizes, community_colors)):
        cx, cy = r*np.cos(ang), r*np.sin(ang)
        radius = 0.08 + 0.05*(sz/57)
        circle = plt.Circle((cx, cy), radius, color=col, alpha=0.7, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, cy, str(sz), ha="center", va="center", fontsize=7,
                fontweight="bold", color="white", zorder=4)

    # PA assets: peripheral (outer)
    PA_angles = np.linspace(0.1, 2*np.pi-0.3, 10)
    for ang in PA_angles:
        x, y = (r+0.3)*np.cos(ang), (r+0.3)*np.sin(ang)
        ax.plot(x, y, "o", ms=9, color=C_GREEN, zorder=5,
                markeredgecolor="white", markeredgewidth=1.5)

    # PB assets: central (inner)
    for ang in np.linspace(0, 2*np.pi, 4, endpoint=False):
        for dr in [0.0, 0.15]:
            x, y = (0.2+dr)*np.cos(ang), (0.2+dr)*np.sin(ang)
            ax.plot(x, y, "s", ms=8, color=C_RED, zorder=5,
                    markeredgecolor="white", markeredgewidth=1.5)

    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=C_GREEN, ms=9,
               label="PA (QUBO peripheral, avg_peri=0.980)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor=C_RED, ms=8,
               label="PB (Central ranking, avg_peri=0.474)"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), fontsize=8, framealpha=0.9)

    # Right: key metrics comparison bar chart
    ax2 = axes[1]
    metrics = {
        "Avg\nperipherality":   [0.980, 0.474, 0.720],
        "Corr\nmean":           [0.176, 0.710, 0.477],
        "Communities\ncovered": [10/11, 4/11, 8/11],
        "Effective N\n(eigen)": [7.28/10, 1.78/10, 3.14/10],
    }
    labels = list(metrics.keys())
    x = np.arange(len(labels))
    w = 0.25
    colors = [C_GREEN, C_RED, C_BLUE]
    names  = ["PA (QUBO)", "PB (Central)", "PC (Manual)"]

    for i, (vals, col, nm) in enumerate(zip(zip(*metrics.values()), colors, names)):
        bars = ax2.bar(x + i*w, vals, w, label=nm, color=col, alpha=0.75, zorder=3)

    ax2.set_xticks(x + w)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("Normalized value", fontsize=10)
    ax2.set_title("Portfolio Universe Quality Metrics\n(normalized to [0,1])", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1.15)

    fig.suptitle("Figure 2: QUBO-Driven Universe Selection via Market Network",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig02_qubo_network.png")


# =============================================================================
# FIG 03 — Correlation matrices PA / PB / PC
# =============================================================================
def fig03_correlation_matrices():
    df = pd.read_csv(ROOT / "data" / "returns_sp500_full.csv", index_col=0, parse_dates=True)
    PA = ["AAL","CLX","DPZ","DVA","FANG","FMC","MKTX","NEM","NFLX","NOC"]
    PB = ["AIG","AMP","BAC","BRK-B","ITW","IVZ","L","MCO","PRU","TROW"]
    PC = ["AAPL","MSFT","GOOGL","JPM","UNH","PG","XOM","CAT","NEE","LIN"]
    oos_start = "2024-01-01"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    labels_info = [
        ("PA — QUBO Peripheral", PA, C_GREEN),
        ("PB — Central Ranking", PB, C_RED),
        ("PC — Mega-Cap Manual", PC, C_BLUE),
    ]

    for ax, (title, tickers, col), letter in zip(axes, labels_info, "ABC"):
        ret = df[tickers][df[tickers].index < oos_start].dropna()
        corr = ret.corr().values
        upper = corr[np.triu_indices_from(corr, k=1)]

        im = ax.imshow(corr, vmin=-0.2, vmax=1.0, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=7.5)
        ax.set_yticklabels(tickers, fontsize=7.5)
        ax.set_title(f"({letter}) {title}", fontsize=10, fontweight="bold", pad=6)
        ax.grid(False)

        stats = (f"μ = {upper.mean():.3f}   max = {upper.max():.3f}\n"
                 f">0.5: {(upper>0.5).mean()*100:.0f}%   "
                 f"eff_N = {1/np.mean(corr**2):.1f}")
        ax.text(0.5, -0.28, stats, transform=ax.transAxes,
                ha="center", fontsize=8.5, color="#374151",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=col, alpha=0.12))

    plt.colorbar(im, ax=axes, shrink=0.8, label="Pearson correlation")
    fig.suptitle("Figure 3: Pairwise Correlation Matrices (Training Period 2020–2023)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig03_correlation_matrices.png")


# =============================================================================
# FIG 04 — Circuit depth diagram
# =============================================================================
def fig04_circuit_depth():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: depth(m) = 4 + 6m
    ax = axes[0]
    m_vals = np.arange(0, 8)
    depths = 4 + 6*m_vals
    eps_ideal = np.pi / (2**(m_vals + 2))

    ax2 = ax.twinx()
    bars = ax.bar(m_vals, depths, color=[C_GREEN if m <= 2 else C_RED for m in m_vals],
                  alpha=0.75, zorder=3)
    ax2.plot(m_vals, eps_ideal, "k--o", ms=6, lw=1.5, label="Ideal ε(m)", zorder=4)

    ax.axvspan(-0.5, 2.5, alpha=0.08, color=C_GREEN, label="Usable on hardware")
    ax.axvspan(2.5, 7.5, alpha=0.08, color=C_RED, label="Aliased on hardware")
    ax.axvline(2.5, color=C_RED, lw=2, ls="--")
    ax.text(2.6, 42, "Aliasing\nonset", fontsize=9, color=C_RED, fontweight="bold")

    ax.set_xlabel("IQAE round m", fontsize=10)
    ax.set_ylabel("Circuit depth (post-transpile IQM)", fontsize=10)
    ax2.set_ylabel("Ideal precision ε(m)", fontsize=10, color="#374151")
    ax.set_title("(A) Circuit depth vs IQAE round\ndepth(m) = 4 + 6m", fontsize=10, fontweight="bold")
    ax.set_xticks(m_vals)
    ax.set_xlim(-0.5, 7.5)

    h1 = mpatches.Patch(color=C_GREEN, alpha=0.75, label="m≤2: usable (depth≤16)")
    h2 = mpatches.Patch(color=C_RED,   alpha=0.75, label="m≥3: aliased (depth≥22)")
    h3 = Line2D([0],[0], color="k", ls="--", marker="o", label="Ideal ε(m)")
    ax.legend(handles=[h1, h2, h3], fontsize=8.5, loc="upper left")

    # Right: precision gap — achievable vs required
    ax3 = axes[1]
    scenarios = ["m=2\n(hw max)", "m=7\n(ε=0.005\ntarget)", "Monte Carlo\nε=0.005"]
    eps_vals  = [np.pi/2**4, 0.005, 0.005]
    q_vals    = [146*9, 146*49, int(1/0.005**2)]
    colors_s  = [C_ORANGE, C_GREEN, C_GREY]

    bars3 = ax3.bar(scenarios, q_vals, color=colors_s, alpha=0.78, zorder=3)
    ax3.set_ylabel("Oracle queries per estimation", fontsize=10)
    ax3.set_title("(B) Query cost comparison\nat target precision ε=0.005", fontsize=10, fontweight="bold")
    ax3.set_yscale("log")

    for bar, q in zip(bars3, q_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, q*1.3, f"{q:,}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax3.text(0, q_vals[0]*0.5, f"ε≈{eps_vals[0]:.2f}\n(imprecise)", ha="center",
             va="center", fontsize=8, color="white", fontweight="bold")
    ax3.text(1, q_vals[1]*0.5, f"ε={eps_vals[1]}", ha="center",
             va="center", fontsize=8, color="white", fontweight="bold")
    ax3.text(2, q_vals[2]*0.5, f"ε={eps_vals[2]}", ha="center",
             va="center", fontsize=8, color="white", fontweight="bold")

    fig.suptitle("Figure 4: IQAE Circuit Architecture and Hardware Depth Constraints",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig04_circuit_depth.png")


# =============================================================================
# FIG 05 — Convergence curves: ideal vs noise
# =============================================================================
def fig05_convergence():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    noise_cfg = {
        "A1": ("Ideal (no noise)",       C_BLUE,   "-",  "o"),
        "E1": ("E1: 1q=1e-4, 2q=1e-3",  C_GREEN,  "--", "s"),
        "E2": ("E2: 1q=5e-4, 2q=5e-3",  C_ORANGE, "-.", "^"),
        "E3": ("E3: 1q=1e-3, 2q=1e-2",  C_RED,    ":",  "D"),
    }
    portfolios = [("PA", "Portfolio PA (QUBO-peripheral)"),
                  ("PB", "Portfolio PB (Central)"),
                  ("PC", "Portfolio PC (Mega-cap)")]

    for ax, (port, title) in zip(axes, portfolios):
        for fam, (label, col, ls, mk) in noise_cfg.items():
            d = load(f"{fam}_{port}")
            hist = d["convergence_history"]["cvar_history"]
            iters = list(range(len(hist)))
            ax.plot(iters, [h*100 for h in hist], color=col, ls=ls,
                    lw=2, alpha=0.9, label=label if port=="PA" else "")
            # Mark final value
            ax.plot(iters[-1], hist[-1]*100, mk, ms=7, color=col, zorder=5)

        ax.set_xlabel("Optimizer iteration", fontsize=10)
        ax.set_ylabel("CVaR loss (%)" if port == "PA" else "", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        # Annotate improvement
        d0 = load(f"A1_{port}")
        init = d0["convergence_history"]["cvar_history"][0]*100
        final = d0["convergence_history"]["cvar_history"][-1]*100
        ax.annotate(f"−{init-final:.2f}pp\n(ideal)", xy=(len(d0['convergence_history']['cvar_history'])-1, final),
                    xytext=(20, 3), textcoords="offset points",
                    fontsize=8, color=C_BLUE, arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1))

    axes[0].legend(fontsize=8.5, loc="upper right")
    fig.suptitle("Figure 5: CVaR Convergence Under Increasing Gate Noise (Adam Optimizer)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig05_convergence.png")


# =============================================================================
# FIG 06 — Cosine similarity + HHI vs noise
# =============================================================================
def fig06_cosine_hhi():
    euler_df = pd.read_csv(ROOT / "results" / "euler_decomposition" / "summary.csv")
    noise_map = {"A1_PA": (0, "Ideal"),
                 "E1_PA": (1, "E1\n1e-4"),
                 "E2_PA": (2, "E2\n5e-4"),
                 "E3_PA": (3, "E3\n1e-3"),
                 "D1_PA": (4, "D1\n1e-3+ZNE")}

    cos_data, hhi_data = [], []
    for exp, (idx, lbl) in noise_map.items():
        d = load(exp)
        cos_sim = d["metrics"]["quantum"]["subgradient"]["cosine_similarity"]
        cos_data.append((idx, lbl, cos_sim))
        row = euler_df[(euler_df["experiment"]==exp) & (euler_df["weight_set"]=="quantum_optimal")]
        hhi = row["herfindahl_index"].values[0] if len(row) else np.nan
        hhi_data.append((idx, lbl, hhi))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    xs = [d[0] for d in cos_data]
    xlabels = [d[1] for d in cos_data]
    cos_vals = [d[2] for d in cos_data]
    hhi_vals = [d[2] for d in hhi_data]

    l1, = ax1.plot(xs, [c*100 for c in cos_vals], "o-", color=C_BLUE,
                   lw=2.5, ms=9, zorder=4, label="Cosine similarity (%)")
    ax1.fill_between(xs, [c*100 for c in cos_vals], 99.9,
                     alpha=0.1, color=C_BLUE)
    l2, = ax2.plot(xs, hhi_vals, "s--", color=C_RED,
                   lw=2, ms=8, zorder=4, label="HHI (risk concentration)")

    ax1.set_ylabel("Cosine similarity gradient direction (%)", fontsize=10, color=C_BLUE)
    ax2.set_ylabel("Herfindahl Index (HHI)", fontsize=10, color=C_RED)
    ax1.set_xticks(xs); ax1.set_xticklabels(xlabels, fontsize=9)
    ax1.set_xlabel("Noise configuration", fontsize=10)
    ax1.set_ylim(99.96, 100.01)
    ax2.set_ylim(0.13, 0.25)
    ax1.tick_params(axis="y", labelcolor=C_BLUE)
    ax2.tick_params(axis="y", labelcolor=C_RED)

    ax1.text(0.02, 0.06,
             "Direction preserved (cos>0.9999)\nbut concentration diluted (HHI↓)",
             transform=ax1.transAxes, fontsize=9, style="italic", color="#374151",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    ax1.set_title("Figure 6: Gradient Direction vs Risk Concentration Under Noise\n"
                  "Gradient direction (cosine similarity) is noise-invariant;\n"
                  "risk concentration (HHI) degrades with increasing gate error",
                  fontsize=10, fontweight="bold")
    plt.tight_layout()
    save(fig, "fig06_cosine_hhi.png")


# =============================================================================
# FIG 07 — Aliasing threshold on IQM hardware
# =============================================================================
def fig07_aliasing():
    hw_dir = ROOT / "results" / "hardware_validation"
    with open(hw_dir / "phase1_direct_results_latest.json") as f:
        p1 = json.load(f)
    with open(hw_dir / "wave1_v2_results_latest.json") as f:
        w1 = json.load(f)
    with open(hw_dir / "wave3_v2_results_latest.json") as f:
        w3 = json.load(f)
    with open(hw_dir / "iqae_garnet_results_latest.json") as f:
        gar = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: p_hw vs m for Emerald and Garnet
    ax = axes[0]
    m_range = np.arange(0, 7)
    depths  = 4 + 6*m_range

    # Theory
    a_ideal = 0.050442
    import math
    p_theory = [math.sin((2*m+1) * math.asin(math.sqrt(a_ideal)))**2 for m in m_range]
    ax.plot(m_range, p_theory, "k-", lw=2, label="Theory p(m)", zorder=2)

    # Emerald data
    em_data = {
        0: p1["phase1"]["grover"]["m0"]["p_hw"],
        1: p1["phase1"]["grover"]["m1"]["p_hw"],
        2: p1["phase1"]["grover"]["m2"]["p_hw"],
        3: p1["phase1"]["grover"]["m3"]["p_hw"],
        4: w3["block_E_extended"]["m4"]["p_hw"],
        5: w3["block_E_extended"]["m5"]["p_hw"],
        6: w3["block_E_extended"]["m6"]["p_hw"],
    }
    aliased_em = {k: w3["block_E_extended"].get(f"m{k}", {}).get("aliased", False)
                  for k in [4, 5, 6]}

    ms_ok  = [m for m in [0,1,2,3] if m in em_data]
    ms_bad = [m for m in [4,5,6] if m in em_data]
    ax.scatter(ms_ok,  [em_data[m] for m in ms_ok],  s=100, marker="o",
               color=C_BLUE, zorder=5, label="IQM Emerald (usable)")
    ax.scatter(ms_bad, [em_data[m] for m in ms_bad], s=100, marker="X",
               color=C_RED, zorder=5, label="IQM Emerald (aliased)")

    # Garnet data
    gar_m0 = w3["block_G_comparison"]["garnet_new"]["m0"]["p_hw"]
    gar_m1 = w3["block_G_comparison"]["garnet_new"]["m1"]["p_hw"]
    ax.scatter([0, 1], [gar_m0, gar_m1], s=130, marker="^",
               color=C_GOLD, zorder=6, label="IQM Garnet")
    ax.annotate("Garnet m=1\nerror=0.27%", xy=(1, gar_m1),
                xytext=(1.6, gar_m1+0.08), fontsize=8, color=C_GOLD, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C_GOLD, lw=1.2))

    ax.axvspan(-0.4, 2.6, alpha=0.08, color=C_GREEN)
    ax.axvspan(2.6, 6.4, alpha=0.08, color=C_RED)
    ax.axvline(2.5, color=C_RED, lw=2, ls="--", label="Aliasing onset m=3")
    ax.set_xlabel("IQAE round m", fontsize=10)
    ax.set_ylabel("Measured probability p_hw", fontsize=10)
    ax.set_title("(A) p_hw vs m: Emerald & Garnet\n"
                 "Aliasing at m=3, depth=22", fontsize=10, fontweight="bold")
    ax.set_xticks(m_range)
    ax.legend(fontsize=8)

    # Right: depth formula + fidelity projection
    ax2 = axes[1]
    depths_all = 4 + 6*m_range

    # Inferred fidelity per gate
    p_em_m1 = em_data[1]; p_th_m1 = p_theory[1]
    fid_per_gate = (p_em_m1 / p_th_m1) ** (1/10)
    fid_em = [fid_per_gate**d for d in depths_all]

    p_gar_m1 = gar_m1; fid_per_gate_gar = (p_gar_m1/p_theory[1])**(1/10)
    fid_gar = [min(fid_per_gate_gar**d, 1.0) for d in depths_all]

    ax2.bar(m_range - 0.2, fid_em,  0.35, color=C_BLUE,  alpha=0.75,
            label="IQM Emerald fidelity")
    ax2.bar(m_range + 0.2, fid_gar, 0.35, color=C_GOLD, alpha=0.75,
            label="IQM Garnet fidelity")
    ax2.axhline(0.7, color=C_RED, ls="--", lw=1.5, label="Fidelity threshold ~0.7")
    ax2.set_xlabel("IQAE round m", fontsize=10)
    ax2.set_ylabel("Estimated circuit fidelity", fontsize=10)
    ax2.set_title("(B) Projected circuit fidelity vs depth\n"
                  "Emerald threshold at m=3; Garnet viable to m=4",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks(m_range)
    ax2.legend(fontsize=8.5)

    for m, d, f in zip(m_range, depths_all, fid_em):
        ax2.text(m-0.2, f+0.01, f"d={d}", ha="center", va="bottom",
                 fontsize=7, color=C_BLUE, rotation=90)

    fig.suptitle("Figure 7: IQAE Aliasing Threshold on IQM Superconducting Hardware\n"
                 "First empirical characterisation: depth(m) = 4+6m, aliasing at m=3 (depth=22)",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig07_aliasing_hardware.png")


# =============================================================================
# FIG 08 — ZNE depth overhead
# =============================================================================
def fig08_zne_depth():
    hw_dir = ROOT / "results" / "hardware_validation"
    with open(hw_dir / "wave1_v2_results_latest.json") as f:
        w1 = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: expected vs actual depth
    ax = axes[0]
    scales = [1.0, 3.0, 5.0]
    hw_depths   = [10, 136, 226]
    exp_depths  = [10, 30, 50]

    x = np.arange(len(scales))
    w = 0.35
    ax.bar(x - w/2, exp_depths, w, color=C_BLUE,   alpha=0.8, label="Expected (scale × base)", zorder=3)
    ax.bar(x + w/2, hw_depths,  w, color=C_RED,    alpha=0.8, label="Actual (opt_level=0)",   zorder=3)

    for i, (exp, act) in enumerate(zip(exp_depths, hw_depths)):
        ax.text(i+w/2, act+3, f"{act}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_RED)
        ax.text(i-w/2, exp+3, f"{exp}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_BLUE)

    ax.annotate("", xy=(2+w/2, 226), xytext=(2+w/2+0.55, 130),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(2+w/2+0.6, 130, "4.5× overhead\n(no optimization)", fontsize=8.5,
            color=C_RED, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([f"Scale {s:.0f}×" for s in scales], fontsize=10)
    ax.set_ylabel("Circuit depth", fontsize=10)
    ax.set_title("(A) ZNE Folding Depth Overhead\nGlobal fold, opt_level=0",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    # Right: p_hw vs scale factor
    ax2 = axes[1]
    zne = w1["block_A_zne"]
    p_hw_vals = [zne["scale_results"][str(s)]["p_hw"] for s in scales]
    p_theory = zne["p_theory"]
    p_extrap  = zne["p_extrap_poly3"]

    ax2.plot(scales, p_hw_vals, "o-", color=C_RED, lw=2, ms=10,
             label="p_hw on Emerald", zorder=4)
    ax2.axhline(p_theory, color="black", lw=1.5, ls="-",
                label=f"p_theory = {p_theory:.4f}")
    ax2.axhline(p_extrap, color=C_ORANGE, lw=2, ls="--",
                label=f"ZNE extrapolation = {p_extrap:.4f}")
    ax2.scatter([0], [p_extrap], s=200, color=C_ORANGE, marker="*", zorder=5)

    ax2.text(0.15, p_extrap+0.02, f"ZNE result\n({p_extrap:.3f})", fontsize=8.5,
             color=C_ORANGE, fontweight="bold")
    ax2.text(2.5, p_theory-0.03, f"Theory\n({p_theory:.3f})", fontsize=8.5,
             ha="right", color="black")

    ax2.annotate("Raw error = 0.055\nZNE error = 0.271\nZNE WORSE",
                 xy=(0, p_extrap), xytext=(1.0, p_extrap+0.12),
                 fontsize=9, color=C_RED, fontweight="bold",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", color=C_RED, lw=1.5))

    ax2.set_xlabel("ZNE scale factor", fontsize=10)
    ax2.set_ylabel("Measured probability p_hw", fontsize=10)
    ax2.set_xticks(scales); ax2.set_xticklabels(["1× (depth=10)", "3× (depth=136)", "5× (depth=226)"])
    ax2.set_title("(B) ZNE Extrapolation Failure\nFolded circuits incoherent at depth≥136",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8.5)

    fig.suptitle("Figure 8: ZNE Circuit Folding is Counterproductive for IQAE on NISQ Hardware\n"
                 "Opt_level=0 inflates depth 4.5× beyond fold factor; extrapolation diverges",
                 fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    save(fig, "fig08_zne_depth.png")


# =============================================================================
# FIG 09 — Euler decomposition (PA, four methods)
# =============================================================================
def fig09_euler():
    euler_df = pd.read_csv(ROOT / "results" / "euler_decomposition" / "summary.csv")

    with open(ROOT / "results" / "euler_decomposition" / "A1_PA_euler.json") as f:
        eu = json.load(f)

    methods = ["initial_equal_weight", "quantum_optimal", "classical_slsqp", "classical_risk_parity"]
    labels  = ["Equal Weight\n(initial)", "Quantum CVaR\n(optimal)", "Classical SLSQP\n(optimal)", "Risk Parity"]
    colors  = [C_GREY, C_BLUE, C_GREEN, C_ORANGE]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Left: bar chart per asset per method
    ax = axes[0]
    assets = eu["decompositions"]["quantum_optimal"]["asset_names"]
    n = len(assets)
    y = np.arange(n)
    h = 0.18

    for i, (meth, lab, col) in enumerate(zip(methods, labels, colors)):
        dec = eu["decompositions"].get(meth)
        if dec is None:
            continue
        pcts = dec["rc_percentage"]
        offset = (i - 1.5) * h
        ax.barh(y + offset, pcts, h, color=col, alpha=0.8, label=lab, zorder=3)

    ax.set_yticks(y); ax.set_yticklabels(assets, fontsize=9)
    ax.set_xlabel("Risk Contribution (%)", fontsize=10)
    ax.set_title("(A) Per-asset CVaR risk contributions\nby optimization method (PA portfolio, train)",
                 fontsize=10, fontweight="bold")
    ax.axvline(10, color=C_GREY, lw=0.8, ls=":", alpha=0.5)
    ax.axvline(20, color=C_GREY, lw=0.8, ls=":", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    # Right: HHI / eff_N comparison
    ax2 = axes[1]
    all_methods = ["initial_equal_weight", "quantum_optimal", "classical_slsqp",
                   "classical_markowitz", "classical_risk_parity"]
    m_labels = ["Equal\nWeight", "Quantum\nCVaR", "SLSQP", "Markowitz", "Risk\nParity"]
    m_colors = [C_GREY, C_BLUE, C_GREEN, C_PURPLE, C_ORANGE]

    hhi_vals  = []
    cvar_vals = []
    for meth in all_methods:
        dec = eu["decompositions"].get(meth)
        if dec:
            hhi_vals.append(dec["herfindahl_index"])
            cvar_vals.append(dec["total_cvar"]*100)
        else:
            hhi_vals.append(np.nan)
            cvar_vals.append(np.nan)

    x2 = np.arange(len(all_methods))
    ax3 = ax2.twinx()

    b1 = ax2.bar(x2 - 0.2, hhi_vals, 0.35, color=m_colors, alpha=0.75, zorder=3)
    l1, = ax3.plot(x2, cvar_vals, "k-D", ms=8, lw=2, zorder=4, label="CVaR (%)")
    ax3.fill_between(x2, cvar_vals, alpha=0.08, color="black")

    ax2.set_ylabel("HHI (risk concentration)", fontsize=10)
    ax3.set_ylabel("CVaR loss (%)", fontsize=10)
    ax2.set_xticks(x2); ax2.set_xticklabels(m_labels, fontsize=9)
    ax2.set_title("(B) Risk concentration vs CVaR quality\nQuantum & SLSQP converge to same HHI",
                  fontsize=10, fontweight="bold")

    for xi, hhi, cvar in zip(x2, hhi_vals, cvar_vals):
        if not np.isnan(hhi):
            ax2.text(xi-0.2, hhi+0.003, f"{hhi:.3f}", ha="center", fontsize=7.5,
                     fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9)

    fig.suptitle("Figure 9: Euler CVaR Decomposition — Portfolio PA\n"
                 "Quantum and SLSQP converge to same risk structure (CLX+CBOE+CHRW = 74% of CVaR)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig09_euler_decomposition.png")


# =============================================================================
# FIG 10 — Cumulative return OOS
# =============================================================================
def fig10_oos_returns():
    df = pd.read_csv(ROOT / "data" / "returns_sp500_full.csv", index_col=0, parse_dates=True)
    d = load("A1_PA")
    oos = d["metrics"]["oos_validation"]
    oos_start = pd.Timestamp(oos["split_date"])

    PA = ["AAL","AAP","CBOE","CHRW","CLX","DXCM","FMC","NEM","NFLX","OXY"]

    portfolios = {
        "Quantum CVaR": (np.array(oos["portfolios"]["Quantum CVaR"]["weights"]), C_BLUE, "-", 2.5),
        "Classical CVaR": (np.array(oos["portfolios"]["Classical CVaR"]["weights"]), C_GREEN, "--", 2),
        "Inverse Vol": (np.array(oos["portfolios"]["Inverse Vol"]["weights"]), C_ORANGE, "-.", 1.5),
        "Equal Weight": (np.array(oos["portfolios"]["Equal Weight"]["weights"]), C_GREY, ":", 1.5),
    }

    ret_oos = df[PA][df[PA].index >= oos_start].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    for name, (w, col, ls, lw) in portfolios.items():
        r = ret_oos.values @ w
        wealth = 100 * np.cumprod(1 + r)
        m = oos["portfolios"][name]["oos_metrics"]
        lbl = (f"{name}\n"
               f"Sh={m['sharpe']:.2f}  CVaR={m['cvar_loss']*100:.2f}%")
        ax.plot(ret_oos.index, wealth, color=col, ls=ls, lw=lw, label=lbl)

    ax.axhline(100, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_ylabel("Portfolio value (base=100)", fontsize=10)
    ax.set_title("(A) Cumulative wealth — OOS (2024–2025)\nPortfolio PA (QUBO peripheral)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, loc="upper left")

    # Right: return distribution comparison
    ax2 = axes[1]
    alpha = 0.05
    for name, (w, col, ls, lw) in portfolios.items():
        r = ret_oos.values @ w
        ax2.hist(r*100, bins=40, color=col, alpha=0.35, density=True, label=name)
        var = np.percentile(r, alpha*100)*100
        ax2.axvline(var, color=col, ls="--", lw=1.5, alpha=0.8)

    ax2.set_xlabel("Daily return (%)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("(B) OOS return distributions\nVaR(5%) marked per method",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    fig.suptitle("Figure 10: Out-of-Sample Financial Performance (501 days, Jan 2024–Dec 2025)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig10_oos_returns.png")


# =============================================================================
# FIG 11 — VaR breach chart + Acerbi-Székely
# =============================================================================
def fig11_backtest():
    df = pd.read_csv(ROOT / "data" / "returns_sp500_full.csv", index_col=0, parse_dates=True)
    d  = load("A1_PA")
    with open(ROOT / "results" / "exp_A1_PA" / "oos_backtest_report_A1_PA.json") as f:
        bt = json.load(f)

    PA = ["AAL","AAP","CBOE","CHRW","CLX","DXCM","FMC","NEM","NFLX","OXY"]
    oos_start = pd.Timestamp(d["metrics"]["oos_validation"]["split_date"])
    w_q = np.array(d["metrics"]["oos_validation"]["portfolios"]["Quantum CVaR"]["weights"])

    ret_oos = df[PA][df[PA].index >= oos_start].dropna()
    port_r  = ret_oos.values @ w_q

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Top-left: daily returns + VaR breach chart
    ax = axes[0][0]
    dates = ret_oos.index
    alpha05 = bt["alpha_0.05"]
    var_pred = alpha05["var_predicted"]

    colors_r = [C_RED if -r > var_pred else "#93C5FD" for r in port_r]
    ax.bar(dates, port_r*100, color=colors_r, width=1, alpha=0.8, zorder=3)
    ax.axhline(-var_pred*100, color=C_RED, lw=2, ls="--",
               label=f"Predicted VaR(5%) = {var_pred*100:.2f}%")

    breach_dates = dates[port_r <= -var_pred]
    ax.scatter(breach_dates, [port_r[i]*100 for i in range(len(port_r)) if port_r[i] <= -var_pred],
               s=50, color="darkred", zorder=5)

    ax.set_ylabel("Daily return (%)", fontsize=9)
    ax.set_title(f"(A) VaR Breach Chart — α=5%\n"
                 f"{alpha05['n_exceedances']} exceedances / 501 ({alpha05['exceedance_rate']*100:.2f}%) | "
                 f"Traffic: {alpha05['traffic_light']}",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.text(0.98, 0.02, f"Expected: {int(501*0.05)} | Observed: {alpha05['n_exceedances']}",
            transform=ax.transAxes, ha="right", fontsize=8.5,
            bbox=dict(boxstyle="round", facecolor="lightyellow"))

    # Top-right: ES predicted vs realized across alpha levels
    ax2 = axes[0][1]
    alpha_levels = [0.01, 0.025, 0.05, 0.10]
    alpha_keys   = ["alpha_0.01", "alpha_0.025", "alpha_0.05", "alpha_0.1"]
    es_pred = [bt[k]['es_predicted']*100 for k in alpha_keys]
    es_real = [bt[k]['es_realized']*100  for k in alpha_keys]
    tls     = [bt[k]['traffic_light']    for k in alpha_keys]
    tl_col  = {"GREEN": C_GREEN, "YELLOW": C_GOLD, "RED": C_RED}

    x3 = np.arange(len(alpha_levels))
    ax2.bar(x3 - 0.2, es_pred, 0.35, color=C_RED,   alpha=0.7, label="Predicted ES")
    ax2.bar(x3 + 0.2, es_real, 0.35, color=C_BLUE,  alpha=0.7, label="Realized ES")
    for i, (tl, pred, real) in enumerate(zip(tls, es_pred, es_real)):
        ax2.text(i, max(pred, real)+0.05, tl[0],
                 ha="center", fontsize=11, fontweight="bold", color=tl_col.get(tl, "black"))

    ax2.set_xticks(x3)
    ax2.set_xticklabels([f"α={a}" for a in alpha_levels], fontsize=9)
    ax2.set_ylabel("Expected Shortfall (%)", fontsize=9)
    ax2.set_title("(B) ES Predicted vs Realized\nG=Green, Y=Yellow, R=Red (Basel traffic light)",
                  fontsize=9, fontweight="bold")
    ax2.legend(fontsize=8.5)

    # Bottom-left: Kupiec / Christoffersen / Acerbi summary table
    ax3 = axes[1][0]
    ax3.axis("off")
    rows = []
    for a, key in zip(alpha_levels, alpha_keys):
        b   = bt[key]
        kup = "PASS" if b["kupiec"]["pass"] else "FAIL"
        chr = "PASS" if b["christoffersen"]["pass_cc"] else "FAIL"
        ace = "PASS" if b["acerbi_szekely"]["pass"] else "FAIL"
        rows.append([f"α={a}", str(b["n_exceedances"]),
                     f"{b['exceedance_rate']*100:.2f}%",
                     kup, chr, ace, b["traffic_light"]])

    col_labels = ["α", "Exceed", "Rate", "Kupiec", "Christoffersen\nCC", "Acerbi-\nSzékely", "Traffic"]
    table = ax3.table(cellText=rows, colLabels=col_labels,
                      loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#1E3A5F")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 5:  # Acerbi column
            txt = cell.get_text().get_text()
            cell.set_facecolor(C_GREEN if txt == "PASS" else C_RED)
            cell.set_text_props(color="white", fontweight="bold")
    ax3.set_title("(C) Regulatory Test Summary (Basel IV)\nAcerbi-Székely PASS all levels",
                  fontsize=9, fontweight="bold")

    # Bottom-right: Acerbi Z1 detail
    ax4 = axes[1][1]
    z1_vals = [bt[k]['acerbi_szekely']['z1']       for k in alpha_keys]
    pvals   = [bt[k]['acerbi_szekely']['pvalue_z1'] for k in alpha_keys]
    bar_colors = [C_GREEN if p > 0.05 else C_RED for p in pvals]

    bars = ax4.bar([f"α={a}" for a in alpha_levels], z1_vals, color=bar_colors, alpha=0.8, zorder=3)
    ax4.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax4.axhline(-0.5, color=C_GREY, lw=1, ls=":", alpha=0.7, label="Expected Z1 ≈ −0.5")
    for bar, p in zip(bars, pvals):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() - 0.04 if bar.get_height() < -0.1 else bar.get_height() + 0.02,
                 f"p={p:.3f}", ha="center", va="top", fontsize=8.5, fontweight="bold",
                 color="white" if bar.get_height() < -0.2 else "black")

    ax4.set_ylabel("Acerbi-Székely Z1 statistic", fontsize=9)
    ax4.set_title("(D) Acerbi-Székely ES Test (Z1)\n"
                  "Z1 ~ −0.5 under H₀ (correct ES). p>0.05 → PASS",
                  fontsize=9, fontweight="bold")
    ax4.legend(fontsize=8.5)
    ax4.set_ylim(-1.1, 0.2)

    fig.suptitle("Figure 11: Regulatory Backtesting — Quantum CVaR Portfolio PA (Basel IV)\n"
                 "Kupiec fails (conservative model); Acerbi-Székely PASS at all α levels",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig11_backtest_regulatory.png")


# =============================================================================
# FIG 12 — Scalability
# =============================================================================
def fig12_scalability():
    scale_data = []
    for exp, N in [("A1_PA", 10), ("S1_PA_30", 30), ("S1_PA_100", 100)]:
        d = load(exp)
        iters = d["metrics"]["optimization"]["num_iterations"]
        impr  = d["metrics"]["optimization"]["cvar_improvement_pct"]
        t     = d["timings"]["optimization_s"]
        q     = d["metrics"]["quantum"]["total_oracle_queries"]["iqae"]
        q_oos = d["metrics"]["oos_validation"]["portfolios"]["Quantum CVaR"]["oos_metrics"]
        nq    = d["metadata"]["configuration"]["phase2"]["num_qubits"]
        scale_data.append((N, nq, iters, impr, t, q, q_oos["sharpe"], q_oos["cvar_loss"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    Ns    = [x[0] for x in scale_data]
    imprs = [x[3] for x in scale_data]
    times = [x[4]/3600 for x in scale_data]
    nq    = [x[1] for x in scale_data]
    sharpes = [x[6] for x in scale_data]

    # Left: CVaR improvement vs N (bubble = Sharpe)
    ax = axes[0]
    sizes = [s*300 for s in sharpes]
    scatter = ax.scatter(Ns, imprs, s=sizes, c=Ns, cmap="Blues",
                         edgecolors=C_BLUE, linewidths=2, zorder=4, alpha=0.85)
    for N, impr, nq_val, sh in zip(Ns, imprs, nq, sharpes):
        ax.annotate(f"N={N}\n{nq_val} qubits\nSh={sh:.2f}",
                    (N, impr), textcoords="offset points",
                    xytext=(10, -15), fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    ax.set_xlabel("Portfolio size N (assets)", fontsize=10)
    ax.set_ylabel("CVaR improvement over equal weight (%)", fontsize=10)
    ax.set_title("(A) CVaR improvement vs N\nbubble size ∝ OOS Sharpe ratio",
                 fontsize=10, fontweight="bold")
    ax.set_xticks(Ns)
    ax.text(0.05, 0.95, "4 qubits in all cases",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=C_PURPLE, va="top",
            bbox=dict(boxstyle="round", facecolor="lavender", alpha=0.9))

    # Middle: wall-clock time vs N (log-log)
    ax2 = axes[1]
    ax2.loglog(Ns, times, "o-", color=C_ORANGE, lw=2.5, ms=10, zorder=4)
    for N, t in zip(Ns, times):
        ax2.annotate(f"{t:.1f}h", (N, t),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Fit line
    slope = np.polyfit(np.log(Ns), np.log(times), 1)
    N_fit = np.logspace(np.log10(8), np.log10(120), 50)
    t_fit = np.exp(slope[1]) * N_fit**slope[0]
    ax2.loglog(N_fit, t_fit, "--", color=C_GREY, lw=1.5,
               label=f"O(N^{slope[0]:.2f}) fit")
    ax2.set_xlabel("Portfolio size N (log scale)", fontsize=10)
    ax2.set_ylabel("Wall-clock time (hours, log scale)", fontsize=10)
    ax2.set_title(f"(B) Compute time scales O(N^{slope[0]:.2f})\nnearly linear in N",
                  fontsize=10, fontweight="bold")
    ax2.set_xticks(Ns); ax2.set_xticklabels(Ns)
    ax2.legend(fontsize=8.5)

    # Right: qubits fixed illustration
    ax3 = axes[2]
    ax3.axis("off")
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)
    ax3.set_title("(C) Qubit count is independent of N\n"
                  "4 qubits encode the loss distribution, not the assets",
                  fontsize=10, fontweight="bold")

    for i, (N, col) in enumerate([(10, C_BLUE), (30, C_GREEN), (100, C_ORANGE)]):
        y = 7.5 - i*2.8
        # N boxes
        n_show = min(N, 8)
        for j in range(n_show):
            rect = mpatches.FancyBboxPatch((0.1 + j*0.55, y-0.3), 0.45, 0.6,
                                            boxstyle="round,pad=0.05",
                                            facecolor=col, alpha=0.3, edgecolor=col)
            ax3.add_patch(rect)
        if N > 8:
            ax3.text(0.1 + 8*0.55, y, "...", fontsize=12, color=col, va="center")
        ax3.text(0.3, y+0.5, f"N={N} assets", fontsize=9, color=col, fontweight="bold")

        # Arrow
        ax3.annotate("", xy=(6.5, y), xytext=(5.2, y),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        # 4 qubit boxes (always the same)
        for j in range(4):
            rect = mpatches.FancyBboxPatch((6.7 + j*0.65, y-0.3), 0.55, 0.6,
                                            boxstyle="round,pad=0.05",
                                            facecolor=C_PURPLE, alpha=0.5, edgecolor=C_PURPLE)
            ax3.add_patch(rect)
            ax3.text(6.7 + j*0.65 + 0.27, y, f"q{j}", ha="center", va="center",
                     fontsize=7.5, color="white", fontweight="bold")
        ax3.text(7.2, y+0.5, "4 qubits", fontsize=9, color=C_PURPLE, fontweight="bold")

    fig.suptitle("Figure 12: Scalability — 4 Qubits Optimize N∈{10,30,100} Assets\n"
                 "Circuit width independent of N; compute time scales O(N^0.93)",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, "fig12_scalability.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generating all 12 paper figures...")
    print(f"Output: {OUTDIR}\n")

    tasks = [
        ("01 — Pipeline diagram",           fig01_pipeline),
        ("02 — QUBO network",               fig02_qubo_network),
        ("03 — Correlation matrices",       fig03_correlation_matrices),
        ("04 — Circuit depth",              fig04_circuit_depth),
        ("05 — Convergence curves",         fig05_convergence),
        ("06 — Cosine sim + HHI",           fig06_cosine_hhi),
        ("07 — Aliasing threshold",         fig07_aliasing),
        ("08 — ZNE depth overhead",         fig08_zne_depth),
        ("09 — Euler decomposition",        fig09_euler),
        ("10 — OOS cumulative return",      fig10_oos_returns),
        ("11 — Backtest regulatory",        fig11_backtest),
        ("12 — Scalability",               fig12_scalability),
    ]

    failed = []
    for label, fn in tasks:
        print(f"  Fig {label}...")
        try:
            fn()
        except Exception as e:
            print(f"    ERROR: {e}")
            failed.append((label, str(e)))

    print(f"\n{'='*50}")
    print(f"Done: {len(tasks)-len(failed)}/{len(tasks)} figures generated")
    if failed:
        print("Failed:")
        for lbl, err in failed:
            print(f"  {lbl}: {err}")
    print(f"Output directory: {OUTDIR}")
