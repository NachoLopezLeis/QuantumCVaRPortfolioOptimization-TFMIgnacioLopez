#!/usr/bin/env python3
# =============================================================================
# Project : Hybrid Quantum-Classical CVaR Portfolio Optimization — TFM
# Author  : Ignacio Lopez Leis — Universidad Autonoma de Madrid
# Script  : generate_visual_figures.py
#
# Generates two publication-quality figures (300 dpi) that match the
# interactive widget designs:
#
#   fig01_pipeline.png  — 4-stage pipeline with in-block illustrations
#   fig02_sp500_network.png — real S&P500 MST with 374 nodes, communities,
#                             PA (peripheral) and PB (central) highlighted
#
# Output : results/paper_figures/
# Usage  : python3 scripts/generate_visual_figures.py
# =============================================================================

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── project paths ─────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "paper_figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
})

# palette
C = dict(
    purple="#6D28D9", lpurple="#DDD6FE",
    blue  ="#1D4ED8", lblue ="#BFDBFE",
    green ="#15803D", lgreen="#BBF7D0",
    amber ="#B45309", lamber="#FDE68A",
    grey  ="#6B7280", lgrey ="#F3F4F6",
    text  ="#1F2937",
    red   ="#B91C1C",
)

def save(fig: plt.Figure, name: str) -> None:
    path = OUTDIR / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved → {path.name}")


# =============================================================================
# FIG 01 — Pipeline with in-block visual illustrations
# =============================================================================
def _draw_block(ax, x, y, w, h, color, lcolor, title, subtitle):
    """Rounded block with header bar."""
    body = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                          facecolor=lcolor, edgecolor=color, linewidth=1.8, zorder=2)
    ax.add_patch(body)
    hdr = FancyBboxPatch((x, y + h - 0.12), w, 0.12, boxstyle="round,pad=0.01",
                         facecolor=color, edgecolor=color, linewidth=0, zorder=3)
    ax.add_patch(hdr)
    ax.text(x + w / 2, y + h - 0.06, title,
            ha="center", va="center", fontsize=11, fontweight="bold",
            color="white", zorder=4)
    ax.text(x + w / 2, y + h - 0.20, subtitle,
            ha="center", va="center", fontsize=8.5, color=color,
            style="italic", zorder=4)


def _arrow(ax, x0, y0, x1, y1, label="", color=C["grey"]):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->,head_width=0.08,head_length=0.06",
                                color=color, lw=1.8))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx, my + 0.06, label, ha="center", fontsize=8,
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, alpha=0.92, linewidth=0.8), zorder=6)


def fig01_pipeline() -> None:
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16); ax.set_ylim(0, 7)
    ax.axis("off")

    BW, BH = 3.4, 5.2
    BXS = [0.25, 4.1, 7.95, 11.8]
    BY = 1.2

    titles   = ["QUBO",        "IQAE",            "Adam",           "OOS"]
    subs     = ["D-Wave hybrid", "IQM 4-qubit",   "Hybrid optimizer", "Basel IV · 501 days"]
    colors   = [C["purple"], C["blue"],  C["green"],  C["amber"]]
    lcolors  = [C["lpurple"], C["lblue"], C["lgreen"], C["lamber"]]

    for i, (bx, t, s, col, lc) in enumerate(zip(BXS, titles, subs, colors, lcolors)):
        _draw_block(ax, bx, BY, BW, BH, col, lc, t, s)

    # ── BLOCK 0: mini network ─────────────────────────────────────────────────
    np.random.seed(7)
    bx0 = BXS[0] + BW / 2
    base_y = BY + 2.4
    outer_r = 0.72
    n_outer = 8
    for i in range(n_outer):
        ang = 2 * np.pi * i / n_outer + 0.3
        nx_ = bx0 + outer_r * np.cos(ang)
        ny_ = base_y + outer_r * np.sin(ang) * 0.72
        ax.plot(nx_, ny_, "o", ms=6, color=C["lpurple"], markeredgecolor=C["purple"],
                markeredgewidth=0.8, zorder=5)
    inner_pts = [(bx0, base_y), (bx0 - 0.22, base_y + 0.18),
                 (bx0 + 0.20, base_y + 0.15), (bx0, base_y - 0.20)]
    for ip in inner_pts:
        ax.plot(*ip, "o", ms=9, color=C["purple"], zorder=6)
    for i, ip in enumerate(inner_pts):
        for j, jp in enumerate(inner_pts):
            if i < j:
                ax.plot([ip[0], jp[0]], [ip[1], jp[1]], color=C["purple"],
                        lw=0.9, alpha=0.5, zorder=4)
    for i in range(n_outer):
        ang = 2 * np.pi * i / n_outer + 0.3
        nx_ = bx0 + outer_r * np.cos(ang)
        ny_ = base_y + outer_r * np.sin(ang) * 0.72
        closest = min(inner_pts, key=lambda p: (p[0]-nx_)**2 + (p[1]-ny_)**2)
        ax.plot([nx_, closest[0]], [ny_, closest[1]], color=C["lpurple"],
                lw=0.7, alpha=0.6, zorder=3)
    # PA selected (green rings)
    pa_angs = [0.3, 1.8, 3.5, 5.1]
    for ang in pa_angs:
        nx_ = bx0 + outer_r * np.cos(ang)
        ny_ = base_y + outer_r * np.sin(ang) * 0.72
        ax.plot(nx_, ny_, "o", ms=11, markerfacecolor="none",
                markeredgecolor=C["green"], markeredgewidth=2.2, zorder=7)
    # text annotations
    for txt, ty in [("374 → 10 assets", -0.95), ("peripherality 0.98", -1.22),
                    ("10/11 communities", -1.49), ("energy = −54,998", -1.76)]:
        ax.text(bx0, base_y + ty, txt, ha="center", fontsize=8.5,
                color=C["text"], zorder=5)

    # ── BLOCK 1: quantum circuit ──────────────────────────────────────────────
    bx1 = BXS[1]
    cx = bx1 + BW / 2
    wire_y = [BY + 3.8, BY + 3.1, BY + 2.4, BY + 1.7]
    wire_x0, wire_x1 = bx1 + 0.22, bx1 + BW - 0.22
    for wy in wire_y:
        ax.plot([wire_x0, wire_x1], [wy, wy], color=C["lblue"], lw=1.2, zorder=3)
    for i, (wy, lbl) in enumerate(zip(wire_y, ["q₀", "q₁", "q₂", "q₃"])):
        ax.text(wire_x0 - 0.05, wy, lbl, ha="right", va="center",
                fontsize=9, color=C["blue"])
        hg = FancyBboxPatch((wire_x0 + 0.05, wy - 0.12), 0.28, 0.24,
                            boxstyle="round,pad=0.02",
                            facecolor=C["lblue"], edgecolor=C["blue"], linewidth=0.8, zorder=5)
        ax.add_patch(hg)
        ax.text(wire_x0 + 0.19, wy, "H", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=C["blue"], zorder=6)
    oracle = FancyBboxPatch((bx1 + 0.62, wire_y[-1] - 0.14), 0.38,
                            wire_y[0] - wire_y[-1] + 0.28,
                            boxstyle="round,pad=0.02",
                            facecolor=C["blue"], edgecolor=C["blue"],
                            linewidth=0, zorder=5, alpha=0.85)
    ax.add_patch(oracle)
    ax.text(bx1 + 0.81, BY + 2.75, "Oracle", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color="white",
            rotation=90, zorder=6)
    # IQAE round arc
    arc_cx = bx1 + 1.5
    arc_top = wire_y[0] + 0.45
    theta = np.linspace(np.pi * 0.15, np.pi * 0.85, 40)
    ax.plot(arc_cx + 0.42 * np.cos(theta), arc_top - 0.22 + 0.28 * np.sin(theta),
            color=C["blue"], lw=1.5, zorder=5)
    ax.annotate("", xy=(arc_cx + 0.42 * np.cos(np.pi * 0.85),
                        arc_top - 0.22 + 0.28 * np.sin(np.pi * 0.85)),
                xytext=(arc_cx + 0.42 * np.cos(np.pi * 0.82),
                        arc_top - 0.22 + 0.28 * np.sin(np.pi * 0.82)),
                arrowprops=dict(arrowstyle="->,head_width=0.05", color=C["blue"], lw=1.2))
    ax.text(arc_cx, arc_top + 0.12, "Qᵐ", ha="center", fontsize=9,
            color=C["blue"], fontweight="bold")
    for txt, ty in [("4 qubits — fixed ∀N", -0.92), ("depth(m) = 4 + 6m", -1.20),
                    ("ε = 0.005", -1.48), ("aliasing at m=3", -1.76)]:
        ax.text(cx, BY + 2.0 + ty + 0.92, txt, ha="center", fontsize=8.5,
                color=C["text"], zorder=5)

    # ── BLOCK 2: convergence curve ────────────────────────────────────────────
    bx2 = BXS[2]
    cx2 = bx2 + BW / 2
    ax_in = ax.inset_axes([bx2 / 16 + 0.01, (BY + 1.55) / 7, (BW - 0.3) / 16, 1.55 / 7])
    iters = np.linspace(0, 38, 80)
    cvar_curve = 0.0371 - (0.0371 - 0.0277) * (1 - np.exp(-iters / 8))
    ax_in.plot(iters, cvar_curve * 100, color=C["green"], lw=2.2)
    ax_in.axhline(3.71, color=C["grey"], lw=0.8, ls="--", alpha=0.5)
    ax_in.axhline(2.77, color=C["green"], lw=0.8, ls="--", alpha=0.7)
    ax_in.fill_between(iters, cvar_curve * 100, 2.77, alpha=0.12, color=C["green"])
    ax_in.set_xlabel("iteration", fontsize=7.5, labelpad=1)
    ax_in.set_ylabel("CVaR %", fontsize=7.5, labelpad=1)
    ax_in.tick_params(labelsize=7)
    ax_in.set_ylim(2.5, 4.0)
    ax_in.text(19, 3.86, "initial 3.72%", ha="center", fontsize=7, color=C["grey"])
    ax_in.text(35, 2.84, "2.77%", ha="left", fontsize=7, color=C["green"])
    ax_in.text(14, 3.24, "−25.6%", ha="center", fontsize=8,
               color=C["green"], fontweight="bold")
    ax_in.spines["top"].set_visible(False)
    ax_in.spines["right"].set_visible(False)
    for txt, ty in [("38 iterations", BY + 2.62), ("100% converged", BY + 2.35),
                    ("N = 10 / 30 / 100", BY + 2.08), ("time ∝ O(N⁰·⁹³)", BY + 1.81)]:
        ax.text(cx2, ty, txt, ha="center", fontsize=8.5, color=C["text"], zorder=5)

    # ── BLOCK 3: breach chart + traffic lights ────────────────────────────────
    bx3 = BXS[3]
    cx3 = bx3 + BW / 2
    ax_b = ax.inset_axes([bx3 / 16 + 0.01, (BY + 2.6) / 7, (BW - 0.3) / 16, 1.35 / 7])
    np.random.seed(42)
    n_days = 80
    rets = np.random.normal(0.0005, 0.009, n_days)
    var_line = 0.016
    breach_idx = np.where(-rets > var_line)[0]
    colors_bar = [C["red"] if -r > var_line else "#93C5FD" for r in rets]
    ax_b.bar(range(n_days), rets * 100, color=colors_bar, width=1.0, alpha=0.85)
    ax_b.axhline(-var_line * 100, color=C["red"], lw=1.5, ls="--")
    ax_b.set_xlim(-1, n_days); ax_b.set_ylim(-3.5, 2.5)
    ax_b.tick_params(labelsize=6.5)
    ax_b.set_xlabel("OOS days", fontsize=7, labelpad=1)
    ax_b.set_ylabel("return %", fontsize=7, labelpad=1)
    ax_b.text(1, -1.7, "VaR 5%", fontsize=6.5, color=C["red"])
    ax_b.text(60, -3.2, f"{len(breach_idx)} breaches / 501 days", fontsize=6.5,
              color=C["red"], fontweight="bold")
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    # traffic light dots
    tl_y = BY + 2.48
    for j, (col, lbl) in enumerate([(C["green"], "α=1%"),
                                     ("#B45309", "α=5%"),
                                     (C["red"],  "α=10%")]):
        tx = bx3 + 0.55 + j * 1.0
        circle = plt.Circle((tx, tl_y), 0.22, color=col, zorder=6)
        ax.add_patch(circle)
        ax.text(tx, tl_y, lbl, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", zorder=7)
    # Acerbi badge
    badge = FancyBboxPatch((bx3 + 0.18, BY + 1.9), BW - 0.36, 0.34,
                           boxstyle="round,pad=0.03",
                           facecolor=C["lgreen"], edgecolor=C["green"], linewidth=1.2, zorder=5)
    ax.add_patch(badge)
    ax.text(cx3, BY + 2.07, "Acerbi-Székely  PASS  all α",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=C["green"], zorder=6)
    for txt, ty in [("CVaR OOS = 1.84%", BY + 1.70), ("Sharpe = 0.68 (PA)",   BY + 1.43),
                    ("Sharpe = 1.10 (N=100)", BY + 1.16)]:
        ax.text(cx3, ty, txt, ha="center", fontsize=8.5, color=C["text"], zorder=5)

    # ── arrows between blocks ─────────────────────────────────────────────────
    mid_y = BY + BH / 2
    _arrow(ax, BXS[0] + BW, mid_y, BXS[1],       mid_y, "tickers", C["purple"])
    _arrow(ax, BXS[1] + BW, mid_y, BXS[2],       mid_y, "∇CVaR",  C["blue"])
    _arrow(ax, BXS[2] + BW, mid_y, BXS[3],       mid_y, "w*",     C["green"])

    # feedback loop arc
    loop_y = BY - 0.22
    for bx in BXS[1:3]:
        ax.annotate("", xy=(bx, loop_y + 0.28), xytext=(bx + BW, loop_y + 0.28),
                    arrowprops=dict(arrowstyle="<-,head_width=0.06",
                                   color=C["green"], lw=1.2, ls="dashed",
                                   connectionstyle="arc3,rad=0.32"))
    ax.text((BXS[1] + BXS[2] + BW) / 2, loop_y + 0.02,
            "Adam iteration", ha="center", fontsize=8, color=C["green"], style="italic")

    # ── hardware badges at bottom ─────────────────────────────────────────────
    hw_labels = ["D-Wave hybrid", "IQM Emerald + Garnet", "Classical simulator", "2024–2025 live data"]
    for bx, col, lbl in zip(BXS, colors, hw_labels):
        badge_b = FancyBboxPatch((bx, 0.12), BW, 0.62, boxstyle="round,pad=0.04",
                                 facecolor=col, edgecolor=col, linewidth=0, zorder=4, alpha=0.85)
        ax.add_patch(badge_b)
        ax.text(bx + BW / 2, 0.43, lbl, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=5)

    ax.set_title("Hybrid Quantum-Classical CVaR Portfolio Optimization Pipeline",
                 fontsize=14, fontweight="bold", pad=10, color=C["text"])
    save(fig, "fig01_pipeline.png")


# =============================================================================
# FIG 02 — S&P500 MST Network
# =============================================================================
def _build_network_data() -> dict:
    """Compute MST layout from returns data. Returns dict with positions, communities, etc."""
    import networkx as nx
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.sparse import csr_matrix
    from networkx.algorithms.community import greedy_modularity_communities

    df = pd.read_csv(ROOT / "data" / "returns_sp500_full.csv",
                     index_col=0, parse_dates=True)
    train = df[df.index < "2024-01-01"].dropna(axis=1)
    tickers = list(train.columns)

    corr = train.corr().values
    dist = np.sqrt(2 * (1 - corr))
    mst  = minimum_spanning_tree(csr_matrix(dist))
    arr  = mst.toarray()

    G = nx.Graph()
    G.add_nodes_from(range(len(tickers)))
    edges = []
    for i, j in zip(*np.nonzero(arr)):
        if i < j:
            G.add_edge(i, j, weight=float(arr[i, j]))
            edges.append((i, j))

    print("  Computing Kamada-Kawai layout (this may take ~30 s)...")
    pos = nx.kamada_kawai_layout(G, weight="weight")
    pos_arr = np.array([pos[i] for i in range(len(tickers))])
    pos_arr -= pos_arr.min(axis=0)
    pos_arr /= pos_arr.max(axis=0)

    print("  Detecting communities...")
    communities = list(greedy_modularity_communities(G))
    communities.sort(key=len, reverse=True)
    comm = np.zeros(len(tickers), dtype=int)
    for ci, c in enumerate(communities[:12]):
        for node in c:
            comm[int(node)] = ci

    degrees = np.array([G.degree(i) for i in range(len(tickers))])

    return dict(tickers=tickers, pos=pos_arr, comm=comm,
                degrees=degrees, edges=edges)


def fig02_sp500_network() -> None:
    print("  Building network (loading returns + MST)...")
    nd = _build_network_data()
    tickers = nd["tickers"]
    pos     = nd["pos"]
    comm    = nd["comm"]
    degrees = nd["degrees"]
    edges   = nd["edges"]

    PA = ["AAL", "CLX", "DPZ", "DVA", "FANG", "FMC", "MKTX", "NEM", "NFLX", "NOC"]
    PB = ["AIG", "AMP", "BAC", "BRK-B", "ITW", "IVZ", "L", "MCO", "PRU", "TROW"]
    pa_set = set(PA)
    pb_set = set(PB)

    # 12 community colours (colour-blind-friendly palette)
    COMM_COLS = [
        "#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66",
        "#77BEDB", "#E0785C", "#5EAD8A", "#D4A04F", "#7EA1CC",
        "#9E7EC4", "#8CB576",
    ]

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    # canvas margin
    M = 0.06
    ax.set_xlim(-M, 1 + M)
    ax.set_ylim(-M - 0.12, 1 + M)  # extra bottom for legend

    # draw edges
    for i, j in edges:
        xi, yi = pos[i]
        xj, yj = pos[j]
        ti, tj = tickers[i], tickers[j]
        # highlight edges connecting PA or PB nodes
        if ti in pa_set or tj in pa_set:
            ecol, ew, ea = "#16A34A", 1.0, 0.55
        elif ti in pb_set or tj in pb_set:
            ecol, ew, ea = "#DC2626", 0.8, 0.40
        else:
            ecol, ew, ea = "#94A3B8", 0.4, 0.30
        ax.plot([xi, xj], [yi, yj], color=ecol, lw=ew, alpha=ea, zorder=1)

    # draw regular nodes
    for i, t in enumerate(tickers):
        if t in pa_set or t in pb_set:
            continue
        xi, yi = pos[i]
        deg     = degrees[i]
        r       = 18 + min(deg, 8) * 8         # point size
        col     = COMM_COLS[comm[i] % len(COMM_COLS)]
        ax.scatter(xi, yi, s=r, c=col, alpha=0.70, linewidths=0.3,
                   edgecolors=col, zorder=2)

    # draw PA nodes — peripheral, bright green
    for t in PA:
        if t not in tickers:
            continue
        i = tickers.index(t)
        xi, yi = pos[i]
        ax.scatter(xi, yi, s=260, c="#22C55E", alpha=1.0, linewidths=2.0,
                   edgecolors="#15803D", zorder=5)
        ax.text(xi, yi, t if len(t) <= 4 else t[:4],
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                color="white", zorder=6)
        # label offset
        dy = 0.038 if yi < 0.85 else -0.038
        ax.text(xi, yi + dy, t, ha="center", va="center",
                fontsize=7.5, color="#15803D", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="#15803D", alpha=0.85, linewidth=0.8))

    # draw PB nodes — central, red
    for t in PB:
        if t not in tickers:
            continue
        i = tickers.index(t)
        xi, yi = pos[i]
        ax.scatter(xi, yi, s=220, c="#F87171", alpha=1.0, linewidths=2.0,
                   edgecolors="#B91C1C", zorder=5)
        ax.text(xi, yi, t if len(t) <= 4 else t[:4],
                ha="center", va="center", fontsize=6.5, fontweight="bold",
                color="white", zorder=6)
        dy = 0.038 if yi < 0.85 else -0.038
        ax.text(xi, yi + dy, t, ha="center", va="center",
                fontsize=7.5, color="#B91C1C", fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="#B91C1C", alpha=0.85, linewidth=0.8))

    # ── legend ────────────────────────────────────────────────────────────────
    legend_y = -0.10
    # community swatches
    for ci in range(12):
        col = COMM_COLS[ci]
        lx  = 0.01 + ci * (1.0 / 13)
        ax.scatter(lx, legend_y, s=50, c=col, alpha=0.85,
                   linewidths=0.5, edgecolors=col, zorder=8,
                   transform=ax.transData)
        ax.text(lx, legend_y - 0.038, f"C{ci+1}",
                ha="center", fontsize=6.5, color="#6B7280", zorder=8)

    # PA / PB legend
    ax.scatter(0.72, legend_y, s=140, c="#22C55E", linewidths=1.5,
               edgecolors="#15803D", zorder=8)
    ax.text(0.745, legend_y, "PA — QUBO peripheral (avg peri = 0.98)",
            va="center", fontsize=8.5, color="#15803D", fontweight="bold", zorder=8)
    ax.scatter(0.72, legend_y - 0.06, s=120, c="#F87171", linewidths=1.5,
               edgecolors="#B91C1C", zorder=8)
    ax.text(0.745, legend_y - 0.06, "PB — Central ranking (avg peri = 0.47)",
            va="center", fontsize=8.5, color="#B91C1C", fontweight="bold", zorder=8)

    ax.set_title(
        "S&P 500 Minimum Spanning Tree  ·  374 assets  ·  373 edges  ·  12 communities\n"
        "Kamada-Kawai layout by correlation distance  $d_{ij}=\\sqrt{2(1-\\rho_{ij})}$",
        fontsize=12, fontweight="bold", pad=10, color=C["text"]
    )
    save(fig, "fig02_sp500_network.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Generating enhanced visual figures...")
    print(f"Output: {OUTDIR}\n")

    tasks = [
        ("01 — Pipeline (illustrated blocks)", fig01_pipeline),
        ("02 — S&P500 MST network",            fig02_sp500_network),
    ]

    failed = []
    for label, fn in tasks:
        print(f"  Fig {label}...")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"    ERROR: {e}")
            traceback.print_exc()
            failed.append(label)

    print(f"\n{'='*50}")
    print(f"Done: {len(tasks)-len(failed)}/{len(tasks)} figures generated.")
    if failed:
        print("Failed:", failed)
    print(f"Output: {OUTDIR}")
