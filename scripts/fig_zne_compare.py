"""
fig_zne_compare.py — Figure for the Sec. 7.2 comparative ZNE analysis
====================================================================
Independent figure generator. Reads the JSON produced by
scripts/hw_sim_zne_compare.py and writes a two-panel figure:

  (A) actual vs intended post-transpile depth for each condition / scale,
      highlighting that identity insertion (FIIM) keeps the lowest depth
      overhead.
  (B) bias of the ZNE-extrapolated estimate vs noise level for each
      condition, with the raw bias as reference, showing where (if anywhere)
      mitigation beats the raw estimate.

Usage
-----
    python scripts/fig_zne_compare.py

Output
------
    results/paper_figures/fig_zne_compare.png
    results/paper_figures/lopez8b.png   (paper-naming convention copy)
"""

import json
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parent.parent
IN_JSON  = ROOT / "results" / "hardware_validation" / "sim_zne_compare_latest.json"
OUT_DIR  = ROOT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

C_BLUE, C_GREEN, C_ORANGE, C_RED, C_GREY = (
    "#2563EB", "#16A34A", "#EA580C", "#DC2626", "#6B7280")
COND_STYLE = {
    "global_fold": (C_RED,    "o", "Global fold (U U\u2020 U)"),
    "gate_fold":   (C_ORANGE, "s", "Gate fold (local)"),
    "iim_fiim":    (C_GREEN,  "^", "Identity insertion (FIIM)"),
}
SCALES = [1.0, 3.0, 5.0]

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def main():
    if not IN_JSON.exists():
        raise SystemExit(f"Missing {IN_JSON}. Run scripts/hw_sim_zne_compare.py first.")
    d = json.loads(IN_JSON.read_text())
    comp = d["comparison"]
    p_th = d["metadata"]["p_theory_m1"]
    base = d["metadata"]["base_depth_opt1"]
    noises = list(d["metadata"]["noise_levels"].keys())
    p2q = [d["metadata"]["noise_levels"][n] for n in noises]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))

    # ----- Panel A: actual vs intended depth (noise-independent; use E3) -----
    ref = comp[noises[-1]]
    x = np.arange(len(SCALES))
    width = 0.22
    intended = [round(s * base) for s in SCALES]
    axA.plot(x, intended, "k--", marker="D", lw=1.5, label="Intended (λ·depth₀)", zorder=5)
    for i, (cond, (color, mk, lab)) in enumerate(COND_STYLE.items()):
        depths = [ref[cond]["scale_results"][str(s)]["depth_actual"] for s in SCALES]
        axA.bar(x + (i - 1) * width, depths, width, color=color, alpha=0.85,
                label=lab, zorder=3)
        for xi, dv in zip(x, depths):
            axA.text(xi + (i - 1) * width, dv + 0.6, str(dv), ha="center",
                     va="bottom", fontsize=8, color=color, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels([f"λ = {int(s)}" for s in SCALES])
    axA.set_ylabel("Post-transpile depth (opt_level = 0)")
    axA.set_title("(A) Depth overhead by noise-scaling method\n"
                  "Identity insertion keeps the lowest overhead",
                  fontsize=10, fontweight="bold")
    axA.legend(fontsize=8, loc="upper left")

    # ----- Panel B: extrapolated bias vs noise level -----
    raw_bias = [comp[n]["raw"]["scale_results"]["1.0"]["bias_raw"] for n in noises]
    axB.plot(p2q, raw_bias, "k--", marker="D", lw=1.6, label="Raw (λ=1, no ZNE)", zorder=5)
    for cond, (color, mk, lab) in COND_STYLE.items():
        bias = [comp[n][cond]["bias_zne_richardson"] for n in noises]
        axB.plot(p2q, bias, color=color, marker=mk, lw=1.8, ms=8, label=lab, zorder=4)
    axB.set_xscale("log")
    axB.set_xlabel("Two-qubit gate error $p_{2q}$")
    axB.set_ylabel("|estimate − theory|  (Richardson, m=1)")
    axB.set_title("(B) Mitigated bias vs noise\n"
                  "Raw beats ZNE below threshold; ZNE helps only at high noise",
                  fontsize=10, fontweight="bold")
    axB.set_xticks(p2q); axB.set_xticklabels([f"{p:g}\n({n})" for p, n in zip(p2q, noises)])
    axB.legend(fontsize=8)

    fig.suptitle("ZNE methods on IQM IQAE: a comparative analysis (simulator)\n"
                 f"m=1 IQAE circuit, p_theory = {p_th:.4f}, 4096 shots, 3 replicas",
                 fontsize=11, fontweight="bold", y=1.03)
    plt.tight_layout()
    out = OUT_DIR / "fig_zne_compare.png"
    fig.savefig(out); plt.close(fig)
    shutil.copyfile(out, OUT_DIR / "lopez8b.png")
    print(f"saved -> {out}")
    print(f"saved -> {OUT_DIR / 'lopez8b.png'}  (paper naming)")


if __name__ == "__main__":
    main()
