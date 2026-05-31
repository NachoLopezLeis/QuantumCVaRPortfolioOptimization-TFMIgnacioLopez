#!/usr/bin/env python3
"""
# =============================================================================
# F-005 — Paper Figure Exporter
# =============================================================================
# Reads all completed experiment JSONs and generates all paper figures
# directly in PDF/PNG, ready for LaTeX inclusion.
#
# Usage:
#   python scripts/11_export_paper_figures.py
#   python scripts/11_export_paper_figures.py --exp A1_PA A1_PB A1_PC
# Output:
#   results/paper_figures/fig01_oos_comparison_table.png
#   results/paper_figures/fig02_convergence_curves.png
#   results/paper_figures/fig03_qae_error_heatmap.png
#   results/paper_figures/fig06_benchmark_comparison.png
#   results/paper_figures/fig07_backtest_rolling.png
#   results/paper_figures/paper_figures_data.json
# =============================================================================
"""
import json, sys, argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except ImportError:
    MPL = False
    print("WARNING: matplotlib not available — only data JSON will be exported")

def find_project_root():
    for p in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (p / "config").is_dir(): return p
    raise FileNotFoundError("Cannot find project root")

PROJECT_ROOT = find_project_root()
OUT_DIR = PROJECT_ROOT / "results" / "paper_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_experiments(filter_ids=None):
    results_dir = PROJECT_ROOT / "results"
    experiments = {}
    for exp_dir in sorted(results_dir.iterdir()):
        if not (exp_dir.is_dir() and exp_dir.name.startswith("exp_")): continue
        exp_id = exp_dir.name.replace("exp_", "")
        if filter_ids and exp_id not in filter_ids: continue
        mf = exp_dir / "tfm_comprehensive_metrics_latest.json"
        if mf.exists():
            try:
                experiments[exp_id] = json.loads(mf.read_text())
            except Exception as e:
                print(f"  SKIP {exp_id}: {e}")
    return experiments


# ── Figure 1: OOS comparison table ───────────────────────────────────────────
def fig01_oos_table(experiments):
    if not MPL: return
    rows = []
    for exp_id, m in experiments.items():
        oos = m.get("metrics", {}).get("oos_validation", {})
        ports = oos.get("portfolios", {})
        q_cvar  = ports.get("Quantum CVaR", {}).get("oos_metrics", {}).get("cvar_loss")
        c_cvar  = ports.get("Classical CVaR", {}).get("oos_metrics", {}).get("cvar_loss")
        ew_cvar = ports.get("Equal Weight", {}).get("oos_metrics", {}).get("cvar_loss")
        if q_cvar and c_cvar:
            rows.append([exp_id, f"{q_cvar*100:.3f}%", f"{c_cvar*100:.3f}%",
                         f"{ew_cvar*100:.3f}%" if ew_cvar else "—",
                         f"{(q_cvar-c_cvar)/c_cvar*100:+.2f}%"])

    if not rows:
        print("fig01: no OOS data found")
        return

    fig, ax = plt.subplots(figsize=(12, max(3, len(rows)*0.5 + 1.5)))
    ax.axis("off")
    headers = ["Experiment", "Quantum CVaR OOS", "Classical CVaR OOS", "Equal Weight OOS", "Δ (Q−C)/C"]
    t = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9)
    t.scale(1, 1.5)
    ax.set_title("OOS CVaR Comparison: Quantum vs Classical Methods", fontsize=13, pad=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig01_oos_comparison_table.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig01: OOS comparison table saved")


# ── Figure 2: Convergence curves ─────────────────────────────────────────────
def fig02_convergence(experiments):
    if not MPL: return
    groups = {"A1": [], "A3": [], "B2": [], "C2": [], "D1": [], "D2": []}
    for exp_id, m in experiments.items():
        grp = exp_id.split("_")[0]
        hist = m.get("convergence_history", {}).get("cvar_history", [])
        if hist and grp in groups:
            groups[grp].append((exp_id, hist))

    active = {g: v for g, v in groups.items() if v}
    if not active:
        print("fig02: no convergence history found")
        return

    n = len(active)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=False)
    if n == 1: axes = [axes]
    colors = ["#378ADD","#1D9E75","#E24B4A","#EF9F27","#888780","#A32D2D"]

    for ax, (grp, runs) in zip(axes, active.items()):
        for (exp_id, hist), col in zip(runs[:6], colors):
            ax.plot(hist, color=col, linewidth=1.5, label=exp_id, alpha=0.85)
        ax.set_title(f"Group {grp}", fontsize=11)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel("CVaR (loss)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Convergence Curves by Noise Group", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig02_convergence_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig02: convergence curves saved")


# ── Figure 3: QAE error heatmap ───────────────────────────────────────────────
def fig03_qae_heatmap(experiments):
    if not MPL: return
    groups = ["A1","A3","B2","C2","D1","D2"]
    universes = ["PA","PB","PC"]
    matrix = np.full((len(groups), len(universes)), np.nan)

    for exp_id, m in experiments.items():
        parts = exp_id.split("_")
        grp = parts[0]
        uni = parts[-1]
        if grp in groups and uni in universes:
            qae_err = m.get("metrics",{}).get("quantum",{}).get("qae",{}).get("cvar_error")
            if qae_err is not None:
                gi = groups.index(grp)
                ui = universes.index(uni)
                matrix[gi, ui] = qae_err

    if np.all(np.isnan(matrix)):
        print("fig03: no QAE error data found")
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=0.5)
    ax.set_xticks(range(len(universes))); ax.set_xticklabels(universes, fontsize=11)
    ax.set_yticks(range(len(groups)));   ax.set_yticklabels(groups, fontsize=11)
    for i in range(len(groups)):
        for j in range(len(universes)):
            if not np.isnan(matrix[i,j]):
                ax.text(j, i, f"{matrix[i,j]*100:.1f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if matrix[i,j] > 0.3 else "black")
    plt.colorbar(im, ax=ax, label="QAE CVaR Error")
    ax.set_title("QAE CVaR Relative Error Heatmap\n(noise group × universe)", fontsize=12)
    ax.set_xlabel("Universe", fontsize=11); ax.set_ylabel("Noise Group", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig03_qae_error_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig03: QAE error heatmap saved")


# ── Figure 6: Benchmark comparison ───────────────────────────────────────────
def fig06_benchmark(experiments):
    if not MPL: return
    records = []
    methods = ["slsqp","cobyla","subgradient","monte_carlo","markowitz","risk_parity"]
    for exp_id, m in experiments.items():
        bench = m.get("metrics",{}).get("benchmarks",{})
        if not bench.get("computed"): continue
        q_cvar = bench.get("quantum_cvar", 0)
        row = {"exp_id": exp_id, "Quantum": q_cvar}
        for meth in methods:
            row[meth] = bench.get("classical_results",{}).get(meth,{}).get("optimal_cvar")
        records.append(row)

    if not records:
        print("fig06: no benchmark data found")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(records))
    width = 0.12
    colors = ["#378ADD","#1D9E75","#639922","#EF9F27","#888780","#A32D2D","#D85A30"]
    all_methods = ["Quantum"] + methods
    for k, (meth, col) in enumerate(zip(all_methods, colors)):
        vals = [r.get(meth) or np.nan for r in records]
        ax.bar(x + k*width, vals, width, label=meth, color=col, alpha=0.85)

    ax.set_xticks(x + width*3); ax.set_xticklabels([r["exp_id"] for r in records], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CVaR (training)")
    ax.set_title("Benchmark Comparison: Quantum vs All Classical Methods", fontsize=12)
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig06_benchmark_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("fig06: benchmark comparison saved")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="*", help="Filter experiment IDs")
    args = parser.parse_args()

    print("Loading experiments...")
    experiments = load_all_experiments(args.exp)
    print(f"Found {len(experiments)} experiments with metrics")

    if not experiments:
        print("No experiments found — run experiments first (step 3)")
        return

    # Save raw data
    data_out = {}
    for exp_id, m in experiments.items():
        data_out[exp_id] = {
            "oos": m.get("metrics",{}).get("oos_validation",{}),
            "benchmark": m.get("metrics",{}).get("benchmarks",{}),
            "qae_error": m.get("metrics",{}).get("quantum",{}).get("qae",{}).get("cvar_error"),
            "convergence_fingerprint": m.get("convergence_fingerprint", {}),
        }
    (OUT_DIR / "paper_figures_data.json").write_text(json.dumps(data_out, indent=2, default=str))

    print("Generating figures...")
    fig01_oos_table(experiments)
    fig02_convergence(experiments)
    fig03_qae_heatmap(experiments)
    fig06_benchmark(experiments)
    print(f"\nAll figures saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
