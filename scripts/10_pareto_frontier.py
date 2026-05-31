#!/usr/bin/env python3
"""
# =============================================================================
# F-004 — Pareto Frontier: QAE error vs Oracle queries
# =============================================================================
# Extracts (total_oracle_queries, qae_relative_error) from all completed runs
# and builds the error-vs-cost Pareto frontier figure.
#
# Usage:
#   python scripts/10_pareto_frontier.py
# Output:
#   results/paper_figures/fig04_pareto_frontier.png
#   results/paper_figures/pareto_data.json
# =============================================================================
"""
import json, sys
import numpy as np
from pathlib import Path

def find_project_root():
    for p in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (p / "config").is_dir(): return p
    raise FileNotFoundError("Cannot find project root")

PROJECT_ROOT = find_project_root()

def collect_data():
    results_dir = PROJECT_ROOT / "results"
    data_points = []

    for exp_dir in sorted(results_dir.iterdir()):
        if not (exp_dir.is_dir() and exp_dir.name.startswith("exp_")): continue
        mf = exp_dir / "tfm_comprehensive_metrics_latest.json"
        if not mf.exists(): continue

        try:
            m = json.loads(mf.read_text())
            exp_id = exp_dir.name.replace("exp_", "")
            parts  = exp_id.split("_")

            # Parse experiment structure: group_universe or group_n_universe
            group   = parts[0] if parts else "unknown"
            universe = parts[-1] if parts else "unknown"

            qae = m.get("metrics", {}).get("quantum", {}).get("qae", {})
            qry = m.get("metrics", {}).get("quantum", {}).get("total_oracle_queries", {})
            opt = m.get("metrics", {}).get("optimization", {})
            diag = m.get("qae_diagnostics_t001", {})

            total_queries = (qry.get("iqae", 0) if isinstance(qry, dict) else 0)
            qae_error     = qae.get("cvar_error", None)
            n_qubits      = diag.get("n_qubits_used") or m.get("metadata", {}).get("configuration", {}).get("phase2", {}).get("qae", {}).get("n_qubits", 4)
            cvar_improv   = opt.get("cvar_improvement_pct", 0)

            if qae_error is not None and total_queries > 0:
                data_points.append({
                    "exp_id": exp_id, "group": group, "universe": universe,
                    "total_queries": int(total_queries),
                    "qae_error": float(qae_error),
                    "n_qubits": int(n_qubits) if n_qubits else 4,
                    "cvar_improvement_pct": float(cvar_improv) if cvar_improv else 0,
                })
        except Exception as e:
            print(f"  SKIP {exp_dir.name}: {e}")

    return data_points

def make_figure(data_points):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not available — saving data only")
        return

    out_dir = PROJECT_ROOT / "results" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_points:
        print("No data points collected — run experiments first")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = {"A1":"#378ADD","A3":"#1D9E75","B2":"#639922","C2":"#EF9F27","D1":"#E24B4A","D2":"#A32D2D","S1":"#888780"}
    markers = {"PA":"o","PB":"s","PC":"^","PA_100":"D","PB_100":"v","PC_100":"P"}

    for dp in data_points:
        c = colors.get(dp["group"], "#888780")
        mk = markers.get(dp["universe"], "o")
        size = 60 + dp["n_qubits"] * 15
        ax.scatter(dp["total_queries"], dp["qae_error"] * 100,
                   c=c, marker=mk, s=size, alpha=0.8, edgecolors="white", linewidths=0.5)
        ax.annotate(dp["exp_id"], (dp["total_queries"], dp["qae_error"]*100),
                    fontsize=6, alpha=0.7, xytext=(4, 4), textcoords="offset points")

    # Monte Carlo reference line: error ~ 1/sqrt(N)
    q_range = np.linspace(100, max((d["total_queries"] for d in data_points), default=10000), 200)
    mc_error = 1 / np.sqrt(q_range) * 100
    ax.plot(q_range, mc_error, "k--", alpha=0.4, linewidth=1, label="Monte Carlo 1/√N reference")

    ax.set_xlabel("Total Oracle Queries (computational cost)", fontsize=12)
    ax.set_ylabel("QAE CVaR Relative Error (%)", fontsize=12)
    ax.set_title("Pareto Frontier: QAE Error vs Computational Cost\nby noise group and universe", fontsize=12)
    ax.legend(handles=[
        mpatches.Patch(color=c, label=g) for g, c in colors.items()
    ] + [plt.Line2D([0],[0], color="k", linestyle="--", label="MC reference")],
    fontsize=8, loc="upper right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    path = out_dir / "fig04_pareto_frontier.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {path}")

def main():
    print("Collecting Pareto data...")
    data = collect_data()
    print(f"Found {len(data)} data points")

    out_dir = PROJECT_ROOT / "results" / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pareto_data.json").write_text(json.dumps(data, indent=2))

    make_figure(data)
    print("Done.")

if __name__ == "__main__":
    main()
