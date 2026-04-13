#!/usr/bin/env python3
"""
# =============================================================================
# F-006 — Sensitivity Analysis: config parameters → results
# =============================================================================
# After all 34 experiments are complete, runs ANOVA + regression to quantify
# which parameters (n_qubits, noise_level, epsilon, universe) explain variance
# in CVaR improvement and QAE error.
#
# Usage:
#   python scripts/12_sensitivity_analysis.py
# Output:
#   results/sensitivity/sensitivity_report.txt
#   results/sensitivity/sensitivity_data.json
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
OUT_DIR = PROJECT_ROOT / "results" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Noise level mapping (group → approximate sq_error_rate)
NOISE_LEVELS = {
    "A1": 0.0, "A3": 0.0, "B2": 1e-4, "C2": 5e-4, "D1": 1e-3, "D2": 5e-4, "S1": 0.0
}

def collect():
    results_dir = PROJECT_ROOT / "results"
    rows = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not (exp_dir.is_dir() and exp_dir.name.startswith("exp_")): continue
        mf = exp_dir / "tfm_comprehensive_metrics_latest.json"
        if not mf.exists(): continue
        try:
            m = json.loads(mf.read_text())
            exp_id = exp_dir.name.replace("exp_", "")
            parts  = exp_id.split("_")
            group  = parts[0]
            universe = parts[-1]

            diag  = m.get("qae_diagnostics_t001", {})
            opt   = m.get("metrics", {}).get("optimization", {})
            qae_d = m.get("metrics", {}).get("quantum", {}).get("qae", {})
            cfg   = m.get("metadata", {}).get("configuration", {})

            n_qubits    = diag.get("n_qubits_used") or cfg.get("phase2", {}).get("qae", {}).get("n_qubits", 4)
            noise_level = NOISE_LEVELS.get(group, 0.0)
            epsilon     = cfg.get("phase2", {}).get("qae", {}).get("epsilon", 0.005)
            cvar_improv = opt.get("cvar_improvement_pct", 0)
            qae_error   = qae_d.get("cvar_error")
            n_iter      = opt.get("num_iterations", 0)

            if cvar_improv is not None:
                rows.append({
                    "exp_id": exp_id, "group": group, "universe": universe,
                    "n_qubits": float(n_qubits or 4),
                    "noise_level": noise_level,
                    "epsilon": float(epsilon or 0.005),
                    "universe_code": ["PA","PB","PC","PA_100","PB_100","PC_100"].index(universe) if universe in ["PA","PB","PC","PA_100","PB_100","PC_100"] else 0,
                    "cvar_improvement_pct": float(cvar_improv),
                    "qae_error": float(qae_error) if qae_error is not None else None,
                    "n_iterations": int(n_iter or 0),
                })
        except Exception as e:
            print(f"  SKIP {exp_dir.name}: {e}")
    return rows

def analyse(rows):
    if len(rows) < 4:
        return "Insufficient data for sensitivity analysis (need ≥ 4 experiments)"

    lines = [
        "=" * 70,
        "SENSITIVITY ANALYSIS: Config Parameters → Results",
        f"n_experiments = {len(rows)}",
        f"Generated: {__import__('datetime').datetime.now().isoformat()}",
        "=" * 70,
        "",
    ]

    features = ["n_qubits", "noise_level", "epsilon", "universe_code"]
    targets  = ["cvar_improvement_pct", "qae_error"]

    try:
        from scipy import stats as sp_stats
        SCIPY = True
    except ImportError:
        SCIPY = False
        lines.append("WARNING: scipy not available — using correlation only\n")

    for target in targets:
        valid = [r for r in rows if r.get(target) is not None]
        if len(valid) < 4:
            lines.append(f"{target}: insufficient data\n"); continue

        y = np.array([r[target] for r in valid])
        lines.append(f"\n── Target: {target} (n={len(valid)}) ──")
        lines.append(f"  Mean: {np.mean(y):.4f}  Std: {np.std(y):.4f}  Range: [{np.min(y):.4f}, {np.max(y):.4f}]")
        lines.append(f"\n  {'Feature':<18} {'Pearson r':>10} {'p-value':>12} {'Significant?':>14}")
        lines.append(f"  {'-'*56}")

        for feat in features:
            x = np.array([r[feat] for r in valid])
            if np.std(x) < 1e-10:
                lines.append(f"  {feat:<18} {'N/A (no variance)':>40}"); continue
            if SCIPY:
                r_val, p_val = sp_stats.pearsonr(x, y)
                sig = "YES *" if p_val < 0.05 else "no"
                lines.append(f"  {feat:<18} {r_val:>10.4f} {p_val:>12.4f} {sig:>14}")
            else:
                r_val = np.corrcoef(x, y)[0, 1]
                lines.append(f"  {feat:<18} {r_val:>10.4f} {'(no p-val)':>12}")

    lines += [
        "\n" + "=" * 70,
        "INTERPRETATION GUIDE",
        "|r| > 0.5: strong effect   |r| 0.3-0.5: moderate   |r| < 0.3: weak",
        "p < 0.05: statistically significant at 5% level",
        "=" * 70,
    ]
    return "\n".join(lines)

def main():
    print("Collecting experiment data...")
    rows = collect()
    print(f"Collected {len(rows)} experiments")

    (OUT_DIR / "sensitivity_data.json").write_text(json.dumps(rows, indent=2))

    report = analyse(rows)
    print(report)
    (OUT_DIR / "sensitivity_report.txt").write_text(report, encoding="utf-8")
    print(f"\nSaved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
