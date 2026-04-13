#!/usr/bin/env python3
"""
# =============================================================================
# F-002 — Automatic Anomaly Detector
# =============================================================================
# Reads all completed experiment JSONs and flags anomalies before they
# contaminate the final analysis.
#
# Usage:
#   python scripts/09_anomaly_detector.py --exp A1_n3_PA
#   python scripts/09_anomaly_detector.py --all
#   python scripts/09_anomaly_detector.py --watch   # continuous mode
# =============================================================================
"""

import sys, json, time, argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from enum import Enum

def find_project_root():
    for p in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (p / "config").is_dir(): return p
    raise FileNotFoundError("Cannot find project root")

PROJECT_ROOT = find_project_root()


class Severity(str, Enum):
    ERROR   = "ERROR"
    WARNING = "WARNING"
    INFO    = "INFO"


@dataclass
class Anomaly:
    exp_id:    str
    severity:  Severity
    category:  str
    message:   str
    value:     object = None
    threshold: object = None


def check_experiment(exp_dir: Path) -> List[Anomaly]:
    exp_id = exp_dir.name.replace("exp_", "")
    mf = exp_dir / "tfm_comprehensive_metrics_latest.json"
    if not mf.exists():
        return []

    try:
        m = json.loads(mf.read_text())
    except Exception as e:
        return [Anomaly(exp_id, Severity.ERROR, "json_parse", f"Cannot parse metrics: {e}")]

    anomalies: List[Anomaly] = []
    opt   = m.get("metrics", {}).get("optimization", {})
    bench = m.get("metrics", {}).get("benchmarks", {})
    risk  = m.get("metrics", {}).get("risk", {})
    qae   = m.get("metrics", {}).get("quantum", {}).get("qae", {})

    # 1. Benchmark not computed
    if not bench.get("computed", False):
        anomalies.append(Anomaly(exp_id, Severity.ERROR, "benchmark",
            "benchmark.computed=False — classical comparison missing (P0-001)"))

    # 2. CVaR got worse
    improv = opt.get("cvar_improvement_pct", 0)
    if improv is not None and improv < 0:
        anomalies.append(Anomaly(exp_id, Severity.WARNING, "optimization",
            f"CVaR improvement is negative: {improv:.2f}% — optimizer diverged?",
            value=improv, threshold=0))

    # 3. Convergence too fast (< 5 iterations)
    n_iter = opt.get("num_iterations", 100)
    if n_iter is not None and n_iter < 5:
        anomalies.append(Anomaly(exp_id, Severity.WARNING, "convergence",
            f"Converged in only {n_iter} iterations — possible initialization collapse",
            value=n_iter, threshold=5))

    # 4. quantum_cvar = 0 (BUG-002 residual)
    q_cvar = bench.get("quantum_cvar", None)
    if q_cvar == 0.0 or q_cvar is None:
        anomalies.append(Anomaly(exp_id, Severity.ERROR, "benchmark",
            "benchmark.quantum_cvar=0.0 — BUG-002 not fixed or quantum result key mismatch"))

    # 5. QAE error very high
    qae_err = qae.get("cvar_error")
    if qae_err is not None and qae_err > 0.5:
        anomalies.append(Anomaly(exp_id, Severity.WARNING, "qae",
            f"QAE relative error={qae_err:.2%} > 50% — consider increasing shots (T-001)",
            value=qae_err, threshold=0.5))

    # 6. No passport (BUG-001)
    passport_dir = exp_dir / "passports"
    if not passport_dir.exists() or not list(passport_dir.glob("*.json")):
        anomalies.append(Anomaly(exp_id, Severity.WARNING, "passport",
            "No passport files found — BUG-001/BUG-009 not fixed for this run"))

    # 7. Speedup factor impossible
    sf = bench.get("speedup_factor", 0)
    if sf is not None and sf > 1000:
        anomalies.append(Anomaly(exp_id, Severity.ERROR, "timing",
            f"speedup_factor={sf:.0f} — timing data likely incorrect",
            value=sf, threshold=1000))

    return anomalies


def format_report(all_anomalies: dict) -> str:
    lines = [
        "=" * 70,
        f"ANOMALY DETECTION REPORT — {datetime.now().isoformat()}",
        "=" * 70,
    ]
    total_errors = sum(1 for anoms in all_anomalies.values()
                       for a in anoms if a.severity == Severity.ERROR)
    total_warns  = sum(1 for anoms in all_anomalies.values()
                       for a in anoms if a.severity == Severity.WARNING)
    clean = [k for k, v in all_anomalies.items() if not v]

    lines.append(f"Experiments checked: {len(all_anomalies)}")
    lines.append(f"Clean: {len(clean)}  |  Errors: {total_errors}  |  Warnings: {total_warns}")
    lines.append("")

    for exp_id, anoms in sorted(all_anomalies.items()):
        if not anoms:
            lines.append(f"  [OK] {exp_id}")
            continue
        for a in anoms:
            icon = "!!" if a.severity == Severity.ERROR else " !"
            lines.append(f"  [{icon}] {exp_id} | {a.severity.value:<8} | {a.category:<12} | {a.message}")

    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="F-002: Anomaly detector")
    parser.add_argument("--exp", help="Single experiment ID")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--watch", action="store_true", help="Continuous mode (check every 60s)")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"

    def run_once():
        if args.exp:
            exp_dirs = [results_dir / f"exp_{args.exp}"]
        else:
            exp_dirs = sorted([d for d in results_dir.iterdir()
                               if d.is_dir() and d.name.startswith("exp_")])

        all_anomalies = {}
        for exp_dir in exp_dirs:
            if exp_dir.exists():
                exp_id = exp_dir.name.replace("exp_", "")
                all_anomalies[exp_id] = check_experiment(exp_dir)

        report = format_report(all_anomalies)
        print(report)

        # Save
        out_dir = results_dir
        out_dir.mkdir(exist_ok=True)
        (out_dir / "anomaly_report.txt").write_text(report, encoding="utf-8")
        (out_dir / "anomaly_report.json").write_text(
            json.dumps({k: [asdict(a) for a in v] for k, v in all_anomalies.items()}, indent=2),
            encoding="utf-8"
        )

    if args.watch:
        print("Watching for new results (Ctrl+C to stop)...")
        while True:
            run_once()
            time.sleep(60)
    else:
        run_once()

if __name__ == "__main__":
    main()
