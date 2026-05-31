#!/usr/bin/env python3
r"""
# =============================================================================
# STEP 0 -- Full pre-execution setup for TFM experiments
# =============================================================================
# Runs in order:
#   1. Download S&P 500 returns data (2018-01-01 to 2025-12-31)
#   2. Universe selection via D-Wave real hardware (LeapHybridBQMSampler)
#   3. Smoke test: A1_PA, A1_PB, A1_PC
#   4. Anomaly check on smoke test results
#
# Usage:
#   python scripts\00_setup_and_smoketest.py --dwave-token YOUR_TOKEN
#   python scripts\00_setup_and_smoketest.py --dwave-token YOUR_TOKEN --start 2018-01-01 --end 2025-12-31
#   python scripts\00_setup_and_smoketest.py --local   # use SimAnneal instead of D-Wave
#
# The script stops and reports clearly if any step fails.
# Only proceeds to the next step if the previous one succeeded.
# =============================================================================
"""

import sys
import os
import subprocess
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

# =============================================================================
# Helpers
# =============================================================================

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def header(msg: str) -> None:
    print(f"\n{BOLD}{CYAN}{'='*65}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'='*65}{RESET}\n")

def ok(msg: str) -> None:
    print(f"{GREEN}  [OK] {msg}{RESET}")

def warn(msg: str) -> None:
    print(f"{YELLOW}  [!!] {msg}{RESET}")

def fail(msg: str) -> None:
    print(f"{RED}{BOLD}  [FAIL] {msg}{RESET}")

def info(msg: str) -> None:
    print(f"  {msg}")

def run(cmd: list, cwd: Path, label: str, timeout: int = None) -> subprocess.CompletedProcess:
    """Run a command, stream output, return result."""
    info(f"Running: {' '.join(str(c) for c in cmd)}")
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            timeout=timeout,
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            ok(f"{label} completed in {elapsed:.0f}s")
        else:
            fail(f"{label} failed (exit code {result.returncode})")
        return result
    except subprocess.TimeoutExpired:
        fail(f"{label} timed out after {timeout}s")
        r = subprocess.CompletedProcess(cmd, returncode=-1)
        return r
    except Exception as e:
        fail(f"{label} raised exception: {e}")
        r = subprocess.CompletedProcess(cmd, returncode=-1)
        return r


def find_project_root() -> Path:
    for p in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (p / "config").is_dir():
            return p
    raise FileNotFoundError("Cannot find project root (no config/ directory)")


def check_data(project_root: Path, start: str, end: str) -> bool:
    """Check if returns CSV already covers the required date range."""
    csv = project_root / "data" / "returns_sp500_full.csv"
    if not csv.exists():
        return False
    try:
        import pandas as pd
        df = pd.read_csv(csv, index_col=0, parse_dates=True, nrows=5)
        first_date = str(df.index[0].date())
        # Also check the tail
        df_tail = pd.read_csv(csv, index_col=0, parse_dates=True).tail(1)
        last_date = str(df_tail.index[0].date())
        info(f"Existing CSV: {first_date} to {last_date}")
        if first_date <= start[:10] and last_date >= "2024-12-01":
            ok(f"Existing data covers {start} to ~{last_date} — download skipped")
            return True
    except Exception:
        pass
    return False


def check_universe(project_root: Path) -> bool:
    """Check if universe selection results already exist."""
    sel_dir = project_root / "results" / "network_selection"
    required = ["selection_PA.json", "selection_PB.json"]
    existing = [f for f in required if (sel_dir / f).exists()]
    if len(existing) == len(required):
        ok(f"Universe selection results found: {', '.join(existing)}")
        return True
    return False


def check_smoke_results(project_root: Path) -> dict:
    """Check smoke test results for anomalies."""
    results = {}
    for exp_id in ["A1_PA", "A1_PB", "A1_PC"]:
        mf = project_root / "results" / f"exp_{exp_id}" / "tfm_comprehensive_metrics_latest.json"
        if not mf.exists():
            results[exp_id] = {"status": "MISSING"}
            continue
        try:
            m = json.loads(mf.read_text())
            bench = m.get("metrics", {}).get("benchmarks", {})
            opt   = m.get("metrics", {}).get("optimization", {})
            results[exp_id] = {
                "status": "OK",
                "benchmark_computed": bench.get("computed", False),
                "quantum_cvar": bench.get("quantum_cvar", 0),
                "cvar_improvement_pct": opt.get("cvar_improvement_pct", 0),
                "passport_exists": (
                    project_root / "results" / f"exp_{exp_id}" / "passports" / "pipeline_chain.json"
                ).exists(),
            }
        except Exception as e:
            results[exp_id] = {"status": f"ERROR reading metrics: {e}"}
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TFM pre-execution setup: download → universe → smoke test"
    )
    parser.add_argument(
        "--dwave-token", dest="dwave_token",
        default=os.environ.get("DWAVE_API_TOKEN", ""),
        help="D-Wave Leap API token (or set DWAVE_API_TOKEN env var)"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Use SimulatedAnnealing instead of D-Wave real hardware"
    )
    parser.add_argument("--start", default="2018-01-01", help="Data start date")
    parser.add_argument("--end",   default="2025-12-31", help="Data end date")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download if data already exists"
    )
    parser.add_argument(
        "--skip-universe", action="store_true",
        help="Skip universe selection if results already exist"
    )
    args = parser.parse_args()

    project_root = find_project_root()
    py = sys.executable

    print(f"\n{BOLD}TFM — Pre-execution Setup{RESET}")
    print(f"Project root : {project_root}")
    print(f"Python       : {py}")
    print(f"Date range   : {args.start} to {args.end}")
    print(f"Universe HW  : {'SimulatedAnnealing (local)' if args.local else 'D-Wave real hardware'}")
    print(f"Started      : {datetime.now().isoformat()}")

    errors = []

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Download data
    # ─────────────────────────────────────────────────────────────────────────
    header("STEP 1/4 — Download S&P 500 returns data")

    skip_dl = args.skip_download and check_data(project_root, args.start, args.end)
    if skip_dl:
        info("Skipping download (data already present)")
    else:
        result = run(
            [py, "scripts/download_sp500_full.py",
             "--start", args.start,
             "--end",   args.end,
             "--min-obs", "1500"],
            cwd=project_root,
            label="Data download",
            timeout=1200,  # 20 min max
        )
        if result.returncode != 0:
            fail("Data download failed — cannot continue")
            fail("Check your internet connection and yfinance version")
            sys.exit(1)

    # Verify data
    csv_full = project_root / "data" / "returns_sp500_full.csv"
    csv_100  = project_root / "data" / "returns_sp500_100.csv"
    if not csv_full.exists():
        fail(f"Expected file not found: {csv_full}")
        sys.exit(1)
    ok(f"returns_sp500_full.csv exists ({csv_full.stat().st_size / 1e6:.1f} MB)")

    # Generate SP100 subset if missing
    if not csv_100.exists():
        info("Generating SP100 subset...")
        run(
            [py, "-c", """
import pandas as pd, numpy as np
from pathlib import Path
full = pd.read_csv('data/returns_sp500_full.csv', index_col=0, parse_dates=True)
sharpes = full.mean() / full.std() * (252**0.5)
top100 = sharpes.nlargest(100).index
subset = full[top100]
subset.to_csv('data/returns_sp500_100.csv')
print(f'SP100 saved: {subset.shape[0]} x {subset.shape[1]}')
"""],
            cwd=project_root,
            label="SP100 subset",
            timeout=60,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Universe selection
    # ─────────────────────────────────────────────────────────────────────────
    header("STEP 2/4 — Universe selection via QUBO")

    skip_uni = args.skip_universe and check_universe(project_root)
    if skip_uni:
        info("Skipping universe selection (results already present)")
    else:
        if not args.local:
            if not args.dwave_token:
                fail("D-Wave token not provided.")
                fail("  Option 1: python scripts\\00_setup_and_smoketest.py --dwave-token YOUR_TOKEN")
                fail("  Option 2: set DWAVE_API_TOKEN=YOUR_TOKEN before running")
                fail("  Option 3: use --local flag for SimulatedAnnealing (no token needed)")
                sys.exit(1)
            os.environ["DWAVE_API_TOKEN"] = args.dwave_token
            ok(f"D-Wave token set ({len(args.dwave_token)} chars)")
            backend_flag = "--backend"
            backend_val  = "hybrid"
        else:
            warn("Using SimulatedAnnealing (local) — not real D-Wave hardware")
            backend_flag = "--backend"
            backend_val  = "simulated"

        result = run(
            [py, "scripts/run_universe_selection.py",
             backend_flag, backend_val],
            cwd=project_root,
            label="Universe selection",
            timeout=3600,  # 1h max
        )
        if result.returncode != 0:
            fail("Universe selection failed — cannot continue")
            fail("Check D-Wave token and network connection")
            errors.append("universe_selection_failed")

    # Verify outputs
    sel_dir = project_root / "results" / "network_selection"
    for fname in ["selection_PA.json", "selection_PB.json"]:
        fpath = sel_dir / fname
        if fpath.exists():
            ok(f"{fname} present")
        else:
            fail(f"{fname} NOT found — universe selection may have failed")
            errors.append(f"missing_{fname}")

    if errors:
        fail("Universe selection step has errors — stopping before smoke test")
        fail("Fix the issues above and re-run with --skip-download --skip-universe if data is OK")
        sys.exit(1)

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Smoke test
    # ─────────────────────────────────────────────────────────────────────────
    header("STEP 3/4 — Smoke test (A1_PA, A1_PB, A1_PC)")
    info("Running 3 baseline experiments to verify the full pipeline...")
    info("Expected duration: ~30-60 minutes per experiment")

    result = run(
        [py, "scripts/run_all_experiments.py", "A1_PA", "A1_PB", "A1_PC"],
        cwd=project_root,
        label="Smoke test",
        timeout=14400,  # 4h max
    )
    if result.returncode != 0:
        fail("Smoke test returned non-zero exit code")
        warn("Check results/batch_runs/batch_runner_*.log for details")
        errors.append("smoke_test_nonzero_exit")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Verify smoke test results
    # ─────────────────────────────────────────────────────────────────────────
    header("STEP 4/4 — Verify smoke test results")

    smoke_results = check_smoke_results(project_root)

    print(f"\n  {'Experiment':<12} {'Status':<8} {'Benchmark':<12} {'q_CVaR':>10} {'Improv%':>10} {'Passport':<10}")
    print(f"  {'-'*65}")

    all_ok = True
    for exp_id, r in smoke_results.items():
        if r["status"] == "MISSING":
            print(f"  {exp_id:<12} {RED}MISSING{RESET}")
            errors.append(f"{exp_id}_missing")
            all_ok = False
            continue

        if r["status"].startswith("ERROR"):
            print(f"  {exp_id:<12} {RED}ERROR{RESET}  {r['status']}")
            errors.append(f"{exp_id}_error")
            all_ok = False
            continue

        bench_ok = r["benchmark_computed"]
        qcvar_ok = r["quantum_cvar"] > 0
        passport = "YES" if r["passport_exists"] else "NO"

        bench_str = f"{GREEN}OK{RESET}" if bench_ok else f"{RED}FAIL{RESET}"
        qcvar_str = f"{r['quantum_cvar']*100:.3f}%" if r["quantum_cvar"] else f"{RED}0.0!{RESET}"
        improv    = f"{r['cvar_improvement_pct']:.2f}%"

        print(f"  {exp_id:<12} {GREEN}OK{RESET}      {bench_str:<20} {qcvar_str:>10} {improv:>10} {passport:<10}")

        if not bench_ok:
            errors.append(f"{exp_id}_benchmark_not_computed")
            all_ok = False
        if not qcvar_ok:
            errors.append(f"{exp_id}_quantum_cvar_zero")
            all_ok = False

    # ─────────────────────────────────────────────────────────────────────────
    # Final verdict
    # ─────────────────────────────────────────────────────────────────────────
    print()
    if all_ok and not errors:
        print(f"{GREEN}{BOLD}{'='*65}{RESET}")
        print(f"{GREEN}{BOLD}  ALL STEPS PASSED — READY TO RUN ALL 34 EXPERIMENTS{RESET}")
        print(f"{GREEN}{BOLD}{'='*65}{RESET}")
        print(f"\n  Next command:")
        print(f"  {BOLD}python scripts\\run_all_experiments.py{RESET}")
    else:
        print(f"{RED}{BOLD}{'='*65}{RESET}")
        print(f"{RED}{BOLD}  SETUP INCOMPLETE — FIX ERRORS BEFORE RUNNING ALL EXPERIMENTS{RESET}")
        print(f"{RED}{BOLD}{'='*65}{RESET}")
        print(f"\n  Errors detected:")
        for err in errors:
            print(f"  {RED}  - {err}{RESET}")
        print(f"\n  Check logs in results/batch_runs/batch_runner_*.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
