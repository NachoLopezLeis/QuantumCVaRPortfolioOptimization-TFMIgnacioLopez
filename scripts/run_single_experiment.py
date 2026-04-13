#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Script  : run_single_experiment.py
# ============================================================================
#
# Description
# -----------
# Runs the full Phase 1-4 pipeline for a single configuration file.
# Useful for testing a single portfolio without running the full batch.
#
# Usage
# -----
#   cd PROJECT_ROOT
#   python scripts/run_single_experiment.py config_net_PA.yaml
#   python scripts/run_single_experiment.py config_A1.yaml --timeout 3600
#   python scripts/run_single_experiment.py config_net_PD.yaml --kernel eTFM
#
# The script:
#   1. Backs up the current config.yaml.
#   2. Copies the specified config into config.yaml.
#   3. Executes notebooks/main.ipynb via papermill (or nbconvert fallback).
#   4. Restores the original config.yaml.
#
# License : Academic use only - TFM Project
# ============================================================================
"""

import sys
import os
import argparse
import shutil
import time
from pathlib import Path
from datetime import datetime


def find_project_root() -> Path:
    """Find project root by looking for config/ directory."""
    candidates = [Path.cwd(), Path(__file__).resolve().parent.parent]
    for p in candidates:
        if (p / "config").is_dir():
            return p
    raise FileNotFoundError("Cannot find project root (no config/ directory)")


def main():
    parser = argparse.ArgumentParser(
        description="Run a single TFM experiment by config file name"
    )
    parser.add_argument(
        "config_file",
        help="Config YAML filename (e.g., config_net_PA.yaml or config_A1.yaml). "
             "Must be in PROJECT_ROOT/config/.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Execution timeout in seconds (default: 7200 = 2h)",
    )
    parser.add_argument(
        "--kernel",
        default="python3",
        help="Jupyter kernel name (default: python3)",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Do not restore original config.yaml after execution",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing",
    )
    args = parser.parse_args()

    project_root = find_project_root()
    config_dir = project_root / "config"
    notebook_path = project_root / "notebooks" / "main.ipynb"
    active_config = config_dir / "config.yaml"
    source_config = config_dir / args.config_file

    # Validate inputs
    if not source_config.exists():
        print(f"ERROR: Config file not found: {source_config}")
        print(f"Available configs:")
        for f in sorted(config_dir.glob("config_*.yaml")):
            print(f"  {f.name}")
        sys.exit(1)

    if not notebook_path.exists():
        print(f"ERROR: Notebook not found: {notebook_path}")
        sys.exit(1)

    # Determine experiment ID from config filename
    exp_id = args.config_file.replace("config_", "").replace(".yaml", "")

    print("=" * 70)
    print(f"SINGLE EXPERIMENT RUNNER — {exp_id}")
    print("=" * 70)
    print(f"  Project root : {project_root}")
    print(f"  Config       : {args.config_file}")
    print(f"  Notebook     : {notebook_path.name}")
    print(f"  Timeout      : {args.timeout}s")
    print(f"  Kernel       : {args.kernel}")

    if args.dry_run:
        print("\n  [DRY RUN] Would execute the notebook with this config.")
        print("  No changes made.")
        return

    # Setup output
    results_dir = project_root / "results" / "batch_runs"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"main_{exp_id}_{timestamp}.ipynb"

    # Backup config
    backup_path = config_dir / f"config_backup_single_{timestamp}.yaml"
    if active_config.exists():
        shutil.copy2(active_config, backup_path)
        print(f"  Backup       : {backup_path.name}")

    # Swap config
    shutil.copy2(source_config, active_config)
    print(f"  Swapped      : {args.config_file} -> config.yaml")

    # Execute
    print("\n" + "-" * 70)
    print(f"Executing {notebook_path.name}...")
    print("-" * 70)

    start_time = time.time()
    success = False

    try:
        import papermill as pm

        pm.execute_notebook(
            str(notebook_path),
            str(output_path),
            kernel_name=args.kernel,
            cwd=str(project_root / "notebooks"),
            progress_bar=True,
            request_save_on_cell_execute=True,
        )
        success = True
    except ImportError:
        print("papermill not available, falling back to nbconvert...")
        try:
            import subprocess

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "notebook",
                    "--execute",
                    f"--ExecutePreprocessor.timeout={args.timeout}",
                    f"--ExecutePreprocessor.kernel_name={args.kernel}",
                    f"--output={output_path}",
                    str(notebook_path),
                ],
                cwd=str(project_root / "notebooks"),
                capture_output=True,
                text=True,
                timeout=args.timeout + 60,
            )
            success = result.returncode == 0
            if not success:
                print(f"ERROR: {result.stderr[:1000]}")
        except subprocess.TimeoutExpired:
            print(f"ERROR: Execution timed out after {args.timeout}s")
        except Exception as e:
            print(f"ERROR: {e}")
    except Exception as e:
        print(f"ERROR: papermill execution failed: {e}")

    elapsed = time.time() - start_time

    # Restore config
    if not args.no_restore and backup_path.exists():
        shutil.copy2(backup_path, active_config)
        backup_path.unlink()
        print(f"\nConfig restored from backup")

    # Summary
    print("\n" + "=" * 70)
    status = "SUCCESS" if success else "FAILED"
    print(f"Experiment {exp_id}: {status} ({elapsed:.0f}s)")
    if success:
        print(f"Output notebook: {output_path}")
    print("=" * 70)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
