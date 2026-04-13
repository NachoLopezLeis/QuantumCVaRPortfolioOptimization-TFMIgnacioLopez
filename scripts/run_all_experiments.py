#!/usr/bin/env python3
r"""
================================================================================
BATCH EXPERIMENT RUNNER — Execute main.ipynb with each config sequentially
VERSION: 2.0 — WITH PERSISTENT LOGGING
================================================================================

Usage:
    cd E:\\\\\TFM\\\\Codigo\\scripts
    python run_all_experiments.py                    # Run ALL 15 experiments
    python run_all_experiments.py A1 A3 D1           # Run only selected ones
    python run_all_experiments.py --dry-run           # Show plan without executing
    python run_all_experiments.py --resume-from C2    # Skip configs before C2

Place this script in PROJECT_ROOT/scripts/ (e.g., E:\\\\\TFM\\\\Codigo\\scripts\\).
All config_X.yaml files must be in PROJECT_ROOT/config/.
The script copies each config_X.yaml -> config/config.yaml, then executes
notebooks/main.ipynb via papermill (preferred) or nbconvert (fallback).

LOGGING (v2.0):
    - A persistent log file (batch_runner.log) is written to results/batch_runs/
      with real-time flush after every write.
    - The batch_summary JSON is updated incrementally after each experiment,
      so partial progress is always visible even if the process crashes.
    - All events (start, config copy, execution, errors, completion) are logged
      with timestamps.

Requirements:
    pip install papermill   # preferred — better logging, parameter injection
    # OR
    pip install nbconvert   # fallback — simpler but less control

Results for each experiment go to the directory specified by each config's
export.run_subdirectory (e.g., results/exp_A1/, results/exp_B2/, etc.).
A summary JSON is written incrementally and finalized at the end.
================================================================================
"""

import sys
import os
import shutil
import time
import subprocess
import argparse
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta


# ============================================================================
# PERSISTENT LOGGER — writes to file with immediate flush
# ============================================================================

class BatchLogger:
    """
    File-based logger for the batch runner.
    
    Writes every event to a persistent log file with immediate flush,
    so that if the process crashes or is killed, all prior events are
    preserved on disk.
    
    Also mirrors all messages to stdout for terminal visibility.
    """
    
    def __init__(self, log_dir: Path, timestamp: str):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"batch_runner_{timestamp}.log"
        self._fh = open(self.log_file, 'w', encoding='utf-8')
        self._write_header(timestamp)
    
    def _write_header(self, timestamp: str):
        header = (
            f"{'='*80}\n"
            f"BATCH EXPERIMENT RUNNER LOG\n"
            f"Started: {datetime.now().isoformat()}\n"
            f"Timestamp ID: {timestamp}\n"
            f"Python: {sys.executable}\n"
            f"PID: {os.getpid()}\n"
            f"{'='*80}\n"
        )
        self._fh.write(header)
        self._fh.flush()
    
    def _format(self, level: str, msg: str) -> str:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[{ts}] [{level:<7}] {msg}"
    
    def _write(self, level: str, msg: str):
        line = self._format(level, msg)
        # Write to file with immediate flush
        self._fh.write(line + '\n')
        self._fh.flush()
        # Mirror to stdout
        print(line)
    
    def info(self, msg: str):
        self._write("INFO", msg)
    
    def warning(self, msg: str):
        self._write("WARNING", msg)
    
    def error(self, msg: str):
        self._write("ERROR", msg)
    
    def section(self, title: str):
        sep = '─' * 70
        self._fh.write(f"\n{sep}\n  {title}\n{sep}\n")
        self._fh.flush()
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
    
    def banner(self, title: str):
        sep = '=' * 70
        self._fh.write(f"\n{sep}\n{title}\n{sep}\n")
        self._fh.flush()
        print(f"\n{sep}")
        print(title)
        print(sep)
    
    def close(self):
        if self._fh and not self._fh.closed:
            self.info(f"Log closed: {self.log_file}")
            self._fh.close()
    
    def __del__(self):
        self.close()


# ============================================================================
# INCREMENTAL SUMMARY — saves after each experiment
# ============================================================================

class IncrementalSummary:
    """
    Writes the batch summary JSON after every experiment completes,
    so that partial results are always available on disk.
    """
    
    def __init__(self, summary_path: Path, timestamp: str, total_planned: int):
        self.path = summary_path
        self.data = {
            'timestamp': timestamp,
            'started': datetime.now().isoformat(),
            'total_planned': total_planned,
            'total_completed': 0,
            'total_ok': 0,
            'total_failed': 0,
            'total_elapsed': '0:00:00',
            'status': 'RUNNING',
            'experiments': [],
        }
        self._save()
    
    def add_result(self, result: dict, total_elapsed_s: float):
        self.data['experiments'].append(result)
        self.data['total_completed'] = len(self.data['experiments'])
        self.data['total_ok'] = sum(
            1 for e in self.data['experiments'] if 'OK' in e.get('status', '')
        )
        self.data['total_failed'] = (
            self.data['total_completed'] - self.data['total_ok']
        )
        self.data['total_elapsed'] = str(
            timedelta(seconds=int(total_elapsed_s))
        )
        self._save()
    
    def finalize(self, total_elapsed_s: float):
        self.data['total_elapsed'] = str(
            timedelta(seconds=int(total_elapsed_s))
        )
        self.data['finished'] = datetime.now().isoformat()
        self.data['status'] = 'COMPLETED'
        self._save()
    
    def _save(self):
        try:
            with open(self.path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass  # Non-critical: don't crash if summary write fails


# ============================================================================
# EXPERIMENT DEFINITIONS — order matters (fast/simple first, slow/complex last)
# ============================================================================

EXPERIMENTS = [
    # ── Group A1: noiseless eps=0.005 n=4 — algorithmic baseline ─────────────
    # Fast (~3-8 min each). Three universes: PA (peripheral), PB (central), PC (predefined).
    ("A1_PA",     "config_A1_PA.yaml",     "noiseless eps=0.005 n=4 | PA peripheral K=10"),
    ("A1_PB",     "config_A1_PB.yaml",     "noiseless eps=0.005 n=4 | PB central K=10"),
    ("A1_PC",     "config_A1_PC.yaml",     "noiseless eps=0.005 n=4 | PC predefined K=10"),

    # ── Group A1_n3: discretization study n=3 ────────────────────────────────
    ("A1_n3_PA",  "config_A1_n3_PA.yaml",  "noiseless eps=0.005 n=3 | PA peripheral K=10"),
    ("A1_n3_PB",  "config_A1_n3_PB.yaml",  "noiseless eps=0.005 n=3 | PB central K=10"),
    ("A1_n3_PC",  "config_A1_n3_PC.yaml",  "noiseless eps=0.005 n=3 | PC predefined K=10"),

    # ── Group A1_n5: discretization study n=5 ────────────────────────────────
    ("A1_n5_PA",  "config_A1_n5_PA.yaml",  "noiseless eps=0.005 n=5 | PA peripheral K=10"),
    ("A1_n5_PB",  "config_A1_n5_PB.yaml",  "noiseless eps=0.005 n=5 | PB central K=10"),
    ("A1_n5_PC",  "config_A1_n5_PC.yaml",  "noiseless eps=0.005 n=5 | PC predefined K=10"),

    # ── Group A1_n6: discretization ceiling n=6 ──────────────────────────────
    ("A1_n6_PA",  "config_A1_n6_PA.yaml",  "noiseless eps=0.005 n=6 | PA peripheral K=10"),
    ("A1_n6_PB",  "config_A1_n6_PB.yaml",  "noiseless eps=0.005 n=6 | PB central K=10"),
    ("A1_n6_PC",  "config_A1_n6_PC.yaml",  "noiseless eps=0.005 n=6 | PC predefined K=10"),

    # ── Group A3: algorithmic ceiling eps=0.002 ───────────────────────────────
    ("A3_PA",     "config_A3_PA.yaml",     "noiseless eps=0.002 n=4 | PA peripheral K=10"),
    ("A3_PB",     "config_A3_PB.yaml",     "noiseless eps=0.002 n=4 | PB central K=10"),
    ("A3_PC",     "config_A3_PC.yaml",     "noiseless eps=0.002 n=4 | PC predefined K=10"),

    # ── Group D1: high noise eps=0.01 — NISQ shallow (~35-70 min each) ───────
    ("D1_PA",     "config_D1_PA.yaml",     "high noise eps=0.01  n=4 | PA peripheral K=10"),
    ("D1_PB",     "config_D1_PB.yaml",     "high noise eps=0.01  n=4 | PB central K=10"),
    ("D1_PC",     "config_D1_PC.yaml",     "high noise eps=0.01  n=4 | PC predefined K=10"),
    ("D1_n3_PA",  "config_D1_n3_PA.yaml",  "high noise eps=0.01  n=3 | PA peripheral K=10 [P1-002]"),
    ("D1_n6_PA",  "config_D1_n6_PA.yaml",  "high noise eps=0.01  n=6 | PA peripheral K=10 [P1-002]"),

    # ── Group D2: high noise eps=0.005 — NISQ deep ───────────────────────────
    ("D2_PA",     "config_D2_PA.yaml",     "high noise eps=0.005 n=4 | PA peripheral K=10"),
    ("D2_PB",     "config_D2_PB.yaml",     "high noise eps=0.005 n=4 | PB central K=10"),
    ("D2_PC",     "config_D2_PC.yaml",     "high noise eps=0.005 n=4 | PC predefined K=10"),

    # ── Group B2: low noise eps=0.005 — projected hw ~2027 (~1.5-2h each) ────
    ("B2_PA",     "config_B2_PA.yaml",     "low noise  eps=0.005 n=4 | PA peripheral K=10"),
    ("B2_PB",     "config_B2_PB.yaml",     "low noise  eps=0.005 n=4 | PB central K=10"),
    ("B2_PC",     "config_B2_PC.yaml",     "low noise  eps=0.005 n=4 | PC predefined K=10"),
    # --- [P1-002] n-qubit cross-validation under noise ---
    ("B2_n3_PA",  "config_B2_n3_PA.yaml",  "low noise  eps=0.005 n=3 | PA peripheral K=10 [P1-002]"),
    ("B2_n6_PA",  "config_B2_n6_PA.yaml",  "low noise  eps=0.005 n=6 | PA peripheral K=10 [P1-002]"),

    # ── Group C2: medium noise eps=0.005 — current hw 2025 (~1.5-2h each) ────
    ("C2_PA",     "config_C2_PA.yaml",     "med noise  eps=0.005 n=4 | PA peripheral K=10"),
    ("C2_PB",     "config_C2_PB.yaml",     "med noise  eps=0.005 n=4 | PB central K=10"),
    ("C2_PC",     "config_C2_PC.yaml",     "med noise  eps=0.005 n=4 | PC predefined K=10"),

    # ── Group S1: scalability K=100 ───────────────────────────────────────────
    ("S1_PA_100", "config_S1_PA_100.yaml", "noiseless eps=0.005 n=4 | PA_100 peripheral K=100"),
    ("S1_PB_100", "config_S1_PB_100.yaml", "noiseless eps=0.005 n=4 | PB_100 central K=100"),
    ("S1_PC_100", "config_S1_PC_100.yaml", "noiseless eps=0.005 n=4 | PC_100 SP100 K=100"),

    # ── S1 scalability N=30 (intermediate point) ─────────────────────────
    ("S1_PA_30",  "config_S1_PA_30.yaml",  "noiseless eps=0.005 n=4 | PA_30  peripheral K=30"),
    ("S1_PB_30",  "config_S1_PB_30.yaml",  "noiseless eps=0.005 n=4 | PB_30  central    K=30"),
    ("S1_PC_30",  "config_S1_PC_30.yaml",  "noiseless eps=0.005 n=4 | PC_30  predefined K=30"),

    # ── E-group: noise INSIDE optimizer loop ─────────────────────────────
    ("E1_PA",     "config_E1_PA.yaml",     "noise-in-opt p2q=0.001 ZNE | PA peripheral K=10"),
    ("E1_PB",     "config_E1_PB.yaml",     "noise-in-opt p2q=0.001 ZNE | PB central    K=10"),
    ("E1_PC",     "config_E1_PC.yaml",     "noise-in-opt p2q=0.001 ZNE | PC predefined K=10"),
    ("E2_PA",     "config_E2_PA.yaml",     "noise-in-opt p2q=0.005 ZNE | PA peripheral K=10"),
    ("E2_PB",     "config_E2_PB.yaml",     "noise-in-opt p2q=0.005 ZNE | PB central    K=10"),
    ("E2_PC",     "config_E2_PC.yaml",     "noise-in-opt p2q=0.005 ZNE | PC predefined K=10"),
    ("E3_PA",     "config_E3_PA.yaml",     "noise-in-opt p2q=0.010 ZNE | PA peripheral K=10"),
    ("E3_PB",     "config_E3_PB.yaml",     "noise-in-opt p2q=0.010 ZNE | PB central    K=10"),
    ("E3_PC",     "config_E3_PC.yaml",     "noise-in-opt p2q=0.010 ZNE | PC predefined K=10"),
]



def find_project_root():
    """Find project root by looking for config/ directory."""
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir.parent,
        script_dir,
        Path.cwd(),
        Path.cwd().parent,
    ]
    for p in candidates:
        if (p / 'config').is_dir() and (p / 'notebooks').is_dir():
            return p
    raise FileNotFoundError(
        f"Cannot find project root (expected config/ and notebooks/ dirs). "
        f"Searched from: {script_dir}"
    )


def get_executor(logger):
    """Detect available notebook executor: papermill or nbconvert."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "papermill", "--version"],
            capture_output=True, check=True, text=True
        )
        version = result.stdout.strip()
        logger.info(f"Executor detected: papermill {version}")
        return "papermill"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    try:
        result = subprocess.run(
            [sys.executable, "-m", "nbconvert", "--version"],
            capture_output=True, check=True, text=True
        )
        version = result.stdout.strip()
        logger.info(f"Executor detected: nbconvert {version}")
        return "nbconvert"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    logger.error("No notebook executor found (papermill or nbconvert)")
    return None


def run_notebook_papermill(notebook_path, output_path, log_path, logger):
    """Execute notebook via papermill.

    cwd is set to the notebook directory so that PROJECT_ROOT detection
    inside the notebook (Path.cwd().parent) resolves correctly.
    """
    cmd = [
        sys.executable, "-m", "papermill",
        str(notebook_path),
        str(output_path),
        "--no-progress-bar",
        "--log-output",
        "--log-level", "WARNING",
    ]
    cwd = str(notebook_path.parent)
    logger.info(f"Papermill command: {' '.join(cmd)}")
    logger.info(f"Papermill cwd: {cwd}")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=43200,
            cwd=cwd,
        )
    logger.info(f"Papermill exit code: {result.returncode}")
    return result.returncode


def run_notebook_nbconvert(notebook_path, output_path, log_path, logger):
    """Execute notebook via nbconvert."""
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=43200",
        "--output", str(output_path),
        str(notebook_path),
    ]
    cwd = str(notebook_path.parent)
    logger.info(f"nbconvert command: {' '.join(cmd)}")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        result = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            timeout=43200,
            cwd=cwd,
        )
    logger.info(f"nbconvert exit code: {result.returncode}")
    return result.returncode


def robust_copy(src, dst, logger, max_retries=5, delay=3):
    """Copy file with retry logic for Windows file-locking issues."""
    for attempt in range(1, max_retries + 1):
        try:
            shutil.copy2(src, dst)
            logger.info(f"Copied: {src.name} -> {dst.name}")
            return
        except PermissionError as e:
            if attempt == max_retries:
                logger.error(
                    f"Failed to copy after {max_retries} attempts: {e}"
                )
                raise
            wait = delay * attempt
            logger.warning(
                f"File locked, retrying in {wait}s "
                f"(attempt {attempt}/{max_retries}): {e}"
            )
            time.sleep(wait)


def check_experiment_output(project_root, exp_id, logger):
    """Check if the experiment produced expected output files.
    
    Looks for recent JSON metrics files in results/ directory.
    Returns a dict with findings for the log.
    """
    results_base = project_root / 'results'
    findings = {'json_found': False, 'json_files': []}
    
    if not results_base.exists():
        return findings
    
    # Look for metrics JSON files modified in the last 5 minutes
    cutoff = time.time() - 300
    for json_file in results_base.rglob('tfm_comprehensive_metrics_*.json'):
        if json_file.stat().st_mtime > cutoff:
            findings['json_found'] = True
            findings['json_files'].append(str(json_file.name))
    
    if findings['json_found']:
        logger.info(
            f"Output check [{exp_id}]: found {len(findings['json_files'])} "
            f"recent metrics JSON(s)"
        )
    else:
        logger.warning(
            f"Output check [{exp_id}]: no recent metrics JSON found "
            f"in results/"
        )
    
    return findings


def main():
    parser = argparse.ArgumentParser(
        description="Run all TFM experiments sequentially"
    )
    parser.add_argument(
        "experiments", nargs="*",
        help="Specific experiment IDs to run (e.g., A1 D1 C2). "
             "If omitted, runs ALL experiments."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show execution plan without running anything"
    )
    parser.add_argument(
        "--resume-from",
        help="Skip all experiments before this ID (e.g., --resume-from C2)"
    )
    parser.add_argument(
        "--config-dir",
        help="Directory containing config_*.yaml files "
             "(default: PROJECT_ROOT/config/)"
    )
    args = parser.parse_args()

    # --- Setup paths ---
    project_root = find_project_root()
    config_dir = (
        Path(args.config_dir) if args.config_dir
        else project_root / 'config'
    )
    notebook_path = project_root / 'notebooks' / 'main.ipynb'
    active_config = config_dir / 'config.yaml'
    results_dir = project_root / 'results' / 'batch_runs'
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialize logger ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = BatchLogger(results_dir, timestamp)
    
    logger.banner(f"BATCH EXPERIMENT RUNNER v2.0 — {timestamp}")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Config dir:   {config_dir}")
    logger.info(f"Notebook:     {notebook_path}")
    logger.info(f"Results dir:  {results_dir}")
    logger.info(f"Log file:     {logger.log_file}")

    # --- Validate ---
    if not notebook_path.exists():
        logger.error(f"Notebook not found: {notebook_path}")
        sys.exit(1)
    if not config_dir.exists():
        logger.error(f"Config directory not found: {config_dir}")
        sys.exit(1)

    # --- Filter experiments ---
    if args.experiments:
        selected = [e for e in EXPERIMENTS if e[0] in args.experiments]
        if not selected:
            logger.error(
                f"No matching experiments for: {args.experiments}. "
                f"Available: {[e[0] for e in EXPERIMENTS]}"
            )
            sys.exit(1)
        logger.info(f"Selected experiments: {[e[0] for e in selected]}")
    else:
        selected = list(EXPERIMENTS)
        logger.info(f"Running ALL {len(selected)} experiments")

    # --- Resume-from ---
    if args.resume_from:
        idx = next(
            (i for i, e in enumerate(selected) if e[0] == args.resume_from),
            None
        )
        if idx is None:
            logger.error(f"--resume-from '{args.resume_from}' not found")
            sys.exit(1)
        skipped = selected[:idx]
        selected = selected[idx:]
        logger.info(
            f"Resuming from {args.resume_from}, "
            f"skipping: {[e[0] for e in skipped]}"
        )

    # --- Detect executor ---
    executor = get_executor(logger)
    if executor is None and not args.dry_run:
        logger.error(
            "Neither papermill nor nbconvert found. "
            "Install: pip install papermill"
        )
        sys.exit(1)

    # --- Log execution plan ---
    logger.banner(f"EXECUTION PLAN — {len(selected)} experiments")
    for i, (exp_id, config_file, desc) in enumerate(selected, 1):
        src = config_dir / config_file
        exists = "OK" if src.exists() else "MISSING"
        logger.info(
            f"  {i:>2}. [{exp_id:<5}] {config_file:<22} "
            f"{exists:<8} {desc}"
        )

    if args.dry_run:
        logger.info("DRY RUN — no experiments executed.")
        logger.close()
        return

    # --- Check all config files exist before starting ---
    missing = [
        (eid, cf) for eid, cf, _ in selected
        if not (config_dir / cf).exists()
    ]
    if missing:
        logger.error("Missing config files:")
        for eid, cf in missing:
            logger.error(f"  {eid}: {config_dir / cf}")
        logger.close()
        sys.exit(1)
    logger.info("All config files validated OK")

    # --- Backup current config ---
    backup_path = config_dir / 'config_backup_before_batch.yaml'
    if active_config.exists():
        robust_copy(active_config, backup_path, logger)
        logger.info(f"Backed up config.yaml -> {backup_path.name}")

    # --- Initialize incremental summary ---
    summary_path = results_dir / f"batch_summary_{timestamp}.json"
    summary = IncrementalSummary(summary_path, timestamp, len(selected))
    logger.info(f"Incremental summary: {summary_path}")

    # --- Execute experiments ---
    total_start = time.time()

    for i, (exp_id, config_file, desc) in enumerate(selected, 1):
        src_config = config_dir / config_file
        run_log = results_dir / f"run_{exp_id}_{timestamp}.log"
        output_nb = results_dir / f"main_{exp_id}_{timestamp}.ipynb"

        logger.section(
            f"[{i}/{len(selected)}] STARTING: {exp_id} — {desc}"
        )
        logger.info(f"Config file:  {config_file}")
        logger.info(f"Notebook log: {run_log.name}")
        logger.info(f"Output nb:    {output_nb.name}")

        # Copy config
        try:
            robust_copy(src_config, active_config, logger)
        except Exception as e:
            logger.error(f"Failed to copy config for {exp_id}: {e}")
            logger.error(traceback.format_exc())
            summary.add_result({
                'experiment': exp_id,
                'config': config_file,
                'status': f'ERROR: config copy failed — {e}',
                'elapsed': '0:00:00',
                'elapsed_s': 0,
                'description': desc,
            }, time.time() - total_start)
            continue

        # Execute notebook
        exp_start = time.time()
        status = "UNKNOWN"
        rc = -1

        try:
            logger.info(f"Executing notebook with {executor}...")

            if executor == "papermill":
                rc = run_notebook_papermill(
                    notebook_path, output_nb, run_log, logger
                )
            else:
                rc = run_notebook_nbconvert(
                    notebook_path, output_nb, run_log, logger
                )

            elapsed = time.time() - exp_start
            status = "OK" if rc == 0 else f"FAIL (rc={rc})"

            if rc != 0:
                logger.error(
                    f"{exp_id} failed with return code {rc}. "
                    f"Check log: {run_log}"
                )
                # Log last 20 lines of the run log for diagnostics
                try:
                    with open(run_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    tail = lines[-20:] if len(lines) > 20 else lines
                    logger.error(f"Last lines of {run_log.name}:")
                    for line in tail:
                        logger.error(f"  | {line.rstrip()}")
                except Exception:
                    pass

        except subprocess.TimeoutExpired:
            elapsed = time.time() - exp_start
            status = "TIMEOUT (12h limit)"
            logger.error(
                f"{exp_id} TIMED OUT after {elapsed/3600:.1f}h"
            )

        except KeyboardInterrupt:
            elapsed = time.time() - exp_start
            status = "INTERRUPTED (Ctrl+C)"
            logger.warning(f"{exp_id} interrupted by user after {elapsed:.0f}s")
            # Save what we have so far
            summary.add_result({
                'experiment': exp_id,
                'config': config_file,
                'status': status,
                'elapsed': str(timedelta(seconds=int(elapsed))),
                'elapsed_s': int(elapsed),
                'description': desc,
            }, time.time() - total_start)
            summary.finalize(time.time() - total_start)
            logger.banner("BATCH INTERRUPTED BY USER")
            logger.info(f"Partial summary saved: {summary_path}")
            # Restore config
            if backup_path.exists():
                try:
                    robust_copy(backup_path, active_config, logger)
                except Exception:
                    pass
            logger.close()
            sys.exit(130)

        except Exception as e:
            elapsed = time.time() - exp_start
            status = f"ERROR: {e}"
            logger.error(f"{exp_id} raised exception: {e}")
            logger.error(traceback.format_exc())

        elapsed_str = str(timedelta(seconds=int(elapsed)))

        # Check output
        output_info = check_experiment_output(project_root, exp_id, logger)

        result_entry = {
            'experiment': exp_id,
            'config': config_file,
            'status': status,
            'elapsed': elapsed_str,
            'elapsed_s': int(elapsed),
            'description': desc,
            'return_code': rc,
            'output_json_found': output_info['json_found'],
            'output_json_files': output_info['json_files'],
        }

        # Save incrementally
        summary.add_result(result_entry, time.time() - total_start)

        icon = "OK" if "OK" in status else "FAIL"
        logger.info(
            f"[{icon}] {exp_id} completed in {elapsed_str} — {status}"
        )

        # ── F-001: AI run report (auto, non-blocking) ────────────────────────
        if rc == 0:
            try:
                ai_script = project_root / "scripts" / "08_ai_run_report.py"
                if ai_script.exists():
                    logger.info(f"Generating AI report for {exp_id}...")
                    ai_res = subprocess.run(
                        [sys.executable, str(ai_script), "--exp", exp_id],
                        capture_output=True, text=True,
                        timeout=120, cwd=str(project_root),
                    )
                    if ai_res.returncode == 0:
                        logger.info(f"AI report saved: results/exp_{exp_id}/report_ai.md")
                    else:
                        logger.warning(f"AI report failed (non-critical): {ai_res.stderr[:200]}")
            except subprocess.TimeoutExpired:
                logger.warning(f"AI report timed out for {exp_id} (skipped)")
            except Exception as _ai_exc:
                logger.warning(f"AI report skipped for {exp_id}: {_ai_exc}")

        # ── F-002: Anomaly check (auto, non-blocking) ────────────────────────
        if rc == 0:
            try:
                anom_script = project_root / "scripts" / "09_anomaly_detector.py"
                if anom_script.exists():
                    anom_res = subprocess.run(
                        [sys.executable, str(anom_script), "--exp", exp_id],
                        capture_output=True, text=True,
                        timeout=30, cwd=str(project_root),
                    )
                    for line in anom_res.stdout.splitlines():
                        if "[!!]" in line:
                            logger.warning(f"ANOMALY DETECTED: {line.strip()}")
            except Exception:
                pass

        # Log cumulative progress
        completed = i
        remaining = len(selected) - i
        total_so_far = time.time() - total_start
        if remaining > 0 and completed > 0:
            avg_time = total_so_far / completed
            eta = avg_time * remaining
            logger.info(
                f"Progress: {completed}/{len(selected)} done, "
                f"{remaining} remaining, "
                f"ETA: ~{timedelta(seconds=int(eta))}"
            )

    # --- Restore original config ---
    if backup_path.exists():
        try:
            robust_copy(backup_path, active_config, logger)
            logger.info("Restored original config.yaml from backup")
        except Exception as e:
            logger.error(f"Failed to restore config backup: {e}")

    # --- Finalize summary ---
    total_elapsed = time.time() - total_start
    summary.finalize(total_elapsed)

    # --- Print final summary table ---
    logger.banner(
        f"BATCH COMPLETE — "
        f"{summary.data['total_completed']} experiments in "
        f"{timedelta(seconds=int(total_elapsed))}"
    )
    logger.info(f"{'Exp':<6} {'Status':<20} {'Time':>10}  Description")
    logger.info('─' * 70)

    for r in summary.data['experiments']:
        icon = "+" if "OK" in r['status'] else "X"
        logger.info(
            f"  {icon} {r['experiment']:<5} {r['status']:<20} "
            f"{r['elapsed']:>10}  {r['description']}"
        )

    logger.info('─' * 70)
    logger.info(
        f"  {summary.data['total_ok']}/{summary.data['total_completed']} "
        f"succeeded, {summary.data['total_failed']} failed"
    )
    logger.info(f"  Summary JSON: {summary_path}")
    logger.info(f"  Full log:     {logger.log_file}")
    # ── Batch PDF report (auto, non-blocking) ──────────────────────────────
    try:
        batch_report_script = project_root / "scripts" / "08_batch_report.py"
        if batch_report_script.exists():
            logger.info("Generating batch process report...")
            br = subprocess.run(
                [sys.executable, str(batch_report_script),
                 "--summary", str(summary_path),
                 "--exps"] + [r["experiment"] for r in summary.data["experiments"]
                              if "OK" in r.get("status", "")],
                capture_output=True, text=True,
                timeout=300, cwd=str(project_root),
            )
            if br.returncode == 0:
                logger.info("Batch report saved: results/batch_report.pdf")
            else:
                logger.warning("Batch report failed (non-critical): " + br.stderr[:300])
        else:
            logger.info("08_batch_report.py not found — skipping batch report")
    except subprocess.TimeoutExpired:
        logger.warning("Batch report timed out (skipped)")
    except Exception as _br_exc:
        logger.warning("Batch report skipped: " + str(_br_exc))
    # ── End batch report ──────────────────────────────────────────────────────

    logger.banner("DONE")
    logger.close()


if __name__ == "__main__":
    main()
