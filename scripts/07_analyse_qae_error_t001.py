#!/usr/bin/env python3
"""
# =============================================================================
# STEP 7 — QAE error vs n_qubits diagnostic analysis [T-001]
# =============================================================================
# Investigates why QAE error INCREASES with more qubits in noiseless simulations.
# Tests all three hypotheses documented in T-001:
#
#   Hypothesis A — Shot undersampling: fixed budget of S shots spread over
#     n_rounds rounds means high-k rounds become undersampled for large n.
#   Hypothesis B — VaR threshold quantization: the VaR value maps to a bin
#     boundary; quantization error depends on bin_width = range / 2^n_qubits.
#   Hypothesis C — Woerner vs IQAE mixing: error may be computed over Woerner
#     estimates instead of IQAE for some n values.
#
# Produces:
#   results/qae_diagnostics/t001_shot_analysis.json
#   results/qae_diagnostics/t001_threshold_analysis.json
#   results/qae_diagnostics/t001_summary.json
#   results/qae_diagnostics/t001_report.txt
#
# Usage:
#   python scripts/07_analyse_qae_error_t001.py
#   python scripts/07_analyse_qae_error_t001.py --shots 16384   # Hypothesis A test
#   python scripts/07_analyse_qae_error_t001.py --exp A1_n6_PA  # specific experiment
# =============================================================================
"""
"""
"""
import sys
import json
import math
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def find_project_root() -> Path:
    for p in [Path.cwd(), Path(__file__).resolve().parent.parent]:
        if (p / "config").is_dir():
            return p
    raise FileNotFoundError("Cannot find project root")


PROJECT_ROOT = find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "results" / "qae_diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hypothesis A analysis: shots_per_round vs n_qubits
# ---------------------------------------------------------------------------

def analyse_shot_budget(epsilon: float = 0.005, shots_values: list = None) -> dict:
    """
    Compute theoretical shots_per_round for each n_qubits value.

    IQAE with epsilon uses K_max = ceil(pi / (8 * epsilon)) Grover rounds max.
    With n_rounds = ceil(log2(K_max)) and fixed shot budget S:
        shots_per_round = S / n_rounds

    When shots_per_round < 50, the chi-squared test used internally by IQAE
    has insufficient power, and the estimated amplitude variance dominates.
    This is Hypothesis A: the APPARENT increase in error with n is actually
    an artefact of the fixed shot budget spread over more rounds.
    """
    if shots_values is None:
        shots_values = [1024, 4096, 16384]

    k_max = math.ceil(math.pi / (8 * epsilon))
    n_rounds = max(1, math.ceil(math.log2(k_max))) if k_max > 1 else 1

    results = {
        "epsilon": epsilon,
        "k_max": k_max,
        "n_rounds": n_rounds,
        "shot_analysis": {},
    }

    print(f"\n=== Hypothesis A: Shot Budget Analysis ===")
    print(f"epsilon={epsilon}, K_max={k_max}, n_rounds_est={n_rounds}")
    print(f"{'Shots':>8}  {'shots/round':>12}  {'Adequate?':>10}  {'Recommendation'}")
    print("-" * 60)

    for S in shots_values:
        spr = S // max(n_rounds, 1)
        adequate = spr >= 50
        rec = "OK" if adequate else f"Increase to {n_rounds * 200}+"
        print(f"{S:>8}  {spr:>12}  {str(adequate):>10}  {rec}")
        results["shot_analysis"][S] = {
            "shots_per_round": spr,
            "adequate": adequate,
            "recommended_min_shots": n_rounds * 200,
        }

    print()
    return results


# ---------------------------------------------------------------------------
# Hypothesis B analysis: VaR threshold quantization error
# ---------------------------------------------------------------------------

def analyse_threshold_quantization(
    portfolio_losses: np.ndarray,
    var_threshold: float,
    n_qubits_range: list = None,
) -> dict:
    """
    Compute VaR threshold quantization error for each n_qubits value.

    bin_width = (l_max - l_min) / 2^n_qubits
    quant_error = |var_threshold - nearest_bin_edge|

    If the VaR threshold falls close to a bin edge for n=3 but near a bin
    centre for n=6, the error ordering can invert — explaining the observed
    non-monotone QAE error pattern.
    """
    if n_qubits_range is None:
        n_qubits_range = [3, 4, 5, 6]

    l_min = float(np.min(portfolio_losses))
    l_max = float(np.max(portfolio_losses))
    l_range = l_max - l_min

    results = {
        "var_threshold": var_threshold,
        "l_min": l_min,
        "l_max": l_max,
        "l_range": l_range,
        "threshold_analysis": {},
    }

    print("=== Hypothesis B: VaR Threshold Quantization ===")
    print(f"VaR threshold = {var_threshold:.6f}, range = [{l_min:.4f}, {l_max:.4f}]")
    print(f"{'n_qubits':>10}  {'bin_width':>12}  {'threshold_bin':>15}  "
          f"{'quant_error':>14}  {'rel_error':>12}")
    print("-" * 70)

    for n in n_qubits_range:
        n_bins = 2 ** n
        bin_width = l_range / n_bins
        edges = np.linspace(l_min, l_max, n_bins + 1)

        # Find which bin contains the threshold
        bin_idx = int(np.searchsorted(edges, var_threshold, side="right")) - 1
        bin_idx = max(0, min(bin_idx, n_bins - 1))

        # Quantization error: distance from threshold to nearest bin boundary
        lower = edges[bin_idx]
        upper = edges[bin_idx + 1] if bin_idx + 1 < len(edges) else edges[-1]
        quant_err = min(abs(var_threshold - lower), abs(var_threshold - upper))
        rel_err = quant_err / max(abs(var_threshold), 1e-10)

        print(f"{n:>10}  {bin_width:>12.6f}  {bin_idx:>15}/{n_bins}  "
              f"{quant_err:>14.6f}  {rel_err:>12.4f}")

        results["threshold_analysis"][n] = {
            "n_bins": n_bins,
            "bin_width": bin_width,
            "threshold_bin_index": bin_idx,
            "bin_lower": float(lower),
            "bin_upper": float(upper),
            "quantization_error": float(quant_err),
            "relative_quantization_error": float(rel_err),
        }

    print()
    return results


# ---------------------------------------------------------------------------
# Parse existing experiment results for Hypothesis C
# ---------------------------------------------------------------------------

def analyse_method_mixing(exp_results_dir: Path) -> dict:
    """
    Check whether reported QAE error is computed from IQAE or Woerner.

    Hypothesis C: if for some n_qubits the Woerner estimate is used instead
    of IQAE, the error comparison is not apples-to-apples.
    """
    print("=== Hypothesis C: IQAE vs Woerner Method Used ===")
    experiments_checked = {}

    # Look for A1_n* experiments
    pattern_exps = {
        "n3": "exp_A1_n3_PA",
        "n4": "exp_A1_PA",
        "n5": "exp_A1_n5_PA",
        "n6": "exp_A1_n6_PA",
    }

    for label, exp_name in pattern_exps.items():
        metrics_path = exp_results_dir / exp_name / "tfm_comprehensive_metrics_latest.json"
        if not metrics_path.exists():
            print(f"  {label} ({exp_name}): NOT FOUND — run experiment first")
            continue

        try:
            with open(metrics_path) as fh:
                m = json.load(fh)

            # Extract method used and QAE error
            qae_data     = m.get("qae_diagnostics_t001", {})
            metrics      = m.get("metrics", {})
            qae_method   = metrics.get("quantum", {}).get("best_method", "unknown")
            qae_error    = metrics.get("quantum", {}).get("qae_relative_error", None)
            n_qubits_used = qae_data.get("n_qubits_used", None)
            spr          = qae_data.get("shots_per_round_est", None)
            tq_err       = qae_data.get("var_threshold_quantization_error", None)

            print(f"  {label}: method={qae_method}, "
                  f"n_qubits={n_qubits_used}, "
                  f"qae_error={qae_error}, "
                  f"shots_per_round={spr}, "
                  f"threshold_quant_err={tq_err}")

            experiments_checked[label] = {
                "experiment": exp_name,
                "method_used": qae_method,
                "n_qubits": n_qubits_used,
                "qae_error": qae_error,
                "shots_per_round_est": spr,
                "threshold_quantization_error": tq_err,
            }
        except Exception as exc:
            print(f"  {label}: ERROR reading metrics — {exc}")

    print()
    return experiments_checked


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T-001 QAE error diagnostic")
    parser.add_argument("--shots", type=int, nargs="+", default=[1024, 4096, 16384],
                        help="Shot budgets to analyse for Hypothesis A")
    parser.add_argument("--epsilon", type=float, default=0.005,
                        help="IQAE epsilon target (default: 0.005)")
    parser.add_argument("--n-qubits", type=int, nargs="+", default=[3, 4, 5, 6],
                        help="n_qubits values to test")
    args = parser.parse_args()

    print("=" * 65)
    print("T-001: QAE ERROR DIAGNOSTIC — Why does error increase with n?")
    print("=" * 65)
    print()

    results = {
        "timestamp": datetime.now().isoformat(),
        "epsilon": args.epsilon,
        "n_qubits_range": args.n_qubits,
    }

    # Hypothesis A
    hyp_a = analyse_shot_budget(epsilon=args.epsilon, shots_values=args.shots)
    results["hypothesis_A_shot_budget"] = hyp_a

    # Hypothesis B — use synthetic losses to demonstrate the effect
    rng = np.random.default_rng(42)
    synthetic_losses = -rng.normal(0.0005, 0.012, 1257)   # ~1257 trading days
    var_5pct = float(np.percentile(synthetic_losses, 95))

    hyp_b = analyse_threshold_quantization(
        portfolio_losses=synthetic_losses,
        var_threshold=var_5pct,
        n_qubits_range=args.n_qubits,
    )
    results["hypothesis_B_threshold_quantization"] = hyp_b

    # Hypothesis C — from existing experiment results
    exp_dir = PROJECT_ROOT / "results"
    hyp_c = analyse_method_mixing(exp_results_dir=exp_dir)
    results["hypothesis_C_method_mixing"] = hyp_c

    # Summary diagnosis
    print("=== DIAGNOSIS SUMMARY ===")

    # Check Hypothesis A: are shots_per_round < 50 for any shot budget?
    spr_4096 = hyp_a["shot_analysis"].get(4096, {}).get("shots_per_round", 999)
    if spr_4096 < 50:
        print(f"  [HYPOTHESIS A CONFIRMED] shots_per_round={spr_4096} < 50 at 4096 shots.")
        print(f"  => IQAE rounds are undersampled. Increasing shots to "
              f"{hyp_a['n_rounds'] * 200}+ should reduce error for large n.")
        results["primary_diagnosis"] = "hypothesis_A_undersampling"
    else:
        print(f"  [Hypothesis A] shots_per_round={spr_4096} — adequate. "
              f"Undersampling not the primary cause.")

    # Check Hypothesis B: is quantization error non-monotone?
    q_errors = {n: v["relative_quantization_error"]
                for n, v in hyp_b["threshold_analysis"].items()}
    is_non_monotone = (
        len(q_errors) >= 2 and
        not all(q_errors[n] <= q_errors[n+1]
                for n, np_ in zip(sorted(q_errors)[:-1], sorted(q_errors)[1:]))
    )
    if is_non_monotone:
        print(f"  [HYPOTHESIS B PLAUSIBLE] Threshold quantization errors are "
              f"non-monotone in n: {q_errors}")
        print(f"  => VaR threshold position within bins varies non-monotonically.")
    else:
        print(f"  [Hypothesis B] Quantization errors are monotone — not the primary cause.")

    print()
    results["q_errors_by_n"] = q_errors

    # Write outputs
    json_out = OUTPUT_DIR / "t001_summary.json"
    with open(json_out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Full diagnostic saved: {json_out}")

    # Write text report
    report_lines = [
        "T-001 QAE ERROR DIAGNOSTIC REPORT",
        f"Generated: {results['timestamp']}",
        "=" * 65,
        "",
        "QUESTION: Why does QAE error increase with n_qubits in noiseless simulation?",
        "",
        "Observed error (from TODO):",
        "  n=3: 7.6% (PA), 19.2% (PC)",
        "  n=4: 14.4% (PA), 14.5% (PC)",
        "  n=5: 15.8% (PA), 20.5% (PC)",
        "  n=6: 30.8% (PA), 15.5% (PC)",
        "",
        "HYPOTHESIS A — Shot undersampling:",
        f"  epsilon={args.epsilon}, K_max={hyp_a['k_max']}, n_rounds={hyp_a['n_rounds']}",
        f"  shots_per_round @ 4096 shots: {spr_4096}",
        f"  {'CONFIRMED' if spr_4096 < 50 else 'NOT confirmed'}: "
        f"{'Undersampling likely dominant cause' if spr_4096 < 50 else 'Shots adequate'}",
        f"  Action: re-run A1_n6_PA with --shots 16384 and compare error",
        "",
        "HYPOTHESIS B — VaR threshold quantization:",
        "  Quantization errors per n_qubits:",
    ]
    for n in sorted(q_errors):
        report_lines.append(f"    n={n}: rel_error={q_errors[n]:.4f}")
    report_lines += [
        f"  {'NON-MONOTONE — plausible secondary cause' if is_non_monotone else 'Monotone — not primary cause'}",
        "",
        "HYPOTHESIS C — Method mixing (IQAE vs Woerner):",
        "  See hypothesis_C_method_mixing in JSON for per-experiment details.",
        "  Action: verify 'best_method' field in each A1_n* experiment metrics.",
        "",
        "RECOMMENDED ACTIONS:",
        "  1. Re-run A1_n6_PA with shots=16384 (tests Hyp A)",
        "  2. Check 'best_method' in each A1_n* metrics JSON (tests Hyp C)",
        "  3. Inspect var_threshold_bin_index in qae_circuits.txt logs (tests Hyp B)",
        "  4. Once confirmed, document correct narrative in paper Section 5.2",
    ]

    report_path = OUTPUT_DIR / "t001_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Text report saved:     {report_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
