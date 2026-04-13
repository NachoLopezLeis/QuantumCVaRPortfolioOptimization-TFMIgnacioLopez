#!/usr/bin/env python3
# =============================================================================
# Smoke Test Consolidated PDF Report
# =============================================================================
# Generates a single PDF comparing all smoke test experiments (A1_PA/PB/PC).
# Includes AI narrative, metrics comparison table, and OOS figures side by side.
#
# Install: pip install reportlab pillow
#
# Usage:
#   python scripts\08_smoke_report.py
#   python scripts\08_smoke_report.py --exps A1_PA A1_PB A1_PC
#   python scripts\08_smoke_report.py --backend claude --api-key sk-ant-...
# =============================================================================

import sys
import json
import argparse
import urllib.request
import urllib.error
import textwrap
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = None
for _p in [Path.cwd(), Path(__file__).resolve().parent.parent]:
    if (_p / "config").is_dir():
        PROJECT_ROOT = _p
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError("Cannot find project root")


# ---------------------------------------------------------------------------
# AI prompt — comparative analysis
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in quantum portfolio optimization with CVaR. "
    "You are analyzing the smoke test results of a Masters Thesis experiment. "
    "Three portfolio universes were tested: PA (peripheral assets), "
    "PB (central/high-centrality assets), PC (predefined S&P assets). "
    "All three use the same algorithm: IQAE + hybrid subgradient optimizer, "
    "noiseless simulation, n_qubits=4, epsilon=0.005.\n\n"
    "Write a comparative technical report. Plain text only, no markdown, no asterisks.\n\n"
    "SMOKE TEST VERDICT\n"
    "State clearly if the pipeline is working correctly based on these 3 runs. "
    "Mention benchmark.computed, quantum_cvar != 0, and convergence.\n\n"
    "COMPARATIVE ANALYSIS\n"
    "Compare PA vs PB vs PC: which universe benefits most from CVaR optimization "
    "and why (peripheral vs central assets hypothesis).\n\n"
    "KEY METRICS ACROSS UNIVERSES\n"
    "List the most important numbers: CVaR improvement per universe, "
    "OOS CVaR quantum vs classical, quantum rank.\n\n"
    "ANOMALIES\n"
    "Any values outside expected ranges across the 3 experiments. "
    "If none: None detected.\n\n"
    "READY FOR FULL RUN?\n"
    "Based on these smoke test results, state clearly: YES the pipeline is "
    "ready for all 34 experiments, or NO with specific reasons.\n\n"
    "Be precise. Reference specific numbers."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment(exp_dir):
    mf = exp_dir / "tfm_comprehensive_metrics_latest.json"
    if not mf.exists():
        return None
    return json.loads(mf.read_text(encoding="utf-8"))


def extract_key_metrics(m, exp_id):
    opt   = m.get("metrics", {}).get("optimization", {})
    risk  = m.get("metrics", {}).get("risk", {})
    bench = m.get("metrics", {}).get("benchmarks", {})
    oos   = m.get("metrics", {}).get("oos_validation", {})
    qae   = m.get("metrics", {}).get("quantum", {}).get("qae", {})
    fp    = m.get("convergence_fingerprint", {})
    diag  = m.get("qae_diagnostics_t001", {})

    classical = {}
    for meth, res in bench.get("classical_results", {}).items():
        classical[meth] = round(res.get("optimal_cvar", 0) * 100, 4)

    return {
        "exp_id":              exp_id,
        "initial_cvar_pct":    round(risk.get("initial",   {}).get("cvar_loss", 0) * 100, 4),
        "optimized_cvar_pct":  round(risk.get("optimized", {}).get("cvar_loss", 0) * 100, 4),
        "cvar_improvement_pct": round(opt.get("cvar_improvement_pct", 0), 3),
        "num_iterations":      opt.get("num_iterations"),
        "converged":           opt.get("converged"),
        "benchmark_computed":  bench.get("computed"),
        "quantum_cvar_pct":    round(bench.get("quantum_cvar", 0) * 100, 4),
        "best_classical_method": bench.get("best_classical_method"),
        "best_classical_cvar_pct": round((bench.get("best_classical_cvar") or 0) * 100, 4),
        "classical_cvars":     classical,
        "oos_quantum_cvar_pct": round((oos.get("quantum_oos_cvar") or 0) * 100, 4),
        "oos_rank":            oos.get("quantum_cvar_rank"),
        "qae_error_pct":       round(qae.get("cvar_error", 0) * 100, 2),
        "shots_per_round":     diag.get("shots_per_round_est"),
        "n_qubits":            diag.get("n_qubits_used"),
        "iter_to_95pct":       fp.get("iter_to_95pct_improvement"),
        "n_regressions":       fp.get("n_regressions"),
    }


# ---------------------------------------------------------------------------
# AI backends
# ---------------------------------------------------------------------------

def generate_ollama(data_json, model="llama3.1", host="http://localhost:11434"):
    payload = {
        "model":   model,
        "prompt":  SYSTEM_PROMPT + "\n\nSmoke test data:\n" + data_json,
        "stream":  False,
        "options": {"temperature": 0.2, "num_predict": 1800},
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        raise ConnectionError(
            "Cannot reach Ollama at " + host +
            ". Run: docker start ollama. Error: " + str(e)
        )


def generate_claude(data_json, api_key):
    payload = {
        "model":      "claude-sonnet-4-20250514",
        "max_tokens": 1800,
        "system":     SYSTEM_PROMPT,
        "messages":   [{"role": "user",
                        "content": "Smoke test data:\n" + data_json}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["content"][0]["text"].strip()
    except urllib.error.URLError as e:
        raise ConnectionError("Claude API error: " + str(e))


# ---------------------------------------------------------------------------
# ReportLab PDF builder
# ---------------------------------------------------------------------------

def build_pdf(all_metrics, ai_text, output_path, exp_dirs):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image, KeepTogether, PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    BLUE  = colors.HexColor("#185FA5")
    LBLUE = colors.HexColor("#E8F1FB")
    GREEN = colors.HexColor("#0F6E56")
    RED   = colors.HexColor("#A32D2D")
    LGRAY = colors.HexColor("#F5F5F5")
    MGRAY = colors.HexColor("#CCCCCC")
    BLACK = colors.HexColor("#222222")

    styles = getSampleStyleSheet()

    def sty(name, **kw):
        return ParagraphStyle(
            name + "_" + str(abs(hash(str(kw)))),
            parent=styles[name], **kw
        )

    H1   = sty("Heading1", fontSize=17, textColor=BLUE, spaceAfter=4)
    H2   = sty("Heading2", fontSize=13, textColor=BLUE, spaceBefore=14, spaceAfter=4)
    H3   = sty("Heading3", fontSize=10, textColor=BLACK, spaceBefore=8, spaceAfter=2)
    BODY = sty("Normal",   fontSize=9.5, leading=14, textColor=BLACK)
    META = sty("Normal",   fontSize=8,  textColor=colors.HexColor("#666666"))
    CAPN = sty("Normal",   fontSize=8,  textColor=colors.HexColor("#555555"),
               alignment=TA_CENTER, spaceAfter=8)
    FOOT = sty("Normal",   fontSize=7.5,
               textColor=colors.HexColor("#999999"), alignment=TA_CENTER)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=18*mm,  bottomMargin=18*mm,
    )
    story = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Cover ────────────────────────────────────────────────────────────────
    exp_ids = [m["exp_id"] for m in all_metrics]
    story.append(Paragraph("Smoke Test Report", H1))
    story.append(Paragraph(
        "Hybrid Quantum-Classical CVaR Portfolio Optimization &mdash; TFM",
        sty("Normal", fontSize=11, textColor=BLUE, spaceAfter=3)
    ))
    story.append(Paragraph(
        "Experiments: " + " | ".join(exp_ids) + " &nbsp;&nbsp; Generated: " + now,
        META
    ))
    story.append(Paragraph(
        "Algorithm: IQAE + hybrid subgradient &nbsp;|&nbsp; "
        "n_qubits=4, epsilon=0.005, noiseless",
        META
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=10))

    # ── Summary comparison table ──────────────────────────────────────────────
    header = ["Metric", "A1_PA (peripheral)", "A1_PB (central)", "A1_PC (predefined)"]
    metrics_order = [
        ("initial_cvar_pct",       "Initial CVaR (%)"),
        ("optimized_cvar_pct",     "Optimized CVaR (%)"),
        ("cvar_improvement_pct",   "CVaR Improvement (%)"),
        ("best_classical_cvar_pct","Best Classical CVaR (%)"),
        ("best_classical_method",  "Best Classical Method"),
        ("oos_quantum_cvar_pct",   "OOS CVaR - Quantum (%)"),
        ("oos_rank",               "OOS Rank (quantum)"),
        ("qae_error_pct",          "QAE Error (%)"),
        ("shots_per_round",        "Shots/Round (est.)"),
        ("num_iterations",         "Iterations"),
        ("iter_to_95pct",          "Iters to 95% improvement"),
        ("n_regressions",          "N Regressions"),
        ("benchmark_computed",     "Benchmark computed"),
    ]

    data = [header]
    for key, label in metrics_order:
        row = [label]
        for m in all_metrics:
            val = m.get(key)
            if val is None:
                row.append("N/A")
            elif isinstance(val, bool):
                row.append("YES" if val else "NO")
            elif isinstance(val, float):
                row.append(str(round(val, 4)))
            else:
                row.append(str(val))
        data.append(row)

    col_w = [52*mm, 38*mm, 38*mm, 38*mm]
    ct = Table(data, colWidths=col_w, repeatRows=1)

    style_cmds = [
        ("BACKGROUND",   (0, 0), (-1, 0),  BLUE),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
        ("GRID",         (0, 0), (-1, -1), 0.4, MGRAY),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),  [colors.white, LGRAY]),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("FONTNAME",     (0, 0), (0, -1),  "Helvetica-Bold"),
    ]
    # Highlight CVaR improvement row green
    for i, (key, _) in enumerate(metrics_order, 1):
        if key == "cvar_improvement_pct":
            for j in range(1, 4):
                style_cmds.append(("TEXTCOLOR", (j, i), (j, i), GREEN))
                style_cmds.append(("FONTNAME",  (j, i), (j, i), "Helvetica-Bold"))
        if key == "benchmark_computed":
            for j in range(1, 4):
                val = all_metrics[j-1].get(key, False)
                c = GREEN if val else RED
                style_cmds.append(("TEXTCOLOR", (j, i), (j, i), c))
                style_cmds.append(("FONTNAME",  (j, i), (j, i), "Helvetica-Bold"))

    ct.setStyle(TableStyle(style_cmds))
    story.append(KeepTogether([
        Paragraph("Comparative Metrics -- All Smoke Test Experiments", H2),
        ct,
    ]))
    story.append(Spacer(1, 10))

    # ── AI analysis ──────────────────────────────────────────────────────────
    story.append(Paragraph("AI Comparative Analysis", H2))
    section_keys = {
        "SMOKE TEST VERDICT", "COMPARATIVE ANALYSIS",
        "KEY METRICS ACROSS UNIVERSES", "ANOMALIES", "READY FOR FULL RUN?",
    }
    for line in ai_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 3))
            continue
        if line.upper() in section_keys:
            story.append(Spacer(1, 4))
            story.append(Paragraph(line, H3))
        elif line.startswith("-"):
            story.append(Paragraph("&bull; " + line[1:].strip(), BODY))
        elif line and line[0].isdigit() and len(line) > 2 and line[1] in ".):":
            story.append(Paragraph(line, BODY))
        else:
            story.append(Paragraph(textwrap.fill(line, 110), BODY))
    story.append(Spacer(1, 10))

    # ── OOS figures — one per experiment ─────────────────────────────────────
    has_figures = False
    for m in all_metrics:
        exp_id  = m["exp_id"]
        exp_dir = exp_dirs.get(exp_id)
        if exp_dir is None:
            continue
        fig_path = exp_dir / ("oos_triple_distribution_" + exp_id + ".png")
        if not fig_path.exists():
            continue
        try:
            img = Image(str(fig_path), width=165*mm, height=70*mm)
            if not has_figures:
                story.append(Paragraph(
                    "Out-of-Sample CVaR Prediction Accuracy", H2
                ))
                has_figures = True
            story.append(KeepTogether([
                Paragraph(exp_id + " -- Predicted vs Realized Loss Distribution", H3),
                img,
                Paragraph(
                    "Parametric (Normal) | Monte Carlo | Quantum (QAE) &nbsp; "
                    "KS p=0.000 for all methods -- regime shift bias "
                    "(training includes COVID crash, OOS is 2024-2025 bull market)",
                    CAPN
                ),
            ]))
            story.append(Spacer(1, 6))
        except Exception as e:
            story.append(Paragraph("Figure unavailable for " + exp_id + ": " + str(e), BODY))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MGRAY, spaceBefore=12))
    story.append(Paragraph(
        "TFM: Hybrid Quantum-Classical Portfolio Optimization with CVaR -- "
        "Ignacio Lopez Leis, Universidad Autonoma de Madrid",
        FOOT
    ))

    doc.build(story)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smoke test consolidated PDF report"
    )
    parser.add_argument("--exps", nargs="+",
                        default=["A1_PA", "A1_PB", "A1_PC"],
                        help="Experiment IDs to include (default: A1_PA A1_PB A1_PC)")
    parser.add_argument("--backend",     choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--model",       default="llama3.1")
    parser.add_argument("--api-key",     dest="api_key", default="")
    parser.add_argument("--ollama-host", dest="ollama_host",
                        default="http://localhost:11434")
    parser.add_argument("--no-ai", dest="no_ai", action="store_true",
                        help="Skip AI analysis (tables and figures only)")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"

    # Load experiments
    all_metrics = []
    exp_dirs    = {}
    missing     = []

    for exp_id in args.exps:
        exp_dir = results_dir / ("exp_" + exp_id)
        m = load_experiment(exp_dir) if exp_dir.exists() else None
        if m is None:
            missing.append(exp_id)
            print("  WARNING: " + exp_id + " not found -- skipping")
            continue
        all_metrics.append(extract_key_metrics(m, exp_id))
        exp_dirs[exp_id] = exp_dir
        print("  Loaded: " + exp_id)

    if not all_metrics:
        sys.exit("No experiment results found. Run the smoke test first.")

    if missing:
        print("  Missing experiments: " + ", ".join(missing))

    # AI analysis
    ai_text = ""
    if not args.no_ai:
        print("  Generating AI comparative analysis...")
        data_json = json.dumps(all_metrics, indent=2)
        try:
            if args.backend == "ollama":
                ai_text = generate_ollama(data_json, args.model, args.ollama_host)
            else:
                if not args.api_key:
                    sys.exit("--api-key required for --backend claude")
                ai_text = generate_claude(data_json, args.api_key)
            print("  AI analysis complete")
        except ConnectionError as e:
            print("  AI skipped: " + str(e))
            ai_text = (
                "AI analysis unavailable (Ollama not running).\n"
                "Start with: docker start ollama"
            )
    else:
        ai_text = "AI analysis skipped (--no-ai flag)."

    # Build PDF
    out_path = results_dir / "smoke_test_report.pdf"
    print("  Building PDF...")
    try:
        build_pdf(all_metrics, ai_text, out_path, exp_dirs)
        size_kb = out_path.stat().st_size // 1024
        print("PDF saved: " + str(out_path) + " (" + str(size_kb) + " KB)")
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab pillow")

    # Also save JSON summary
    summary_path = results_dir / "smoke_test_summary.json"
    summary_path.write_text(
        json.dumps({
            "generated": datetime.now().isoformat(),
            "experiments": all_metrics,
            "ai_analysis": ai_text,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print("Summary JSON: " + str(summary_path))


if __name__ == "__main__":
    main()
