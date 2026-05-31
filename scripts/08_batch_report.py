#!/usr/bin/env python3
# =============================================================================
# 08_batch_report.py -- Full batch run PDF report
# =============================================================================
# Generates a single PDF summarizing the entire batch run:
#   Page 1: Execution dashboard (pass/fail per experiment, timings, errors)
#   Page 2: CVaR results comparison across all experiments
#   Page 3: OOS results + benchmark summary
#   Page 4: AI narrative analysis of the full batch
#
# Called automatically by run_all_experiments.py at the end of every batch.
#
# Usage:
#   python scripts\08_batch_report.py
#   python scripts\08_batch_report.py --exps A1_PA A1_PB A1_PC
#   python scripts\08_batch_report.py --model phi3
#   python scripts\08_batch_report.py --no-ai
#
# Output: results/batch_report.pdf
# =============================================================================

import sys, json, argparse, urllib.request, urllib.error, textwrap
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
# AI prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert reviewer for a quantum computing Masters Thesis. "
    "You have just received results from a full batch of experiments on "
    "quantum CVaR portfolio optimization using IQAE.\n\n"
    "Write a batch analysis report. Plain text only, no markdown, no asterisks.\n\n"
    "BATCH VERDICT\n"
    "State: how many experiments succeeded, how many failed, "
    "and whether the batch is scientifically valid.\n\n"
    "PERFORMANCE SUMMARY\n"
    "What is the average CVaR improvement across experiments? "
    "Which experiment performed best and worst and why?\n\n"
    "UNIVERSE COMPARISON\n"
    "Compare PA (peripheral), PB (central), PC (predefined) universes. "
    "Which benefits most from quantum optimization?\n\n"
    "NOISE SENSITIVITY\n"
    "If multiple noise groups are present (A1 noiseless, B2 low, C2 medium, "
    "D1 high), describe the trend in CVaR improvement vs noise level.\n\n"
    "QUANTUM ADVANTAGE\n"
    "Based on these results, does quantum optimization outperform the best "
    "classical method? In how many experiments? By how much on average?\n\n"
    "RECOMMENDATIONS\n"
    "2-3 concrete improvements for the next batch or the paper.\n\n"
    "Be precise. Use exact numbers. Max 700 words."
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_experiment(exp_id):
    mf = PROJECT_ROOT / "results" / ("exp_" + exp_id) / "tfm_comprehensive_metrics_latest.json"
    if not mf.exists():
        return None
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None

def extract(m, exp_id):
    opt   = m.get("metrics", {}).get("optimization", {})
    risk  = m.get("metrics", {}).get("risk", {})
    bench = m.get("metrics", {}).get("benchmarks", {})
    oos   = m.get("metrics", {}).get("oos_validation", {})
    qae   = m.get("metrics", {}).get("quantum", {}).get("qae", {})
    cfg   = m.get("metadata", {}).get("configuration", {})

    q_cvar = bench.get("quantum_cvar", 0) or risk.get("optimized", {}).get("cvar_loss", 0)

    classical_cvars = {
        k: round(v.get("optimal_cvar", 0) * 100, 4)
        for k, v in bench.get("classical_results", {}).items()
    }

    oos_ports = oos.get("portfolios", {})

    return {
        "exp_id":              exp_id,
        "group":               exp_id.split("_")[0],
        "universe":            exp_id.split("_")[-1],
        "n_qubits":            cfg.get("phase2", {}).get("qae", {}).get("n_qubits"),
        "n_assets":            cfg.get("data", {}).get("n_effective"),
        "initial_cvar":        round(risk.get("initial",   {}).get("cvar_loss", 0) * 100, 4),
        "optimized_cvar":      round(risk.get("optimized", {}).get("cvar_loss", 0) * 100, 4),
        "cvar_improvement":    round(opt.get("cvar_improvement_pct", 0), 3),
        "converged":           opt.get("converged", False),
        "num_iterations":      opt.get("num_iterations", 0),
        "benchmark_computed":  bench.get("computed", False),
        "quantum_cvar":        round(q_cvar * 100, 4),
        "best_classical":      bench.get("best_classical_method", ""),
        "best_classical_cvar": round((bench.get("best_classical_cvar") or 0) * 100, 4),
        "classical_cvars":     classical_cvars,
        "qae_error":           round(qae.get("cvar_error", 0) * 100, 3),
        "oos_quantum_cvar":    round((oos.get("quantum_oos_cvar") or 0) * 100, 4),
        "oos_rank":            oos.get("quantum_cvar_rank"),
        "oos_quantum_pred_err": round(
            (oos_ports.get("Quantum CVaR", {}).get("cvar_prediction_error") or 0) * 100, 2
        ),
    }

def load_batch_summary(summary_path):
    if not summary_path or not Path(summary_path).exists():
        return None
    try:
        return json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except Exception:
        return None

# ---------------------------------------------------------------------------
# AI backends
# ---------------------------------------------------------------------------
def generate_ollama(data, model="llama3.1", host="http://localhost:11434"):
    payload = {
        "model":   model,
        "prompt":  SYSTEM_PROMPT + "\n\nBatch data:\n" + json.dumps(data, indent=2),
        "stream":  False,
        "options": {"temperature": 0.15, "num_predict": 1400},
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read()).get("response", "").strip()
    except urllib.error.URLError as e:
        raise ConnectionError("Cannot reach Ollama. docker start ollama. " + str(e))

def generate_claude(data, api_key):
    payload = {
        "model": "claude-sonnet-4-20250514", "max_tokens": 1400,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user",
                      "content": "Batch data:\n" + json.dumps(data, indent=2)}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-api-key": api_key,
                 "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["content"][0]["text"].strip()
    except urllib.error.URLError as e:
        raise ConnectionError("Claude API error: " + str(e))

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
def P():
    from reportlab.lib import colors
    return {
        "blue":      colors.HexColor("#1A5FA8"),
        "blue_dark": colors.HexColor("#0F3D6E"),
        "blue_lt":   colors.HexColor("#EAF1FB"),
        "blue_mid":  colors.HexColor("#C5D9F0"),
        "green":     colors.HexColor("#0A6E4F"),
        "green_lt":  colors.HexColor("#E6F4EF"),
        "red":       colors.HexColor("#B02020"),
        "red_lt":    colors.HexColor("#FBEAEA"),
        "orange":    colors.HexColor("#C05A00"),
        "orange_lt": colors.HexColor("#FEF3E6"),
        "gray":      colors.HexColor("#555555"),
        "gray_lt":   colors.HexColor("#F5F5F5"),
        "gray_mid":  colors.HexColor("#CCCCCC"),
        "black":     colors.HexColor("#1A1A1A"),
        "white":     colors.white,
    }

# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------
def build_pdf(experiments, batch_summary, ai_text, output_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether, PageBreak, Image,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

    C   = P()
    STY = getSampleStyleSheet()

    def s(base, **kw):
        return ParagraphStyle(base + str(abs(hash(str(kw)))),
                               parent=STY[base], **kw)

    TITLE = s("Normal",  fontSize=20, textColor=C["blue_dark"],
               fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=2)
    SUBT  = s("Normal",  fontSize=11, textColor=C["blue"],
               alignment=TA_CENTER, spaceAfter=2)
    META  = s("Normal",  fontSize=8,  textColor=C["gray"],
               alignment=TA_CENTER, spaceAfter=8)
    H2    = s("Heading2",fontSize=12, textColor=C["blue_dark"],
               spaceBefore=12, spaceAfter=4)
    H3    = s("Normal",  fontSize=10, textColor=C["blue"],
               fontName="Helvetica-Bold", spaceBefore=8, spaceAfter=3)
    BODY  = s("Normal",  fontSize=9.5,textColor=C["black"], leading=14)
    SMALL = s("Normal",  fontSize=8,  textColor=C["gray"],  leading=11)
    CAPN  = s("Normal",  fontSize=8,  textColor=C["gray"],
               alignment=TA_CENTER, spaceAfter=6)
    FOOT  = s("Normal",  fontSize=7.5,textColor=C["gray_mid"],
               alignment=TA_CENTER)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    story = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    n_ok   = sum(1 for e in experiments if e["converged"])
    n_tot  = len(experiments)
    n_fail = n_tot - n_ok

    # =========================================================================
    # PAGE 1: EXECUTION DASHBOARD
    # =========================================================================
    story.append(Paragraph("Batch Run Report", TITLE))
    story.append(Paragraph(
        "Hybrid Quantum-Classical CVaR Portfolio Optimization", SUBT
    ))
    story.append(Paragraph(
        str(n_tot) + " experiments  |  " + now + "  |  TFM Ignacio Lopez Leis, UAM",
        META
    ))
    story.append(HRFlowable(width="100%", thickness=2,
                             color=C["blue"], spaceAfter=10))

    # ── Batch-level KPIs ─────────────────────────────────────────────────────
    avg_improv = round(sum(e["cvar_improvement"] for e in experiments) / n_tot, 2) if n_tot else 0
    n_bench    = sum(1 for e in experiments if e["benchmark_computed"])
    n_rank1    = sum(1 for e in experiments if e["oos_rank"] == 1)
    avg_qae    = round(sum(e["qae_error"] for e in experiments) / n_tot, 2) if n_tot else 0
    quantum_wins = sum(
        1 for e in experiments
        if e["quantum_cvar"] > 0 and e["best_classical_cvar"] > 0
        and e["quantum_cvar"] <= e["best_classical_cvar"]
    )

    def kpi_block(label, val, unit="", col=None):
        c = col or C["blue_dark"]
        return [
            Paragraph(label, s("Normal", fontSize=7.5, textColor=C["gray"],
                                alignment=TA_CENTER)),
            Paragraph(str(val) + unit,
                      s("Normal", fontSize=16, fontName="Helvetica-Bold",
                        textColor=c, alignment=TA_CENTER)),
        ]

    kpis = [
        kpi_block("Experiments run",   n_tot),
        kpi_block("Succeeded",         n_ok,
                  col=C["green"] if n_ok == n_tot else C["orange"]),
        kpi_block("Failed",            n_fail,
                  col=C["red"] if n_fail > 0 else C["green"]),
        kpi_block("Avg CVaR improvement", ("+" if avg_improv>0 else "") + str(avg_improv),
                  "%", C["green"] if avg_improv > 0 else C["red"]),
        kpi_block("Quantum wins OOS",  str(n_rank1) + "/" + str(n_tot)),
        kpi_block("Benchmark computed",str(n_bench) + "/" + str(n_tot)),
        kpi_block("Avg QAE error",     avg_qae, "%"),
        kpi_block("Quantum beats best classical", str(quantum_wins) + "/" + str(n_tot),
                  col=C["green"] if quantum_wins > n_tot//2 else C["orange"]),
    ]
    row_a = [item for pair in kpis[:4] for item in pair]
    row_b = [item for pair in kpis[4:] for item in pair]

    def kpi_tbl(row):
        t = Table([row[0::2], row[1::2]], colWidths=[43.5*mm]*4)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), C["blue_lt"]),
            ("BACKGROUND",    (0,1), (-1,1), C["white"]),
            ("BOX",           (0,0), (-1,-1), 1,   C["blue_mid"]),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, C["blue_mid"]),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        return t

    story.append(KeepTogether([
        Paragraph("Batch Summary", H2),
        kpi_tbl(row_a), Spacer(1, 4), kpi_tbl(row_b),
    ]))
    story.append(Spacer(1, 10))

    # ── Per-experiment execution status grid ─────────────────────────────────
    story.append(Paragraph("Per-Experiment Execution Status", H2))

    # Read batch summary for timing info
    bs_exps = {}
    if batch_summary:
        for r in batch_summary.get("experiments", []):
            bs_exps[r.get("experiment", "")] = r

    status_hdr = ["Experiment", "Status", "Time", "CVaR improv.", "Converged",
                  "Benchmark", "OOS rank"]
    status_rows = [status_hdr]
    for e in experiments:
        bs = bs_exps.get(e["exp_id"], {})
        ok        = e["converged"]
        bench_ok  = e["benchmark_computed"]
        improv    = e["cvar_improvement"]
        elapsed   = bs.get("elapsed", "--")

        status_rows.append([
            Paragraph(e["exp_id"],
                      s("Normal", fontSize=8.5, fontName="Helvetica-Bold",
                        textColor=C["blue_dark"])),
            Paragraph("OK" if ok else "FAIL",
                      s("Normal", fontSize=8.5, fontName="Helvetica-Bold",
                        textColor=C["green"] if ok else C["red"],
                        alignment=TA_CENTER)),
            elapsed,
            Paragraph(("+" if improv > 0 else "") + str(improv) + "%",
                      s("Normal", fontSize=8.5,
                        textColor=C["green"] if improv > 0 else C["red"],
                        fontName="Helvetica-Bold", alignment=TA_CENTER)),
            Paragraph("YES" if ok else "NO",
                      s("Normal", fontSize=8,
                        textColor=C["green"] if ok else C["red"],
                        alignment=TA_CENTER)),
            Paragraph("YES" if bench_ok else "NO",
                      s("Normal", fontSize=8,
                        textColor=C["green"] if bench_ok else C["red"],
                        alignment=TA_CENTER)),
            Paragraph("#" + str(e["oos_rank"]) if e["oos_rank"] else "--",
                      s("Normal", fontSize=8,
                        textColor=C["green"] if e["oos_rank"]==1 else C["black"],
                        fontName="Helvetica-Bold" if e["oos_rank"]==1 else "Helvetica",
                        alignment=TA_CENTER)),
        ])

    sw = [32*mm, 16*mm, 18*mm, 24*mm, 20*mm, 22*mm, 18*mm]
    st = Table(status_rows, colWidths=sw, repeatRows=1)
    st_cmds = [
        ("BACKGROUND",    (0,0), (-1,0), C["blue_dark"]),
        ("TEXTCOLOR",     (0,0), (-1,0), C["white"]),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("GRID",          (0,0), (-1,-1), 0.4, C["gray_mid"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ALIGN",         (0,0), (0,-1), "LEFT"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C["white"], C["gray_lt"]]),
    ]
    for i, e in enumerate(experiments, 1):
        if not e["converged"]:
            st_cmds.append(("BACKGROUND", (0,i), (-1,i), C["red_lt"]))
    st.setStyle(TableStyle(st_cmds))
    story.append(st)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: CVaR RESULTS COMPARISON
    # =========================================================================
    story.append(Paragraph("CVaR Results Across All Experiments", H2))

    cvar_hdr = ["Experiment", "Group", "Universe", "n_qubits",
                "Initial CVaR", "Optimized CVaR", "Improvement",
                "Best classical", "Best cl. CVaR", "Quantum wins?"]
    cvar_rows = [cvar_hdr]
    for e in experiments:
        q_wins = (e["quantum_cvar"] > 0 and e["best_classical_cvar"] > 0
                  and e["quantum_cvar"] <= e["best_classical_cvar"])
        cvar_rows.append([
            Paragraph(e["exp_id"],
                      s("Normal", fontSize=7.5, fontName="Helvetica-Bold")),
            e["group"],
            e["universe"],
            str(e["n_qubits"] or "?"),
            str(e["initial_cvar"])    + "%",
            str(e["optimized_cvar"])  + "%",
            Paragraph(("+" if e["cvar_improvement"]>0 else "") +
                      str(e["cvar_improvement"]) + "%",
                      s("Normal", fontSize=7.5,
                        textColor=C["green"] if e["cvar_improvement"]>0 else C["red"],
                        fontName="Helvetica-Bold", alignment=TA_CENTER)),
            e["best_classical"],
            str(e["best_classical_cvar"]) + "%",
            Paragraph("YES" if q_wins else "NO",
                      s("Normal", fontSize=7.5,
                        textColor=C["green"] if q_wins else C["red"],
                        fontName="Helvetica-Bold", alignment=TA_CENTER)),
        ])

    cw = [28*mm, 14*mm, 16*mm, 14*mm, 20*mm, 20*mm, 18*mm, 20*mm, 18*mm, 16*mm]
    ct = Table(cvar_rows, colWidths=cw, repeatRows=1)
    ct_cmds = [
        ("BACKGROUND",    (0,0), (-1,0), C["blue_dark"]),
        ("TEXTCOLOR",     (0,0), (-1,0), C["white"]),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("GRID",          (0,0), (-1,-1), 0.4, C["gray_mid"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C["white"], C["blue_lt"]]),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    ct.setStyle(TableStyle(ct_cmds))
    story.append(ct)
    story.append(Spacer(1, 10))

    # ── QAE error across experiments ──────────────────────────────────────────
    story.append(Paragraph("QAE Error and OOS Performance", H2))

    qae_hdr = ["Experiment", "QAE Error (%)", "OOS Quantum CVaR",
               "OOS Rank", "Prediction Error", "Iterations"]
    qae_rows = [qae_hdr]
    for e in experiments:
        qae_rows.append([
            Paragraph(e["exp_id"],
                      s("Normal", fontSize=8, fontName="Helvetica-Bold")),
            Paragraph(str(e["qae_error"]) + "%",
                      s("Normal", fontSize=8,
                        textColor=C["green"] if e["qae_error"] < 10 else
                        (C["orange"] if e["qae_error"] < 25 else C["red"]),
                        fontName="Helvetica-Bold", alignment=TA_CENTER)),
            str(e["oos_quantum_cvar"]) + "%",
            Paragraph("#" + str(e["oos_rank"]) if e["oos_rank"] else "--",
                      s("Normal", fontSize=8,
                        textColor=C["green"] if e["oos_rank"]==1 else C["black"],
                        fontName="Helvetica-Bold" if e["oos_rank"]==1 else "Helvetica",
                        alignment=TA_CENTER)),
            str(e["oos_quantum_pred_err"]) + "%",
            str(e["num_iterations"]),
        ])

    qw = [36*mm, 28*mm, 32*mm, 20*mm, 30*mm, 22*mm]
    qt = Table(qae_rows, colWidths=qw, repeatRows=1)
    qt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0), C["blue_dark"]),
        ("TEXTCOLOR",     (0,0), (-1,0), C["white"]),
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8),
        ("GRID",          (0,0), (-1,-1), 0.4, C["gray_mid"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [C["white"], C["gray_lt"]]),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(qt)
    story.append(Paragraph(
        "QAE error: green < 10% (good), orange 10-25% (acceptable), red > 25% (high). "
        "Prediction error = |CVaR_train - CVaR_OOS| / |CVaR_train|.",
        CAPN
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: AI NARRATIVE
    # =========================================================================
    story.append(Paragraph("AI Batch Analysis", H2))
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=C["blue_mid"], spaceAfter=8))

    section_keys = {
        "BATCH VERDICT", "PERFORMANCE SUMMARY", "UNIVERSE COMPARISON",
        "NOISE SENSITIVITY", "QUANTUM ADVANTAGE", "RECOMMENDATIONS",
    }
    for line in ai_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 3))
            continue
        if line.rstrip(":").upper() in section_keys or line.upper() in section_keys:
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.3,
                                     color=C["blue_mid"], spaceAfter=2))
            story.append(Paragraph(line.rstrip(":"), H3))
        elif line.startswith("-") or line.startswith("*"):
            story.append(Paragraph("&bull; " + line[1:].strip(), BODY))
        elif line and line[0].isdigit() and len(line) > 2 and line[1] in ".):":
            story.append(Paragraph(line, BODY))
        else:
            story.append(Paragraph(line, BODY))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C["gray_mid"]))
    story.append(Paragraph(
        "TFM: Hybrid Quantum-Classical Portfolio Optimization with CVaR -- "
        "Ignacio Lopez Leis, Universidad Autonoma de Madrid -- " + now,
        FOOT
    ))

    doc.build(story)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch run PDF report"
    )
    parser.add_argument("--exps",    nargs="*",
                        help="Experiment IDs (default: all in results/)")
    parser.add_argument("--summary", default="",
                        help="Path to batch_summary JSON")
    parser.add_argument("--backend", choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--model",   default="llama3.1")
    parser.add_argument("--api-key", dest="api_key", default="")
    parser.add_argument("--ollama-host", dest="ollama_host",
                        default="http://localhost:11434")
    parser.add_argument("--no-ai",   dest="no_ai", action="store_true")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"

    # Resolve experiment list
    if args.exps:
        exp_ids = args.exps
    else:
        exp_ids = sorted([
            d.name.replace("exp_", "")
            for d in results_dir.iterdir()
            if d.is_dir() and d.name.startswith("exp_")
            and (d / "tfm_comprehensive_metrics_latest.json").exists()
        ])

    if not exp_ids:
        sys.exit("No experiments found.")

    print("Loading " + str(len(exp_ids)) + " experiments...")
    experiments = []
    for exp_id in exp_ids:
        m = load_experiment(exp_id)
        if m:
            experiments.append(extract(m, exp_id))
            print("  OK: " + exp_id)
        else:
            print("  SKIP (no metrics): " + exp_id)

    if not experiments:
        sys.exit("No experiment metrics found.")

    # Batch summary JSON
    batch_summary = load_batch_summary(args.summary) if args.summary else None
    # Also try to find the most recent one automatically
    if not batch_summary:
        batch_runs = sorted(results_dir.glob("batch_runs/batch_summary_*.json"))
        if batch_runs:
            try:
                batch_summary = json.loads(batch_runs[-1].read_text(encoding="utf-8"))
            except Exception:
                pass

    # AI analysis
    ai_text = ""
    if not args.no_ai:
        print("Generating AI batch analysis...")
        slim = [{k: v for k, v in e.items()
                 if k not in ("classical_cvars",)} for e in experiments]
        try:
            if args.backend == "ollama":
                ai_text = generate_ollama(slim, args.model, args.ollama_host)
            else:
                if not args.api_key:
                    sys.exit("--api-key required for --backend claude")
                ai_text = generate_claude(slim, args.api_key)
            print("AI analysis complete")
        except ConnectionError as e:
            print("AI skipped: " + str(e))
            ai_text = (
                "AI analysis unavailable (Ollama not running).\n"
                "Start with: docker start ollama"
            )
    else:
        ai_text = "AI analysis skipped (--no-ai flag)."

    # Build PDF
    out = results_dir / "batch_report.pdf"
    print("Building PDF...")
    try:
        build_pdf(experiments, batch_summary, ai_text, out)
        size_kb = out.stat().st_size // 1024
        print("Batch report saved: " + str(out) + " (" + str(size_kb) + " KB)")
    except ImportError:
        print("reportlab not installed. Run: pip install reportlab pillow")
    except Exception as e:
        import traceback
        print("PDF build error: " + str(e))
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
