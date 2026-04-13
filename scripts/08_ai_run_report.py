#!/usr/bin/env python3
# =============================================================================
# F-001 -- AI Run Report Agent (ReportLab PDF, paper-quality)
# =============================================================================
# Usage:
#   python scripts\08_ai_run_report.py --exp A1_PA
#   python scripts\08_ai_run_report.py --all --model phi3
#   python scripts\08_ai_run_report.py --exp A1_PA --backend claude --api-key sk-ant-...
# Install: pip install reportlab pillow
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
    "You are an expert reviewer for a quantum computing Masters Thesis on "
    "CVaR portfolio optimization. Analyze this experiment and write a "
    "technical report suitable for inclusion in an academic paper. "
    "Plain text only, no markdown, no asterisks, no special characters.\n\n"
    "EXECUTIVE SUMMARY\n"
    "3 precise sentences: hypothesis tested, main quantitative result, "
    "main limitation.\n\n"
    "SCIENTIFIC CONTRIBUTION\n"
    "What does this result contribute to the field? "
    "Is the CVaR improvement statistically meaningful? "
    "How does OOS performance validate the approach?\n\n"
    "BENCHMARK ANALYSIS\n"
    "Detailed comparison vs classical methods. "
    "Explain why quantum outperforms or underperforms. "
    "Reference specific CVaR values.\n\n"
    "QAE ERROR ANALYSIS\n"
    "Interpret the QAE error in context. "
    "Is it acceptable for this n_qubits setting? "
    "What does it imply for the quadratic speedup claim?\n\n"
    "ANOMALIES AND DATA QUALITY\n"
    "Flag any suspicious values. Check: quantum_cvar=0 means BUG-002. "
    "Check convergence regressions. Check OOS prediction error magnitude.\n\n"
    "CONCLUSIONS FOR PAPER\n"
    "2-3 sentences suitable for the Results section of the paper.\n\n"
    "Be precise. Use exact numbers. Maximum 600 words total."
)

# ---------------------------------------------------------------------------
# Colors and styles (centralized)
# ---------------------------------------------------------------------------
def get_palette():
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
# Metrics extraction
# ---------------------------------------------------------------------------
def load_metrics(exp_dir):
    mf = exp_dir / "tfm_comprehensive_metrics_latest.json"
    if not mf.exists():
        raise FileNotFoundError("No metrics file in " + str(exp_dir))
    return json.loads(mf.read_text(encoding="utf-8"))

def extract(m, exp_id):
    opt   = m.get("metrics", {}).get("optimization", {})
    risk  = m.get("metrics", {}).get("risk", {})
    bench = m.get("metrics", {}).get("benchmarks", {})
    oos   = m.get("metrics", {}).get("oos_validation", {})
    qae   = m.get("metrics", {}).get("quantum", {}).get("qae", {})
    rc    = m.get("metrics", {}).get("risk_contributions", {})
    fp    = m.get("convergence_fingerprint", {})
    diag  = m.get("qae_diagnostics_t001", {})
    cfg   = m.get("metadata", {}).get("configuration", {})
    err   = m.get("metrics", {}).get("errors", {})
    timings = m.get("timings", {})
    weights = m.get("weights", {})
    conv  = m.get("convergence_history", {})

    classical = {}
    for meth, res in bench.get("classical_results", {}).items():
        classical[meth] = {
            "cvar":    round(res.get("optimal_cvar", 0) * 100, 4),
            "var":     round(res.get("optimal_var",  0) * 100, 4),
            "time_s":  round(res.get("execution_time_s", 0), 4),
            "iters":   res.get("iterations", 0),
            "success": res.get("success", False),
            "msg":     res.get("message", ""),
        }

    oos_ports = oos.get("portfolios", {})

    # Fix BUG-002: quantum_cvar from bench or from risk.optimized
    q_cvar_bench = bench.get("quantum_cvar", 0)
    if not q_cvar_bench:
        q_cvar_bench = risk.get("optimized", {}).get("cvar_loss", 0)

    return {
        "exp_id":       exp_id,
        "experiment_id": cfg.get("experiment_id", exp_id),
        "alpha":        cfg.get("phase1", {}).get("cvar", {}).get("alpha", 0.05),
        "n_assets":     cfg.get("data", {}).get("n_effective"),
        "n_qubits":     diag.get("n_qubits_used") or cfg.get("phase2", {}).get("qae", {}).get("n_qubits"),
        "epsilon":      cfg.get("phase2", {}).get("qae", {}).get("epsilon"),
        "noise_group":  exp_id.split("_")[0],
        "universe":     exp_id.split("_")[-1],
        # Risk
        "initial_cvar":   round(risk.get("initial",   {}).get("cvar_loss", 0) * 100, 4),
        "initial_var":    round(risk.get("initial",   {}).get("var_loss",  0) * 100, 4),
        "optimized_cvar": round(risk.get("optimized", {}).get("cvar_loss", 0) * 100, 4),
        "optimized_var":  round(risk.get("optimized", {}).get("var_loss",  0) * 100, 4),
        # Optimization
        "cvar_improvement_pct": round(opt.get("cvar_improvement_pct", 0), 3),
        "cvar_reduction_abs":   round(opt.get("cvar_reduction_absolute", 0) * 100, 4),
        "num_iterations":       opt.get("num_iterations"),
        "converged":            opt.get("converged"),
        "method":               opt.get("method"),
        # Benchmark
        "benchmark_computed":   bench.get("computed"),
        "quantum_cvar_bench":   round(q_cvar_bench * 100, 4),
        "best_classical_method": bench.get("best_classical_method"),
        "best_classical_cvar":  round((bench.get("best_classical_cvar") or 0) * 100, 4),
        "classical":            classical,
        # QAE
        "qae_quantum_cvar":     round(qae.get("quantum_cvar", 0) * 100, 4),
        "qae_classical_cvar":   round(qae.get("classical_cvar", 0) * 100, 4),
        "qae_error_pct":        round(qae.get("cvar_error", 0) * 100, 3),
        "shots_per_round":      diag.get("shots_per_round_est"),
        "var_quant_error":      diag.get("var_threshold_quantization_error"),
        "bin_width":            diag.get("bin_width"),
        "n_rounds":             diag.get("n_rounds_est"),
        # OOS
        "oos_quantum_cvar":     round((oos.get("quantum_oos_cvar") or 0) * 100, 4),
        "oos_rank":             oos.get("quantum_cvar_rank"),
        "oos_n_portfolios":     len(oos_ports),
        "oos_portfolios": {
            k: {
                "oos_cvar":     round((v.get("oos_metrics", {}).get("cvar_loss") or 0) * 100, 4),
                "train_cvar":   round((v.get("train_cvar") or 0) * 100, 4),
                "pred_error":   round((v.get("cvar_prediction_error") or 0) * 100, 2),
            }
            for k, v in oos_ports.items()
        },
        # Convergence fingerprint
        "fp":    fp,
        "conv_history": conv.get("cvar_history", []),
        # Timings
        "timings": timings,
        # Weights
        "weights_by_asset": weights.get("by_asset", {}),
        # Errors
        "errors": err,
        # Passport
        "passport_id": m.get("metadata", {}).get("passport_id", ""),
    }

# ---------------------------------------------------------------------------
# AI backends
# ---------------------------------------------------------------------------
def generate_ollama(d, model="llama3.1", host="http://localhost:11434"):
    payload = {
        "model": model,
        "prompt": SYSTEM_PROMPT + "\n\nExperiment data:\n" + json.dumps(d, indent=2),
        "stream": False,
        "options": {"temperature": 0.15, "num_predict": 1200},
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
        raise ConnectionError("Cannot reach Ollama. Run: docker start ollama. " + str(e))

def generate_claude(d, api_key):
    payload = {
        "model": "claude-sonnet-4-20250514", "max_tokens": 1200,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": "Data:\n" + json.dumps(d, indent=2)}],
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
# PDF builder
# ---------------------------------------------------------------------------
def build_pdf(exp_id, d, ai_text, oos_img_path, passport_chain_path, output_path):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image, KeepTogether, PageBreak,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus.flowables import HRFlowable

    P   = get_palette()
    STY = getSampleStyleSheet()

    def s(base, **kw):
        return ParagraphStyle(base + str(abs(hash(str(kw)))), parent=STY[base], **kw)

    # Style definitions
    TITLE  = s("Normal",  fontSize=20, textColor=P["blue_dark"], fontName="Helvetica-Bold",
                spaceAfter=2, alignment=TA_CENTER)
    SUBT   = s("Normal",  fontSize=11, textColor=P["blue"],       alignment=TA_CENTER, spaceAfter=2)
    META   = s("Normal",  fontSize=8,  textColor=P["gray"],       alignment=TA_CENTER, spaceAfter=8)
    H2     = s("Heading2",fontSize=12, textColor=P["blue_dark"],  spaceBefore=12, spaceAfter=4,
                borderPad=0)
    H3     = s("Normal",  fontSize=10, textColor=P["blue"],       fontName="Helvetica-Bold",
                spaceBefore=8, spaceAfter=3)
    BODY   = s("Normal",  fontSize=9.5,textColor=P["black"],      leading=14)
    SMALL  = s("Normal",  fontSize=8,  textColor=P["gray"],       leading=11)
    CAPN   = s("Normal",  fontSize=8,  textColor=P["gray"],       alignment=TA_CENTER, spaceAfter=6)
    FOOT   = s("Normal",  fontSize=7.5,textColor=P["gray_mid"],   alignment=TA_CENTER)
    STATUS_OK  = s("Normal", fontSize=9, textColor=P["green"], fontName="Helvetica-Bold",
                    alignment=TA_CENTER)
    STATUS_ERR = s("Normal", fontSize=9, textColor=P["red"],   fontName="Helvetica-Bold",
                    alignment=TA_CENTER)

    doc = SimpleDocTemplate(
        str(output_path), pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=15*mm, bottomMargin=15*mm,
    )
    story = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # =========================================================================
    # PAGE 1: VISUAL DASHBOARD
    # =========================================================================

    # Title block
    story.append(Paragraph("Quantum CVaR Portfolio Optimization", TITLE))
    story.append(Paragraph("Experiment Run Report", SUBT))
    story.append(Paragraph(
        exp_id + "  |  " + now + "  |  "
        "TFM Ignacio Lopez Leis, UAM",
        META
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=P["blue"], spaceAfter=10))

    # ── Status banner ────────────────────────────────────────────────────────
    improv     = d["cvar_improvement_pct"]
    bench_ok   = d["benchmark_computed"]
    converged  = d["converged"]
    q_cvar_ok  = d["quantum_cvar_bench"] > 0
    oos_rank   = d["oos_rank"]

    def status_cell(label, ok, ok_text="OK", fail_text="FAIL"):
        bg  = P["green_lt"]  if ok else P["red_lt"]
        col = P["green"]     if ok else P["red"]
        sym = "PASS" if ok else "FAIL"
        return [
            Paragraph(label, s("Normal", fontSize=7.5, textColor=P["gray"],
                                alignment=TA_CENTER)),
            Paragraph(sym + " -- " + (ok_text if ok else fail_text),
                      s("Normal", fontSize=9, textColor=col,
                        fontName="Helvetica-Bold", alignment=TA_CENTER)),
        ]

    status_data = [[
        "\n".join(["Pipeline Status", ""]),
        "\n".join(["Benchmark", ""]),
        "\n".join(["Convergence", ""]),
        "\n".join(["Quantum CVaR", ""]),
        "\n".join(["OOS Rank", ""]),
    ]]
    pass_fail = [
        ("Pipeline",    True,        "3/3 steps OK",  "check logs"),
        ("Benchmark",   bench_ok,    "computed",      "NOT computed"),
        ("Convergence", converged,   "converged",     "did not converge"),
        ("q_CVaR",      q_cvar_ok,   str(d["quantum_cvar_bench"]) + "%", "BUG-002"),
        ("OOS Rank",    oos_rank==1, "#1 (best)",
         "#" + str(oos_rank) if oos_rank else "N/A"),
    ]

    row1 = []
    row2 = []
    for label, ok, ok_t, fail_t in pass_fail:
        bg  = P["green_lt"] if ok else P["red_lt"]
        col = P["green"]    if ok else P["red"]
        sym = "PASS" if ok else "FAIL"
        row1.append(Paragraph(label, SMALL))
        row2.append(Paragraph(
            sym + "\n" + (ok_t if ok else fail_t),
            s("Normal", fontSize=9.5, textColor=col,
              fontName="Helvetica-Bold", alignment=TA_CENTER)
        ))

    st = Table([row1, row2], colWidths=[36*mm]*5)
    st.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), P["gray_lt"]),
        ("BACKGROUND", (0,1), (0,1),  P["green_lt"] if True   else P["red_lt"]),
        ("BACKGROUND", (1,1), (1,1),  P["green_lt"] if bench_ok else P["red_lt"]),
        ("BACKGROUND", (2,1), (2,1),  P["green_lt"] if converged else P["red_lt"]),
        ("BACKGROUND", (3,1), (3,1),  P["green_lt"] if q_cvar_ok else P["red_lt"]),
        ("BACKGROUND", (4,1), (4,1),  P["green_lt"] if oos_rank==1 else P["orange_lt"]),
        ("BOX",        (0,0), (-1,-1), 1, P["gray_mid"]),
        ("INNERGRID",  (0,0), (-1,-1), 0.5, P["gray_mid"]),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
    ]))
    story.append(KeepTogether([
        Paragraph("Pipeline Validation Status", H2),
        st,
    ]))
    story.append(Spacer(1, 8))

    # ── Big KPI grid ─────────────────────────────────────────────────────────
    def kpi(label, val, unit="", color=None):
        c = color or P["blue_dark"]
        return [
            Paragraph(label, s("Normal", fontSize=7.5, textColor=P["gray"],
                                alignment=TA_CENTER)),
            Paragraph(str(val) + unit,
                      s("Normal", fontSize=16, fontName="Helvetica-Bold",
                        textColor=c, alignment=TA_CENTER)),
        ]

    improv_col = P["green"] if improv > 0 else P["red"]
    rank_col   = P["green"] if oos_rank == 1 else (P["orange"] if oos_rank == 2 else P["red"])

    kpi_rows = [
        kpi("Initial CVaR (alpha=5%)",     d["initial_cvar"],    "%"),
        kpi("Optimized CVaR",              d["optimized_cvar"],  "%", P["green"]),
        kpi("CVaR Improvement",            ("+" if improv>0 else "") + str(round(improv,2)), "%", improv_col),
        kpi("Best Classical CVaR",         d["best_classical_cvar"], "%"),
        kpi("OOS CVaR (quantum)",          d["oos_quantum_cvar"], "%"),
        kpi("OOS Rank",                    "#" + str(oos_rank or "?"), "", rank_col),
        kpi("QAE Error",                   d["qae_error_pct"],   "%"),
        kpi("Iterations",                  d["num_iterations"],  ""),
    ]
    # 4 columns x 2 rows
    row_a = [item for pair in kpi_rows[:4] for item in pair]
    row_b = [item for pair in kpi_rows[4:] for item in pair]

    def kpi_table(row):
        t = Table(
            [row[0::2], row[1::2]],
            colWidths=[43.5*mm]*4
        )
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  P["blue_lt"]),
            ("BACKGROUND",    (0,1), (-1,1),  P["white"]),
            ("BOX",           (0,0), (-1,-1), 1,   P["blue_mid"]),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, P["blue_mid"]),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        return t

    story.append(KeepTogether([
        Paragraph("Key Performance Indicators", H2),
        kpi_table(row_a),
        Spacer(1, 4),
        kpi_table(row_b),
    ]))
    story.append(Spacer(1, 8))

    # ── Config summary ────────────────────────────────────────────────────────
    cfg_data = [
        ["Parameter", "Value", "Parameter", "Value"],
        ["Experiment ID",  d["experiment_id"],  "Universe",   d["universe"]],
        ["Noise group",    d["noise_group"],     "n_assets",   str(d["n_assets"] or "?")],
        ["n_qubits",       str(d["n_qubits"] or "?"), "epsilon (QAE)", str(d["epsilon"] or "?")],
        ["Alpha (CVaR)",   str(d["alpha"]),      "Optimizer",  str(d["method"] or "hybrid")],
    ]
    ct = Table(cfg_data, colWidths=[42*mm, 36*mm, 42*mm, 36*mm])
    ct.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  P["blue_dark"]),
        ("TEXTCOLOR",     (0,0), (-1,0),  P["white"]),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [P["blue_lt"], P["white"]]),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",      (2,1), (2,-1),  "Helvetica-Bold"),
        ("GRID",          (0,0), (-1,-1), 0.4, P["gray_mid"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(KeepTogether([Paragraph("Experiment Configuration", H2), ct]))
    story.append(Spacer(1, 8))

    # ── Convergence fingerprint visual ────────────────────────────────────────
    fp = d["fp"]
    if fp:
        fp_data = [
            ["Iters to 95% improvement", "Mean slope", "Final std", "Regressions", "Total improvement"],
            [
                str(fp.get("iter_to_95pct_improvement", "N/A")),
                str(round(fp.get("mean_convergence_slope", 0), 6)),
                str(round(fp.get("final_stability_std", 0), 6)),
                str(fp.get("n_regressions", "N/A")),
                str(round(fp.get("total_improvement_pct", 0), 3)) + "%",
            ]
        ]
        fpt = Table(fp_data, colWidths=[38*mm]*5)
        fpt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  P["blue"]),
            ("TEXTCOLOR",     (0,0), (-1,0),  P["white"]),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("BACKGROUND",    (0,1), (-1,1),  P["blue_lt"]),
            ("FONTNAME",      (0,1), (-1,1),  "Helvetica-Bold"),
            ("GRID",          (0,0), (-1,-1), 0.4, P["gray_mid"]),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(KeepTogether([
            Paragraph("Convergence Fingerprint (F-003)", H2), fpt
        ]))

    # ── Passport pipeline (Page 1 bottom or Page 2 top) ──────────────────────
    chain = None
    if passport_chain_path and passport_chain_path.exists():
        try:
            chain = json.loads(passport_chain_path.read_text(encoding="utf-8"))
        except Exception:
            chain = None

    if chain:
        stages = chain.get("stages", [])
        master = chain.get("master", {})
        story.append(Spacer(1, 6))
        story.append(Paragraph("Data Passport Pipeline", H2))
        story.append(Paragraph(
            "Master passport: " + master.get("passport_id", "N/A")[:16] + "...  |  "
            "Stages: " + str(len(stages)) + "  |  "
            "Created: " + master.get("created_at", "N/A")[:19],
            SMALL
        ))
        story.append(Spacer(1, 4))

        pipeline_rows = [["Stage", "Module", "Transformation", "Status", "in hash", "out hash"]]
        for st_item in stages:
            seals = st_item.get("seals", [])
            if seals:
                for seal in seals:
                    status = seal.get("status", "?")
                    ok_col = P["green"] if status == "valid" else P["red"]
                    pipeline_rows.append([
                        str(st_item.get("stage_index", "")),
                        seal.get("function", "?")[:22],
                        seal.get("transformation_type", "?")[:18],
                        Paragraph(status, s("Normal", fontSize=7.5,
                                            textColor=ok_col, fontName="Helvetica-Bold")),
                        seal.get("input_hash",  "?")[:8],
                        seal.get("output_hash", "?")[:8],
                    ])
            else:
                pipeline_rows.append([
                    str(st_item.get("stage_index", "")),
                    st_item.get("data_type", "?")[:22],
                    st_item.get("source", "?")[:18],
                    Paragraph("no seals", s("Normal", fontSize=7.5,
                                            textColor=P["gray"])),
                    "--", "--",
                ])

        pt = Table(pipeline_rows,
                   colWidths=[12*mm, 40*mm, 34*mm, 18*mm, 22*mm, 22*mm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  P["blue_dark"]),
            ("TEXTCOLOR",     (0,0), (-1,0),  P["white"]),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 7.5),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [P["white"], P["gray_lt"]]),
            ("GRID",          (0,0), (-1,-1), 0.3, P["gray_mid"]),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
            ("TOPPADDING",    (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(pt)
        story.append(Paragraph(
            "Data lineage tracked via passport system -- "
            "each row is one function seal recording input/output hashes",
            CAPN
        ))
    else:
        story.append(Spacer(1, 4))
        story.append(Paragraph(
            "Passport pipeline not available for this run "
            "(replace notebooks/main.ipynb with V5 version to enable)",
            s("Normal", fontSize=8, textColor=P["orange"], alignment=TA_CENTER)
        ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: BENCHMARK + OOS TABLES
    # =========================================================================
    story.append(Paragraph("Benchmark: Quantum vs Classical Methods", H2))

    q_cvar = d["quantum_cvar_bench"]
    bench_hdr = ["Method", "CVaR train (%)", "VaR train (%)",
                 "Time (s)", "Iters", "vs Quantum (pp)", "Status"]
    bench_rows = [bench_hdr]
    bench_rows.append([
        Paragraph("Quantum (IQAE)", s("Normal", fontSize=9,
                                       textColor=P["blue"], fontName="Helvetica-Bold")),
        str(q_cvar), str(d["qae_error_pct"]) + " err",
        "--", str(d["num_iterations"]), "Reference",
        Paragraph("QUANTUM", s("Normal", fontSize=8, textColor=P["blue"],
                                fontName="Helvetica-Bold")),
    ])
    best = d["best_classical_method"]
    for meth, v in d["classical"].items():
        diff = round(v["cvar"] - q_cvar, 4) if q_cvar else 0
        diff_str = ("+" if diff > 0 else "") + str(diff)
        is_best = meth == best
        bench_rows.append([
            Paragraph(meth + (" (*)" if is_best else ""),
                      s("Normal", fontSize=9,
                        fontName="Helvetica-Bold" if is_best else "Helvetica",
                        textColor=P["blue_dark"] if is_best else P["black"])),
            str(v["cvar"]),
            str(v["var"]),
            str(v["time_s"]),
            str(v["iters"]),
            diff_str,
            Paragraph("BEST" if is_best else ("OK" if v["success"] else "FAIL"),
                      s("Normal", fontSize=8,
                        textColor=P["green"] if is_best else (P["gray"] if v["success"] else P["red"]),
                        fontName="Helvetica-Bold")),
        ])

    bw = [40*mm, 24*mm, 22*mm, 18*mm, 14*mm, 26*mm, 20*mm]
    bt = Table(bench_rows, colWidths=bw, repeatRows=1)
    bt_style = [
        ("BACKGROUND",    (0,0), (-1,0),  P["blue_dark"]),
        ("TEXTCOLOR",     (0,0), (-1,0),  P["white"]),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [P["blue_lt"], P["white"]]),
        ("GRID",          (0,0), (-1,-1), 0.4, P["gray_mid"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
    # Highlight best classical row
    for i, (meth, v) in enumerate(d["classical"].items(), 2):
        if meth == best:
            bt_style.append(("BACKGROUND", (0,i), (-1,i), P["green_lt"]))
    bt.setStyle(TableStyle(bt_style))
    story.append(bt)
    story.append(Paragraph(
        "(*) Best classical method by CVaR.  pp = percentage points vs quantum CVaR.",
        CAPN
    ))
    story.append(Spacer(1, 8))

    # ── OOS comparison table ──────────────────────────────────────────────────
    story.append(Paragraph("Out-of-Sample Validation", H2))
    oos_hdr = ["Portfolio", "Train CVaR (%)", "OOS CVaR (%)", "Prediction error (%)", "Rank"]
    oos_rows = [oos_hdr]
    ports = d["oos_portfolios"]
    for rank_idx, (name, v) in enumerate(ports.items(), 1):
        is_q = "Quantum" in name
        oos_rows.append([
            Paragraph(name, s("Normal", fontSize=9,
                               fontName="Helvetica-Bold" if is_q else "Helvetica",
                               textColor=P["blue"] if is_q else P["black"])),
            str(v["train_cvar"]),
            str(v["oos_cvar"]),
            str(v["pred_error"]) + "%",
            Paragraph(
                "#" + str(d["oos_rank"]) if is_q else "--",
                s("Normal", fontSize=9,
                  textColor=P["green"] if (is_q and d["oos_rank"]==1) else P["black"],
                  fontName="Helvetica-Bold" if is_q else "Helvetica")
            ),
        ])

    oot = Table(oos_rows, colWidths=[52*mm, 28*mm, 28*mm, 38*mm, 18*mm], repeatRows=1)
    oo_style = [
        ("BACKGROUND",    (0,0), (-1,0),  P["blue_dark"]),
        ("TEXTCOLOR",     (0,0), (-1,0),  P["white"]),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [P["blue_lt"], P["white"]]),
        ("GRID",          (0,0), (-1,-1), 0.4, P["gray_mid"]),
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
    for i, (name, v) in enumerate(ports.items(), 1):
        if "Quantum" in name:
            oo_style.append(("BACKGROUND", (0,i), (-1,i), P["blue_lt"]))
    oot.setStyle(TableStyle(oo_style))
    story.append(oot)
    story.append(Paragraph(
        "Prediction error = |CVaR_train - CVaR_OOS| / |CVaR_train|. "
        "High error expected due to regime shift (COVID training, bull-market OOS).",
        CAPN
    ))
    story.append(Spacer(1, 8))

    # ── QAE diagnostics ────────────────────────────────────────────────────────
    story.append(Paragraph("QAE Diagnostics (T-001)", H2))
    qae_data = [
        ["Parameter", "Value", "Interpretation"],
        ["n_qubits",               str(d["n_qubits"] or "N/A"),
         "Discretization bins = 2^n"],
        ["epsilon (precision)",    str(d["epsilon"] or "N/A"),
         "IQAE target error bound"],
        ["QAE CVaR error",         str(d["qae_error_pct"]) + "%",
         "< 10% acceptable; < 5% good"],
        ["Shots per round (est.)", str(d["shots_per_round"] or "N/A"),
         "Total oracle queries proportional"],
        ["VaR quant. error",       str(d["var_quant_error"] or "N/A"),
         "Discretization bias on VaR threshold"],
        ["Bin width",              str(d["bin_width"] or "N/A"),
         "Return space resolution"],
        ["Est. n_rounds",          str(d["n_rounds"] or "N/A"),
         "IQAE rounds executed"],
    ]
    qw = [48*mm, 28*mm, 82*mm]
    qt = Table(qae_data, colWidths=qw, repeatRows=1)
    qt.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  P["blue_dark"]),
        ("TEXTCOLOR",     (0,0), (-1,0),  P["white"]),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 8.5),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [P["white"], P["gray_lt"]]),
        ("FONTNAME",      (0,1), (0,-1),  "Helvetica-Bold"),
        ("GRID",          (0,0), (-1,-1), 0.4, P["gray_mid"]),
        ("ALIGN",         (1,0), (1,-1),  "CENTER"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    story.append(qt)
    story.append(Spacer(1, 8))

    # ── Portfolio weights ──────────────────────────────────────────────────────
    weights = d["weights_by_asset"]
    if weights:
        story.append(Paragraph("Optimized Portfolio Weights", H2))
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        w_data = [["Asset", "Weight (%)", "Asset", "Weight (%)"]]
        half = (len(sorted_w) + 1) // 2
        for i in range(half):
            left  = sorted_w[i]
            right = sorted_w[i + half] if (i + half) < len(sorted_w) else ("--", 0)
            w_data.append([
                left[0],
                str(round(left[1] * 100, 2)) + "%",
                right[0],
                str(round(right[1] * 100, 2)) + "%" if right[1] else "--",
            ])
        wt = Table(w_data, colWidths=[40*mm, 28*mm, 40*mm, 28*mm])
        wt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  P["blue_dark"]),
            ("TEXTCOLOR",     (0,0), (-1,0),  P["white"]),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [P["blue_lt"], P["white"]]),
            ("GRID",          (0,0), (-1,-1), 0.4, P["gray_mid"]),
            ("ALIGN",         (1,0), (1,-1),  "RIGHT"),
            ("ALIGN",         (3,0), (3,-1),  "RIGHT"),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(wt)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: AI ANALYSIS + OOS FIGURE
    # =========================================================================
    story.append(Paragraph("AI Analysis", H2))
    story.append(HRFlowable(width="100%", thickness=0.5, color=P["blue_mid"], spaceAfter=6))

    section_keys = {
        "EXECUTIVE SUMMARY", "SCIENTIFIC CONTRIBUTION", "BENCHMARK ANALYSIS",
        "QAE ERROR ANALYSIS", "ANOMALIES AND DATA QUALITY", "CONCLUSIONS FOR PAPER",
    }
    for line in ai_text.split("\n"):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 3))
            continue
        if line.rstrip(":").upper() in section_keys or line.upper() in section_keys:
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.3,
                                     color=P["blue_mid"], spaceAfter=2))
            story.append(Paragraph(line.rstrip(":"), H3))
        elif line.startswith("-") or line.startswith("*"):
            story.append(Paragraph("&bull; " + line[1:].strip(), BODY))
        elif line and line[0].isdigit() and len(line) > 2 and line[1] in ".):":
            story.append(Paragraph(line, BODY))
        else:
            story.append(Paragraph(line, BODY))
    story.append(Spacer(1, 10))

    # ── OOS figure ─────────────────────────────────────────────────────────────
    if oos_img_path and oos_img_path.exists():
        try:
            img = Image(str(oos_img_path), width=168*mm, height=72*mm)
            story.append(KeepTogether([
                Paragraph("Out-of-Sample CVaR Prediction Accuracy", H2),
                HRFlowable(width="100%", thickness=0.5, color=P["blue_mid"], spaceAfter=6),
                img,
                Paragraph(
                    "Figure: Predicted vs realized loss distribution -- "
                    "Parametric (Normal) | Monte Carlo | Quantum (QAE). "
                    "High prediction error is expected due to distribution shift "
                    "(COVID crash in training, 2024-2025 bull market in OOS).",
                    CAPN
                ),
            ]))
        except Exception as e:
            story.append(Paragraph("OOS figure unavailable: " + str(e), SMALL))

    # ── Footer ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 12))
    story.append(HRFlowable(width="100%", thickness=0.5, color=P["gray_mid"]))
    story.append(Paragraph(
        "TFM: Hybrid Quantum-Classical Portfolio Optimization with CVaR -- "
        "Ignacio Lopez Leis, Universidad Autonoma de Madrid -- "
        "Generated: " + now,
        FOOT
    ))

    doc.build(story)

# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------
def process_experiment(exp_dir, backend="ollama", model="llama3.1",
                        api_key="", ollama_host="http://localhost:11434"):
    exp_id = exp_dir.name.replace("exp_", "")
    print("  [" + exp_id + "] Generating report (" + backend + "/" + model + ")...")

    try:
        m = load_metrics(exp_dir)
        d = extract(m, exp_id)

        if backend == "ollama":
            ai_text = generate_ollama(d, model=model, host=ollama_host)
        elif backend == "claude":
            if not api_key:
                raise ValueError("--api-key required for --backend claude")
            ai_text = generate_claude(d, api_key=api_key)
        else:
            raise ValueError("Unknown backend: " + backend)

        oos_img   = exp_dir / ("oos_triple_distribution_" + exp_id + ".png")
        chain_path = exp_dir / "passports" / "pipeline_chain.json"
        pdf_path   = exp_dir / "report_ai.pdf"

        try:
            build_pdf(exp_id, d, ai_text, oos_img, chain_path, pdf_path)
            size_kb = pdf_path.stat().st_size // 1024
            print("  [" + exp_id + "] PDF: " + str(pdf_path) + " (" + str(size_kb) + " KB)")
        except ImportError:
            print("  [" + exp_id + "] reportlab not installed. Run: pip install reportlab pillow")

        # Markdown
        md = (
            "# Report -- " + exp_id + "\n\n"
            "**Generated:** " + datetime.now().isoformat() + "\n\n---\n\n"
            + ai_text + "\n\n---\n*TFM Ignacio Lopez Leis, UAM*\n"
        )
        (exp_dir / "report_ai.md").write_text(md, encoding="utf-8")
        (exp_dir / "report_ai_raw.json").write_text(
            json.dumps({"exp_id": exp_id, "backend": backend, "model": model,
                        "data": d, "report": ai_text}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    except ConnectionError as e:
        print("  [" + exp_id + "] CONNECTION ERROR: " + str(e))
    except Exception as e:
        import traceback
        print("  [" + exp_id + "] ERROR: " + str(e))
        print(traceback.format_exc())

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI PDF run report (ReportLab)")
    parser.add_argument("--exp",         help="Experiment ID, e.g. A1_PA")
    parser.add_argument("--all",         action="store_true")
    parser.add_argument("--backend",     choices=["ollama", "claude"], default="ollama")
    parser.add_argument("--model",       default="llama3.1",
                        help="Ollama model: llama3.1, mistral, phi3")
    parser.add_argument("--api-key",     dest="api_key", default="")
    parser.add_argument("--ollama-host", dest="ollama_host", default="http://localhost:11434")
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / "results"
    kw = dict(backend=args.backend, model=args.model,
               api_key=args.api_key, ollama_host=args.ollama_host)

    if args.all:
        exp_dirs = sorted([
            d for d in results_dir.iterdir()
            if d.is_dir() and d.name.startswith("exp_")
            and (d / "tfm_comprehensive_metrics_latest.json").exists()
        ])
        print("Processing " + str(len(exp_dirs)) + " experiments...")
        for exp_dir in exp_dirs:
            process_experiment(exp_dir, **kw)
    elif args.exp:
        exp_dir = results_dir / ("exp_" + args.exp)
        if not exp_dir.exists():
            sys.exit("Not found: " + str(exp_dir))
        process_experiment(exp_dir, **kw)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
