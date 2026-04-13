#!/usr/bin/env python3
"""
# ============================================================================
# Project : Hybrid Quantum-Classical Portfolio Optimization with CVaR
#           Master's Thesis (TFM) - MSc in Quantum Computing
# Author  : Ignacio Lopez Leis
# Affil.  : Universidad Autonoma de Madrid (UAM)
# Date    : March 2026
# Module  : passport_pipeline_viewer.py  (Utils - Passport Visualization)
# ============================================================================
#
# Reconstructs and renders the full data pipeline journey from passport
# chain JSON files. Shows every transformation the data went through.
#
# Usage:
#   python src/utils/passport_pipeline_viewer.py --exp A1_n3_PA
#   python src/utils/passport_pipeline_viewer.py --chain results/exp_A1_n3_PA/passports/pipeline_chain.json
# ============================================================================
"""

import sys, json, argparse
from pathlib import Path
from datetime import datetime

def find_project_root():
    for p in [Path.cwd(), Path(__file__).resolve().parent.parent.parent]:
        if (p / "config").is_dir(): return p
    raise FileNotFoundError("Cannot find project root")


def render_pipeline_text(chain: dict) -> str:
    """Render pipeline chain as ASCII pipeline diagram."""
    lines = [
        "=" * 80,
        "DATA PIPELINE JOURNEY — Passport Chain Visualization",
        f"Generated: {datetime.now().isoformat()}",
        "=" * 80,
    ]

    master = chain.get("master", {})
    if master:
        lines += [
            "",
            f"MASTER PASSPORT: {master.get('passport_id','?')[:16]}...",
            f"  Created: {master.get('created_at','?')}",
            f"  Type:    {master.get('data_type','?')}",
            f"  Source:  {master.get('source','?')}",
            f"  Seals:   {master.get('n_seals', 0)}",
            "",
        ]

    stages = chain.get("stages", [])
    if not stages:
        lines.append("No pipeline stages recorded (passport system not active during run)")
        return "\n".join(lines)

    lines.append(f"PIPELINE STAGES ({len(stages)} recorded):")
    lines.append("")

    for i, stage in enumerate(stages):
        connector = "│" if i < len(stages) - 1 else " "
        arrow     = "├──" if i < len(stages) - 1 else "└──"

        lines.append(f"  {arrow} Stage {stage.get('stage_index', i+1)}: {stage.get('data_type','?')} [{stage.get('source','?')}]")
        lines.append(f"  {connector}     Passport: {stage.get('passport_id','?')[:12]}...")
        lines.append(f"  {connector}     Created:  {stage.get('created_at','?')}")

        seals = stage.get("seals", [])
        if seals:
            lines.append(f"  {connector}     Seals ({len(seals)}):")
            for seal in seals:
                status_icon = "✓" if seal.get("status") == "valid" else "⚠"
                lines.append(
                    f"  {connector}       {status_icon} [{seal.get('phase','?')}] {seal.get('function','?')} "
                    f"— {seal.get('execution_time_ms',0):.1f}ms "
                    f"— in={seal.get('input_hash','?')[:8]} → out={seal.get('output_hash','?')[:8]}"
                )
                if seal.get("transformation_type") and seal["transformation_type"] != "unknown":
                    lines.append(f"  {connector}         type: {seal['transformation_type']}")

        checkpoints = stage.get("checkpoints", [])
        if checkpoints:
            lines.append(f"  {connector}     Checkpoints ({len(checkpoints)}):")
            for cp in checkpoints:
                lines.append(f"  {connector}       ◉ {cp.get('name','?')} — {cp.get('data_hash','?')[:8]}...")

        if i < len(stages) - 1:
            lines.append(f"  │")

    lines += [
        "",
        "=" * 80,
        f"Total stages: {len(stages)}  |  "
        f"Total seals: {sum(len(s.get('seals',[])) for s in stages)}  |  "
        f"Total checkpoints: {sum(len(s.get('checkpoints',[])) for s in stages)}",
        "=" * 80,
    ]
    return "\n".join(lines)


def render_pipeline_html(chain: dict) -> str:
    """Render pipeline chain as a self-contained HTML file."""
    stages = chain.get("stages", [])
    master = chain.get("master", {})

    stage_html = ""
    for stage in stages:
        seals_html = ""
        for seal in stage.get("seals", []):
            status_color = "#27500A" if seal.get("status") == "valid" else "#A32D2D"
            status_bg    = "#EAF3DE" if seal.get("status") == "valid" else "#FCEBEB"
            seals_html += f"""
            <div style="margin:4px 0 4px 16px; padding:6px 10px;
                        background:{status_bg}; border-radius:4px;
                        border-left:3px solid {status_color}; font-size:12px;">
              <strong>[{seal.get('phase','?')}]</strong> {seal.get('function','?')}
              &nbsp;|&nbsp; {seal.get('execution_time_ms',0):.1f}ms
              &nbsp;|&nbsp; type: <code>{seal.get('transformation_type','?')}</code>
              <br>
              <span style="color:#888;font-size:10px;">
                in: <code>{seal.get('input_hash','?')[:12]}</code> →
                out: <code>{seal.get('output_hash','?')[:12]}</code>
              </span>
            </div>"""

        cps_html = ""
        for cp in stage.get("checkpoints", []):
            cps_html += f"""
            <div style="margin:3px 0 3px 16px; font-size:11px; color:#555;">
              ◉ <strong>{cp.get('name','?')}</strong> — hash: <code>{cp.get('data_hash','?')[:12]}</code>
            </div>"""

        stage_html += f"""
        <div style="border:1px solid #ddd; border-radius:8px; margin:8px 0; padding:12px;
                    background:#fafafa;">
          <div style="font-weight:500; font-size:13px; margin-bottom:6px;">
            Stage {stage.get('stage_index','?')}: {stage.get('data_type','?')}
            <span style="color:#888; font-size:11px; margin-left:8px;">
              [{stage.get('source','?')}]
            </span>
          </div>
          <div style="font-size:10px; color:#888; margin-bottom:6px;">
            passport: <code>{stage.get('passport_id','?')[:16]}...</code>
            &nbsp;|&nbsp; {stage.get('created_at','?')}
          </div>
          {seals_html}
          {cps_html}
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>Data Pipeline Journey — TFM Passport Viewer</title>
<style>
  body {{ font-family: monospace; margin: 2em; background:#f8f8f8; }}
  h1 {{ color:#333; border-bottom:2px solid #666; }}
  code {{ background:#f0f0f0; padding:1px 4px; border-radius:3px; font-size:11px; }}
</style>
</head>
<body>
<h1>Data Pipeline Journey — Passport Chain</h1>
<p style="color:#666;">
  Master: <code>{master.get('passport_id','?')}</code> &nbsp;|&nbsp;
  Generated: {datetime.now().isoformat()} &nbsp;|&nbsp;
  Stages: {len(stages)}
</p>
<div style="border-left:3px solid #378ADD; padding-left:12px; margin:16px 0;">
{stage_html}
</div>
<p style="font-size:11px; color:#999; margin-top:32px;">
  TFM: Hybrid Quantum-Classical Portfolio Optimization with CVaR — Ignacio Lopez Leis, UAM
</p>
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Passport pipeline viewer")
    parser.add_argument("--exp", help="Experiment ID (e.g. A1_n3_PA)")
    parser.add_argument("--chain", help="Direct path to pipeline_chain.json")
    parser.add_argument("--html", action="store_true", help="Output HTML instead of text")
    args = parser.parse_args()

    if args.chain:
        chain_path = Path(args.chain)
    elif args.exp:
        chain_path = find_project_root() / "results" / f"exp_{args.exp}" / "passports" / "pipeline_chain.json"
    else:
        parser.print_help(); return

    if not chain_path.exists():
        sys.stderr.write(f"Chain file not found: {chain_path}\n")
        sys.stderr.write("Run the experiment with the V4 passport fixes applied first.\n")
        return

    chain = json.loads(chain_path.read_text())

    if args.html:
        html = render_pipeline_html(chain)
        out = chain_path.parent / "pipeline_view.html"
        out.write_text(html, encoding="utf-8")
        sys.stdout.write(f"HTML report: {out}\n")
    else:
        sys.stdout.write(render_pipeline_text(chain) + "\n")


if __name__ == "__main__":
    main()
