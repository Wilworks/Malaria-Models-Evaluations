#!/usr/bin/env python3
"""
Automatically ingest YOLO PELP and Full Retrain metrics into Table 5 across
both paper/main.tex and paper/collaborative_draft_guide.tex, then recompile PDFs.
"""

import re
import json
import subprocess
from pathlib import Path

ROOT = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
RESULTS_DIR = ROOT / "results" / "finetuning"
MAIN_TEX = ROOT / "paper" / "main.tex"
GUIDE_TEX = ROOT / "paper" / "collaborative_draft_guide.tex"

def format_row(name, training_regime, metrics_file):
    p = RESULTS_DIR / metrics_file
    if not p.exists():
        return f"{name:<28} & {training_regime:<38} & \\textit{{[Evaluating]}} & \\textit{{[Evaluating]}} & \\textit{{[Evaluating]}} & \\textit{{[Evaluating]}} & \\textit{{[Evaluating]}} \\\\"
    
    with open(p) as f:
        d = json.load(f)
        
    sens = d.get("sensitivity_pct", d.get("sensitivity", 0.0))
    sens_ci = d.get("sens_95ci", [0.0, 0.0])
    spec = d.get("specificity_pct", d.get("specificity", 0.0))
    spec_ci = d.get("spec_95ci", [0.0, 0.0])
    f1 = d.get("f1_score", 0.0)
    
    # Baseline zero-shot YOLO F1 was 0.6390
    delta_f1 = f1 - 0.6390
    delta_str = f"+{delta_f1:.4f}" if delta_f1 >= 0 else f"{delta_f1:.4f}"
    
    status = "\\textbf{Pass (High-Sens Triage)}" if sens >= 90.0 else ("\\textbf{Pass (\\ge 90\\%)}" if (sens >= 90.0 and spec >= 90.0) else "Borderline ($<90\\%$)")
    
    sens_str = f"{sens:.2f}\\% [{sens_ci[0]:.1f}--{sens_ci[1]:.1f}]"
    spec_str = f"{spec:.2f}\\% [{spec_ci[0]:.1f}--{spec_ci[1]:.1f}]"
    f1_str = f"{f1:.4f}"
    
    return f"{name:<28} & {training_regime:<38} & {sens_str} & {spec_str} & {f1_str} & {delta_str} & {status} \\\\"

def update_tex(tex_path):
    content = tex_path.read_text()
    
    pelp_row = format_row("fbononi-YOLOv8 (PELP)", "Frozen Backbone (\\texttt{freeze}=10)", "yolo_thick_pelp_metrics.json")
    full_row = format_row("fbononi-YOLOv8 (Full Retrain)", "End-to-End Anchor-Free ($D_4$)", "yolo_thick_full_metrics.json")
    
    # Replace PELP row
    content = re.sub(
        r"fbononi-YOLOv8 \(PELP\)\s+&[^\\]+\\\\([^\n]*)",
        pelp_row + r"\1",
        content
    )
    
    # Replace Full row
    content = re.sub(
        r"fbononi-YOLOv8 \(Full Retrain\)\s+&[^\\]+\\\\([^\n]*)",
        full_row + r"\1",
        content
    )
    
    tex_path.write_text(content)
    print(f"[Updated] Synced Table 5 in: {tex_path.name}")

def main():
    update_tex(MAIN_TEX)
    update_tex(GUIDE_TEX)
    
    print("\n[Compiling] Running Tectonic build on both manuscripts...")
    subprocess.run(["/Users/wilfredayineasumboya/.local/bin/tectonic", "paper/main.tex"], cwd=ROOT, check=True)
    subprocess.run(["/Users/wilfredayineasumboya/.local/bin/tectonic", "paper/collaborative_draft_guide.tex"], cwd=ROOT, check=True)
    print("\n[Complete] Both PDF manuscripts successfully recompiled and updated with YOLO metrics!")

if __name__ == "__main__":
    main()
