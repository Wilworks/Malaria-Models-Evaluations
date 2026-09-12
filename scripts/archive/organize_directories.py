import shutil
from pathlib import Path

root = Path(__file__).resolve().parent.parent
fig_root = root / "manuscript" / "figures"
tab_root = root / "manuscript" / "tables"

# 1. Define new directory layout
subdirs = [
    fig_root / "01_cross_modality",
    fig_root / "02_thick_smears",
    fig_root / "03_thin_smears",
    fig_root / "04_failure_modes",
    fig_root / "05_imaging_physics",
    tab_root / "01_primary_benchmarks",
    tab_root / "02_stratified_quality",
    tab_root / "03_stage_sensitivity",
]

for d in subdirs:
    d.mkdir(parents=True, exist_ok=True)
    print(f"[Created] {d.relative_to(root)}")

# 2. Move existing figures into their respective home
fig_moves = [
    (fig_root / "fig4_crossmodality_heatmap.png", fig_root / "01_cross_modality" / "fig_cross_modality_heatmap.png"),
    (fig_root / "fig2_sensitivity_comparison.png", fig_root / "02_thick_smears" / "fig_thick_sensitivity_comparison.png"),
    (fig_root / "fig3_quality_sensitivity_lines.png", fig_root / "02_thick_smears" / "fig_thick_quality_sensitivity_lines.png"),
    (fig_root / "fig6_crossmodel_comparison.png", fig_root / "04_failure_modes" / "fig_crossmodel_5panel_comparison.png"),
]

for src, dst in fig_moves:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"[Organized] {src.name} -> {dst.relative_to(root)}")

# 3. Move existing tables
tab_moves = [
    (tab_root / "main_results_table.csv", tab_root / "01_primary_benchmarks" / "table_thick_results.csv"),
    (tab_root / "stratified_sensitivity.csv", tab_root / "02_stratified_quality" / "table_thick_stratified.csv"),
]

for src, dst in tab_moves:
    if src.exists():
        shutil.copy2(src, dst)
        print(f"[Organized] {src.name} -> {dst.relative_to(root)}")

print("\n=== Directory Restructuring Complete ===")
