#!/usr/bin/env python3
"""
Generate All Manuscript Figures with Identical Canvas Dimensions, Font Scales,
and Aspect Ratios (16:8.5 at 300 DPI) for Uniform Publication Presentation.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np
import cv2
from pathlib import Path

import sys
ROOT = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
sys.path.insert(0, str(ROOT))
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
W, H = 14.0, 7.5
DPI = 300

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: D4 Dihedral Group Augmentation Invariance Grid (2 x 4 layout)
# ══════════════════════════════════════════════════════════════════════════════
from scripts.visualize_bounding_box_transforms import (
    transform_image_d4, transform_box_d4, load_yolo_labels
)

img_p = ROOT / "data" / "raw" / "lacuna_ghana" / "thick_smear" / "positive" / "2.jpg"
lbl_p = ROOT / "data" / "raw" / "lacuna_ghana" / "thick_smear" / "labels_yolo" / "2.txt"

img = cv2.imread(str(img_p))
boxes = load_yolo_labels(lbl_p)

ops = [
    ("identity", "1. Original Micrograph (Identity I)", "xc'=xc, yc'=yc"),
    ("hflip", "2. Horizontal Reflection (F_h)", "xc'=1-xc, yc'=yc"),
    ("vflip", "3. Vertical Reflection (F_v)", "xc'=xc, yc'=1-yc"),
    ("rot90", "4. 90 deg Orthogonal Rotation (R_90)", "xc'=1-yc, yc'=xc"),
    ("rot180", "5. 180 deg Inversion (R_180)", "xc'=1-xc, yc'=1-yc"),
    ("rot270", "6. 270 deg Orthogonal Rotation (R_270)", "xc'=yc, yc'=1-xc"),
    ("diag_main", "7. Main Diagonal Reflection (D_main)", "xc'=yc, yc'=xc"),
    ("diag_anti", "8. Anti-Diagonal Reflection (D_anti)", "xc'=1-yc, yc'=1-xc"),
]

fig, axes = plt.subplots(2, 4, figsize=(W, H), dpi=DPI)
axes = axes.flatten()

for i, (op_name, title, sub) in enumerate(ops):
    ax = axes[i]
    t_img = transform_image_d4(img, op_name)
    t_boxes = [transform_box_d4(b, op_name) for b in boxes]
    h_img, w_img = t_img.shape[:2]
    
    ax.imshow(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
    for cls_id, xc, yc, bw, bh in t_boxes:
        xmin = (xc - bw / 2.0) * w_img
        ymin = (yc - bh / 2.0) * h_img
        rect = patches.Rectangle((xmin, ymin), bw * w_img, bh * h_img,
                                 linewidth=1.8, edgecolor="#00FF00" if cls_id==0 else "#FF3333",
                                 facecolor="none")
        ax.add_patch(rect)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=4)
    ax.set_xlabel(sub, fontsize=9.5, style="italic")
    ax.set_xticks([])
    ax.set_yticks([])

plt.suptitle("Validation of Dihedral Group D4 Data Augmentation Protocol Across Thick Smears",
             fontsize=13.5, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_DIR / "fig1_d4_transforms.png", dpi=DPI)
plt.close()
print("✓ Generated fig1_d4_transforms.png (14x7.5)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Zero-Shot Diagnostic Performance vs WHO 90% Benchmark
# ══════════════════════════════════════════════════════════════════════════════
df1 = pd.read_csv(ROOT / "results" / "run_1" / "table1_master_diagnostic_performance.csv")
df_pooled = df1[df1["Data_Source"] == "Pooled (All West Africa)"].copy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, H), dpi=DPI)
labels = {
    "MalariaScreener_Sudan": "MS Sudan\n(NIH, Sudan)",
    "MalariaScreener_Thick": "MS Thick\n(NIH, Bangladesh)", 
    "MalariaScreener_Thin": "MS Thin\n(NIH, Bangladesh)",
    "fbononibelloepoch_YOLOv8": "YOLOv8 Nano\n(Field Detector)"
}

x = np.arange(4)
bar_w = 0.35

for idx, (modality, ax) in enumerate([("Thick", ax1), ("Thin", ax2)]):
    sub = df_pooled[df_pooled["Modality"] == modality]
    models = list(sub["Model"])
    sens = list(sub["Sensitivity (%)"])
    spec = list(sub["Specificity (%)"])
    
    b1 = ax.bar(x - bar_w/2, sens, bar_w, label="Sensitivity (%)", color="#1F77B4", alpha=0.9, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + bar_w/2, spec, bar_w, label="Specificity (%)", color="#FF7F0E", alpha=0.9, edgecolor="black", linewidth=0.8)
    
    ax.axhline(90, color="crimson", linestyle="--", linewidth=1.8, label="WHO Level-1 Benchmark (90%)" if idx==0 else "")
    ax.set_title(f"{modality} Smear Cohort Performance (Zero-Shot)", fontsize=12.5, fontweight="bold", pad=10)
    ax.set_ylabel("Diagnostic Metric (%)" if idx == 0 else "", fontsize=11.5)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(m, m) for m in models], fontsize=10.0)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    if idx == 0:
        ax.legend(frameon=True, fontsize=10, loc="upper right")

plt.suptitle("Zero-Shot Cross-Domain Clinical Evaluation vs. WHO Level-1 Triage Benchmark",
             fontsize=13.5, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_DIR / "fig2_zero_shot_performance.png", dpi=DPI)
plt.close()
print("✓ Generated fig2_zero_shot_performance.png (14x7.5)")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Optical Quality Sensitivity Degradation Curves
# ══════════════════════════════════════════════════════════════════════════════
df3 = pd.read_csv(ROOT / "results" / "run_1" / "table3_quality_stratified_sensitivity.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(W, H), dpi=DPI)

strata = ["Low Quality", "Medium Quality", "High Quality"]
strata_labels = ["Blurry Fields\n(Low Focus)", "Moderate Focus\n(Medium)", "Sharp In-Focus\n(High)"]

colors = {
    "MalariaScreener_Sudan": "#D55E00", 
    "MalariaScreener_Thick": "#0072B2",
    "MalariaScreener_Thin": "#009E73", 
    "fbononibelloepoch_YOLOv8": "#CC79A7"
}

# Thick Smear
for model in ["MalariaScreener_Sudan", "MalariaScreener_Thick", "fbononibelloepoch_YOLOv8"]:
    sub = df3[(df3["smear_type"] == "thick") & (df3["model_name"] == model)]
    sens_vals = [sub[sub["quality_strata"] == s]["sensitivity"].values[0] for s in strata]
    ax1.plot(strata_labels, sens_vals, marker="o", markersize=8, linewidth=2.4,
             label=labels.get(model, model).split("\n")[0], color=colors.get(model, "#333"))

ax1.set_title("Thick Smears: Focus Blur vs. Sensitivity", fontsize=12.5, fontweight="bold", pad=10)
ax1.set_ylabel("Diagnostic Sensitivity (%)", fontsize=11.5)
ax1.set_ylim(0, 100)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(frameon=True, fontsize=10, loc="lower right")

# Thin Smear
for model in ["MalariaScreener_Sudan", "MalariaScreener_Thin", "fbononibelloepoch_YOLOv8"]:
    sub = df3[(df3["smear_type"] == "thin") & (df3["model_name"] == model)]
    sens_vals = [sub[sub["quality_strata"] == s]["sensitivity"].values[0] for s in strata]
    ax2.plot(strata_labels, sens_vals, marker="s", markersize=8, linewidth=2.4,
             label=labels.get(model, model).split("\n")[0], color=colors.get(model, "#333"))

ax2.set_title("Thin Smears: Focus Blur vs. Sensitivity", fontsize=12.5, fontweight="bold", pad=10)
ax2.set_ylim(0, 100)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(frameon=True, fontsize=10, loc="lower right")

plt.suptitle("Impact of Laplacian Optical Focus Blur on Diagnostic Sensitivity",
             fontsize=13.5, fontweight="bold", y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(FIG_DIR / "fig3_optical_quality_curves.png", dpi=DPI)
plt.close()
print("✓ Generated fig3_optical_quality_curves.png (14x7.5)")
