#!/usr/bin/env python3
"""
Generate Figure 4: Multi-Center Micrograph Morphology, Rouleaux Formation,
and Optical Defocus Blur Gallery.
Standardized to 14.0 x 7.5 inches at 300 DPI matching Figures 1, 2, and 3.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
from pathlib import Path

ROOT = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
W, H = 14.0, 7.5
DPI = 300


def crop_center_square(img, crop_fraction=0.55):
    """Crops a square region from the center of the micrograph."""
    h, w = img.shape[:2]
    side = int(min(h, w) * crop_fraction)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    return img[y1 : y1 + side, x1 : x1 + side], (x1, y1, side, side)


def main():
    p_ghana = ROOT / "data" / "raw" / "lacuna_ghana" / "thick_smear" / "positive" / "2.jpg"
    p_osun = ROOT / "data" / "raw" / "adeleke_nigeria" / "thick_smear" / "positive" / "Pos_PH_35.jpg"
    p_kano_rouleaux = ROOT / "data" / "raw" / "muhammad_kano_nigeria" / "thin_smear" / "positive" / "AR2_4.jpg"
    p_kano_healthy = ROOT / "data" / "raw" / "muhammad_kano_nigeria" / "thin_smear" / "negative" / "IMG_20211006_141101.jpg"
    p_sharp = ROOT / "data" / "raw" / "muhammad_kano_nigeria" / "thin_smear" / "positive" / "IMG_0667.JPG"
    p_blurry = ROOT / "data" / "raw" / "lacuna_ghana" / "thin_smear" / "positive" / "622.jpg"

    paths = [p_ghana, p_osun, p_kano_rouleaux, p_kano_healthy, p_sharp, p_blurry]
    titles = [
        "(a) Accra Thick Smear (PML Hospital, Ghana)\nHigh Parasitemia with Lysed Stroma",
        "(b) Osun Thick Smear (Adeleke Cohort, Nigeria)\nPrimary Center Smear & Giemsa Precipitate",
        "(c) Kano Thin Smear (Acute Symptomatic Malaria)\nMarked Erythrocyte Rouleaux Stacking",
        "(d) Kano Thin Smear (Healthy Control Cohort)\nUniform Biconcave Discocyte Monolayer",
        "(e) High-Focus In-Distribution Field\nSharp High-Frequency Chromatin (\u03c3\u00b2_Lap = 360.0)",
        "(f) Defocused Field (Clinical Hardware Failure)\nSevere Blur Suppression (\u03c3\u00b2_Lap = 2.1)"
    ]
    subtitles = [
        "Lysed RBC stroma allows clear ring detection (green boxes)",
        "Denser manual smear thickness & artisanal staining artifacts",
        "Erythrocyte clumping obscures single-cell boundaries (RQ2/RQ3)",
        "Intact RBC rings mimic trophozoites, triggering false proposals",
        "Resolves minute purple chromatin dot & pale cytoplasm",
        "Parasite rings vanish into stroma; pre-inference gate mandatory"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(W, H), dpi=DPI)
    axes = axes.flatten()

    for idx, (p, title, sub) in enumerate(zip(paths, titles, subtitles)):
        ax = axes[idx]
        img = cv2.imread(str(p))
        if img is None:
            ax.text(0.5, 0.5, f"Image Not Found:\n{p.name}", ha="center", va="center")
            continue

        cropped, (cx, cy, cw, ch) = crop_center_square(img, crop_fraction=0.60 if idx < 4 else 0.50)
        cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        ax.imshow(cropped_rgb)

        # Highlight ground truth bounding boxes in panel (a) if available
        if idx == 0:
            lbl_p = ROOT / "data" / "raw" / "lacuna_ghana" / "thick_smear" / "labels_yolo" / "2.txt"
            if lbl_p.exists():
                h_orig, w_orig = img.shape[:2]
                with open(lbl_p) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id, xc, yc, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            bx_px = xc * w_orig
                            by_px = yc * h_orig
                            bw_px = bw * w_orig
                            bh_px = bh * h_orig
                            if cx <= bx_px <= cx + cw and cy <= by_px <= cy + ch:
                                rx = (bx_px - bw_px / 2.0) - cx
                                ry = (by_px - bh_px / 2.0) - cy
                                rect = patches.Rectangle((rx, ry), bw_px, bh_px, linewidth=2.0,
                                                         edgecolor="#00FF00" if cls_id == 0 else "#FF3333",
                                                         facecolor="none")
                                ax.add_patch(rect)

        # Status badge
        badge_color = "#2CA02C" if idx in [0, 4] else ("#D62728" if idx == 5 else "#1F77B4")
        ax.set_title(title, fontsize=10.0, fontweight="bold", pad=5)
        ax.set_xlabel(sub, fontsize=8.5, style="italic", labelpad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(badge_color)
            spine.set_linewidth(1.8)

    plt.suptitle("Multi-Center Micrograph Morphology, Erythrocyte Rouleaux Formation, and Defocus Blur Degradation",
                 fontsize=13.0, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = FIG_DIR / "fig4_qualitative_gallery.png"
    plt.savefig(out_path, dpi=DPI)
    plt.close()
    print(f"✓ Generated {out_path} ({W}x{H} in at {DPI} DPI)")


if __name__ == "__main__":
    main()
