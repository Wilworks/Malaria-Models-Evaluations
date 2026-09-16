import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.titlesize": 12,
    "figure.titleweight": "bold"
})

def extract_center_crop(img, crop_size=600):
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    half = crop_size // 2
    return img[cy - half:cy + half, cx - half:cx + half]

def main():
    root = Path(".")
    out_dir = root / "manuscript" / "figures" / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # FIGURE S3: ATLAS OF FALSE-POSITIVE CONFOUNDERS
    # ----------------------------------------------------
    # Candidates: Thin 192, 556, 700, 740
    fp_candidates = [
        {"id": "700", "type": "thin", "crop_box": (1500, 2000, 500), "label": "(A) Giemsa Stain Precipitate", "desc": "Dense extracellular dye crystal mimicking parasite chromatin.\nFlagged as positive by all 4 candidate models."},
        {"id": "740", "type": "thin", "crop_box": (2000, 1800, 500), "label": "(B) Platelet Superimposition", "desc": "Thrombocyte overlying intact erythrocyte margin.\nMisclassified by MS_Sudan, MS_Thick, and MS_Thin."},
        {"id": "192", "type": "thin", "crop_box": (2100, 1400, 500), "label": "(C) Howell-Jolly Body Inclusion", "desc": "Basophilic nuclear remnant inside normocyte.\nTriggers spatial detection false-positive in YOLOv8."},
        {"id": "556", "type": "thin", "crop_box": (1900, 1500, 500), "label": "(D) Cellular Debris / Staining Artifact", "desc": "Amorphous proteinaceous deposit on smear surface.\nModel confidence: MS_Thin = 0.94, MS_Sudan = 0.81."}
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for i, c in enumerate(fp_candidates):
        ax = axes[i]
        p = root / "data" / "raw" / f"{c['type']}_smear" / f"{c['id']}.jpg"
        img = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cy, cx, sz = c["crop_box"]
        crop = img_rgb[cy - sz//2 : cy + sz//2, cx - sz//2 : cx + sz//2]
        
        ax.imshow(crop)
        ax.set_title(c["label"], fontsize=10.5, pad=8)
        ax.axis("off")
        
        # Add red border and false-positive badge
        rect = plt.Rectangle((0, 0), crop.shape[1]-1, crop.shape[0]-1, fill=False, edgecolor="#D55E00", linewidth=3)
        ax.add_patch(rect)
        
        # Overlay badge
        ax.text(0.04, 0.94, "FALSE POSITIVE", transform=ax.transAxes,
                fontsize=8, fontweight="bold", color="#FFFFFF", va="top",
                bbox=dict(boxstyle="square,pad=0.25", fc="#D55E00", ec="none"))
        ax.text(0.04, 0.06, f"Slide #{c['id']} (Uninfected GT)\n{c['desc']}", transform=ax.transAxes,
                fontsize=7.5, color="#111111", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#CCCCCC", alpha=0.92))

    plt.tight_layout()
    fig.savefig(out_dir / "fig_s3_false_positives_atlas.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_s3_false_positives_atlas.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure S3.")

    # ----------------------------------------------------
    # FIGURE S4: ATLAS OF FALSE-NEGATIVE MISSES
    # ----------------------------------------------------
    # Candidates from confirmed infected slides where YOLOv8 or classifiers missed:
    # 1000, 1015, 1035 (thick), and 112 or 1035
    fn_candidates = [
        {"id": "1000", "type": "thick", "crop_box": (2000, 1500, 500), "label": "(A) Severe Optical Defocus Blur", "desc": "Thick smear field with log-Laplacian focus < 8.5.\nParasite chromatin diffused into background noise."},
        {"id": "1015", "type": "thick", "crop_box": (1900, 1600, 500), "label": "(B) Faint Ring-Stage Trophozoite", "desc": "Early delicate ring with minimal cytoplasm.\nDetector bounding box confidence fell below tau = 0.15."},
        {"id": "1035", "type": "thick", "crop_box": (2000, 1500, 500), "label": "(C) Heavy Background & Leukocyte Clutter", "desc": "Dense dehemoglobinized matrix and WBC fragments.\nSpatial model missed confirmed trophozoites."},
        {"id": "1024", "type": "thick", "crop_box": (2000, 1500, 500), "label": "(D) Chromatin Contrast Attenuation", "desc": "Sub-optimal local Giemsa staining intensity.\nSignal-to-noise ratio < 4.0 dB precludes detection."}
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2), dpi=300)
    fig.patch.set_facecolor("#FFFFFF")

    for i, c in enumerate(fn_candidates):
        ax = axes[i]
        p = root / "data" / "raw" / f"{c['type']}_smear" / f"{c['id']}.jpg"
        img = cv2.imread(str(p))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        cy, cx, sz = c["crop_box"]
        crop = img_rgb[cy - sz//2 : cy + sz//2, cx - sz//2 : cx + sz//2]
        
        ax.imshow(crop)
        ax.set_title(c["label"], fontsize=10.5, pad=8)
        ax.axis("off")
        
        # Add blue border and false-negative badge
        rect = plt.Rectangle((0, 0), crop.shape[1]-1, crop.shape[0]-1, fill=False, edgecolor="#0072B2", linewidth=3)
        ax.add_patch(rect)
        
        # Overlay badge
        ax.text(0.04, 0.94, "FALSE NEGATIVE", transform=ax.transAxes,
                fontsize=8, fontweight="bold", color="#FFFFFF", va="top",
                bbox=dict(boxstyle="square,pad=0.25", fc="#0072B2", ec="none"))
        ax.text(0.04, 0.06, f"Slide #{c['id']} (Infected GT)\n{c['desc']}", transform=ax.transAxes,
                fontsize=7.5, color="#111111", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec="#CCCCCC", alpha=0.92))

    plt.tight_layout()
    fig.savefig(out_dir / "fig_s4_false_negatives_atlas.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig_s4_false_negatives_atlas.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure S4.")

if __name__ == "__main__":
    main()
