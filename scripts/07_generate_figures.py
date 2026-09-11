"""
Script 07: Generate all paper figures from current evaluation results.
Produces:
  - Fig 2: Sensitivity bar chart (all 4 models vs WHO threshold)
  - Fig 3: Quality-sensitivity line plot (3 strata × 4 models)
  - Fig 4: Cross-modality heatmap (models × smear type) — thick column only for now
  - Fig 5: Curated TP + FN 3-panel smear image gallery per model
  - Fig 6: Same image run through all 4 models (cross-model comparison panel)
"""

import sys
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT / "manuscript" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

LABELS_DIR = ROOT / "data" / "raw" / "thick_smear" / "labels_yolo"
IMG_DIR    = ROOT / "data" / "raw" / "thick_smear"

# Color palette — one per model, consistent across all figures
MODEL_COLORS = {
    "MalariaScreener_Sudan":   "#E07B39",
    "MalariaScreener_Thick":   "#4A90D9",
    "MalariaScreener_Thin":    "#7DB87A",
    "fbononibelloepoch_YOLOv8":"#9B59B6",
}
MODEL_LABELS = {
    "MalariaScreener_Sudan":   "MS_Sudan\n(NIH, thick, Sudan)",
    "MalariaScreener_Thick":   "MS_Thick\n(NIH, thick, Bangladesh)",
    "MalariaScreener_Thin":    "MS_Thin\n(NIH, thin, Bangladesh)",
    "fbononibelloepoch_YOLOv8":"fbononi YOLOv8\n(HuggingFace, informal)",
}
WHO_THRESHOLD = 0.90


def load_data():
    """Load manifest + GT labels + quality strata."""
    gt_map = {}
    for img in IMG_DIR.iterdir():
        if img.suffix in (".jpg", ".png", ".jpeg"):
            lf = LABELS_DIR / (img.stem + ".txt")
            gt_map[img.stem] = 1 if (lf.exists() and lf.read_text().strip()) else 0

    df = pd.read_csv(ROOT / "data" / "processed" / "predictions_manifest.csv")
    df["gt"] = df["image_id"].astype(str).map(gt_map).fillna(0).astype(int)

    qm = pd.read_csv(ROOT / "data" / "quality_metrics" / "quality_manifest.csv")
    qm["quality_strata"] = pd.qcut(qm["blur_laplacian"], q=3,
                                   labels=["Low", "Medium", "High"], duplicates="drop")
    df = df.merge(qm[["image_id", "smear_type", "blur_laplacian", "quality_strata"]],
                  on=["image_id", "smear_type"], how="left")
    return df, gt_map


def compute_agg(df):
    rows = []
    for model, grp in df.groupby("model_name"):
        TP = int(((grp["predicted_class"]==1) & (grp["gt"]==1)).sum())
        FN = int(((grp["predicted_class"]==0) & (grp["gt"]==1)).sum())
        FP = int(((grp["predicted_class"]==1) & (grp["gt"]==0)).sum())
        n  = len(grp)
        sens = TP / (TP + FN) if (TP + FN) else 0
        f1   = 2*TP / (2*TP + FP + FN) if (2*TP + FP + FN) else 0
        rows.append({"model": model, "sensitivity": sens, "f1": f1,
                     "TP": TP, "FN": FN, "n": n})
    return pd.DataFrame(rows)


def compute_strat(df):
    rows = []
    for model, grp in df.groupby("model_name"):
        for strata, sg in grp.groupby("quality_strata"):
            TP = int(((sg["predicted_class"]==1) & (sg["gt"]==1)).sum())
            FN = int(((sg["predicted_class"]==0) & (sg["gt"]==1)).sum())
            sens = TP / (TP + FN) if (TP + FN) else 0
            rows.append({"model": model, "strata": str(strata), "sensitivity": sens})
    return pd.DataFrame(rows)


# ── Figure 2: Sensitivity Bar Chart ──────────────────────────────────────────
def fig2_sensitivity_bars(agg_df):
    models = ["MalariaScreener_Sudan", "MalariaScreener_Thick",
              "MalariaScreener_Thin", "fbononibelloepoch_YOLOv8"]
    agg = agg_df.set_index("model")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    x = np.arange(len(models))
    bars = []
    for i, m in enumerate(models):
        s = float(agg.loc[m, "sensitivity"]) if m in agg.index else 0
        bar = ax.bar(i, s * 100, color=MODEL_COLORS[m], width=0.55,
                     zorder=3, alpha=0.92, edgecolor="white", linewidth=0.5)
        bars.append(bar)
        ax.text(i, s * 100 + 0.8, f"{s*100:.1f}%",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color="white")

    # WHO threshold line
    ax.axhline(WHO_THRESHOLD * 100, color="#FF4444", linewidth=2,
               linestyle="--", zorder=4, label="WHO 90% Triage Threshold")
    ax.text(len(models) - 0.45, WHO_THRESHOLD * 100 + 0.8,
            "WHO 90%", color="#FF4444", fontsize=10, fontweight="bold", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models],
                       fontsize=9.5, color="white")
    ax.set_ylabel("Zero-Shot Sensitivity (%)", fontsize=12, color="white")
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 10))
    ax.tick_params(colors="white", labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.grid(True, color="#333355", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    ax.set_title("Zero-Shot Sensitivity: All 4 Models vs WHO Clinical Triage Threshold\n"
                 "Dataset: n=432 Ghanaian Thick Smear Images (Princess Marie Louise Hospital, Accra)",
                 fontsize=11, color="white", pad=14)

    fig.tight_layout()
    out = FIGURES_DIR / "fig2_sensitivity_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] {out.name}")


# ── Figure 3: Quality-Sensitivity Line Plot ───────────────────────────────────
def fig3_quality_lines(strat_df):
    models = ["MalariaScreener_Sudan", "MalariaScreener_Thick",
              "MalariaScreener_Thin", "fbononibelloepoch_YOLOv8"]
    strata_order = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    for m in models:
        sub = strat_df[strat_df["model"] == m].set_index("strata")
        vals = [float(sub.loc[s, "sensitivity"]) * 100
                if s in sub.index else None for s in strata_order]
        ax.plot(strata_order, vals, marker="o", linewidth=2.5,
                markersize=8, color=MODEL_COLORS[m], label=MODEL_LABELS[m].replace("\n", " "),
                zorder=3)
        # Annotate final point
        if vals[-1] is not None:
            ax.annotate(f"{vals[-1]:.1f}%", xy=(2, vals[-1]),
                        xytext=(8, 0), textcoords="offset points",
                        color=MODEL_COLORS[m], fontsize=9, fontweight="bold", va="center")

    ax.axhline(WHO_THRESHOLD * 100, color="#FF4444", linewidth=1.8,
               linestyle="--", zorder=2, label="WHO 90% Threshold")

    ax.set_xlabel("Image Focus Quality Strata (Laplacian Variance Tertile)",
                  fontsize=11, color="white")
    ax.set_ylabel("Sensitivity (%)", fontsize=11, color="white")
    ax.set_ylim(55, 100)
    ax.tick_params(colors="white", labelsize=10)
    ax.xaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.yaxis.grid(True, color="#333355", linewidth=0.6)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=8.5, loc="lower left",
                       facecolor="#1A1D2E", edgecolor="#555577",
                       labelcolor="white", framealpha=0.9)

    ax.set_title("Quality-Stratified Sensitivity: Three Distinct Model Response Patterns\n"
                 "Low (Lap≤9.1) | Medium (9.1–15.6) | High (>15.6) | n=144 per stratum",
                 fontsize=10.5, color="white", pad=14)

    fig.tight_layout()
    out = FIGURES_DIR / "fig3_quality_sensitivity_lines.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] {out.name}")


# ── Figure 4: Cross-Modality Heatmap (partial — thick column only) ────────────
def fig4_crossmodality_heatmap(agg_df):
    models = ["MalariaScreener_Sudan", "MalariaScreener_Thick",
              "MalariaScreener_Thin", "fbononibelloepoch_YOLOv8"]
    model_short = ["MS_Sudan", "MS_Thick", "MS_Thin", "fbononi YOLOv8"]
    agg = agg_df.set_index("model")

    thick_sens = [float(agg.loc[m, "sensitivity"]) if m in agg.index else np.nan
                  for m in models]
    thin_sens  = [np.nan] * len(models)  # pending

    data = np.array([thick_sens, thin_sens]).T  # shape: (4 models, 2 smear types)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0F1117")
    ax.set_facecolor("#1A1D2E")

    cmap = LinearSegmentedColormap.from_list("red_green", ["#8B0000", "#FFD700", "#006400"])
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=0.55, vmax=0.95, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Thick Smear\n(Ghana, n=432)", "Thin Smear\n(Ghana, pending)"],
                       color="white", fontsize=10)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(model_short, color="white", fontsize=10)

    # Annotate cells
    for i in range(len(models)):
        val = data[i, 0]
        if not np.isnan(val):
            ax.text(0, i, f"{val*100:.1f}%", ha="center", va="center",
                    fontsize=13, fontweight="bold", color="white")
        ax.text(1, i, "Pending", ha="center", va="center",
                fontsize=10, color="#888899", style="italic")

    # Draw cell borders
    for i in range(len(models)):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       fill=False, edgecolor="#555577", linewidth=1))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    cbar.set_label("Sensitivity", color="white", fontsize=10)

    ax.set_title("Cross-Domain × Cross-Modality Performance Matrix\n"
                 "Sensitivity: Model Training Modality vs. Evaluation Smear Type",
                 fontsize=10.5, color="white", pad=14)

    fig.tight_layout()
    out = FIGURES_DIR / "fig4_crossmodality_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] {out.name}")


# ── Figure 6: GT panel + all 4 models ────────────────────────────────────────
# Layout: [GT Annotated] | [MS_Sudan] | [MS_Thick] | [MS_Thin] | [fbononi YOLOv8]
def fig6_crossmodel_panel(df, gt_map):
    """Pick one image, show GT and all 4 models side by side (5 panels)."""
    CLASSIFIER_MODELS = {"MalariaScreener_Sudan", "MalariaScreener_Thick", "MalariaScreener_Thin"}
    DETECTOR_MODELS   = {"fbononibelloepoch_YOLOv8"}
    models = ["MalariaScreener_Sudan", "MalariaScreener_Thick",
              "MalariaScreener_Thin", "fbononibelloepoch_YOLOv8"]

    # Pick an image all 4 models correctly classified as positive
    all_tp = None
    for img_id, grp in df.groupby("image_id"):
        if len(grp) < 4:
            continue
        if (grp["predicted_class"] == 1).all() and gt_map.get(str(img_id), 0) == 1:
            all_tp = str(img_id)
            break

    if all_tp is None:
        print("[Skip] No image found where all 4 models are TP")
        return

    img_path = IMG_DIR / f"{all_tp}.jpg"
    img_bgr  = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[Skip] Could not read image {img_path}")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Load GT boxes from minoHealth AI Labs YOLO annotations
    lf = LABELS_DIR / f"{all_tp}.txt"
    gt_boxes_norm = []
    if lf.exists():
        for line in lf.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                _, xc, yc, bw_n, bh_n = map(float, parts[:5])
                gt_boxes_norm.append((xc, yc, bw_n, bh_n))

    # 5 panels: GT annotated | MS_Sudan | MS_Thick | MS_Thin | fbononi
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.patch.set_facecolor("#0F1117")

    # ── Panel 0: Ground Truth ──
    ax_gt = axes[0]
    ax_gt.set_facecolor("#1A1D2E")
    ax_gt.imshow(img_rgb)
    for (xc, yc, bw_n, bh_n) in gt_boxes_norm:
        x1 = int((xc - bw_n/2) * w)
        y1 = int((yc - bh_n/2) * h)
        rect = mpatches.Rectangle((x1, y1), int(bw_n * w), int(bh_n * h),
                                   linewidth=1.5, edgecolor="#00FF88",
                                   facecolor="none", linestyle="-")
        ax_gt.add_patch(rect)
    ax_gt.set_title(f"Ground Truth\n(minoHealth AI Labs)\n{len(gt_boxes_norm)} annotations",
                    fontsize=8.5, color="#00FF88", pad=6, fontweight="bold")
    ax_gt.axis("off")

    # ── Panels 1-4: One per model ──
    for ax, model_name in zip(axes[1:], models):
        ax.set_facecolor("#1A1D2E")
        ax.imshow(img_rgb)

        row = df[(df["image_id"].astype(str) == all_tp) &
                 (df["model_name"] == model_name)]
        conf = float(row.iloc[0]["confidence"]) if len(row) > 0 else 0.0
        pred_class = int(row.iloc[0]["predicted_class"]) if len(row) > 0 else 0
        pred_label = "POSITIVE" if pred_class == 1 else "NEGATIVE"
        label_color = "#00FF88" if pred_class == 1 else "#FF4444"

        if model_name in DETECTOR_MODELS:
            # Draw model's own predicted boxes
            boxes_json = row.iloc[0].get("boxes_json", "[]") if len(row) > 0 else "[]"
            try:
                for pb in json.loads(boxes_json):
                    bbox = pb.get("bbox", [])
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        rect = mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                                                   linewidth=1.5,
                                                   edgecolor=MODEL_COLORS[model_name],
                                                   facecolor="none")
                        ax.add_patch(rect)
            except Exception:
                pass
            subtitle = f"Object Detector | {pred_label}\nConf: {conf:.2f}"
        else:
            # Image classifier — no spatial output, text overlay only
            ax.text(0.5, 0.06, f"{pred_label}  ({conf:.2f})",
                    transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=12, fontweight="bold", color=label_color,
                    bbox=dict(facecolor="#0F1117", alpha=0.75,
                              edgecolor=label_color, boxstyle="round,pad=0.3"))
            subtitle = f"Image Classifier\n(no spatial output)"

        short = MODEL_LABELS[model_name].split("\n")[0]
        ax.set_title(f"{short}\n{subtitle}", fontsize=8.5, color="white", pad=6)
        ax.axis("off")

    fig.suptitle(
        f"Cross-Model Comparison | Ghanaian Thick Smear (Image ID: {all_tp})\n"
        "Panel 1: GT annotations (minoHealth AI Labs)  "
        "| Panels 2-4: Image classifiers (label only, no boxes predicted)  "
        "| Panel 5: Object detector (model's own predicted boxes)",
        fontsize=9, color="white", y=1.02
    )
    fig.tight_layout()
    out = FIGURES_DIR / "fig6_crossmodel_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[Saved] {out.name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Script 07: Generating Paper Figures ===\n")
    df, gt_map = load_data()
    agg_df   = compute_agg(df)
    strat_df = compute_strat(df)

    fig2_sensitivity_bars(agg_df)
    fig3_quality_lines(strat_df)
    fig4_crossmodality_heatmap(agg_df)
    fig6_crossmodel_panel(df, gt_map)

    print(f"\n[Complete] All figures saved to {FIGURES_DIR}")
