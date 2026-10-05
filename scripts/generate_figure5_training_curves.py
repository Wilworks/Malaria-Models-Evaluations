#!/usr/bin/env python3
"""
Generate Figure 5 for WAM-Bench Manuscript:
Training convergence dynamics, learning rate decay, and clinical diagnostic
recovery trajectories across frontline adaptation tiers (RQ5).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Setup paths
ROOT = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
RESULTS_DIR = ROOT / "results" / "finetuning"
FIGURES_DIR = ROOT / "paper" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9,
    "figure.titlesize": 13.5
})


def load_history(slug):
    csv_p = RESULTS_DIR / f"{slug}_history.csv"
    if csv_p.exists():
        return pd.read_csv(csv_p)
    json_p = RESULTS_DIR / f"{slug}_metrics.json"
    if json_p.exists():
        with open(json_p) as f:
            d = json.load(f)
            if "history" in d:
                return pd.DataFrame(d["history"])
    return None


def generate_figure5():
    df_t1a = load_history("mobilenet_thick_linear_probe")
    df_t1b = load_history("mobilenet_thick_linear_probe_d4")
    if df_t1a is None:
        sys.exit("[Error] Could not load Tier 1A history.")

    fig = plt.figure(figsize=(14.0, 8.5), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # ─────────────────────────────────────────────────────────────
    # PANEL A: Loss Minimization Dynamics (Tier 1A vs Tier 1B)
    # ─────────────────────────────────────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    epochs_a = df_t1a["epoch"]
    
    ax_a.plot(epochs_a, df_t1a["train_loss"], color="#1f77b4", lw=2.0, marker="o", markersize=3.0, label="Tier 1A Train Loss (Loss-Weighted)")
    ax_a.plot(epochs_a, df_t1a["val_loss"], color="#d62728", lw=1.8, linestyle="--", marker="s", markersize=3.0, label="Tier 1A Val Loss (Loss-Weighted)")

    if df_t1b is not None:
        epochs_b = df_t1b["epoch"]
        ax_a.plot(epochs_b, df_t1b["train_loss"], color="#7b1fa2", lw=2.2, marker="^", markersize=3.5, label="Tier 1B Train Loss (D4-Balanced)")
        ax_a.plot(epochs_b, df_t1b["val_loss"], color="#2e7d32", lw=1.8, linestyle="-.", marker="d", markersize=3.5, label="Tier 1B Val Loss (D4-Balanced)")

    ax_a.set_title("(a) Cross-Entropy Loss Convergence (Tier 1A vs. Tier 1B)", fontweight="bold")
    ax_a.set_xlabel("Epoch", fontweight="bold")
    ax_a.set_ylabel("Loss", fontweight="bold")
    ax_a.set_xlim(1, 30)
    ax_a.set_ylim(0.12, 0.70)
    ax_a.grid(True, linestyle=":", alpha=0.6)
    ax_a.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=8.5)

    # ─────────────────────────────────────────────────────────────
    # PANEL B: Per-Epoch Sensitivity & Specificity vs WHO Benchmark
    # ─────────────────────────────────────────────────────────────
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(epochs_a, df_t1a["val_sensitivity"], color="#2ca02c", lw=1.8, linestyle="--", label="Tier 1A Val Sensitivity")
    ax_b.plot(epochs_a, df_t1a["val_specificity"], color="#ff7f0e", lw=1.8, linestyle="--", label="Tier 1A Val Specificity")
    
    if df_t1b is not None:
        ax_b.plot(epochs_b, df_t1b["val_sensitivity"], color="#00796b", lw=2.2, marker="^", markersize=3.5, label="Tier 1B Val Sensitivity (D4)")
        ax_b.plot(epochs_b, df_t1b["val_specificity"], color="#e65100", lw=2.2, marker="v", markersize=3.5, label="Tier 1B Val Specificity (D4)")

    # WHO Level-1 Benchmark Target line
    ax_b.axhline(90.0, color="#d62728", lw=2.0, linestyle=":", label="WHO Level-1 Benchmark (90%)")
    ax_b.axhspan(90.0, 100.0, color="#2ca02c", alpha=0.10, label=r"WHO Triage Target Zone ($\geq 90$%)")

    ax_b.set_title("(b) Clinical Validation Metrics vs. WHO Level-1 Standard", fontweight="bold")
    ax_b.set_xlabel("Epoch", fontweight="bold")
    ax_b.set_ylabel("Diagnostic Score (%)", fontweight="bold")
    ax_b.set_xlim(1, 30)
    ax_b.set_ylim(50, 102)
    ax_b.grid(True, linestyle=":", alpha=0.6)
    ax_b.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=8.0)

    # ─────────────────────────────────────────────────────────────
    # PANEL C: 2D Clinical Recovery Trajectory (The Path to Clinical Viability)
    # ─────────────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, 0])
    
    # Shade WHO Level-1 Target Zone
    who_rect = Rectangle((90, 90), 14, 16, color="#2ca02c", alpha=0.15, ec="#2ca02c", lw=1.5, linestyle="--")
    ax_c.add_patch(who_rect)
    # Place WHO text cleanly at top right of the target zone so it never extends left towards Tier 1B
    ax_c.text(96.5, 104.5, "WHO Level-1 Target Zone\n(Sens & Spec $\\geq 90\\%$)", 
              color="#005500", fontsize=7.5, fontweight="bold", ha="center", va="top",
              bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#2ca02c", alpha=0.92, lw=0.7))

    # Zero-shot baseline points
    baselines = [
        ("MalariaScreener-Sudan", 55.69, 63.88, "#7f7f7f", "s", (8, -4)),
        ("MalariaScreener-Thick", 83.18, 30.29, "#bcbd22", "D", (8, -4)),
        ("fbononi-YOLOv8", 96.92, 47.13, "#8c564b", "P", (-98, -4))  # offset left to stay inside plot
    ]
    for name, spec, sens, col, marker, offset in baselines:
        ax_c.scatter(spec, sens, color=col, s=85, marker=marker, edgecolors="black", zorder=4)
        ax_c.annotate(name, (spec, sens), textcoords="offset points", xytext=offset,
                      fontsize=7.8, fontweight="bold", color=col)

    # Training trajectory for Tier 1A PELP
    traj_spec_a = df_t1a["val_specificity"].values
    traj_sens_a = df_t1a["val_sensitivity"].values
    ax_c.plot(traj_spec_a, traj_sens_a, color="#1f77b4", lw=1.8, linestyle="-", alpha=0.75, zorder=3, label=r"Tier 1A Trajectory (Loss-Weighted)")
    
    # Final test point Tier 1A
    ax_c.scatter(93.55, 92.15, color="#1f77b4", s=150, marker="*", edgecolors="black", zorder=6, label="Tier 1A Test (92.2% / 93.6%)")
    # Position Tier 1A label straight below with vertical arrow (clean empty space)
    ax_c.annotate("Tier 1A: Balanced Triage\n(Sens: 92.15%, Spec: 93.55%)",
                  xy=(93.55, 92.15), xytext=(93.55, 78.0),
                  arrowprops=dict(facecolor="#1f77b4", edgecolor="#1f77b4", shrink=0.08, width=1.0, headwidth=4),
                  ha="center", fontsize=7.8, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.25", fc="#f0f7ff", ec="#1f77b4", lw=0.8))

    if df_t1b is not None:
        traj_spec_b = df_t1b["val_specificity"].values
        traj_sens_b = df_t1b["val_sensitivity"].values
        ax_c.plot(traj_spec_b, traj_sens_b, color="#7b1fa2", lw=2.0, linestyle="-", alpha=0.85, zorder=3, label=r"Tier 1B Trajectory (D4-Balanced)")
        ax_c.scatter(83.53, 96.79, color="#7b1fa2", s=140, marker="D", edgecolors="black", zorder=6, label="Tier 1B Test (96.8% / 83.5%)")
        # Position Tier 1B label slightly above-left with short, crisp arrow
        ax_c.annotate("Tier 1B: High-Sens Screening\n(Sens: 96.79%, Spec: 83.53%)",
                      xy=(83.53, 96.79), xytext=(68.0, 102.5),
                      arrowprops=dict(facecolor="#7b1fa2", edgecolor="#7b1fa2", shrink=0.08, width=1.0, headwidth=4),
                      ha="center", fontsize=7.6, fontweight="bold",
                      bbox=dict(boxstyle="round,pad=0.22", fc="#faf0fa", ec="#7b1fa2", lw=0.8))

    ax_c.set_title("(c) 2D Diagnostic Recovery Trajectories in ROC Space", fontweight="bold")
    ax_c.set_xlabel("Specificity (%)", fontweight="bold")
    ax_c.set_ylabel("Sensitivity (%)", fontweight="bold")
    ax_c.set_xlim(38, 104)
    ax_c.set_ylim(18, 107)
    ax_c.grid(True, linestyle=":", alpha=0.6)
    ax_c.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=7.5)

    # ─────────────────────────────────────────────────────────────
    # PANEL D: Learning Rate Schedule & Parameter Efficiency Inset
    # ─────────────────────────────────────────────────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Plot Cosine Annealing Learning Rate clamped to minimum 1e-5 to avoid log(0) dive
    lr_clamped = np.maximum(df_t1a["lr"], 1e-5)
    ax_d.plot(epochs_a, lr_clamped, color="#17becf", lw=2.2, label=r"Cosine Annealing LR ($\eta_{\min} = 10^{-5}$)")
    ax_d.set_title("(d) Learning Rate Schedule & Adaptation Efficiency", fontweight="bold")
    ax_d.set_xlabel("Epoch", fontweight="bold")
    ax_d.set_ylabel("Learning Rate", fontweight="bold", color="#17becf")
    ax_d.set_yscale("log")
    ax_d.set_xlim(1, 30)
    ax_d.set_ylim(8e-6, 1.8e-3)
    ax_d.grid(True, linestyle=":", alpha=0.6)
    ax_d.tick_params(axis="y", labelcolor="#17becf")
    ax_d.legend(loc="upper right", frameon=True, framealpha=0.92)

    # Inset Bar Chart positioned cleanly in the bottom-left void (below learning rate curve)
    # At epochs 3-13, LR is > 7e-4 (top 20% of plot); the inset reaches up to only 7e-5.
    # Placed at y=0.22 so its tick labels ("PELP", "Tier 1") never overlap the main epoch x-axis.
    ax_inset = ax_d.inset_axes([0.10, 0.22, 0.35, 0.30])
    x_pos = [0, 1]
    param_counts = [0.0026, 2.22]
    bars = ax_inset.bar(x_pos, param_counts, color=["#2ca02c", "#7f7f7f"], edgecolor="black", lw=0.8, width=0.55)
    ax_inset.set_xticks(x_pos)
    ax_inset.set_xticklabels(["PELP\n(Tier 1)", "Full Retrain\n(Tier 2)"])
    ax_inset.set_ylabel("Params (M)", fontsize=7.5, fontweight="bold")
    ax_inset.set_title("Edge Adaptation Footprint", fontsize=8.0, fontweight="bold")
    ax_inset.set_yscale("log")
    ax_inset.set_ylim(0.001, 10.0)
    ax_inset.tick_params(axis="x", labelsize=7.0, pad=1)
    ax_inset.tick_params(axis="y", labelsize=7.0)
    ax_inset.grid(True, linestyle=":", alpha=0.5, axis="y")
    # Annotate bar values cleanly
    ax_inset.text(0, 0.004, "2.6k\n(0.1%)", ha="center", fontsize=6.8, fontweight="bold", color="#006600")
    ax_inset.text(1, 3.2, "2.22M\n(100%)", ha="center", fontsize=6.8, fontweight="bold")

    # Overall Figure Suptitle
    fig.suptitle("Figure 5: Frontline Regional Adaptation Optimization Dynamics, Epoch Convergence, and 2D Clinical Triage Trajectories (RQ5)",
                 fontsize=13.5, fontweight="bold", y=0.99)

    out_png = FIGURES_DIR / "fig5_training_curves.png"
    out_pdf = FIGURES_DIR / "fig5_training_curves.pdf"
    
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[Success] Generated Figure 5:\n  PNG: {out_png}\n  PDF: {out_pdf}")


if __name__ == "__main__":
    generate_figure5()
