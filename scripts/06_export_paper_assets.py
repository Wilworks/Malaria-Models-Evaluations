"""
Script 06: Visual Asset & Paper Figure Exporter.
Generates per-model isolated 3-panel figures with Ground Truth Class and Model Prediction Class in titles.
"""

import sys
import cv2
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import LacunaGhanaDataset
from src.visualizer import MicrographVisualizer


def generate_paper_visuals():
    root_dir = Path(__file__).resolve().parent.parent
    figures_dir = root_dir / "manuscript" / "figures"
    panel_base_dir = figures_dir / "3panel_figures"
    tables_dir = root_dir / "manuscript" / "tables"
    
    figures_dir.mkdir(parents=True, exist_ok=True)
    panel_base_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    visualizer = MicrographVisualizer()

    print("=== Script 06: Generating & Saving Per-Model Paper Visuals ===")

    # 1. Plot Quality Metrics Distribution Figure
    quality_path = root_dir / "data" / "quality_metrics" / "quality_manifest.csv"
    if quality_path.exists():
        df_q = pd.read_csv(quality_path)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        sns.histplot(df_q['blur_laplacian'], kde=True, ax=axes[0], color='#2b5c8f')
        axes[0].set_title('Focus Blur (Laplacian Variance)', fontsize=11, fontweight='bold')
        axes[0].set_xlabel('Focus Sharpness', fontsize=10)
        
        sns.histplot(df_q['michelson_contrast'], kde=True, ax=axes[1], color='#8f2b2b')
        axes[1].set_title('Michelson Contrast', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Dynamic Range', fontsize=10)

        sns.histplot(df_q['fov_coverage_ratio'] * 100, kde=True, ax=axes[2], color='#2b8f5c')
        axes[2].set_title('Circular FOV Coverage (%)', fontsize=11, fontweight='bold')
        axes[2].set_xlabel('Microscope Field Area (%)', fontsize=10)

        plt.tight_layout()
        fig_path = figures_dir / "fig1_quality_distributions.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"[Success] Saved Figure 1 (Quality Distributions) -> {fig_path}")

    # 2. Render Authentic 3-Panel Figures Per Model with Titles & Quality Strata
    pred_path = root_dir / "data" / "processed" / "predictions_manifest.csv"
    quality_path = root_dir / "data" / "quality_metrics" / "quality_manifest.csv"

    if pred_path.exists():
        df_p = pd.read_csv(pred_path)
        
        # Merge quality metrics if available
        q_dict = {}
        if quality_path.exists():
            df_q = pd.read_csv(quality_path)
            df_q['quality_strata'] = pd.qcut(df_q['blur_laplacian'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
            for _, row in df_q.iterrows():
                q_dict[str(row['image_id'])] = {
                    'strata': str(row.get('quality_strata', 'N/A')),
                    'blur': float(row.get('blur_laplacian', 0.0)),
                    'contrast': float(row.get('michelson_contrast', 0.0))
                }

        unique_models = df_p['model_name'].unique()

        for model_name in unique_models:
            model_panel_dir = panel_base_dir / model_name
            model_panel_dir.mkdir(parents=True, exist_ok=True)

            sub_df = df_p[df_p['model_name'] == model_name]
            sample_ids = sub_df['image_id'].unique()[:5]

            print(f"Generating authentic 3-panel figures for model '{model_name}' ({len(sample_ids)} samples)...")
            for img_id in sample_ids:
                img_path = root_dir / "data" / "raw" / "thick_smear" / f"{img_id}.jpg"
                label_path = root_dir / "data" / "raw" / "thick_smear" / "labels_yolo" / f"{img_id}.txt"
                
                if not img_path.exists():
                    continue

                gt_boxes = []
                gt_class_str = "Uninfected / Negative"
                if label_path.exists():
                    with open(label_path, "r") as f:
                        lines = f.readlines()
                        if len(lines) > 0:
                            gt_class_str = "Parasite Positive (Infected)"
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id, xc, yc, bw, bh = map(float, parts[:5])
                                gt_boxes.append({"bbox": [xc, yc, bw, bh], "confidence": 1.0, "class_id": int(cls_id)})

                sub_img = sub_df[sub_df['image_id'] == img_id]
                pred_boxes = []
                pred_class_val = 0
                max_conf_val = 0.0
                if len(sub_img) > 0:
                    r = sub_img.iloc[0]
                    pred_class_val = r.get('predicted_class', 0)
                    max_conf_val = r.get('confidence', 0.0)
                    boxes_str = r.get("boxes_json", "[]")
                    if isinstance(boxes_str, str) and boxes_str.startswith("["):
                        try:
                            pred_boxes = json.loads(boxes_str)
                        except Exception:
                            pass

                pred_class_str = f"Infected (Conf: {max_conf_val:.2f})" if pred_class_val == 1 else f"Uninfected (Conf: {max_conf_val:.2f})"

                q_info = q_dict.get(str(img_id), {'strata': 'N/A', 'blur': 0.0, 'contrast': 0.0})

                out_fig_path = model_panel_dir / f"fig_3panel_{img_id}.png"
                visualizer.save_3panel_figure(
                    image_path=str(img_path),
                    pred_boxes=pred_boxes,
                    gt_boxes=gt_boxes,
                    out_path=str(out_fig_path),
                    image_id=str(img_id),
                    gt_class_name=gt_class_str,
                    pred_class_name=pred_class_str,
                    model_name=model_name,
                    quality_strata=q_info['strata'],
                    blur_val=q_info['blur'],
                    contrast_val=q_info['contrast']
                )

            print(f"[Success] Saved 3-panel visual figures for '{model_name}' -> {model_panel_dir}")

    print(f"\n[Complete] Per-model paper figures exported to {figures_dir}")


if __name__ == "__main__":
    generate_paper_visuals()
