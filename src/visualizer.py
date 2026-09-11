"""
Visual Analysis Module.
Renders authentic 3-panel figures with explicit Ground Truth Class and Model Prediction Class in titles:
Panel A: Raw Smartphone Micrograph (Image ID & Ground Truth Label)
Panel B: Ground-Truth Bounding Boxes (GT Class & Annotations Count)
Panel C: Model Zero-Shot Predictions (Model Predicted Class & Confidence)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class MicrographVisualizer:
    """Generates authentic 3-panel figures for manuscript publication."""

    @staticmethod
    def draw_yolo_boxes(image: np.ndarray, boxes: List[Dict], color: Tuple[int, int, int] = (0, 255, 0), label_prefix: str = "Pred") -> np.ndarray:
        """
        Draws bounding boxes and confidence score badges on an image array.
        """
        img_copy = image.copy()
        h, w = img_copy.shape[:2]

        for box in boxes:
            bbox = box.get("bbox", [])
            conf = box.get("confidence", 0.0)
            cls_id = box.get("class_id", 0)

            if len(bbox) == 4:
                if all(0.0 <= c <= 1.0 for c in bbox):
                    xc, yc, bw, bh = bbox
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                else:
                    x1, y1, x2, y2 = map(int, bbox)

                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 3)
                text = f"{label_prefix} Class {cls_id}: {conf:.2f}"
                t_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(img_copy, (x1, y1 - t_size[1] - 8), (x1 + t_size[0] + 6, y1), color, -1)
                cv2.putText(img_copy, text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        return img_copy

    def save_3panel_figure(
        self, 
        image_path: str, 
        pred_boxes: List[Dict], 
        gt_boxes: List[Dict], 
        out_path: str, 
        image_id: str,
        gt_class_name: str = "Parasite Positive",
        pred_class_name: str = "Uninfected (0.00)",
        model_name: str = "Model",
        quality_strata: str = "N/A",
        blur_val: float = 0.0,
        contrast_val: float = 0.0
    ):
        """
        Generates an authentic 3-panel publication figure with explicit class labels & quality metrics in titles:
        Panel A: Raw Smartphone Micrograph (with Image ID, GT Label, & Quality Strata)
        Panel B: Ground-Truth Bounding Boxes (with Annotation Count & GT Class)
        Panel C: Model Zero-Shot Predictions (with Model Name, Object Count, & Predicted Class)
        """
        img = cv2.imread(image_path)
        if img is None:
            return

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_gt = self.draw_yolo_boxes(img, gt_boxes, color=(255, 0, 0), label_prefix="GT")
        img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)

        img_pred = self.draw_yolo_boxes(img, pred_boxes, color=(0, 255, 0), label_prefix="Pred")
        img_pred_rgb = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Quality annotation line
        quality_info = f"Quality: {quality_strata.upper()} (Blur: {blur_val:.1f}, Contrast: {contrast_val:.2f})" if quality_strata != "N/A" else ""

        # Panel A Title
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"(A) Raw Micrograph (Image #{image_id})\nGT Label: {gt_class_name}\n{quality_info}", fontsize=11, fontweight='bold')
        axes[0].axis('off')

        # Panel B Title
        axes[1].imshow(img_gt_rgb)
        axes[1].set_title(f"(B) Ground-Truth BBoxes ({len(gt_boxes)} objects)\nClass: {gt_class_name}", fontsize=11, fontweight='bold')
        axes[1].axis('off')

        # Panel C Title
        axes[2].imshow(img_pred_rgb)
        if len(pred_boxes) > 0:
            panel_c_title = f"(C) {model_name} Object Detections ({len(pred_boxes)} objects)\nPred Class: {pred_class_name}"
        else:
            panel_c_title = f"(C) {model_name} Image Classification\nPred Class: {pred_class_name}"
            
        axes[2].set_title(panel_c_title, fontsize=11, fontweight='bold')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Success] Saved 3-panel visual figure with titles -> {out_path}")
