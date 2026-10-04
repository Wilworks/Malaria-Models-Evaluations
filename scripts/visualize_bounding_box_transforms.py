#!/usr/bin/env python3
"""
Visual Validation of Bounding Box Transformations Under Dihedral Group D4 & Affine Augmentation.

Corrected D4 group mappings:
  - Identity (I): (xc, yc, w, h)
  - Horizontal Flip (Fx): (1-xc, yc, w, h)
  - Vertical Flip (Fy): (xc, 1-yc, w, h)
  - 90 deg Clockwise (R90): (1-yc, xc, h, w)
  - 180 deg (R180): (1-xc, 1-yc, w, h)
  - 270 deg Clockwise (R270): (yc, 1-xc, h, w)
  - Main Diagonal (Transpose, D_main): (yc, xc, h, w)
  - Anti-Diagonal (D_anti): (1-yc, 1-xc, h, w)
  - Continuous Affine: Vertex affine multiplication via matrix M
"""

import sys
import math
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Color palette for classes (Okabe-Ito inspired)
CLASS_COLORS = {
    0: "#E69F00",  # Orange
    1: "#56B4E9",  # Sky Blue
    2: "#009E73",  # Bluish Green
    3: "#F0E442",  # Yellow
    4: "#0072B2",  # Blue
    5: "#D55E00",  # Vermillion
}

CLASS_NAMES_DEFAULT = {
    0: "Gametocyte",
    1: "Trophozoite",
    2: "Other",
    3: "WBC",
    4: "Artefact",
    5: "Ring Stage",
}


def load_yolo_labels(label_path: Path):
    """Loads YOLO boxes: List of (class_id, xc, yc, w, h)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
                boxes.append((cls_id, xc, yc, w, h))
    return boxes


def transform_box_d4(box, op_name):
    """
    Transforms a normalized YOLO box (cls_id, xc, yc, w, h) under D4 operations.
    Returns: (cls_id, xc_new, yc_new, w_new, h_new)
    """
    cls_id, xc, yc, w, h = box
    if op_name == "identity":
        return (cls_id, xc, yc, w, h)
    elif op_name == "hflip":
        return (cls_id, 1.0 - xc, yc, w, h)
    elif op_name == "vflip":
        return (cls_id, xc, 1.0 - yc, w, h)
    elif op_name == "rot90":
        # Clockwise 90: (x, y) -> (1-y, x), w and h swap
        return (cls_id, 1.0 - yc, xc, h, w)
    elif op_name == "rot180":
        return (cls_id, 1.0 - xc, 1.0 - yc, w, h)
    elif op_name == "rot270":
        # Clockwise 270: (x, y) -> (y, 1-x), w and h swap
        return (cls_id, yc, 1.0 - xc, h, w)
    elif op_name == "diag_main":
        # Main Diagonal Reflection (cv2.transpose): (x, y) -> (y, x), w and h swap
        return (cls_id, yc, xc, h, w)
    elif op_name == "diag_anti":
        # Anti-Diagonal Reflection (transpose + rotate 180): (x, y) -> (1-y, 1-x), w and h swap
        return (cls_id, 1.0 - yc, 1.0 - xc, h, w)
    else:
        raise ValueError(f"Unknown D4 operation: {op_name}")


def transform_image_d4(img, op_name):
    """Transforms an image under D4 operations."""
    if op_name == "identity":
        return img.copy()
    elif op_name == "hflip":
        return cv2.flip(img, 1)
    elif op_name == "vflip":
        return cv2.flip(img, 0)
    elif op_name == "rot90":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif op_name == "rot180":
        return cv2.rotate(img, cv2.ROTATE_180)
    elif op_name == "rot270":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif op_name == "diag_main":
        # Transpose (reflection across main diagonal y = x)
        return cv2.transpose(img)
    elif op_name == "diag_anti":
        # Transpose then rotate 180 (reflection across anti-diagonal y = 1 - x)
        t = cv2.transpose(img)
        return cv2.rotate(t, cv2.ROTATE_180)
    else:
        raise ValueError(f"Unknown D4 operation: {op_name}")


def transform_affine(img, boxes, angle_deg=20.0, scale=1.05, tx_px=15, ty_px=-10):
    """
    Applies continuous affine transform to image and bounding boxes using vertex mapping.
    """
    h_img, w_img = img.shape[:2]
    center = (w_img / 2.0, h_img / 2.0)
    
    # 2x3 affine matrix
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx_px
    M[1, 2] += ty_px
    
    transformed_img = cv2.warpAffine(img, M, (w_img, h_img), borderMode=cv2.BORDER_REFLECT)
    
    transformed_boxes = []
    for cls_id, xc, yc, w, h in boxes:
        # Convert to pixel corners
        xmin = (xc - w / 2.0) * w_img
        xmax = (xc + w / 2.0) * w_img
        ymin = (yc - h / 2.0) * h_img
        ymax = (yc + h / 2.0) * h_img
        
        corners = np.array([
            [xmin, ymin, 1.0],
            [xmax, ymin, 1.0],
            [xmax, ymax, 1.0],
            [xmin, ymax, 1.0]
        ]).T  # shape (3, 4)
        
        # Multiply by M (2x3)
        new_corners = M @ corners  # shape (2, 4)
        new_x = new_corners[0, :]
        new_y = new_corners[1, :]
        
        n_xmin = float(np.min(new_x))
        n_xmax = float(np.max(new_x))
        n_ymin = float(np.min(new_y))
        n_ymax = float(np.max(new_y))
        
        # Clip to image boundaries
        c_xmin = max(0.0, min(float(w_img), n_xmin))
        c_xmax = max(0.0, min(float(w_img), n_xmax))
        c_ymin = max(0.0, min(float(h_img), n_ymin))
        c_ymax = max(0.0, min(float(h_img), n_ymax))
        
        box_w = c_xmax - c_xmin
        box_h = c_ymax - c_ymin
        
        if box_w > 6 and box_h > 6:
            new_xc = (c_xmin + c_xmax) / (2.0 * w_img)
            new_yc = (c_ymin + c_ymax) / (2.0 * h_img)
            new_w = box_w / float(w_img)
            new_h = box_h / float(h_img)
            transformed_boxes.append((cls_id, new_xc, new_yc, new_w, new_h))
            
    return transformed_img, transformed_boxes


def draw_panel(ax, img, boxes, title_str, subtitle_str):
    """Renders an image panel with overlay bounding boxes."""
    h_img, w_img = img.shape[:2]
    # Convert BGR to RGB for matplotlib
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(rgb)
    
    for cls_id, xc, yc, w, h in boxes:
        xmin = (xc - w / 2.0) * w_img
        ymin = (yc - h / 2.0) * h_img
        bw = w * w_img
        bh = h * h_img
        
        col = CLASS_COLORS.get(cls_id, "#FF0000")
        name = CLASS_NAMES_DEFAULT.get(cls_id, f"Cls_{cls_id}")
        
        rect = patches.Rectangle(
            (xmin, ymin), bw, bh,
            linewidth=2.5, edgecolor=col, facecolor="none", linestyle="-"
        )
        ax.add_patch(rect)
        
        # Center marker crosshair
        cx = xc * w_img
        cy = yc * h_img
        ax.plot(cx, cy, marker="+", markersize=9, color=col, markeredgewidth=2.2)
        
        # Label text background
        ax.text(
            xmin, max(0, ymin - 4),
            f"{name}",
            color="black", fontsize=8.5, weight="bold",
            bbox=dict(boxstyle="square,pad=0.2", facecolor=col, edgecolor="none", alpha=0.92)
        )
        
    ax.set_title(title_str, fontsize=11, weight="bold", pad=5)
    ax.set_xlabel(subtitle_str, fontsize=9, style="italic")
    ax.set_xticks([])
    ax.set_yticks([])


def generate_validation_figure(img_path: Path, label_path: Path, out_path: Path, cohort_title: str):
    """Generates a 3x3 multi-panel figure demonstrating all transformations."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Error: Unable to load {img_path}")
        return
        
    boxes = load_yolo_labels(label_path)
    print(f"[{cohort_title}] Loaded {len(boxes)} boxes from {label_path.name}")
    
    ops = [
        ("identity", "1. Original View (Identity I)", f"xc'=xc, yc'=yc ({len(boxes)} boxes)"),
        ("hflip", "2. Horizontal Flip (Fx)", "xc'=1-xc, yc'=yc"),
        ("vflip", "3. Vertical Flip (Fy)", "xc'=xc, yc'=1-yc"),
        ("rot90", "4. 90 deg Clockwise (R90)", "xc'=1-yc, yc'=xc, w'=h, h'=w"),
        ("rot180", "5. 180 deg Rotation (R180)", "xc'=1-xc, yc'=1-yc"),
        ("rot270", "6. 270 deg Clockwise (R270)", "xc'=yc, yc'=1-xc, w'=h, h'=w"),
        ("diag_main", "7. Main Diagonal (Transpose)", "xc'=yc, yc'=xc, w'=h, h'=w"),
        ("diag_anti", "8. Anti-Diagonal Reflection", "xc'=1-yc, yc'=1-xc, w'=h, h'=w"),
    ]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, (op_name, title, sub) in enumerate(ops):
        t_img = transform_image_d4(img, op_name)
        t_boxes = [transform_box_d4(b, op_name) for b in boxes]
        draw_panel(axes[i], t_img, t_boxes, title, sub)
        
    # 9th panel: General Affine transformation (rotation + scale + translation)
    aff_img, aff_boxes = transform_affine(img, boxes, angle_deg=-6.0, scale=0.92, tx_px=20, ty_px=-35)
    draw_panel(
        axes[8], aff_img, aff_boxes,
        "9. Continuous Affine Transform",
        f"M: theta=-6 deg, scale=0.92x, tx=+20, ty=-35 ({len(aff_boxes)}/{len(boxes)} boxes)"
    )
    
    fig.suptitle(
        f"Geometric Coordinate Invariance & Bounding Box Preservation\n{cohort_title} (Slide: {img_path.name})",
        fontsize=14, weight="bold", y=0.99
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved validation figure to: {out_path}")


def main():
    root = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
    
    # 1. Ghana Thin Smear Sample
    ghana_img = root / "data" / "raw" / "lacuna_ghana" / "thin_smear" / "positive" / "1014.jpg"
    ghana_lbl = root / "data" / "raw" / "lacuna_ghana" / "thin_smear" / "labels_yolo" / "1014.txt"
    ghana_out = root / "results" / "figures" / "fig_s1_bounding_box_transforms_ghana.png"
    
    if ghana_img.exists() and ghana_lbl.exists():
        generate_validation_figure(ghana_img, ghana_lbl, ghana_out, "Princess Marie Louise Hospital, Ghana (Thin Smear)")
        
    # 2. Adeleke Nigeria Thick Smear Sample
    adeleke_img = root / "data" / "raw" / "adeleke_nigeria" / "thick_smear" / "positive" / "Pos_Iwo_42.jpg"
    adeleke_lbl = root / "data" / "raw" / "adeleke_nigeria" / "thick_smear" / "labels_yolo" / "Pos_Iwo_42.txt"
    adeleke_out = root / "results" / "figures" / "fig_s2_bounding_box_transforms_adeleke.png"
    
    if adeleke_img.exists() and adeleke_lbl.exists():
        generate_validation_figure(adeleke_img, adeleke_lbl, adeleke_out, "Osun State Health Centers, Nigeria (Thick Smear)")

if __name__ == "__main__":
    main()
