#!/usr/bin/env python3
"""
Fine-Tune YOLOv8 Nano on West African Thick Blood Smears (WAM-Bench).
Uses Apple Silicon MPS GPU acceleration, checkpointing, and rigorous
held-out slide-level diagnostic evaluation (Sensitivity, Specificity, F1).
"""

import sys
import json
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT = Path("/Users/wilfredayineasumboya/Desktop/Projects/bcm-projects/Malaria-Models-Evaluations")
DATASET_YAML = ROOT / "data" / "splits" / "yolo_thick" / "dataset.yaml"
BASE_MODEL_PATH = ROOT / "models" / "external" / "fbononibelloepoch" / "best_yolo.pt"
OUTPUT_DIR = ROOT / "models" / "finetuned" / "yolo_thick"
RESULTS_DIR = ROOT / "results" / "finetuning"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on thick blood smears.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate (default: 0.001)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers (default: 4)")
    return parser.parse_args()


def select_device():
    if torch.backends.mps.is_available():
        print("[Device] Apple Silicon MPS GPU Acceleration enabled.")
        return "mps"
    elif torch.cuda.is_available():
        print("[Device] CUDA GPU Acceleration enabled.")
        return "cuda"
    else:
        print("[Device] Running on CPU.")
        return "cpu"


def evaluate_test_set(model, test_images_dir, conf_threshold=0.25):
    """
    Evaluates slide-level sensitivity and specificity on held-out physical slides.
    A slide is classified as POSITIVE if >= 1 bounding box detection >= conf_threshold.
    """
    print("\n" + "=" * 60)
    print("[Evaluation] Auditing Held-Out Test Set (Unaugmented Slides)...")
    print("=" * 60)

    test_imgs = sorted(list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.JPG")) + list(test_images_dir.glob("*.png")))
    if not test_imgs:
        print("[Warning] No test images found in", test_images_dir)
        return {}

    tp, fp, tn, fn = 0, 0, 0, 0
    test_labels_dir = test_images_dir.parent.parent / "labels" / "test"

    for img_path in test_imgs:
        lbl_path = test_labels_dir / f"{img_path.stem}.txt"
        has_gt = lbl_path.exists() and lbl_path.stat().st_size > 0
        gt_label = 1 if has_gt else 0

        # Run inference
        results = model.predict(source=str(img_path), conf=conf_threshold, verbose=False)
        pred_boxes = len(results[0].boxes) if results and len(results) > 0 else 0
        pred_label = 1 if pred_boxes > 0 else 0

        if gt_label == 1 and pred_label == 1:
            tp += 1
        elif gt_label == 0 and pred_label == 1:
            fp += 1
        elif gt_label == 0 and pred_label == 0:
            tn += 1
        elif gt_label == 1 and pred_label == 0:
            fn += 1

    total = tp + fp + tn + fn
    sensitivity = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)) * 100 if (tn + fp) > 0 else 0.0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    accuracy = ((tp + tn) / total) * 100 if total > 0 else 0.0
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

    metrics = {
        "total_test_samples": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity_pct": round(sensitivity, 2),
        "specificity_pct": round(specificity, 2),
        "precision_pct": round(precision, 2),
        "accuracy_pct": round(accuracy, 2),
        "f1_score": round(f1, 4),
        "conf_threshold": conf_threshold
    }

    print(f"\n[Test Set Results - Slide Level Classification]")
    print(f"  Total Test Micrographs: {total}")
    print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"  Sensitivity : {sensitivity:.2f}%")
    print(f"  Specificity : {specificity:.2f}%")
    print(f"  Accuracy    : {accuracy:.2f}%")
    print(f"  F1-Score    : {f1:.4f}")
    print("=" * 60)
    return metrics


def main():
    args = parse_args()
    device = select_device()

    if not DATASET_YAML.exists():
        sys.exit(f"[Error] Dataset specification not found at {DATASET_YAML}. Run scripts/create_finetune_dataset_splits.py first.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("WAM-Bench YOLOv8 Thick Smear Fine-Tuning Pipeline")
    print(f"  Base Checkpoint : {BASE_MODEL_PATH}")
    print(f"  Dataset Spec    : {DATASET_YAML}")
    print(f"  Device          : {device}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  Batch Size      : {args.batch}")
    print(f"  Image Size      : {args.imgsz}")
    print("=" * 60 + "\n")

    # Load pre-trained model
    model = YOLO(str(BASE_MODEL_PATH))

    # Fine-tune model
    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        patience=args.patience,
        device=device,
        workers=args.workers,
        project=str(OUTPUT_DIR),
        name="run_wam_thick",
        exist_ok=True,
        save=True,
        plots=True,
        verbose=True
    )

    # Load best checkpoint for test set evaluation
    best_weights = OUTPUT_DIR / "run_wam_thick" / "weights" / "best.pt"
    if best_weights.exists():
        print(f"\n[Success] Loading best fine-tuned weights from: {best_weights}")
        best_model = YOLO(str(best_weights))
    else:
        best_model = model

    test_imgs_dir = ROOT / "data" / "splits" / "yolo_thick" / "images" / "test"
    metrics = evaluate_test_set(best_model, test_imgs_dir)

    # Save metrics JSON
    metrics_path = RESULTS_DIR / "yolo_thick_finetune_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] Test evaluation metrics saved to: {metrics_path}\n")


if __name__ == "__main__":
    main()
