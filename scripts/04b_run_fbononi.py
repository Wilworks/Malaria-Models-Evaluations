"""
Script 04b: Zero-Shot Evaluation for fbononibelloepoch/malaria-detection YOLOv8.
Runs inference on Lacuna Ghana thick smear dataset and computes sensitivity metrics.
Model: YOLOv8 object detection, classes: {0: Trophozoite, 1: WBC}
Source: https://huggingface.co/fbononibelloepoch/malaria-detection
"""

import sys
import cv2
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import LacunaGhanaDataset
from models.wrappers.yolo_wrapper import YOLOMalariaWrapper


def run_fbononi_eval():
    root = Path(__file__).resolve().parent.parent
    processed_dir = root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    weights = root / "models" / "external" / "fbononibelloepoch" / "best_yolo.pt"
    if not weights.exists():
        print(f"[Error] Weights not found at {weights}. Run download step first.")
        return

    print("=== Script 04b: Evaluating fbononibelloepoch/malaria-detection YOLOv8 ===")
    print(f"Weights: {weights} ({weights.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"Provenance: HuggingFace fbononibelloepoch/malaria-detection")
    print(f"Model classes: Trophozoite (0), WBC (1)")
    print(f"Task: Object Detection — binary positive if any Trophozoite box predicted")
    print()

    wrapper = YOLOMalariaWrapper(
        model_name="fbononibelloepoch_YOLOv8",
        model_path=str(weights),
        confidence_threshold=0.15
    )

    ds = LacunaGhanaDataset(str(root / "data" / "raw"), smear_type="thick")
    items = ds.annotations
    print(f"Dataset: {len(items)} thick smear images (Lacuna Ghana, Princess Marie Louise Hospital)")
    print()

    results = []
    for item in tqdm(items, desc="Evaluating fbononibelloepoch_YOLOv8"):
        img = cv2.imread(item["image_path"])
        if img is None:
            continue
        try:
            pred = wrapper.predict(img)
            results.append({
                "image_id": item["image_id"],
                "smear_type": "thick",
                "model_name": wrapper.model_name,
                "predicted_class": pred.get("predicted_class", 0),
                "confidence": pred.get("confidence", 0.0),
                "detected_objects": pred.get("detected_objects", 0),
                "boxes_json": json.dumps(pred.get("boxes", [])),
                "image_path": item["image_path"]
            })
        except Exception as e:
            print(f"[Warning] Inference error on {item['image_id']}: {e}")

    df = pd.DataFrame(results)
    out_path = processed_dir / "predictions_fbononibelloepoch_YOLOv8.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[Success] Saved {len(df)} predictions -> {out_path}")

    # Ground-truth labels from YOLO annotations
    labels_dir = root / "data" / "raw" / "thick_smear" / "labels_yolo"
    gt_map = {}
    for img_path in (root / "data" / "raw" / "thick_smear").iterdir():
        if img_path.suffix in (".jpg", ".png", ".jpeg"):
            lf = labels_dir / (img_path.stem + ".txt")
            gt_map[img_path.stem] = 1 if (lf.exists() and lf.read_text().strip()) else 0

    df["gt"] = df["image_id"].astype(str).map(gt_map).fillna(0).astype(int)

    TP = int(((df["predicted_class"] == 1) & (df["gt"] == 1)).sum())
    FP = int(((df["predicted_class"] == 1) & (df["gt"] == 0)).sum())
    FN = int(((df["predicted_class"] == 0) & (df["gt"] == 1)).sum())
    TN = int(((df["predicted_class"] == 0) & (df["gt"] == 0)).sum())
    n = len(df)
    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
    acc = (TP + TN) / n
    mean_conf = float(df["confidence"].mean())
    pos_rate = int((df["predicted_class"] == 1).sum())

    print()
    print("=" * 60)
    print("  fbononibelloepoch_YOLOv8 — Zero-Shot Results Summary")
    print("=" * 60)
    print(f"  n                 : {n}")
    print(f"  TP (detected +ve) : {TP}")
    print(f"  FN (missed +ve)   : {FN}")
    print(f"  FP                : {FP}")
    print(f"  TN                : {TN}")
    print(f"  Sensitivity       : {sens:.3f} ({sens*100:.1f}%)")
    print(f"  F1-Score          : {f1:.3f}")
    print(f"  Accuracy          : {acc:.3f} ({acc*100:.1f}%)")
    print(f"  Mean Confidence   : {mean_conf:.3f} ({mean_conf*100:.1f}%)")
    print(f"  Positive rate     : {pos_rate}/{n}")
    print("=" * 60)

    # Append to master manifest
    master_path = processed_dir / "predictions_manifest.csv"
    if master_path.exists():
        master = pd.read_csv(master_path)
        # Remove any stale fbononi rows if re-running
        master = master[master["model_name"] != "fbononibelloepoch_YOLOv8"]
        combined = pd.concat([master, df[["image_id", "smear_type", "model_name",
                                          "predicted_class", "confidence",
                                          "detected_objects", "boxes_json", "image_path"]]],
                              ignore_index=True)
        combined.to_csv(master_path, index=False)
        print(f"\n[Success] Appended to master manifest -> {master_path}")
        print(f"          Total rows now: {len(combined)}")


if __name__ == "__main__":
    run_fbononi_eval()
