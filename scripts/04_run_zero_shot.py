"""
Script 04: Zero-Shot Benchmark Execution Engine.
Evaluates all candidate models without retraining against the Lacuna Ghana dataset.

Supports:
  - Multiple smear types via --smear-type flag (thick | thin | both)
  - Incremental manifest updates (does NOT overwrite existing results)
  - All 4 models: MalariaScreener_Thick, MalariaScreener_Thin,
                  MalariaScreener_Sudan, fbononibelloepoch_YOLOv8

Usage:
  # Run on thick smears only (default)
  python scripts/04_run_zero_shot.py

  # Run on thin smears only
  python scripts/04_run_zero_shot.py --smear-type thin

  # Run on both thick and thin
  python scripts/04_run_zero_shot.py --smear-type both

  # Limit samples for testing
  python scripts/04_run_zero_shot.py --limit-samples 20
"""

import sys
import cv2
import json
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import LacunaGhanaDataset
from models.wrappers.malariascreener_wrapper import MalariaScreenerWrapper
from models.wrappers.yolo_wrapper import YOLOMalariaWrapper


def build_model_list(root_dir: Path) -> list:
    """Instantiates all available model wrappers from confirmed weight paths."""
    models = []

    ms_thick_path = root_dir / "models" / "external" / "MalariaScreener" / "assets" / "malaria_thick_model.tflite"
    ms_thin_path  = root_dir / "models" / "external" / "MalariaScreener" / "assets" / "malaria_thin_model.tflite"
    ms_sudan_path = root_dir / "models" / "external" / "MalariaScreener" / "assets" / "malaria_sudan_model.pb"
    fbononi_path  = root_dir / "models" / "external" / "fbononibelloepoch" / "best_yolo.pt"

    if ms_thick_path.exists():
        models.append(MalariaScreenerWrapper(str(ms_thick_path), smear_type="thick"))
    else:
        print(f"[Warning] MalariaScreener_Thick not found at {ms_thick_path}")

    if ms_thin_path.exists():
        models.append(MalariaScreenerWrapper(str(ms_thin_path), smear_type="thin"))
    else:
        print(f"[Warning] MalariaScreener_Thin not found at {ms_thin_path}")

    if ms_sudan_path.exists():
        models.append(MalariaScreenerWrapper(str(ms_sudan_path), smear_type="sudan"))
    else:
        print(f"[Warning] MalariaScreener_Sudan not found at {ms_sudan_path}")

    if fbononi_path.exists():
        models.append(YOLOMalariaWrapper(
            model_name="fbononibelloepoch_YOLOv8",
            model_path=str(fbononi_path),
            confidence_threshold=0.15
        ))
    else:
        print(f"[Warning] fbononibelloepoch_YOLOv8 not found at {fbononi_path}")

    return models


def run_zero_shot_eval(smear_types: list, pilot_fraction: float = 1.0, limit_samples: int = None):
    root_dir = Path(__file__).resolve().parent.parent
    data_raw = root_dir / "data" / "raw"
    processed_dir = root_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=== Script 04: Zero-Shot Evaluation Pipeline ===")
    print(f"Smear types: {smear_types}")

    models = build_model_list(root_dir)
    if not models:
        print("[Error] No models loaded. Check weight paths.")
        return

    print(f"Models loaded: {[m.model_name for m in models]}")

    # Load existing manifest to enable incremental appending
    master_path = processed_dir / "predictions_manifest.csv"
    if master_path.exists():
        master_df = pd.read_csv(master_path)
        print(f"Existing manifest: {len(master_df)} rows. Will append new results.")
    else:
        master_df = pd.DataFrame()

    new_results = []

    for smear_type in smear_types:
        img_dir = data_raw / f"{smear_type}_smear"
        if not img_dir.exists() or not any(img_dir.glob("*.jpg")):
            print(f"\n[Skip] No images found for smear_type='{smear_type}' at {img_dir}")
            continue

        ds = LacunaGhanaDataset(str(data_raw), smear_type=smear_type)
        items = ds.annotations

        if limit_samples:
            items = items[:limit_samples]
        elif pilot_fraction < 1.0:
            items = items[:max(1, int(len(items) * pilot_fraction))]

        print(f"\n--- Smear type: {smear_type.upper()} | {len(items)} images ---")

        for model_wrapper in models:
            # Skip if this model+smear_type combo already fully evaluated
            if not master_df.empty:
                already = master_df[
                    (master_df["model_name"] == model_wrapper.model_name) &
                    (master_df["smear_type"] == smear_type)
                ]
                evaluated_ids = set(already["image_id"].astype(str))
                remaining = [i for i in items if str(i["image_id"]) not in evaluated_ids]
                if not remaining:
                    print(f"[Skip] {model_wrapper.model_name} on {smear_type}: already fully evaluated ({len(already)} rows)")
                    continue
                print(f"\n  Model: {model_wrapper.model_name} | {len(remaining)} new images to evaluate")
                items_to_run = remaining
            else:
                print(f"\n  Model: {model_wrapper.model_name} | {len(items)} images")
                items_to_run = items

            model_results = []
            for item in tqdm(items_to_run, desc=f"  {model_wrapper.model_name}"):
                img = cv2.imread(item["image_path"])
                if img is None:
                    continue
                try:
                    pred = model_wrapper.predict(img)
                    record = {
                        "image_id":         item["image_id"],
                        "smear_type":       smear_type,
                        "model_name":       model_wrapper.model_name,
                        "predicted_class":  pred.get("predicted_class", 0),
                        "confidence":       pred.get("confidence", 0.0),
                        "detected_objects": pred.get("detected_objects", 0),
                        "boxes_json":       json.dumps(pred.get("boxes", [])),
                        "image_path":       item["image_path"]
                    }
                    model_results.append(record)
                    new_results.append(record)
                except Exception as e:
                    print(f"[Warning] {model_wrapper.model_name} on {item['image_id']}: {e}")

            if model_results:
                per_model_file = processed_dir / f"predictions_{model_wrapper.model_name}_{smear_type}.csv"
                pd.DataFrame(model_results).to_csv(per_model_file, index=False)
                print(f"  [Saved] {per_model_file.name} ({len(model_results)} rows)")

    # Append new results to master manifest
    if new_results:
        combined = pd.concat([master_df, pd.DataFrame(new_results)], ignore_index=True)
        combined.to_csv(master_path, index=False)
        print(f"\n[Success] Master manifest updated -> {master_path}")
        print(f"          Total rows: {len(combined)} (+{len(new_results)} new)")
    else:
        print("\n[Info] No new results to append — manifest unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Shot Malaria Benchmark Evaluator")
    parser.add_argument("--smear-type", type=str, default="thick",
                        choices=["thick", "thin", "both"],
                        help="Smear type to evaluate (default: thick)")
    parser.add_argument("--pilot-fraction", type=float, default=1.0,
                        help="Fraction of dataset to evaluate (default: 1.0 = all)")
    parser.add_argument("--limit-samples", type=int, default=None,
                        help="Max number of samples per smear type (for quick tests)")
    args = parser.parse_args()

    smear_types = ["thick", "thin"] if args.smear_type == "both" else [args.smear_type]
    run_zero_shot_eval(
        smear_types=smear_types,
        pilot_fraction=args.pilot_fraction,
        limit_samples=args.limit_samples
    )
