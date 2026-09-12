"""
Script 03: Pre-computes image quality proxy metrics (blur, contrast, SNR) for all dataset images
and saves metadata manifest to data/quality_metrics/quality_manifest.json.
"""

import sys
import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import LacunaGhanaDataset
from src.quality_assessment import ImageQualityAssessor


def compute_quality_strata():
    root_dir = Path(__file__).resolve().parent.parent
    data_raw = root_dir / "data" / "raw"
    out_dir = root_dir / "data" / "quality_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "quality_manifest.csv"
    out_json = out_dir / "quality_manifest.json"

    results = []
    existing_keys = set()
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict(orient="records")
        for r in results:
            existing_keys.add((str(r["image_id"]), str(r["smear_type"])))
        print(f"Loaded {len(results)} existing quality records from {out_csv.name}.")

    assessor = ImageQualityAssessor()
    print("=== Script 03: Computing Image Quality Strata (Incremental) ===")

    new_count = 0
    for smear in ["thick", "thin"]:
        ds = LacunaGhanaDataset(str(data_raw), smear_type=smear)
        pending = [item for item in ds.annotations if (str(item["image_id"]), smear) not in existing_keys]
        print(f"Found {len(pending)} unassessed {smear} smear images (out of {len(ds)} total).")
        
        for item in tqdm(pending, desc=f"{smear.capitalize()} Smears"):
            img_path = item["image_path"]
            try:
                metrics = assessor.assess_image(img_path)
                metrics.update({
                    "image_id": item["image_id"],
                    "smear_type": smear,
                    "image_path": img_path
                })
                results.append(metrics)
                new_count += 1
            except Exception as e:
                print(f"[Warning] Could not process {img_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Success] Processed {new_count} new images. Total quality metrics in manifest: {len(df)}")
    print(f"Saved to:\n  - {out_csv}\n  - {out_json}")


if __name__ == "__main__":
    compute_quality_strata()

