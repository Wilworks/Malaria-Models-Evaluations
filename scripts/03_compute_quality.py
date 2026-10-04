"""
Pre-computes optical quality metrics (blur, contrast, SNR, circular FOV coverage)
for all images across all discovered West African sources and exports metadata manifest
to data/quality_metrics/quality_manifest.csv.
"""

import sys
import json
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import MultiSourceMalariaDataset
from src.quality_assessment import ImageQualityAssessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_quality_strata(data_root: Path = None):
    data_root = data_root or (ROOT / "data" / "raw")
    out_dir = ROOT / "data" / "quality_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "quality_manifest.csv"
    out_json = out_dir / "quality_manifest.json"

    results = []
    existing_keys = set()
    if out_csv.exists():
        existing_df = pd.read_csv(out_csv)
        results = existing_df.to_dict(orient="records")
        for r in results:
            existing_keys.add((str(r.get("data_source", "")), str(r["image_id"]), str(r["smear_type"])))
        logger.info("Loaded %d existing quality records from %s", len(results), out_csv.name)

    assessor = ImageQualityAssessor()
    logger.info("=== Computing Image Quality Metrics (Circular FOV Masking) ===")

    new_count = 0
    for smear in ["thick", "thin"]:
        ds = MultiSourceMalariaDataset(str(data_root), smear_type=smear, shuffle=False)
        pending = [
            item for item in ds.annotations
            if (str(item.get("data_source", "")), str(item["image_id"]), smear) not in existing_keys
        ]
        logger.info("Found %d unassessed %s smear images (out of %d total across sources).",
                    len(pending), smear, len(ds))

        for item in tqdm(pending, desc=f"{smear.capitalize()} Smears"):
            img_path = item["image_path"]
            try:
                metrics = assessor.assess_image(img_path)
                metrics.update({
                    "image_id": item["image_id"],
                    "data_source": item.get("data_source", "lacuna_ghana"),
                    "smear_type": smear,
                    "image_path": img_path
                })
                results.append(metrics)
                new_count += 1
            except Exception as e:
                logger.warning("Could not assess quality for %s: %s", img_path, e)

    df = pd.DataFrame(results)
    if not df.empty:
        df.to_csv(out_csv, index=False)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    logger.info("[Success] Computed quality for %d new micrographs. Total in manifest: %d", new_count, len(df))
    logger.info("Saved manifest to: %s", out_csv)


if __name__ == "__main__":
    compute_quality_strata()
