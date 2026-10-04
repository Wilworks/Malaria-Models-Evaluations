#!/usr/bin/env python3
"""
Extracts and standardizes the Muhammad et al. (Kano, Nigeria) thin blood smear dataset
from ~/Downloads/Thin_blood_smear_images.zip into WAM-Bench standard directory structure:
  data/raw/muhammad_kano_nigeria/thin_smear/positive/  (772 micrographs, acute infection / rouleaux)
  data/raw/muhammad_kano_nigeria/thin_smear/negative/  (772 micrographs, healthy control / normal)
"""

import os
import sys
import time
import zipfile
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
ZIP_PATH = Path.home() / "Downloads" / "Thin_blood_smear_images.zip"
TARGET_DIR = ROOT / "data" / "raw" / "muhammad_kano_nigeria" / "thin_smear"

def main():
    if not ZIP_PATH.exists():
        print(f"Error: Archive not found at {ZIP_PATH}")
        sys.exit(1)

    pos_dir = TARGET_DIR / "positive"
    neg_dir = TARGET_DIR / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    print(f"Opening archive: {ZIP_PATH} ({ZIP_PATH.stat().st_size / (1024**3):.2f} GB)")
    print(f"Target directory: {TARGET_DIR}")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_members = [
            m for m in z.infolist()
            if not m.filename.startswith("__MACOSX")
            and not m.is_dir()
            and m.filename.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        roul_members = [m for m in all_members if "Rouleaux morphology RBC original form" in m.filename]
        norm_members = [m for m in all_members if "Normal morphology RBC original form" in m.filename]

        print(f"Found {len(roul_members)} Rouleaux (Positive) micrographs")
        print(f"Found {len(norm_members)} Normal (Negative) micrographs")

        # Extract Positive (Rouleaux)
        print("\nExtracting Positive micrographs (Rouleaux morphology)...")
        for m in tqdm(roul_members, desc="Positive"):
            fname = Path(m.filename).name
            target_path = pos_dir / fname
            if not target_path.exists() or target_path.stat().st_size != m.file_size:
                with z.open(m) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

        # Extract Negative (Normal)
        print("\nExtracting Negative micrographs (Normal morphology)...")
        for m in tqdm(norm_members, desc="Negative"):
            fname = Path(m.filename).name
            target_path = neg_dir / fname
            if not target_path.exists() or target_path.stat().st_size != m.file_size:
                with z.open(m) as src, open(target_path, "wb") as dst:
                    dst.write(src.read())

    print("\nExtraction complete!")
    pos_count = len(list(pos_dir.glob("*.jpg")))
    neg_count = len(list(neg_dir.glob("*.jpg")))
    print(f"Verified files in {TARGET_DIR}:")
    print(f"  Positive (Malaria / Rouleaux): {pos_count}")
    print(f"  Negative (Healthy / Normal) : {neg_count}")
    print(f"  Total Kano Micrographs      : {pos_count + neg_count}")

if __name__ == "__main__":
    main()
