"""
Script to ingest the thin smear images and annotations from Downloads
into data/raw/thin_smear/.
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def ingest_thin_smear():
    downloads = Path(os.environ.get("USERPROFILE", "")) / "Downloads"
    source_root = downloads / "Thin Images Ghana" / "Thin Images With Annotations"
    
    src_images = source_root / "images"
    src_labels = source_root / "labels_yolo"
    src_label_file = source_root / "label.txt"
    
    if not src_images.exists():
        raise FileNotFoundError(f"Source images directory not found: {src_images}")
        
    project_root = Path(__file__).resolve().parent.parent
    dest_thin = project_root / "data" / "raw" / "thin_smear"
    dest_labels = dest_thin / "labels_yolo"
    
    dest_thin.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)
    
    # 1. Copy label.txt
    if src_label_file.exists():
        shutil.copy2(src_label_file, dest_labels / "label.txt")
        print(f"[Copied] {src_label_file.name} -> {dest_labels}")
        
    # 2. Copy labels
    label_files = list(src_labels.glob("*.txt"))
    print(f"Copying {len(label_files)} label files to {dest_labels}...")
    for lf in tqdm(label_files, desc="Copying Labels"):
        shutil.copy2(lf, dest_labels / lf.name)
        
    # 3. Copy images
    image_files = list(src_images.glob("*.jpg"))
    print(f"Copying {len(image_files)} image files to {dest_thin}...")
    for img in tqdm(image_files, desc="Copying Images"):
        shutil.copy2(img, dest_thin / img.name)
        
    # Verification
    copied_images = list(dest_thin.glob("*.jpg"))
    copied_labels = list(dest_labels.glob("*.txt"))
    
    print("\n=== Ingestion Complete ===")
    print(f"Total Thin Smear Images in {dest_thin}: {len(copied_images)}")
    print(f"Total Thin Smear Labels in {dest_labels}: {len(copied_labels)}")
    
    if len(copied_images) == len(image_files) and len(copied_labels) >= len(label_files):
        print("[SUCCESS] All files copied and verified successfully.")
    else:
        print("[WARNING] Discrepancy detected in copied file counts.")

if __name__ == "__main__":
    ingest_thin_smear()
