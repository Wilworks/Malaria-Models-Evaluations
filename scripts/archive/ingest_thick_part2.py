"""
Script to extract and ingest images from Thick_Ghana.part2.rar into data/raw/thick_smear/.
"""

import os
import shutil
import subprocess
from pathlib import Path

def ingest_part2():
    downloads = Path(os.environ.get("USERPROFILE", "")) / "Downloads"
    rar_path = downloads / "Thick_Ghana.part2.rar"
    
    if not rar_path.exists():
        raise FileNotFoundError(f"Archive not found: {rar_path}")
        
    project_root = Path(__file__).resolve().parent.parent
    dest_dir = project_root / "data" / "raw" / "thick_smear"
    staging_dir = project_root / "data" / "raw" / "_staging_part2"
    staging_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Extracting {rar_path.name} ({rar_path.stat().st_size / (1024*1024):.2f} MB) ===")
    print(f"Staging directory: {staging_dir}")
    
    # Extract using tar.exe
    cmd = ["tar.exe", "-xf", str(rar_path), "-C", str(staging_dir)]
    print("Running extraction command...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[Warning] tar.exe stderr: {res.stderr}")
    else:
        print("[Extraction Complete] Unpacked archive into staging.")
        
    # Find all extracted .jpg images
    extracted_images = list(staging_dir.rglob("*.jpg")) + list(staging_dir.rglob("*.jpeg")) + list(staging_dir.rglob("*.png"))
    print(f"Extracted image files found: {len(extracted_images)}")
    
    # Check for any label files
    extracted_labels = list(staging_dir.rglob("*.txt"))
    print(f"Extracted label files found: {len(extracted_labels)}")
    
    # Move images into data/raw/thick_smear/
    moved_count = 0
    already_exists = 0
    for img in extracted_images:
        target_path = dest_dir / img.name
        if target_path.exists():
            already_exists += 1
        else:
            shutil.copy2(img, target_path)
            moved_count += 1
            
    print(f"\nIngestion summary:")
    print(f"  - Newly ingested images: {moved_count}")
    print(f"  - Already existing images: {already_exists}")
    
    # If any labels were extracted, copy them to labels_yolo
    if extracted_labels:
        lbl_dir = dest_dir / "labels_yolo"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for lf in extracted_labels:
            shutil.copy2(lf, lbl_dir / lf.name)
        print(f"  - Copied {len(extracted_labels)} label files into labels_yolo/")
        
    # Clean up staging directory
    print("Cleaning up staging directory...")
    shutil.rmtree(staging_dir, ignore_errors=True)
    
    # Total count in dest_dir
    total_thick = len(list(dest_dir.glob("*.jpg")))
    print(f"\n[SUCCESS] Total Thick Smear Images now in {dest_dir}: {total_thick}")

if __name__ == "__main__":
    ingest_part2()
