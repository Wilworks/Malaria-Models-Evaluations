"""
Script to extract and ingest images from Thick_Ghana.part1.rar into data/raw/thick_smear/.
Directly extracts into data/raw/ and moves images into thick_smear/ to conserve disk space.
"""

import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image

def ingest_part1():
    downloads = Path(os.environ.get("USERPROFILE", "")) / "Downloads"
    rar_path = downloads / "Thick_Ghana.part1.rar"
    
    if not rar_path.exists():
        raise FileNotFoundError(f"Archive not found: {rar_path}")
        
    project_root = Path(__file__).resolve().parent.parent
    dest_dir = project_root / "data" / "raw" / "thick_smear"
    staging_base = project_root / "data" / "raw"
    
    print(f"=== Extracting {rar_path.name} ({rar_path.stat().st_size / (1024*1024):.2f} MB) ===")
    
    # Run tar.exe to extract directly into data/raw/
    cmd = ["tar.exe", "-xf", str(rar_path), "-C", str(staging_base)]
    print("Executing tar.exe extraction (this may take 1-2 minutes)...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(f"tar.exe return code: {res.returncode}")
    if res.stderr:
        print(f"tar.exe note: {res.stderr[:200]}...")
        
    extracted_folder = staging_base / "Ghana" / "Thick" / "images"
    if not extracted_folder.exists():
        # Look for any extracted Ghana folder
        candidates = list(staging_base.glob("**/Ghana/**/images"))
        if candidates:
            extracted_folder = candidates[0]
        else:
            print(f"[Error] Could not locate extracted Ghana folder in {staging_base}")
            return
            
    extracted_images = list(extracted_folder.glob("*.jpg")) + list(extracted_folder.glob("*.png"))
    print(f"\nExtracted image count in {extracted_folder.name}: {len(extracted_images)}")
    
    # Audit integrity & move into data/raw/thick_smear
    moved_count = 0
    already_exists = 0
    corrupt_count = 0
    
    for img in extracted_images:
        target = dest_dir / img.name
        if target.exists():
            already_exists += 1
            img.unlink(missing_ok=True)
        else:
            try:
                with Image.open(img) as im:
                    im.verify()
                shutil.move(str(img), str(target))
                moved_count += 1
            except Exception as e:
                print(f"[Warning] Corrupt image {img.name}: {e}")
                corrupt_count += 1
                img.unlink(missing_ok=True)
                
    print(f"\nIngestion Summary for Part 1:")
    print(f"  - Successfully moved new images: {moved_count}")
    print(f"  - Already existed (skipped): {already_exists}")
    print(f"  - Corrupt images discarded: {corrupt_count}")
    
    # Clean up Ghana directory structure in data/raw
    ghana_dir = staging_base / "Ghana"
    if ghana_dir.exists():
        shutil.rmtree(ghana_dir, ignore_errors=True)
        print("Cleaned up temporary extraction tree.")
        
    total_thick = len(list(dest_dir.glob("*.jpg")))
    print(f"\n=======================================================")
    print(f"[SUCCESS] TOTAL THICK SMEAR IMAGES NOW: {total_thick}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    ingest_part1()
