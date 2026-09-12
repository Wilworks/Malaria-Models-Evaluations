import os
from pathlib import Path

thin_dir = Path(os.environ.get("USERPROFILE", "")) / "Downloads" / "Thin Images Ghana"
print(f"=== Inspecting Directory: {thin_dir} ===")

if not thin_dir.exists():
    print("Directory does not exist!")
else:
    # Walk and summarize
    all_files = list(thin_dir.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    dirs = [d for d in all_files if d.is_dir()]
    
    print(f"Total subdirectories: {len(dirs)}")
    for d in dirs:
        print(f"  - Subdir: {d.relative_to(thin_dir)}")
        
    print(f"\nTotal files: {len(files)}")
    
    # Extensions breakdown
    exts = {}
    for f in files:
        ext = f.suffix.lower()
        exts[ext] = exts.get(ext, 0) + 1
    print("File types / extensions:", exts)
    
    # Sample files
    print("\nSample files:")
    for f in files[:15]:
        print(f"  {f.relative_to(thin_dir)} ({f.stat().st_size / 1024:.1f} KB)")
        
    # Check for any non-image files (labels, txt, json, csv, xml)
    non_img = [f for f in files if f.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]]
    if non_img:
        print(f"\nNon-image files ({len(non_img)}):")
        for f in non_img[:20]:
            print(f"  - {f.name}")
