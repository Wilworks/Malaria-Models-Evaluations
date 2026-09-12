import os
from pathlib import Path

downloads = Path(os.environ.get("USERPROFILE", "")) / "Downloads"
print(f"=== Inspecting Downloads folder: {downloads} ===")

keywords = ["thin", "thick", "malaria", "lacuna", "ghana"]
matches = []

try:
    for item in downloads.iterdir():
        name_lower = item.name.lower()
        if any(k in name_lower for k in keywords):
            is_dir = item.is_dir()
            size = item.stat().st_size if not is_dir else 0
            size_mb = size / (1024 * 1024)
            matches.append((item.name, is_dir, size_mb, item))
            print(f"[{'DIR' if is_dir else 'FILE'}] {item.name} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"Error accessing downloads: {e}")

if not matches:
    print("No matching files found with keywords in Downloads.")
