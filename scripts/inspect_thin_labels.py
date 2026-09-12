import os
from pathlib import Path
import json

thin_annot_dir = Path(os.environ.get("USERPROFILE", "")) / "Downloads" / "Thin Images Ghana" / "Thin Images With Annotations"

label_txt = thin_annot_dir / "label.txt"
if label_txt.exists():
    print(f"=== Content of {label_txt} ===")
    print(label_txt.read_text(encoding="utf-8", errors="ignore"))
    print("="*40)

# Check sample txt in labels_yolo
labels_dir = thin_annot_dir / "labels_yolo"
sample_txts = list(labels_dir.glob("*.txt"))[:5]
print(f"\nTotal label txt files in labels_yolo: {len(list(labels_dir.glob('*.txt')))}")
for st in sample_txts:
    content = st.read_text(encoding="utf-8", errors="ignore").strip()
    print(f"\n--- {st.name} ---")
    print(content[:300] if content else "(EMPTY - Negative case!)")

# Check class distribution across all txt files
class_counts = {}
empty_count = 0
for txt_file in labels_dir.glob("*.txt"):
    content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        empty_count += 1
        continue
    for line in content.splitlines():
        parts = line.strip().split()
        if parts:
            cls_id = parts[0]
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

print("\n=== Class Distribution in labels_yolo ===")
print(f"Empty label files (potential negatives/uninfected): {empty_count}")
print(f"Non-empty label files (positives with bounding boxes): {len(list(labels_dir.glob('*.txt'))) - empty_count}")
print(f"Total annotations by class ID: {class_counts}")
