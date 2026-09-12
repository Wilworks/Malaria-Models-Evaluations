import os
from pathlib import Path

thin_annot_dir = Path(os.environ.get("USERPROFILE", "")) / "Downloads" / "Thin Images Ghana" / "Thin Images With Annotations"
labels_dir = thin_annot_dir / "labels_yolo"

parasite_classes = {"0", "1", "2", "5"} # gametocyte, trophozoite, other stage, ring stage
wbc_artefact = {"3", "4"}

has_parasite = 0
only_wbc_artefact = 0
empty = 0

for txt_file in labels_dir.glob("*.txt"):
    content = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
    if not content:
        empty += 1
        continue
    classes_in_file = set()
    for line in content.splitlines():
        parts = line.strip().split()
        if parts:
            classes_in_file.add(parts[0])
    if classes_in_file.intersection(parasite_classes):
        has_parasite += 1
    else:
        only_wbc_artefact += 1

print(f"=== Ground Truth Slide-Level Classification for Thin Smears ===")
print(f"Total thin smear files: {len(list(labels_dir.glob('*.txt')))}")
print(f"Contains Parasite (GT = 1): {has_parasite}")
print(f"Contains ONLY WBC / Artefacts (GT = 0): {only_wbc_artefact}")
print(f"Empty files (GT = 0): {empty}")
