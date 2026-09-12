from pathlib import Path

labels_dir = Path("data/raw/thick_smear/labels_yolo")
txt_files = list(labels_dir.glob("*.txt"))

empty_files = []
wbc_only_files = []
parasite_files = []

# Class mapping in Lacuna Ghana Thick Smear: 0: Parasite, 1: WBC (or check unique classes)
classes_found = set()

for tf in txt_files:
    lines = tf.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not lines:
        empty_files.append(tf.stem)
        continue
    file_classes = set()
    for l in lines:
        p = l.strip().split()
        if p:
            file_classes.add(p[0])
            classes_found.add(p[0])
    
    # In minoHealth Lacuna Ghana: 0 is Parasite, 1 is WBC
    if "0" in file_classes:
        parasite_files.append(tf.stem)
    else:
        wbc_only_files.append(tf.stem)

print(f"=== Total Thick Smear Label Files: {len(txt_files)} ===")
print(f"Unique class IDs found: {sorted(list(classes_found))}")
print(f"Files containing Parasite (GT = 1): {len(parasite_files)}")
print(f"Files containing ONLY WBC (GT = 0): {len(wbc_only_files)}")
print(f"Empty label files (no boxes / negative, GT = 0): {len(empty_files)}")
print(f"Total Negative Ground Truth Slides: {len(wbc_only_files) + len(empty_files)}")
