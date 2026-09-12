import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent

# Load GT
thin_labels = root / "data/raw/thin_smear/labels_yolo"
parasite_classes = {"0", "1", "2", "5"}
gt_map = {}
for tf in thin_labels.glob("*.txt"):
    if tf.name == "label.txt":
        continue
    content = tf.read_text(encoding="utf-8", errors="ignore").strip()
    is_pos = 0
    if content:
        for line in content.splitlines():
            parts = line.strip().split()
            if parts and parts[0] in parasite_classes:
                is_pos = 1
                break
    gt_map[tf.stem] = is_pos

models = ["MalariaScreener_Thick", "MalariaScreener_Thin", "MalariaScreener_Sudan"]
print("=== Preliminary Thin Smear Results (Completed Models) ===")
print(f"Total Thin Smears: {len(gt_map)} (Positives: {sum(gt_map.values())}, Negatives: {len(gt_map) - sum(gt_map.values())})\n")

for m in models:
    csv_file = root / f"data/processed/predictions_{m}_thin.csv"
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        df["gt"] = df["image_id"].astype(str).map(gt_map).fillna(0).astype(int)
        TP = int(((df["predicted_class"] == 1) & (df["gt"] == 1)).sum())
        FP = int(((df["predicted_class"] == 1) & (df["gt"] == 0)).sum())
        TN = int(((df["predicted_class"] == 0) & (df["gt"] == 0)).sum())
        FN = int(((df["predicted_class"] == 0) & (df["gt"] == 1)).sum())
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        acc = (TP + TN) / len(df)
        print(f"[{m}]")
        print(f"  Sensitivity: {sens*100:.2f}% ({TP}/{TP+FN})")
        print(f"  Specificity: {spec*100:.2f}% ({TN}/{TN+FP})")
        print(f"  Accuracy:    {acc*100:.2f}%")
        print(f"  TP={TP}, FP={FP}, TN={TN}, FN={FN}")
        print()
