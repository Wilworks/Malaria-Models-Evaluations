from pathlib import Path

labels_dir = Path("data/raw/thick_smear/labels_yolo")
c0_count = 0
c1_count = 0
c0_sizes = []
c1_sizes = []

for tf in list(labels_dir.glob("*.txt"))[:500]:
    for line in tf.read_text().splitlines():
        p = line.strip().split()
        if len(p) >= 5:
            cls_id, xc, yc, bw, bh = p[0], float(p[1]), float(p[2]), float(p[3]), float(p[4])
            area = bw * bh
            if cls_id == "0":
                c0_count += 1
                c0_sizes.append(area)
            elif cls_id == "1":
                c1_count += 1
                c1_sizes.append(area)

print(f"Sample (500 files): Class 0 count: {c0_count}, mean bbox area: {sum(c0_sizes)/len(c0_sizes):.6f}")
print(f"Sample (500 files): Class 1 count: {c1_count}, mean bbox area: {sum(c1_sizes)/len(c1_sizes) if c1_sizes else 0:.6f}")
