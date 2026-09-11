"""Compute full 4-model metrics table and stratified sensitivity. Saves CSVs."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent
labels_dir = root / "data/raw/thick_smear/labels_yolo"

# Build GT map
gt_map = {}
for img in (root / "data/raw/thick_smear").iterdir():
    if img.suffix in (".jpg", ".png", ".jpeg"):
        lf = labels_dir / (img.stem + ".txt")
        gt_map[img.stem] = 1 if (lf.exists() and lf.read_text().strip()) else 0

df_all = pd.read_csv(root / "data/processed/predictions_manifest.csv")
df_all["gt"] = df_all["image_id"].astype(str).map(gt_map).fillna(0).astype(int)

qm = pd.read_csv(root / "data/quality_metrics/quality_manifest.csv")
qm["quality_strata"] = pd.qcut(qm["blur_laplacian"], q=3, labels=["Low","Medium","High"], duplicates="drop")
df_q = df_all.merge(qm[["image_id","smear_type","blur_laplacian","quality_strata"]],
                    on=["image_id","smear_type"], how="left")

# ---- Aggregate metrics ----
print("=" * 70)
print("  TABLE 1: AGGREGATE ZERO-SHOT PERFORMANCE (ALL 4 MODELS)")
print(f"  Dataset: n=432 Ghanaian thick smear images (all GT positive)")
print("=" * 70)
agg_rows = []
for model, grp in df_all.groupby("model_name"):
    TP = int(((grp["predicted_class"]==1) & (grp["gt"]==1)).sum())
    FP = int(((grp["predicted_class"]==1) & (grp["gt"]==0)).sum())
    TN = int(((grp["predicted_class"]==0) & (grp["gt"]==0)).sum())
    FN = int(((grp["predicted_class"]==0) & (grp["gt"]==1)).sum())
    n = len(grp)
    sens = TP/(TP+FN) if (TP+FN) else 0
    f1   = 2*TP/(2*TP+FP+FN) if (2*TP+FP+FN) else 0
    acc  = (TP+TN)/n
    mc   = float(grp["confidence"].mean())
    gap  = 0.90 - sens
    agg_rows.append({
        "Model": model, "n": n, "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Sensitivity": round(sens, 3), "F1": round(f1, 3),
        "Accuracy": round(acc, 3), "MeanConf": round(mc, 3),
        "Gap_vs_WHO_90pct": round(gap, 3)
    })
    flag = "[PASS]" if sens >= 0.90 else "[WARN]"
    print(f"  {flag} {model}")
    print(f"      TP={TP}  FP={FP}  TN={TN}  FN={FN}  n={n}")
    print(f"      Sensitivity={sens:.3f} ({sens*100:.1f}%)  F1={f1:.3f}  Acc={acc:.3f}  MeanConf={mc:.3f}")
    print(f"      Gap vs WHO 90%: {gap*100:+.1f} pp")
    print()

agg_df = pd.DataFrame(agg_rows)
agg_df.to_csv(root / "manuscript/tables/main_results_table.csv", index=False)
print(f"  Saved -> manuscript/tables/main_results_table.csv")

# ---- Stratified sensitivity ----
print()
print("=" * 70)
print("  TABLE 2: STRATIFIED SENSITIVITY BY IMAGE FOCUS QUALITY")
print("  (Low: Laplacian <= 9.14 | Medium: 9.14-15.65 | High: >15.65)")
print("=" * 70)
strat_rows = []
for model, grp in df_q.groupby("model_name"):
    for strata, sg in grp.groupby("quality_strata"):
        TP = int(((sg["predicted_class"]==1) & (sg["gt"]==1)).sum())
        FN = int(((sg["predicted_class"]==0) & (sg["gt"]==1)).sum())
        sens = TP/(TP+FN) if (TP+FN) else 0
        mb = float(sg["blur_laplacian"].mean())
        strat_rows.append({
            "Model": model, "Quality_Strata": strata, "n": len(sg),
            "TP": TP, "FN": FN, "Sensitivity": round(sens, 3),
            "Mean_Blur": round(mb, 2)
        })
        print(f"  {model} [{strata}]: Sens={sens:.3f} ({sens*100:.1f}%)  TP={TP}  FN={FN}  MeanBlur={mb:.2f}")

strat_df = pd.DataFrame(strat_rows)
strat_df.to_csv(root / "manuscript/tables/stratified_sensitivity.csv", index=False)
print(f"\n  Saved -> manuscript/tables/stratified_sensitivity.csv")

# ---- Delta computation for each model ----
print()
print("=" * 70)
print("  QUALITY RESPONSE PATTERN (delta Low->High per model)")
print("=" * 70)
for model in strat_df["Model"].unique():
    sub = strat_df[strat_df["Model"] == model].set_index("Quality_Strata")
    if "Low" in sub.index and "High" in sub.index:
        delta = sub.loc["High", "Sensitivity"] - sub.loc["Low", "Sensitivity"]
        direction = "POSITIVE (sharper=better)" if delta > 0.01 else ("NEGATIVE (blurry=better)" if delta < -0.01 else "FLAT (quality-invariant)")
        print(f"  {model}: delta={delta*100:+.1f} pp -> {direction}")

print("\nDone.")
