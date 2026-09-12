"""
Compute comprehensive multi-modal benchmark metrics:
  - Table 1a: Thick Smear Zero-Shot Evaluation (Ghana, n=432)
  - Table 1b: Thin Smear Zero-Shot Evaluation (Ghana, n=1,011) with Sensitivity & Specificity
  - Table 1c: Cross-Modality Performance Matrix (Models x Smear Modalities)
  - Table 2: Quality-Stratified Robustness across Laplacian Focus Tertiles
"""

import sys
import io
import json
import pandas as pd
import numpy as np
from pathlib import Path

# UTF-8 stdout for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "manuscript" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# 1. Build Ground Truth Maps for both modalities
def build_ground_truth_maps():
    gt_map = {} # (image_id, smear_type) -> binary label (1: positive, 0: negative)
    
    # Thick smears: 0 is parasite, 1 is WBC
    thick_labels = DATA_RAW / "thick_smear" / "labels_yolo"
    if thick_labels.exists():
        for tf in thick_labels.glob("*.txt"):
            content = tf.read_text(encoding="utf-8", errors="ignore").strip()
            # Positive if non-empty and contains parasite class 0
            is_pos = 0
            if content:
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts and parts[0] == "0":
                        is_pos = 1
                        break
            gt_map[(tf.stem, "thick")] = is_pos
            
    # Thin smears: 0: gametocyte, 1: trophozoite, 2: other, 5: ring are parasites
    # 3: WBC, 4: artefacts are negative controls
    thin_labels = DATA_RAW / "thin_smear" / "labels_yolo"
    parasite_classes = {"0", "1", "2", "5"}
    if thin_labels.exists():
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
            gt_map[(tf.stem, "thin")] = is_pos
            
    return gt_map

def compute_metrics():
    gt_map = build_ground_truth_maps()
    print(f"Ground Truth index built: {len(gt_map)} entries indexed.")
    
    pred_path = PROCESSED / "predictions_manifest.csv"
    if not pred_path.exists():
        print(f"[Error] Predictions manifest not found at {pred_path}")
        return
        
    df_all = pd.read_csv(pred_path)
    
    # Map ground truth
    df_all["gt"] = df_all.apply(lambda r: gt_map.get((str(r["image_id"]), str(r["smear_type"])), 0), axis=1)
    
    # Merge quality metrics if available
    qm_path = ROOT / "data" / "quality_metrics" / "quality_manifest.csv"
    if qm_path.exists():
        qm = pd.read_csv(qm_path)
        # Compute strata within each smear type
        qm["quality_strata"] = qm.groupby("smear_type")["blur_laplacian"].transform(
            lambda x: pd.qcut(x, q=3, labels=["Low", "Medium", "High"], duplicates="drop")
        )
        df_all = df_all.merge(
            qm[["image_id", "smear_type", "blur_laplacian", "quality_strata"]],
            on=["image_id", "smear_type"],
            how="left"
        )
    else:
        df_all["quality_strata"] = "Unknown"
        df_all["blur_laplacian"] = np.nan

    # -------------------------------------------------------------
    # 2. Compute Metrics per Smear Type (Thick & Thin)
    # -------------------------------------------------------------
    for smear in ["thick", "thin"]:
        sub_df = df_all[df_all["smear_type"] == smear]
        if sub_df.empty:
            print(f"\n[Notice] No predictions recorded yet for smear_type='{smear}'.")
            continue
            
        print("\n" + "=" * 75)
        print(f"  BENCHMARK RESULTS: {smear.upper()} SMEAR MICROGRAPHS")
        print(f"  Total evaluations: {len(sub_df)} rows | Models: {sub_df['model_name'].nunique()}")
        print("=" * 75)
        
        rows = []
        for model, grp in sub_df.groupby("model_name"):
            TP = int(((grp["predicted_class"] == 1) & (grp["gt"] == 1)).sum())
            FP = int(((grp["predicted_class"] == 1) & (grp["gt"] == 0)).sum())
            TN = int(((grp["predicted_class"] == 0) & (grp["gt"] == 0)).sum())
            FN = int(((grp["predicted_class"] == 0) & (grp["gt"] == 1)).sum())
            n = len(grp)
            
            sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            spec = TN / (TN + FP) if (TN + FP) > 0 else (1.0 if FP == 0 else 0.0)
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            f1   = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
            acc  = (TP + TN) / n if n > 0 else 0.0
            mc   = float(grp["confidence"].mean()) if "confidence" in grp else 0.0
            
            rows.append({
                "Model": model,
                "Smear_Type": smear,
                "n": n,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4),
                "Precision": round(prec, 4),
                "F1_Score": round(f1, 4),
                "Accuracy": round(acc, 4),
                "Mean_Confidence": round(mc, 4)
            })
            
            who_status = "PASS (>=90%)" if sens >= 0.90 else f"GAP ({(0.90 - sens)*100:+.1f}%)"
            print(f"  [{who_status}] {model}")
            print(f"      n={n} | TP={TP}, FP={FP}, TN={TN}, FN={FN}")
            print(f"      Sensitivity: {sens*100:.2f}% | Specificity: {spec*100:.2f}% | F1: {f1:.3f} | Acc: {acc*100:.2f}%")
            print()
            
        res_df = pd.DataFrame(rows)
        csv_out = TABLES_DIR / f"main_results_table_{smear}.csv"
        res_df.to_csv(csv_out, index=False)
        print(f"  [Saved] Table saved to: {csv_out.name}")

    # -------------------------------------------------------------
    # 3. Cross-Modality Performance Matrix
    # -------------------------------------------------------------
    matrix_rows = []
    models = sorted(df_all["model_name"].unique())
    for m in models:
        row = {"Model": m}
        for smear in ["thick", "thin"]:
            grp = df_all[(df_all["model_name"] == m) & (df_all["smear_type"] == smear)]
            if not grp.empty:
                TP = int(((grp["predicted_class"] == 1) & (grp["gt"] == 1)).sum())
                FN = int(((grp["predicted_class"] == 0) & (grp["gt"] == 1)).sum())
                sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                row[f"{smear}_sensitivity"] = round(sens, 4)
                row[f"{smear}_n"] = len(grp)
            else:
                row[f"{smear}_sensitivity"] = np.nan
                row[f"{smear}_n"] = 0
        matrix_rows.append(row)
        
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_out = TABLES_DIR / "cross_modality_matrix.csv"
    matrix_df.to_csv(matrix_out, index=False)
    print(f"\n[Saved] Cross-modality matrix saved to: {matrix_out.name}")
    print(matrix_df.to_string(index=False))

    # -------------------------------------------------------------
    # 4. Stratified Sensitivity by Focus Quality
    # -------------------------------------------------------------
    if "quality_strata" in df_all and df_all["quality_strata"].notna().any():
        strat_rows = []
        for (model, smear), grp in df_all.groupby(["model_name", "smear_type"]):
            for strata, sg in grp.groupby("quality_strata"):
                if str(strata) == "nan":
                    continue
                TP = int(((sg["predicted_class"] == 1) & (sg["gt"] == 1)).sum())
                FN = int(((sg["predicted_class"] == 0) & (sg["gt"] == 1)).sum())
                sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                mb = float(sg["blur_laplacian"].mean()) if "blur_laplacian" in sg else 0.0
                strat_rows.append({
                    "Model": model,
                    "Smear_Type": smear,
                    "Quality_Strata": strata,
                    "n": len(sg),
                    "TP": TP,
                    "FN": FN,
                    "Sensitivity": round(sens, 4),
                    "Mean_Laplacian": round(mb, 2)
                })
        strat_df = pd.DataFrame(strat_rows)
        strat_out = TABLES_DIR / "stratified_sensitivity.csv"
        strat_df.to_csv(strat_out, index=False)
        print(f"\n[Saved] Stratified quality metrics saved to: {strat_out.name}")

if __name__ == "__main__":
    compute_metrics()
