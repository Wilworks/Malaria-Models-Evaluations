"""
Script 05: Generates error stratification matrices across class, image quality, and smear modality.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.error_analysis import ErrorAnalyzer
from src.deployment_safety import DeploymentSafetyEvaluator


def run_error_analysis():
    root_dir = Path(__file__).resolve().parent.parent
    predictions_path = root_dir / "data" / "processed" / "predictions_manifest.csv"
    quality_path = root_dir / "data" / "quality_metrics" / "quality_manifest.csv"
    output_dir = root_dir / "manuscript" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Script 05: Running Stratified Error Analysis & Safety Translation ===")

    if not predictions_path.exists():
        print(f"[Info] Predictions file {predictions_path} not found. Run scripts/04_run_zero_shot.py first.")
        return

    df = pd.read_csv(predictions_path)

    # Merge with quality metrics manifest if available
    if quality_path.exists():
        q_df = pd.read_csv(quality_path)
        # Select columns to merge
        cols_to_use = [c for c in q_df.columns if c not in df.columns or c in ["image_id", "smear_type"]]
        df = df.merge(q_df[cols_to_use], on=["image_id", "smear_type"], how="left")
        
        # Save merged dataframe with quality metrics tagged per prediction
        merged_out = root_dir / "data" / "processed" / "predictions_with_quality.csv"
        df.to_csv(merged_out, index=False)
        print(f"[Success] Saved predictions tagged with quality metrics -> {merged_out}")

    analyzer = ErrorAnalyzer(df)
    
    quality_summary = analyzer.stratify_by_quality(metric_col="blur_laplacian", bins=3)
    print("\n--- Error Stratification by Focus Blur Quality (Low / Medium / High) ---")
    print(quality_summary.to_string())

    # Save summary table for manuscript
    table_path = output_dir / "quality_error_strata.csv"
    quality_summary.to_csv(table_path, index=False)
    print(f"\n[Success] Table exported to -> {table_path}")


if __name__ == "__main__":
    run_error_analysis()
