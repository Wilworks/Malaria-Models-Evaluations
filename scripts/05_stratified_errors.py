"""
Generates error stratification matrices across class, optical focus blur, and smear modality.
"""

import sys
import logging
import argparse
from pathlib import Path
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.error_analysis import ErrorAnalyzer
from src.deployment_safety import DeploymentSafetyEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_error_analysis(predictions_path: Path = None, output_dir: Path = None):
    output_dir = output_dir or (ROOT / "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if predictions_path is None:
        candidates = [
            ROOT / "results" / "run_1" / "predictions_manifest.csv",
            ROOT / "results" / "predictions_manifest.csv",
            ROOT / "data" / "processed" / "predictions_manifest.csv"
        ]
        for c in candidates:
            if c.exists():
                predictions_path = c
                break

    if not predictions_path or not predictions_path.exists():
        logger.warning("Predictions manifest not found. Run 'python run_pipeline.py' first.")
        return

    logger.info("=== Running Stratified Error Analysis on %s ===", predictions_path)
    df = pd.read_csv(predictions_path)

    # Merge with quality metrics manifest if available
    quality_path = ROOT / "data" / "quality_metrics" / "quality_manifest.csv"
    if quality_path.exists():
        q_df = pd.read_csv(quality_path)
        cols_to_use = [c for c in q_df.columns if c not in df.columns or c in ["image_id", "smear_type", "data_source"]]
        df = df.merge(q_df[cols_to_use], on=["image_id", "smear_type"], how="left")

    analyzer = ErrorAnalyzer(df)
    quality_summary = analyzer.stratify_by_quality(metric_col="blur_laplacian", bins=3)
    logger.info("\n--- Error Stratification by Focus Blur Quality (Low / Medium / High) ---\n%s\n", quality_summary.to_string())

    table_path = output_dir / "table3_quality_stratified_sensitivity.csv"
    quality_summary.to_csv(table_path, index=False)
    logger.info("Exported stratified errors table to: %s", table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stratified Error Analyzer")
    parser.add_argument("--predictions-path", type=str, default=None, help="Path to predictions_manifest.csv")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    run_error_analysis(
        predictions_path=Path(args.predictions_path) if args.predictions_path else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
