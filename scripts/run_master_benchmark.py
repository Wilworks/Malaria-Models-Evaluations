"""
Master Benchmark and Clinical Reproducibility Orchestrator.
Executes the zero-shot empirical evaluation pipeline across candidate malaria models:
  1. Ingests and validates multi-source clinical blood smear cohorts (thick and thin smears).
  2. Executes zero-shot inference for available external models (Sudan, Thick, Thin, YOLOv8).
  3. Computes comprehensive diagnostic metrics with 95% Wilson Score Confidence Intervals.
  4. Evaluates clinical deployment readiness against WHO screening thresholds.
  5. Computes quality-stratified performance across Laplacian focus blur tertiles.
  6. Exports structured CSV, JSON, and summary tables disaggregated by data source.
"""

import sys
import os
import cv2
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import MultiSourceMalariaDataset
from src.metrics import ClassificationMetrics, wilson_score_interval
from src.deployment_safety import DeploymentSafetyEvaluator
from models.wrappers.malariascreener_wrapper import MalariaScreenerWrapper
from models.wrappers.yolo_wrapper import YOLOMalariaWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum of a file for deterministic verification."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_models(
    root_dir: Path,
    selected_models: Optional[List[str]] = None,
    conf_threshold: float = 0.15
) -> List:
    """Instantiates verified model wrappers."""
    models = []
    
    ms_thick = root_dir / "models" / "external" / "MalariaScreener" / "assets" / "malaria_thick_model.tflite"
    ms_thin = root_dir / "models" / "external" / "MalariaScreener" / "assets" / "malaria_thin_model.tflite"
    ms_sudan = root_dir / "models" / "external" / "MalariaScreener" / "assets" / "malaria_sudan_model.pb"
    fbononi = root_dir / "models" / "external" / "fbononibelloepoch" / "best_yolo.pt"
    
    candidates = [
        ("sudan", "MalariaScreener_Sudan", ms_sudan, lambda p: MalariaScreenerWrapper(str(p), smear_type="sudan")),
        ("thick", "MalariaScreener_Thick", ms_thick, lambda p: MalariaScreenerWrapper(str(p), smear_type="thick")),
        ("thin", "MalariaScreener_Thin", ms_thin, lambda p: MalariaScreenerWrapper(str(p), smear_type="thin")),
        ("yolo", "fbononibelloepoch_YOLOv8", fbononi, lambda p: YOLOMalariaWrapper("fbononibelloepoch_YOLOv8", str(p), confidence_threshold=conf_threshold))
    ]

    for key, name, path, factory in candidates:
        if selected_models and "all" not in selected_models and key not in selected_models:
            continue
        if path.exists() and path.stat().st_size > 0:
            try:
                models.append(factory(path))
                logger.info("Loaded model candidate: %s", name)
            except Exception as e:
                logger.warning("Failed to initialize %s: %s", name, e)
        else:
            logger.warning("Model checkpoint not found for %s at %s", name, path)
            
    return models


def run_master_benchmark(
    data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    smear_types: Optional[List[str]] = None,
    selected_models: Optional[List[str]] = None,
    conf_threshold: float = 0.15,
    limit_samples: Optional[int] = None,
    shuffle: bool = True,
    seed: int = 42,
    run_id: int = 1
) -> pd.DataFrame:
    """Executes the complete multi-source evaluation benchmark."""
    start_time = time.time()
    data_dir = Path(data_dir or (ROOT / "data" / "raw"))
    output_dir = Path(output_dir or (ROOT / "results" / f"run_{run_id}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    smear_types = smear_types or ["thick", "thin"]

    logger.info("=" * 70)
    logger.info("STARTING MALARIA AI MULTI-SOURCE BENCHMARK RUN #%d", run_id)
    logger.info("Data Directory  : %s", data_dir)
    logger.info("Output Directory: %s", output_dir)
    logger.info("Smear Modalities: %s", smear_types)
    logger.info("=" * 70)

    # 1. Load Candidate Models
    models = build_models(ROOT, selected_models=selected_models, conf_threshold=conf_threshold)
    if not models:
        logger.error("No model checkpoints available to benchmark! Run scripts/download_models.py first.")
        return pd.DataFrame()

    # 2. Ingest Cohorts across Data Sources
    datasets = {}
    for smear in smear_types:
        ds = MultiSourceMalariaDataset(str(data_dir), smear_type=smear, shuffle=shuffle, seed=seed)
        if len(ds) > 0:
            datasets[smear] = ds
        else:
            logger.warning("No %s smear micrographs discovered in %s", smear, data_dir)

    if not datasets:
        logger.error("No images found in data directory: %s. Please add dataset images.", data_dir)
        return pd.DataFrame()

    # 3. Inference Loop
    predictions = []
    for smear, ds in datasets.items():
        items = ds.annotations
        if limit_samples and limit_samples > 0:
            items = items[:limit_samples]
            logger.info("Limited %s evaluation to %d samples for rapid testing", smear, len(items))

        logger.info("Running zero-shot inference on %s smears (%d slides)...", smear.upper(), len(items))

        for model in models:
            for item in tqdm(items, desc=f"{model.model_name} [{smear}]"):
                img_path = item["image_path"]
                img = cv2.imread(img_path)
                if img is None:
                    continue

                try:
                    res = model.predict(img)
                    predictions.append({
                        "image_id": item["image_id"],
                        "data_source": item.get("data_source", "lacuna_ghana"),
                        "smear_type": smear,
                        "model_name": model.model_name,
                        "ground_truth": item["ground_truth"],
                        "predicted_class": res.get("predicted_class", 0),
                        "confidence": float(res.get("confidence", 0.0)),
                        "parasite_count": res.get("parasite_count", 0),
                        "wbc_count": res.get("wbc_count", 0),
                        "detected_objects": res.get("detected_objects", 0),
                        "boxes_json": json.dumps(res.get("boxes", [])),
                        "image_path": img_path
                    })
                except Exception as e:
                    logger.warning("Inference failure on %s (%s): %s", img_path, model.model_name, e)

    if not predictions:
        logger.warning("No predictions generated.")
        return pd.DataFrame()

    pred_df = pd.DataFrame(predictions)
    pred_df = pred_df.sort_values(by=["data_source", "smear_type", "model_name", "image_id"]).reset_index(drop=True)
    pred_df.to_csv(output_dir / "predictions_manifest.csv", index=False)
    logger.info("Saved %d predictions to %s", len(pred_df), output_dir / "predictions_manifest.csv")

    # 4. Diagnostic Metrics & Confidence Intervals
    logger.info("Computing diagnostic performance metrics (overall and per-source)...")
    table_rows = []
    safety_evaluator = DeploymentSafetyEvaluator()
    safety_audit = {}

    distinct_sources = sorted(pred_df["data_source"].unique().tolist())
    has_multiple_sources = len(distinct_sources) > 1

    for smear in smear_types:
        smear_sub = pred_df[pred_df["smear_type"] == smear]
        if smear_sub.empty:
            continue

        for model_name, grp in smear_sub.groupby("model_name"):
            # A. Overall Pooled Benchmark
            pooled_metrics = ClassificationMetrics.compute_binary_metrics(
                grp["ground_truth"].values,
                grp["predicted_class"].values,
                y_prob=grp["confidence"].values
            )
            safety = safety_evaluator.evaluate_safety_profile(pooled_metrics)
            sens_low, sens_high = pooled_metrics["sensitivity_ci_95"]
            spec_low, spec_high = pooled_metrics["specificity_ci_95"]

            table_rows.append({
                "Modality": smear.capitalize(),
                "Data_Source": "Pooled (All West Africa)" if has_multiple_sources else distinct_sources[0],
                "Model": model_name,
                "N": pooled_metrics["n_samples"],
                "TP": pooled_metrics["tp"],
                "FP": pooled_metrics["fp"],
                "TN": pooled_metrics["tn"],
                "FN": pooled_metrics["fn"],
                "Sensitivity (%)": round(pooled_metrics["sensitivity"] * 100, 2),
                "Sens_95CI": f"[{sens_low*100:.1f}-{sens_high*100:.1f}]",
                "Specificity (%)": round(pooled_metrics["specificity"] * 100, 2) if (pooled_metrics["tn"] + pooled_metrics["fp"]) > 0 else "N/A",
                "Spec_95CI": f"[{spec_low*100:.1f}-{spec_high*100:.1f}]" if (pooled_metrics["tn"] + pooled_metrics["fp"]) > 0 else "N/A",
                "F1_Score": round(pooled_metrics["f1_score"], 4),
                "Accuracy (%)": round(pooled_metrics["accuracy"] * 100, 2),
                "Clinical_Safety": safety["risk_level"]
            })

            # B. Per-Source Disaggregated Performance (if multiple sources present)
            if has_multiple_sources:
                for src, src_grp in grp.groupby("data_source"):
                    src_metrics = ClassificationMetrics.compute_binary_metrics(
                        src_grp["ground_truth"].values,
                        src_grp["predicted_class"].values,
                        y_prob=src_grp["confidence"].values
                    )
                    s_low, s_high = src_metrics["sensitivity_ci_95"]
                    sp_low, sp_high = src_metrics["specificity_ci_95"]

                    table_rows.append({
                        "Modality": smear.capitalize(),
                        "Data_Source": src,
                        "Model": model_name,
                        "N": src_metrics["n_samples"],
                        "TP": src_metrics["tp"],
                        "FP": src_metrics["fp"],
                        "TN": src_metrics["tn"],
                        "FN": src_metrics["fn"],
                        "Sensitivity (%)": round(src_metrics["sensitivity"] * 100, 2),
                        "Sens_95CI": f"[{s_low*100:.1f}-{s_high*100:.1f}]",
                        "Specificity (%)": round(src_metrics["specificity"] * 100, 2) if (src_metrics["tn"] + src_metrics["fp"]) > 0 else "N/A",
                        "Spec_95CI": f"[{sp_low*100:.1f}-{sp_high*100:.1f}]" if (src_metrics["tn"] + src_metrics["fp"]) > 0 else "N/A",
                        "F1_Score": round(src_metrics["f1_score"], 4),
                        "Accuracy (%)": round(src_metrics["accuracy"] * 100, 2),
                        "Clinical_Safety": safety_evaluator.evaluate_safety_profile(src_metrics)["risk_level"]
                    })

            safety_audit[f"{model_name}_{smear}"] = safety

    summary_df = pd.DataFrame(table_rows)
    logger.info("\n%s\n", summary_df.to_string(index=False))

    # Save summary tables
    summary_df.to_csv(output_dir / "table1_master_diagnostic_performance.csv", index=False)
    with open(output_dir / "deployment_safety_audit.json", "w") as f:
        json.dump(safety_audit, f, indent=2)

    # 5. Quality-Stratified Analysis
    qual_path = ROOT / "data" / "quality_metrics" / "quality_manifest.csv"
    if qual_path.exists():
        logger.info("Merging quality metrics from %s...", qual_path)
        qual_df = pd.read_csv(qual_path)
        qual_df["image_id"] = qual_df["image_id"].astype(str)
        merged_df = pred_df.merge(qual_df, on=["image_id", "smear_type"], how="left")
        
        if "blur_laplacian" in merged_df.columns:
            from src.error_analysis import ErrorAnalyzer
            analyzer = ErrorAnalyzer(merged_df)
            strat_df = analyzer.stratify_by_quality(metric_col="blur_laplacian", bins=3)
            strat_df.to_csv(output_dir / "table3_quality_stratified_sensitivity.csv", index=False)
            logger.info("Saved quality-stratified performance to %s", output_dir / "table3_quality_stratified_sensitivity.csv")

    # 6. Checksums for Determinism Audit
    checksums = {}
    for p in output_dir.glob("*.csv"):
        checksums[p.name] = compute_sha256(p)
    with open(output_dir / "run_checksum.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - start_time, 2),
            "checksums": checksums
        }, f, indent=2)

    logger.info("Run #%d successfully completed in %.2fs. Outputs saved to %s",
                run_id, time.time() - start_time, output_dir)
    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Malaria AI Benchmark Runner")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to output directory")
    parser.add_argument("--smear-type", type=str, default="both", choices=["thick", "thin", "both"], help="Smear modality to benchmark")
    parser.add_argument("--models", nargs="+", default=["all"], help="Models to benchmark (sudan, thick, thin, yolo, all)")
    parser.add_argument("--conf-threshold", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit sample count for testing")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable random shuffling of samples across sources")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--run-id", type=int, default=1, help="Run ID number")
    args = parser.parse_args()

    smears = ["thick", "thin"] if args.smear_type == "both" else [args.smear_type]
    run_master_benchmark(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        smear_types=smears,
        selected_models=args.models,
        conf_threshold=args.conf_threshold,
        limit_samples=args.limit_samples,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        run_id=args.run_id
    )
