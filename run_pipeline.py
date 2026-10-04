#!/usr/bin/env python3
"""
Master Malaria Benchmark Pipeline Orchestrator.
Orchestrates environment pre-flight, model checkpoint validation, dataset auditing,
optical quality profiling, and zero-shot clinical evaluation.
"""

import sys
import os
import time
import logging
import argparse
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List

# Ensure repository root is in sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("malaria_pipeline")


def stage_1_environment_check() -> bool:
    """Stage 1: Pre-flight Python runtime and package validation."""
    logger.info("-" * 65)
    logger.info("STAGE 1: Environment & Dependency Verification")
    logger.info("-" * 65)

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    logger.info("Python Runtime : %s (%s)", py_ver, platform.architecture()[0])
    logger.info("Python Binary  : %s", sys.executable)
    
    in_venv = sys.prefix != sys.base_prefix
    logger.info("Virtual Env    : %s (Active: %s)", Path(sys.prefix).name if in_venv else "None", in_venv)

    required_pkgs = ["torch", "torchvision", "ultralytics", "cv2", "pandas", "sklearn", "scipy"]
    missing = []
    for pkg in required_pkgs:
        try:
            if pkg == "cv2":
                import cv2
            elif pkg == "sklearn":
                import sklearn
            else:
                __import__(pkg)
        except ImportError:
            missing.append(pkg)

    # Verify TFLite runtime (ai-edge-litert, tflite_runtime, or tensorflow)
    has_tflite = False
    for tfl_pkg in ["ai_edge_litert", "tflite_runtime", "tensorflow"]:
        try:
            __import__(tfl_pkg)
            has_tflite = True
            logger.info("TFLite Runtime : %s", tfl_pkg)
            break
        except ImportError:
            pass

    if not has_tflite:
        missing.append("ai-edge-litert (or tensorflow)")

    if missing:
        logger.warning("Missing dependencies: %s", ", ".join(missing))
        logger.info("Install required dependencies using: pip install -r requirements.txt")
        return False

    logger.info("Core machine learning libraries verified.")
    return True


def stage_2_hardware_and_checkpoints(conf_threshold: float = 0.15) -> List[Any]:
    """Stage 2: Hardware acceleration probe and model checkpoint validation."""
    logger.info("-" * 65)
    logger.info("STAGE 2: Hardware Acceleration & Model Checkpoints Probe")
    logger.info("-" * 65)

    # Real hardware probe
    gpu_info = "CPU Inference Mode"
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info = f"NVIDIA CUDA {torch.version.cuda} ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            gpu_info = "Apple Silicon MPS (Metal Performance Shaders)"
    except Exception:
        pass

    logger.info("Host Platform : %s %s (%s)", platform.system(), platform.release(), platform.machine())
    logger.info("Logical Cores : %s", os.cpu_count())
    logger.info("Accelerator   : %s", gpu_info)

    # Verify model checkpoints
    from scripts.download_models import verify_or_download_models
    from scripts.run_master_benchmark import build_models

    verify_or_download_models()
    models = build_models(ROOT, conf_threshold=conf_threshold)
    logger.info("Active Benchmark Models: %d loaded", len(models))
    for m in models:
        logger.info("  - %s (%s)", m.model_name, m.task_type)

    return models


def stage_3_dataset_audit(data_dir: Path) -> Dict[str, Any]:
    """Stage 3: Clinical multi-source dataset audit."""
    logger.info("-" * 65)
    logger.info("STAGE 3: Clinical Multi-Source Dataset Audit")
    logger.info("-" * 65)
    logger.info("Target Data Directory: %s", data_dir)

    from src.data_loader import MultiSourceMalariaDataset

    thick_ds = MultiSourceMalariaDataset(data_dir, smear_type="thick", shuffle=False)
    thin_ds = MultiSourceMalariaDataset(data_dir, smear_type="thin", shuffle=False)

    sources = set(r["data_source"] for r in thick_ds.annotations).union(
        r["data_source"] for r in thin_ds.annotations
    )

    total = len(thick_ds) + len(thin_ds)
    logger.info("Discovered Data Sources : %s", sorted(list(sources)) if sources else "None")
    logger.info("Thick Smear Micrographs : %d", len(thick_ds))
    logger.info("Thin Smear Micrographs  : %d", len(thin_ds))
    logger.info("Total Cohort Size       : %d micrographs", total)

    if sources:
        for src in sorted(sources):
            src_thick = sum(1 for r in thick_ds.annotations if r["data_source"] == src)
            src_thin = sum(1 for r in thin_ds.annotations if r["data_source"] == src)
            logger.info("  ↳ Source '%s': %d thick, %d thin", src, src_thick, src_thin)

    if total == 0:
        logger.warning(
            "No micrographs detected in %s.\n"
            "Supported folder structures:\n"
            "  1. Multi-source: %s/<source_name>/thick_smear/ and thin_smear/\n"
            "  2. Direct:       %s/thick_smear/ and thin_smear/\n"
            "Or specify --data-dir <path>",
            data_dir, data_dir, data_dir
        )

    return {"thick": len(thick_ds), "thin": len(thin_ds), "total": total, "sources": list(sources)}


def stage_4_optical_quality(data_dir: Path, force_recompute: bool = False) -> None:
    """Stage 4: Optical blur, contrast, and SNR calculation."""
    logger.info("-" * 65)
    logger.info("STAGE 4: Optical Quality Assessment (Circular FOV Masking)")
    logger.info("-" * 65)

    qual_path = ROOT / "data" / "quality_metrics" / "quality_manifest.csv"
    if qual_path.exists() and not force_recompute:
        logger.info("Cached optical quality manifest found at: %s", qual_path)
        return

    script = ROOT / "scripts" / "03_compute_quality.py"
    if not script.exists():
        script = ROOT / "scripts" / "compute_quality.py"

    if script.exists():
        logger.info("Executing optical quality profiling on %s...", data_dir)
        subprocess.run([sys.executable, str(script)], check=False)
    else:
        logger.info("Optical quality script not found, continuing without cached metrics.")


def stage_5_execute_benchmark(
    data_dir: Path,
    output_dir: Path,
    smear_types: List[str],
    conf_threshold: float,
    limit_samples: int = None,
    shuffle: bool = True,
    seed: int = 42,
    run_id: int = 1
) -> Any:
    """Stage 5: Multi-model zero-shot evaluation and statistical reporting."""
    logger.info("-" * 65)
    logger.info("STAGE 5: Zero-Shot Multi-Model Benchmark Execution")
    logger.info("-" * 65)

    from scripts.run_master_benchmark import run_master_benchmark
    summary_df = run_master_benchmark(
        data_dir=data_dir,
        output_dir=output_dir,
        smear_types=smear_types,
        conf_threshold=conf_threshold,
        limit_samples=limit_samples,
        shuffle=shuffle,
        seed=seed,
        run_id=run_id
    )
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Master Malaria AI Benchmark & Reproducibility Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory where results will be stored")
    parser.add_argument("--smear-type", type=str, default="both", choices=["thick", "thin", "both"], help="Smear modality")
    parser.add_argument("--conf-threshold", type=float, default=0.15, help="YOLO confidence threshold")
    parser.add_argument("--limit-samples", type=int, default=None, help="Limit sample count for rapid verification")
    parser.add_argument("--no-shuffle", action="store_true", help="Disable random shuffling of samples across sources")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed for shuffling")
    parser.add_argument("--force-quality", action="store_true", help="Force recomputation of optical quality metrics")
    parser.add_argument("--run-id", type=int, default=1, help="Benchmark run identifier")
    args = parser.parse_args()

    start_time = time.time()
    logger.info("=" * 65)
    logger.info("WAM-BENCH: WEST AFRICAN MALARIA AI CLINICAL BENCHMARK")
    logger.info("Multi-Center Evaluation Across Ghanaian and Nigerian Micrographs")
    logger.info("=" * 65)

    # Stage 1: Environment check
    env_ok = stage_1_environment_check()
    if not env_ok:
        logger.error("Environment verification failed. Please resolve dependencies.")
        sys.exit(1)

    # Stage 2: Hardware & Models
    models = stage_2_hardware_and_checkpoints(conf_threshold=args.conf_threshold)

    # Stage 3: Dataset Audit
    data_dir = Path(args.data_dir)
    cohort = stage_3_dataset_audit(data_dir)

    if cohort["total"] == 0:
        logger.warning("Pipeline halted at Stage 3: Dataset is empty.")
        logger.info("Add images and labels to %s to execute inference.", data_dir)
        sys.exit(0)

    # Stage 4: Optical Quality
    stage_4_optical_quality(data_dir, force_recompute=args.force_quality)

    # Stage 5: Benchmark Execution
    smear_types = ["thick", "thin"] if args.smear_type == "both" else [args.smear_type]
    out_dir = Path(args.output_dir) / f"run_{args.run_id}"
    summary_df = stage_5_execute_benchmark(
        data_dir=data_dir,
        output_dir=out_dir,
        smear_types=smear_types,
        conf_threshold=args.conf_threshold,
        limit_samples=args.limit_samples,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        run_id=args.run_id
    )

    elapsed = time.time() - start_time
    logger.info("=" * 65)
    logger.info("PIPELINE COMPLETED IN %.2f SECONDS", elapsed)
    logger.info("Artifacts saved to: %s", out_dir)
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
