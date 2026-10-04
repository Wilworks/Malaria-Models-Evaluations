"""
Model Checkpoints Downloader and Verifier.
Ensures external benchmark weights are available locally in models/external/.
"""

import sys
import os
import logging
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "name": "fbononi_YOLOv8",
        "filename": "best_yolo.pt",
        "dir": ROOT / "models" / "external" / "fbononibelloepoch",
        "url": "https://huggingface.co/fbononibelloepoch/malaria-detection/resolve/main/best_yolo.pt",
        "description": "YOLOv8 Nano object detector trained on field blood smears"
    },
    {
        "name": "MalariaScreener_Sudan",
        "filename": "malaria_sudan_model.pb",
        "dir": ROOT / "models" / "external" / "MalariaScreener" / "assets",
        "url": None,  # Provided in NIH MalariaScreener repository
        "description": "MobileNetV2 trained on East African smears (Sudan)"
    },
    {
        "name": "MalariaScreener_Thick",
        "filename": "malaria_thick_model.tflite",
        "dir": ROOT / "models" / "external" / "MalariaScreener" / "assets",
        "url": None,
        "description": "MobileNetV2 trained on thick smears (Chittagong, Bangladesh)"
    },
    {
        "name": "MalariaScreener_Thin",
        "filename": "malaria_thin_model.tflite",
        "dir": ROOT / "models" / "external" / "MalariaScreener" / "assets",
        "url": None,
        "description": "MobileNetV2 trained on thin smears (Chittagong, Bangladesh)"
    }
]


def verify_or_download_models():
    """Audits local model weights and downloads remote checkpoints if missing."""
    logger.info("Auditing candidate model checkpoints...")
    all_ready = True

    for m in MODELS:
        m["dir"].mkdir(parents=True, exist_ok=True)
        file_path = m["dir"] / m["filename"]

        if file_path.exists() and file_path.stat().st_size > 0:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info("  [READY] %s (%s, %.2f MB) at %s", m["name"], m["filename"], size_mb, file_path)
        elif m["url"]:
            logger.info("  [DOWNLOADING] %s from %s...", m["name"], m["url"])
            try:
                urllib.request.urlretrieve(m["url"], str(file_path))
                size_mb = file_path.stat().st_size / (1024 * 1024)
                logger.info("  [DOWNLOADED] %s (%.2f MB)", m["name"], size_mb)
            except Exception as e:
                logger.error("  [FAILED] Could not download %s: %s", m["name"], e)
                all_ready = False
        else:
            logger.warning("  [MISSING] %s checkpoint not found at %s", m["name"], file_path)
            all_ready = False

    return all_ready


if __name__ == "__main__":
    success = verify_or_download_models()
    sys.exit(0 if success else 1)
