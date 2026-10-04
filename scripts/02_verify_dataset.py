"""
Dataset verification and multi-source structure checker for West African cohorts.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import MultiSourceMalariaDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def verify_dataset(data_root: Path = None):
    data_root = data_root or (ROOT / "data" / "raw")
    logger.info("=== Verifying Multi-Source Dataset in %s ===", data_root)

    thick_ds = MultiSourceMalariaDataset(str(data_root), smear_type="thick", shuffle=False)
    thin_ds = MultiSourceMalariaDataset(str(data_root), smear_type="thin", shuffle=False)

    sources = set(r["data_source"] for r in thick_ds.annotations).union(
        r["data_source"] for r in thin_ds.annotations
    )

    logger.info("Discovered Sources: %s", sorted(list(sources)) if sources else "None")
    logger.info("Total Thick Smear Micrographs : %d", len(thick_ds))
    logger.info("Total Thin Smear Micrographs  : %d", len(thin_ds))

    if sources:
        for src in sorted(sources):
            th_cnt = sum(1 for r in thick_ds.annotations if r["data_source"] == src)
            th_pos = sum(1 for r in thick_ds.annotations if r["data_source"] == src and r["ground_truth"] == 1)
            tn_cnt = sum(1 for r in thin_ds.annotations if r["data_source"] == src)
            tn_pos = sum(1 for r in thin_ds.annotations if r["data_source"] == src and r["ground_truth"] == 1)
            logger.info("  ↳ Source '%s':", src)
            logger.info("      Thick: %d (Positives: %d, Negatives: %d)", th_cnt, th_pos, th_cnt - th_pos)
            logger.info("      Thin : %d (Positives: %d, Negatives: %d)", tn_cnt, tn_pos, tn_cnt - tn_pos)

    if len(thin_ds) == 0 and len(thick_ds) == 0:
        logger.warning(
            "No images found. Place West African cohort folders into:\n"
            "  - %s/<source_name>/thick_smear/\n"
            "  - %s/<source_name>/thin_smear/",
            data_root, data_root
        )
    else:
        logger.info("[OK] Dataset verification complete and ready for benchmarking.")


if __name__ == "__main__":
    verify_dataset()
