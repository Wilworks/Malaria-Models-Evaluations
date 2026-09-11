"""
Script 02: Dataset verification and structure checker for Lacuna Ghana subset.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import LacunaGhanaDataset


def verify_dataset():
    data_root = Path(__file__).resolve().parent.parent / "data" / "raw"
    print(f"=== Script 02: Verifying Dataset in {data_root} ===")

    thin_ds = LacunaGhanaDataset(str(data_root), smear_type="thin")
    thick_ds = LacunaGhanaDataset(str(data_root), smear_type="thick")

    print(f"Thin Smear Images Found:  {len(thin_ds)}")
    print(f"Thick Smear Images Found: {len(thick_ds)}")

    if len(thin_ds) == 0 and len(thick_ds) == 0:
        print("\n[Action Required] No images found. Download Lacuna Ghana data into:")
        print(f"  - {data_root}/thin_smear/\n  - {data_root}/thick_smear/")
    else:
        print("\n[OK] Dataset verification complete.")


if __name__ == "__main__":
    verify_dataset()
