"""
Unit tests for multi-source clinical dataset loader and annotation parser.
"""

import tempfile
import unittest
from pathlib import Path
from src.data_loader import MultiSourceMalariaDataset


class TestMultiSourceDataLoader(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

        # Source 1: Lacuna Ghana with YOLO annotations
        src1_thick = self.root / "lacuna_ghana" / "thick_smear"
        src1_labels = src1_thick / "labels_yolo"
        src1_labels.mkdir(parents=True, exist_ok=True)

        # Slide 1: Parasite (Class 0) -> Positive
        img1 = src1_thick / "slide_pos.jpg"
        img1.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        (src1_labels / "slide_pos.txt").write_text("0 0.5 0.5 0.1 0.1\n")

        # Slide 2: WBC ONLY (Class 1) -> Negative
        img2 = src1_thick / "slide_wbc_neg.jpg"
        img2.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        (src1_labels / "slide_wbc_neg.txt").write_text("1 0.3 0.3 0.2 0.2\n")

        # Source 2: Another West African cohort with binary subfolders (positive/ vs negative/)
        src2_thick = self.root / "cohort_two" / "thick_smear"
        pos_dir = src2_thick / "positive"
        neg_dir = src2_thick / "negative"
        pos_dir.mkdir(parents=True, exist_ok=True)
        neg_dir.mkdir(parents=True, exist_ok=True)

        img3 = pos_dir / "field_pos.jpg"
        img3.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

        img4 = neg_dir / "field_neg.jpg"
        img4.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_multi_source_discovery(self):
        """Verifies automatic discovery of multiple data sources."""
        ds = MultiSourceMalariaDataset(str(self.root), smear_type="thick", shuffle=False)
        self.assertEqual(len(ds), 4)

        sources = set(r["data_source"] for r in ds.annotations)
        self.assertEqual(sources, {"lacuna_ghana", "cohort_two"})

    def test_wbc_negative_and_folder_binary(self):
        """Verifies correct ground truth parsing across YOLO and folder structures."""
        ds = MultiSourceMalariaDataset(str(self.root), smear_type="thick", shuffle=False)
        mapping = {r["image_id"]: r for r in ds.annotations}

        # Source 1 (YOLO)
        self.assertEqual(mapping["slide_pos"]["ground_truth"], 1)
        self.assertEqual(mapping["slide_pos"]["data_source"], "lacuna_ghana")

        self.assertEqual(mapping["slide_wbc_neg"]["ground_truth"], 0)
        self.assertEqual(mapping["slide_wbc_neg"]["data_source"], "lacuna_ghana")

        # Source 2 (Folder binary)
        self.assertEqual(mapping["field_pos"]["ground_truth"], 1)
        self.assertEqual(mapping["field_pos"]["data_source"], "cohort_two")

        self.assertEqual(mapping["field_neg"]["ground_truth"], 0)
        self.assertEqual(mapping["field_neg"]["data_source"], "cohort_two")

    def test_shuffling(self):
        """Verifies deterministic shuffling produces a mixed order."""
        ds_unshuffled = MultiSourceMalariaDataset(str(self.root), smear_type="thick", shuffle=False)
        ds_shuffled = MultiSourceMalariaDataset(str(self.root), smear_type="thick", shuffle=True, seed=123)

        ids_unshuffled = [r["image_id"] for r in ds_unshuffled.annotations]
        ids_shuffled = [r["image_id"] for r in ds_shuffled.annotations]

        self.assertEqual(set(ids_unshuffled), set(ids_shuffled))
        self.assertEqual(len(ds_shuffled), 4)


if __name__ == "__main__":
    unittest.main()
