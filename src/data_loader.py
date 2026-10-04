"""
Multi-Source Clinical Dataset Loader for Malaria Blood Smears.
Supports heterogeneous multi-center West African datasets across diverse annotation formats:
  1. YOLO format (.txt bounding boxes)
  2. Pascal VOC format (.xml bounding boxes)
  3. Binary classification subdirectories (positive/ vs. negative/)
  4. CSV annotation manifests (annotations.csv / metadata.csv)

Enables tracking 'data_source' per micrograph while pooling and deterministically
shuffling images across sources at inference time.
"""

import os
import random
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
import pandas as pd

logger = logging.getLogger(__name__)


class MultiSourceMalariaDataset:
    """
    Robust multi-center microscopy dataset indexing and parsing engine.
    
    Automatically discovers multiple data sources (e.g. 'lacuna_ghana', 'nigeria_cohort', etc.),
    standardizes heterogeneous annotation formats, and aggregates samples into a unified
    dataset with source-level provenance.
    """

    # Official Lacuna Ghana class mappings
    THICK_CLASSES = {0: "Parasite", 1: "White Blood Cells"}
    THIN_CLASSES = {
        0: "Trophozoites",
        1: "Ring stage",
        2: "Gametocytes",
        3: "White Blood Cells",
        4: "Artifacts",
        5: "Schizont"
    }

    THICK_PARASITE_IDS: Set[int] = {0}
    THIN_PARASITE_IDS: Set[int] = {0, 1, 2, 5}

    # Official Adeleke Nigeria class mappings (PMC11827117 / CVAT)
    ADELEKE_CLASSES = {
        0: "Plasmodium falciparum",
        1: "Plasmodium malariae",
        2: "Plasmodium ovale",
        4: "Plasmodium parasite"
    }
    ADELEKE_PARASITE_IDS: Set[int] = {0, 1, 2, 4}

    POSITIVE_FOLDER_NAMES: Set[str] = {
        "positive", "pos", "parasite", "parasites", "infected", "malaria", "trophozoite",
        "slides_with_positive_cells"
    }
    NEGATIVE_FOLDER_NAMES: Set[str] = {
        "negative", "neg", "uninfected", "clean", "normal", "healthy", "non_parasite",
        "slides_with_negative_cells"
    }

    def __init__(
        self,
        data_root: Union[str, Path],
        smear_type: str = "thick",
        data_sources: Optional[List[str]] = None,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Args:
            data_root: Root data directory (e.g., 'data/raw').
            smear_type: 'thick' or 'thin'.
            data_sources: Optional list of source subfolder names to include. If None, auto-discovers all.
            shuffle: Whether to shuffle pooled samples across sources before inference.
            seed: Deterministic random seed for shuffling.
        """
        self.data_root = Path(data_root)
        self.smear_type = smear_type.lower()
        if self.smear_type not in ["thin", "thick"]:
            raise ValueError(f"smear_type must be 'thin' or 'thick', got '{smear_type}'")

        self.data_sources = data_sources
        self.shuffle = shuffle
        self.seed = seed

        self.parasite_ids = self.THICK_PARASITE_IDS if self.smear_type == "thick" else self.THIN_PARASITE_IDS
        self.class_map = self.THICK_CLASSES if self.smear_type == "thick" else self.THIN_CLASSES
        self.annotations = self._discover_and_index()

    def _discover_and_index(self) -> List[Dict]:
        """Discovers data sources and indexes all matching micrographs."""
        if not self.data_root.exists():
            logger.warning("Data root '%s' does not exist.", self.data_root)
            return []

        # Detect whether data_root contains multi-source subdirectories
        source_dirs = self._find_source_directories()
        if not source_dirs:
            logger.warning("No %s smear data discovered in %s", self.smear_type, self.data_root)
            return []

        all_records = []
        for src_name, smear_dir in source_dirs.items():
            records = self._index_smear_directory(src_name, smear_dir)
            logger.info("  [%s] Discovered %d %s smear micrographs (Positives: %d, Negatives: %d)",
                        src_name, len(records), self.smear_type,
                        sum(r["ground_truth"] for r in records),
                        len(records) - sum(r["ground_truth"] for r in records))
            all_records.extend(records)

        # Deterministic shuffle across sources at inference time
        if self.shuffle and len(all_records) > 0:
            rng = random.Random(self.seed)
            rng.shuffle(all_records)
            logger.info("Deterministically shuffled %d total %s micrographs (seed=%d)",
                        len(all_records), self.smear_type, self.seed)

        return all_records

    def _find_source_directories(self) -> Dict[str, Path]:
        """Maps data_source_name -> path_to_smear_dir."""
        target_names = [f"{self.smear_type}_smear", self.smear_type]
        source_map = {}

        # Case 1: Direct layout (e.g. data/raw/thick_smear or data/raw/thick)
        for tname in target_names:
            direct_dir = self.data_root / tname
            if direct_dir.exists():
                source_map["lacuna_ghana"] = direct_dir
                break

        # Case 2: Multi-source layout (e.g. data/raw/lacuna_ghana/thick_smear, data/raw/adeleke_nigeria/thick_smear)
        for child in sorted(self.data_root.iterdir()):
            if child.is_dir() and child.name not in target_names and not child.name.startswith("."):
                for tname in target_names:
                    sub_smear = child / tname
                    if sub_smear.exists():
                        source_name = child.name
                        if self.data_sources is None or source_name in self.data_sources:
                            source_map[source_name] = sub_smear
                        break

        return source_map

    def _index_smear_directory(self, source_name: str, smear_dir: Path) -> List[Dict]:
        """Indexes all images and resolves annotations for a given source directory."""
        records = []
        image_extensions = {".jpg", ".jpeg", ".png"}

        # Search for images recursively
        img_paths = [p for p in smear_dir.rglob("*") if p.suffix.lower() in image_extensions and not p.name.startswith(".")]

        # Look for CSV manifest in smear_dir or parent
        csv_label_map = self._load_csv_manifest(smear_dir)

        # Search paths for YOLO or VOC label files
        possible_label_dirs = [
            smear_dir / "labels_yolo",
            smear_dir / "labels",
            smear_dir / "annotations",
            smear_dir
        ]

        for img_path in sorted(img_paths):
            stem = img_path.stem
            annot_path = None
            annot_type = None

            # 1. Try YOLO format
            for ldir in possible_label_dirs:
                cand = ldir / f"{stem}.txt"
                if cand.exists() and cand.name != "label.txt":
                    annot_path = cand
                    annot_type = "yolo"
                    break

            # 2. Try Pascal VOC format
            if annot_path is None:
                for ldir in possible_label_dirs:
                    cand = ldir / f"{stem}.xml"
                    if cand.exists():
                        annot_path = cand
                        annot_type = "voc"
                        break

            parent_folder = img_path.parent.name.lower()
            stem_lower = stem.lower()
            is_pos_prefix = stem_lower.startswith("pos_") or stem_lower.startswith("pos-")
            is_neg_prefix = stem_lower.startswith("neg_") or stem_lower.startswith("neg-")

            ground_truth = 0
            parasite_count = 0
            wbc_count = 0
            artifact_count = 0
            boxes = []

            if annot_path:
                boxes, parasite_count, wbc_count, artifact_count = self._parse_annotation(annot_path, annot_type, source_name=source_name)
                if parasite_count > 0 or parent_folder in self.POSITIVE_FOLDER_NAMES or is_pos_prefix:
                    ground_truth = 1
                elif parent_folder in self.NEGATIVE_FOLDER_NAMES or is_neg_prefix:
                    ground_truth = 0
                else:
                    ground_truth = 1 if parasite_count > 0 else 0
            elif csv_label_map and (stem in csv_label_map or img_path.name in csv_label_map):
                annot_type = "csv_manifest"
                ground_truth = csv_label_map.get(stem, csv_label_map.get(img_path.name))
            else:
                # 4. Check folder name hierarchy (binary folders: positive/ vs negative/) or file prefix
                if parent_folder in self.POSITIVE_FOLDER_NAMES or is_pos_prefix:
                    annot_type = "folder_binary"
                    ground_truth = 1
                elif parent_folder in self.NEGATIVE_FOLDER_NAMES or is_neg_prefix:
                    annot_type = "folder_binary"
                    ground_truth = 0
                else:
                    # Unlabeled / default
                    annot_type = "unlabeled"
                    ground_truth = 0

            records.append({
                "image_id": stem,
                "image_path": str(img_path),
                "data_source": source_name,
                "smear_type": self.smear_type,
                "annotation_path": str(annot_path) if annot_path else None,
                "annotation_type": annot_type,
                "ground_truth": int(ground_truth),
                "parasite_count": parasite_count,
                "wbc_count": wbc_count,
                "artifact_count": artifact_count,
                "total_boxes": len(boxes),
                "boxes": boxes
            })

        return records

    def _load_csv_manifest(self, smear_dir: Path) -> Dict[str, int]:
        """Loads optional CSV label map if present."""
        candidates = [
            smear_dir / "labels.csv",
            smear_dir / "annotations.csv",
            smear_dir / "metadata.csv",
            smear_dir.parent / "labels.csv"
        ]
        label_map = {}
        for c in candidates:
            if c.exists():
                try:
                    df = pd.read_csv(c)
                    cols = [col.lower() for col in df.columns]
                    # Identify image column and label column
                    id_col = next((df.columns[i] for i, col in enumerate(cols) if any(k in col for k in ["image", "id", "file", "filename"])), None)
                    lbl_col = next((df.columns[i] for i, col in enumerate(cols) if any(k in col for k in ["label", "gt", "malaria", "status", "infected"])), None)

                    if id_col and lbl_col:
                        for _, row in df.iterrows():
                            key = str(row[id_col]).strip()
                            raw_val = str(row[lbl_col]).strip().lower()
                            val = 1 if raw_val in ["1", "true", "positive", "pos", "parasite", "infected"] else 0
                            label_map[Path(key).stem] = val
                            label_map[key] = val
                        logger.info("Loaded CSV annotation manifest with %d entries from %s", len(label_map), c)
                        break
                except Exception as e:
                    logger.warning("Could not parse CSV manifest at %s: %s", c, e)
        return label_map

    def _parse_annotation(
        self,
        annot_path: Optional[Path],
        annot_type: Optional[str],
        source_name: str = ""
    ) -> Tuple[List[Dict], int, int, int]:
        """Parses annotations from file with source-aware class ontology."""
        if annot_path is None or not annot_path.exists():
            return [], 0, 0, 0

        boxes = []
        parasite_count = 0
        wbc_count = 0
        artifact_count = 0

        # Resolve ontology
        if "adeleke" in source_name.lower():
            target_parasite_ids = self.ADELEKE_PARASITE_IDS
            target_class_map = self.ADELEKE_CLASSES
        else:
            target_parasite_ids = self.parasite_ids
            target_class_map = self.class_map

        if annot_type == "yolo":
            try:
                content = annot_path.read_text(encoding="utf-8", errors="ignore").strip()
                if content:
                    for line in content.splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        coords = [float(x) for x in parts[1:5]] if len(parts) >= 5 else []
                        boxes.append({"class_id": cls_id, "bbox_yolo": coords})

                        if cls_id in target_parasite_ids:
                            parasite_count += 1
                        elif "white" in target_class_map.get(cls_id, "").lower():
                            wbc_count += 1
                        elif "artifact" in target_class_map.get(cls_id, "").lower():
                            artifact_count += 1
            except Exception as e:
                logger.warning("Failed parsing YOLO file %s: %s", annot_path, e)

        elif annot_type == "voc":
            try:
                tree = ET.parse(str(annot_path))
                root = tree.getroot()
                for obj in root.findall("object"):
                    name = obj.find("name")
                    label = name.text.strip().lower() if name is not None and name.text else ""
                    bndbox = obj.find("bndbox")
                    bbox = []
                    if bndbox is not None:
                        bbox = [
                            float(bndbox.find("xmin").text),
                            float(bndbox.find("ymin").text),
                            float(bndbox.find("xmax").text),
                            float(bndbox.find("ymax").text)
                        ]
                    boxes.append({"label": label, "bbox_pascal": bbox})

                    if any(p in label for p in ["parasite", "trophozoite", "ring", "gametocyte", "schizont"]):
                        parasite_count += 1
                    elif "white" in label or "wbc" in label:
                        wbc_count += 1
                    elif "artifact" in label:
                        artifact_count += 1
            except Exception as e:
                logger.warning("Failed parsing VOC XML %s: %s", annot_path, e)

        return boxes, parasite_count, wbc_count, artifact_count

    def to_dataframe(self) -> pd.DataFrame:
        """Exports indexed cohort metadata to DataFrame."""
        return pd.DataFrame(self.annotations)

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        return self.annotations[idx]


# Backwards compatibility alias
LacunaGhanaDataset = MultiSourceMalariaDataset
