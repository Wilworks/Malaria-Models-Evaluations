"""
Dataset loader and parser for the Ghanaian subset of the Lacuna Malaria Dataset.
Source: Princess Marie Louise Hospital, Accra, Ghana (minoHealth AI Labs collection).
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np


class LacunaGhanaDataset:
    """Parser and PyTorch/TF adapter for Lacuna Ghana blood smear images."""

    THIN_CLASSES = ["gametocyte", "trophozoite", "ring", "white_blood_cell", "artifact"]
    THICK_CLASSES = ["parasite", "white_blood_cell"]

    def __init__(self, data_root: str, smear_type: str = "thin"):
        """
        Args:
            data_root: Path to raw dataset root (e.g. data/raw)
            smear_type: 'thin' or 'thick'
        """
        self.data_root = Path(data_root)
        self.smear_type = smear_type.lower()
        if self.smear_type not in ["thin", "thick"]:
            raise ValueError("smear_type must be either 'thin' or 'thick'")
            
        self.img_dir = self.data_root / f"{self.smear_type}_smear"
        self.annotations = self._load_annotation_index()

    def _load_annotation_index(self) -> List[Dict]:
        """Indexes all images and corresponding annotations in the directory."""
        if not self.img_dir.exists():
            print(f"[Warning] Directory {self.img_dir} does not exist yet. Ensure dataset is downloaded.")
            return []

        records = []
        image_extensions = {".jpg", ".jpeg", ".png"}
        
        for img_path in self.img_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                xml_path = img_path.with_suffix(".xml")
                json_path = img_path.with_suffix(".json")
                
                annot_file = None
                if xml_path.exists():
                    annot_file = xml_path
                elif json_path.exists():
                    annot_file = json_path

                records.append({
                    "image_id": img_path.stem,
                    "image_path": str(img_path),
                    "annotation_path": str(annot_file) if annot_file else None,
                    "smear_type": self.smear_type
                })
        return records

    def __len__(self) -> int:
        return len(self.annotations)

    def parse_pascal_voc(self, xml_path: str) -> List[Dict]:
        """Parses Pascal VOC XML annotations into bounding box dicts."""
        boxes = []
        if not xml_path or not os.path.exists(xml_path):
            return boxes

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            label = obj.find("name").text
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            boxes.append({
                "label": label,
                "bbox": [xmin, ymin, xmax, ymax]
            })
        return boxes
