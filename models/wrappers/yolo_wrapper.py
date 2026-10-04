"""
Wrapper Adapter for Ultralytics YOLOv8 Malaria Detection Models.
Handles PyTorch (.pt) weights, class-specific filtering (parasites vs. WBCs/artifacts),
and slide-level clinical diagnosis aggregation.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from models.wrappers.base_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)


class YOLOMalariaWrapper(BaseModelWrapper):
    """
    Adapter for YOLOv8 object detection checkpoints.
    
    Distinguishes target malaria parasites from host white blood cells (WBCs)
    and artifacts, preventing false-positive slide diagnoses from benign host cells.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str,
        confidence_threshold: float = 0.15,
        parasite_class_ids: Optional[List[int]] = None,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Args:
            model_name: Identifier for the model.
            model_path: Path to model checkpoint file (.pt).
            confidence_threshold: Minimum detection confidence threshold.
            parasite_class_ids: List of class IDs that denote malaria parasites.
                                Defaults to [0] (Trophozoite in fbononibelloepoch).
            class_names: Optional mapping from class ID to human-readable label.
        """
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            task_type="detection"
        )
        self.conf_thresh = confidence_threshold
        # Default: Class 0 is Trophozoite (parasite), Class 1 is WBC (host cell)
        self.parasite_class_ids = set(parasite_class_ids if parasite_class_ids is not None else [0])
        self.class_names = class_names or {0: "Trophozoite", 1: "WBC"}
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Loads Ultralytics YOLO model from checkpoint."""
        if YOLO is None:
            logger.error("Ultralytics library is not installed. Please install ultralytics.")
            return

        path = Path(self.model_path)
        try:
            self.model = YOLO(str(path) if path.exists() else self.model_path)
            logger.info("Loaded YOLO model '%s' from %s", self.model_name, self.model_path)
        except Exception as e:
            logger.warning("Could not load YOLO model from %s: %s", self.model_path, e)

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs object detection inference on an image array.

        Returns a dictionary containing:
            - predicted_class: 1 if ANY parasite is detected, 0 otherwise (WBCs do NOT trigger positive)
            - confidence: Max parasite confidence if positive; (1 - max_parasite_conf) if negative
            - parasite_count: Number of detected parasite objects
            - wbc_count: Number of detected white blood cells
            - detected_objects: Total detected bounding boxes
            - boxes: Detailed list of bounding boxes with labels and confidences
        """
        if self.model is None:
            return {
                "model_name": self.model_name,
                "predicted_class": 0,
                "confidence": 0.0,
                "parasite_count": 0,
                "wbc_count": 0,
                "detected_objects": 0,
                "boxes": [],
                "error": "Model not loaded"
            }

        results = self.model.predict(image, conf=self.conf_thresh, verbose=False)
        all_boxes = []
        parasite_boxes = []
        wbc_boxes = []

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy_val = box.xyxy[0]
                xyxy = xyxy_val.cpu().numpy().tolist() if hasattr(xyxy_val, "cpu") else (xyxy_val.tolist() if hasattr(xyxy_val, "tolist") else list(xyxy_val))
                
                conf_val = box.conf[0]
                conf = float(conf_val.cpu().numpy()) if hasattr(conf_val, "cpu") else float(conf_val)
                
                cls_val = box.cls[0]
                cls_id = int(cls_val.cpu().numpy()) if hasattr(cls_val, "cpu") else int(cls_val)
                label_name = self.class_names.get(cls_id, f"class_{cls_id}")

                box_info = {
                    "bbox": xyxy,
                    "confidence": round(conf, 4),
                    "class_id": cls_id,
                    "class_name": label_name,
                    "is_parasite": cls_id in self.parasite_class_ids
                }
                all_boxes.append(box_info)

                if cls_id in self.parasite_class_ids:
                    parasite_boxes.append(box_info)
                else:
                    wbc_boxes.append(box_info)

        has_parasite = len(parasite_boxes) > 0
        if has_parasite:
            max_parasite_conf = max(b["confidence"] for b in parasite_boxes)
            slide_confidence = max_parasite_conf
            predicted_class = 1
        else:
            # If no parasites detected, slide confidence is high that it's negative
            predicted_class = 0
            slide_confidence = 1.0

        return {
            "model_name": self.model_name,
            "predicted_class": predicted_class,
            "confidence": round(float(slide_confidence), 4),
            "parasite_count": len(parasite_boxes),
            "wbc_count": len(wbc_boxes),
            "detected_objects": len(all_boxes),
            "boxes": all_boxes
        }
