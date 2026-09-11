"""
Wrapper Adapter for Ultralytics YOLOv8 Malaria Object Detection Models
(kossisoroyce, Mmpk, fbononibelloepoch).
Handles PyTorch (.pt) weights and bounding box object detection outputs.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from ultralytics import YOLO
from models.wrappers.base_wrapper import BaseModelWrapper


class YOLOMalariaWrapper(BaseModelWrapper):
    """Adapter for YOLOv8 object detection model checkpoints."""

    def __init__(self, model_name: str, model_path: str, confidence_threshold: float = 0.15):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            task_type="detection"
        )
        self.conf_thresh = confidence_threshold
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """Loads Ultralytics YOLO model from checkpoint or Hugging Face repo."""
        path = Path(self.model_path)
        try:
            self.model = YOLO(str(path) if path.exists() else self.model_path)
            print(f"[Success] Loaded YOLO model '{self.model_name}' from {self.model_path}")
        except Exception as e:
            print(f"[Warning] Could not load YOLO model from {self.model_path}: {e}")

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Runs object detection inference on input image array."""
        if self.model is None:
            return {"boxes": [], "predicted_class": 0, "confidence": 0.0, "error": "Model not loaded"}

        results = self.model.predict(image, conf=self.conf_thresh, verbose=False)
        boxes_list = []
        has_detection = False
        max_conf = 0.0

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                boxes_list.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "class_id": cls_id
                })
                if conf > max_conf:
                    max_conf = conf
                has_detection = True

        return {
            "model_name": self.model_name,
            "predicted_class": 1 if has_detection else 0,
            "confidence": max_conf,
            "detected_objects": len(boxes_list),
            "boxes": boxes_list
        }
