"""
Base Abstract Evaluator Wrapper Interface.
Establishes a unified predict(image) API contract for classification and detection models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


class BaseModelWrapper(ABC):
    """Abstract base class for all benchmark model evaluation wrappers."""

    def __init__(self, model_name: str, model_path: str, task_type: str = "classification"):
        self.model_name = model_name
        self.model_path = model_path
        self.task_type = task_type.lower()  # 'classification' or 'detection'

    @abstractmethod
    def load_model(self) -> None:
        """Loads model weights into memory."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on an image array (BGR or RGB).
        Returns:
            Dict containing:
                - 'predicted_class': int or str
                - 'confidence': float
                - 'boxes': Optional[List[List[float]]] for object detection
        """
        pass
