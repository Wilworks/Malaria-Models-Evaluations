"""
Base Abstract Evaluator Wrapper Interface.
Establishes a unified predict(image) API contract across classification and object detection architectures.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseModelWrapper(ABC):
    """Abstract base class for all benchmark model evaluation wrappers."""

    def __init__(self, model_name: str, model_path: str, task_type: str = "classification"):
        self.model_name = model_name
        self.model_path = model_path
        self.task_type = task_type.lower()

    @abstractmethod
    def load_model(self) -> None:
        """Loads model weights into memory."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Runs inference on an image array.

        Returns:
            Dict containing at minimum:
                - 'model_name': str
                - 'predicted_class': int (0 for negative, 1 for positive)
                - 'confidence': float (between 0.0 and 1.0)
        """
        pass
