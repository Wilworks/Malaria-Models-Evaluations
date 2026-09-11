"""
Image Quality Proxy Metrics Module.
Calculates Laplacian variance (blur proxy), Michelson contrast, and Signal-to-Noise Ratio (SNR)
to stratify model performance against smartphone-camera capture degradation.
"""

import cv2
import numpy as np
from typing import Dict, Union


class ImageQualityAssessor:
    """Computes proxy image quality metrics for blood smear micrographs."""

    @staticmethod
    def laplacian_variance(image: np.ndarray) -> float:
        """
        Computes Laplacian variance as a focus/blur proxy metric.
        Lower values indicate heavier focus blur.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def michelson_contrast(image: np.ndarray) -> float:
        """Computes Michelson contrast: (I_max - I_min) / (I_max + I_min)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        i_max = float(np.max(gray))
        i_min = float(np.min(gray))
        if i_max + i_min == 0:
            return 0.0
        return (i_max - i_min) / (i_max + i_min)

    @staticmethod
    def signal_to_noise_ratio(image: np.ndarray) -> float:
        """Computes mean image intensity divided by standard deviation."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        if std == 0:
            return 0.0
        return mean / std

    def assess_image(self, image_path: str) -> Dict[str, float]:
        """Loads an image and computes all quality proxies."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")

        return {
            "blur_laplacian": self.laplacian_variance(img),
            "michelson_contrast": self.michelson_contrast(img),
            "snr": self.signal_to_noise_ratio(img)
        }
