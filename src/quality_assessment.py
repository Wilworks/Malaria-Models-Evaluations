"""
Optical Quality Assessment Engine for Smartphone-Captured Blood Smears.
Implements circular microscope Field of View (FOV) masking to isolate biological specimens
from dark outer vignetting, and computes Laplacian blur variance, Michelson contrast, and SNR.
"""

import logging
from typing import Dict, Optional, Union
from pathlib import Path
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImageQualityAssessor:
    """Computes physics-grounded optical quality metrics on microscopy micrographs."""

    @staticmethod
    def detect_circular_fov_mask(image: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask isolating the circular microscope field of view
        from black outer vignetting caused by mobile phone eyepiece adapters.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold out black vignetted borders
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Morphological ellipse closing and opening to eliminate noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def laplacian_variance(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Computes Laplacian variance (focus/sharpness proxy).
        Higher values indicate sharp, in-focus cellular structures;
        lower values indicate optical defocus blur.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        if mask is not None:
            valid_pixels = laplacian[mask > 0]
            if len(valid_pixels) == 0:
                return 0.0
            return float(np.var(valid_pixels))

        return float(laplacian.var())

    def michelson_contrast(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Computes Michelson contrast: (I_max - I_min) / (I_max + I_min).
        Evaluated strictly within the illuminated circular FOV.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if mask is not None:
            valid_pixels = gray[mask > 0]
            if len(valid_pixels) == 0:
                return 0.0
            i_max = float(np.max(valid_pixels))
            i_min = float(np.min(valid_pixels))
        else:
            i_max = float(np.max(gray))
            i_min = float(np.min(gray))

        if i_max + i_min == 0:
            return 0.0
        return float((i_max - i_min) / (i_max + i_min))

    def signal_to_noise_ratio(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Computes Signal-to-Noise Ratio (mean / std) inside the circular FOV."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if mask is not None:
            valid_pixels = gray[mask > 0]
            if len(valid_pixels) == 0:
                return 0.0
            mean = float(np.mean(valid_pixels))
            std = float(np.std(valid_pixels))
        else:
            mean = float(np.mean(gray))
            std = float(np.std(gray))

        if std == 0:
            return 0.0
        return float(mean / std)

    def assess_image(self, image_input: Union[str, Path, np.ndarray]) -> Dict[str, float]:
        """
        Loads an image (or accepts an existing ndarray), computes the circular FOV mask,
        and extracts optical quality metrics.
        """
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise FileNotFoundError(f"Failed to read image at: {image_input}")
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            raise TypeError("image_input must be a file path or numpy.ndarray")

        mask = self.detect_circular_fov_mask(img)
        total_pixels = img.shape[0] * img.shape[1]
        fov_coverage_ratio = float(np.sum(mask > 0) / total_pixels) if total_pixels > 0 else 0.0

        return {
            "blur_laplacian": round(self.laplacian_variance(img, mask=mask), 2),
            "michelson_contrast": round(self.michelson_contrast(img, mask=mask), 4),
            "snr": round(self.signal_to_noise_ratio(img, mask=mask), 4),
            "fov_coverage_ratio": round(fov_coverage_ratio, 4)
        }
