"""
Image Quality Proxy Metrics Module.
Implements Circular Ocular FOV Masking to isolate the biological microscope field
from dark outer vignetting caused by smartphone camera capture.
Calculates Laplacian variance (blur proxy), Michelson contrast, and SNR strictly inside the circular FOV.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ImageQualityAssessor:
    """Computes FOV-masked image quality metrics for smartphone blood smear micrographs."""

    @staticmethod
    def detect_circular_fov_mask(image: np.ndarray) -> np.ndarray:
        """
        Creates a binary mask (1 inside microscope circle, 0 outside black vignetting)
        using Otsu thresholding and morphological operations.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Threshold out pure black background
        _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        
        # Morphological closing to fill small gaps inside the circle
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask

    def laplacian_variance(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """
        Computes Laplacian variance as a focus/blur proxy metric.
        If mask is provided, computes variance strictly over pixels inside the mask.
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
        """Computes Michelson contrast inside the circular FOV mask."""
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
        return (i_max - i_min) / (i_max + i_min)

    def signal_to_noise_ratio(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
        """Computes SNR (mean / std) inside the circular FOV mask."""
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
        return mean / std

    def assess_image(self, image_path: str) -> Dict[str, float]:
        """Loads an image, detects the circular ocular mask, and computes FOV-masked quality proxies."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {image_path}")

        mask = self.detect_circular_fov_mask(img)
        fov_coverage_ratio = float(np.sum(mask > 0) / (img.shape[0] * img.shape[1]))

        return {
            "blur_laplacian": self.laplacian_variance(img, mask=mask),
            "michelson_contrast": self.michelson_contrast(img, mask=mask),
            "snr": self.signal_to_noise_ratio(img, mask=mask),
            "fov_coverage_ratio": fov_coverage_ratio
        }
