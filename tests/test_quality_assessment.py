"""
Unit tests for optical quality assessment engine.
"""

import numpy as np
import cv2
import pytest
from src.quality_assessment import ImageQualityAssessor


def test_circular_fov_mask():
    """Verify FOV mask isolates bright circle from black background."""
    assessor = ImageQualityAssessor()

    # Create synthetic image: 100x100 black image with a bright circle in center
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(img, (50, 50), 30, 200, -1)

    mask = assessor.detect_circular_fov_mask(img)
    assert mask[50, 50] == 255  # Center inside circle
    assert mask[5, 5] == 0      # Corner outside circle


def test_laplacian_variance_sharp_vs_blur():
    """Sharp synthetic edges must produce strictly higher Laplacian variance than blurred edges."""
    assessor = ImageQualityAssessor()

    # Sharp image with high-contrast checkerboard
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[::10, :] = 255
    sharp[:, ::10] = 255

    # Gaussian-blurred version of the same image
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    var_sharp = assessor.laplacian_variance(sharp)
    var_blur = assessor.laplacian_variance(blurred)

    assert var_sharp > var_blur
    assert var_blur > 0.0


def test_michelson_contrast():
    """Verify Michelson contrast computation."""
    assessor = ImageQualityAssessor()

    # Image with min=50, max=150 -> contrast = (150-50)/(150+50) = 100/200 = 0.5
    img = np.full((50, 50), 50, dtype=np.uint8)
    img[20:30, 20:30] = 150

    contrast = assessor.michelson_contrast(img)
    assert contrast == pytest.approx(0.5, abs=1e-3)
