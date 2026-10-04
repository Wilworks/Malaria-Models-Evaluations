"""
Unit tests for YOLO malaria wrapper logic.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from models.wrappers.yolo_wrapper import YOLOMalariaWrapper


class MockBox:
    def __init__(self, cls_id: int, conf: float, xyxy: list):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class MockResults:
    def __init__(self, boxes: list):
        self.boxes = boxes


def test_yolo_wrapper_wbc_rejection():
    """
    CRITICAL TEST: Ensures that if YOLO detects a White Blood Cell (Class 1),
    the slide diagnosis remains negative (0), NOT false-positive.
    """
    wrapper = YOLOMalariaWrapper.__new__(YOLOMalariaWrapper)
    wrapper.model_name = "test_yolo"
    wrapper.model_path = "mock.pt"
    wrapper.task_type = "detection"
    wrapper.conf_thresh = 0.15
    wrapper.parasite_class_ids = {0}
    wrapper.class_names = {0: "Trophozoite", 1: "WBC"}

    # Mock YOLO model detecting ONLY a WBC (class 1)
    wbc_box = MockBox(cls_id=1, conf=0.88, xyxy=[10.0, 10.0, 50.0, 50.0])
    mock_model = MagicMock()
    mock_model.predict.return_value = [MockResults(boxes=[wbc_box])]
    wrapper.model = mock_model

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    res = wrapper.predict(dummy_image)

    assert res["predicted_class"] == 0  # Must be negative!
    assert res["parasite_count"] == 0
    assert res["wbc_count"] == 1
    assert res["detected_objects"] == 1


def test_yolo_wrapper_parasite_positive():
    """Verifies that parasite detection (Class 0) triggers a positive slide diagnosis."""
    wrapper = YOLOMalariaWrapper.__new__(YOLOMalariaWrapper)
    wrapper.model_name = "test_yolo"
    wrapper.model_path = "mock.pt"
    wrapper.task_type = "detection"
    wrapper.conf_thresh = 0.15
    wrapper.parasite_class_ids = {0}
    wrapper.class_names = {0: "Trophozoite", 1: "WBC"}

    parasite_box = MockBox(cls_id=0, conf=0.92, xyxy=[20.0, 20.0, 40.0, 40.0])
    wbc_box = MockBox(cls_id=1, conf=0.75, xyxy=[60.0, 60.0, 90.0, 90.0])
    mock_model = MagicMock()
    mock_model.predict.return_value = [MockResults(boxes=[parasite_box, wbc_box])]
    wrapper.model = mock_model

    dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
    res = wrapper.predict(dummy_image)

    assert res["predicted_class"] == 1
    assert res["parasite_count"] == 1
    assert res["wbc_count"] == 1
    assert res["detected_objects"] == 2
    assert res["confidence"] == pytest.approx(0.92)
