"""
Unit tests for clinical diagnostic and object detection metrics.
"""

import unittest
import numpy as np
from src.metrics import (
    wilson_score_interval,
    ClassificationMetrics,
    ObjectDetectionMetrics,
    calculate_metrics
)


class TestMetrics(unittest.TestCase):
    def test_wilson_score_interval_bounds(self):
        """Wilson Score interval must always lie strictly within [0.0, 1.0]."""
        # Edge case: zero trials
        low, high = wilson_score_interval(0, 0)
        self.assertEqual(low, 0.0)
        self.assertEqual(high, 0.0)

        # Edge case: 0 out of 100
        low, high = wilson_score_interval(0, 100)
        self.assertTrue(0.0 <= low <= high <= 1.0)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.0)

        # Edge case: 100 out of 100
        low, high = wilson_score_interval(100, 100)
        self.assertTrue(0.0 <= low <= high <= 1.0)
        self.assertAlmostEqual(high, 1.0)
        self.assertLess(low, 1.0)

        # Typical case
        low, high = wilson_score_interval(70, 100)
        self.assertTrue(0.60 < low < 0.70)
        self.assertTrue(0.70 < high < 0.80)

    def test_classification_metrics_perfect(self):
        """Verify metrics on a perfect diagnostic prediction."""
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        y_prob = np.array([0.9, 0.85, 0.95, 0.1, 0.05, 0.2])

        res = ClassificationMetrics.compute_binary_metrics(y_true, y_pred, y_prob)

        self.assertEqual(res["sensitivity"], 1.0)
        self.assertEqual(res["specificity"], 1.0)
        self.assertEqual(res["precision"], 1.0)
        self.assertEqual(res["npv"], 1.0)
        self.assertEqual(res["f1_score"], 1.0)
        self.assertEqual(res["accuracy"], 1.0)
        self.assertEqual(res["tp"], 3)
        self.assertEqual(res["tn"], 3)
        self.assertEqual(res["fp"], 0)
        self.assertEqual(res["fn"], 0)
        self.assertEqual(res["auc_roc"], 1.0)

    def test_classification_metrics_zero_division(self):
        """Verify metrics handle all-zero or all-one inputs gracefully without crashing."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])

        res = ClassificationMetrics.compute_binary_metrics(y_true, y_pred)
        self.assertEqual(res["sensitivity"], 0.0)
        self.assertEqual(res["specificity"], 1.0)
        self.assertEqual(res["tn"], 4)
        self.assertEqual(res["tp"], 0)

    def test_object_detection_iou(self):
        """Verify IoU calculation for bounding boxes [xmin, ymin, xmax, ymax]."""
        box_a = [0.0, 0.0, 10.0, 10.0]  # Area = 100
        box_b = [0.0, 0.0, 10.0, 10.0]  # Exact match
        self.assertAlmostEqual(ObjectDetectionMetrics.calculate_iou(box_a, box_b), 1.0)

        box_c = [20.0, 20.0, 30.0, 30.0]  # Disjoint
        self.assertAlmostEqual(ObjectDetectionMetrics.calculate_iou(box_a, box_c), 0.0)

        box_d = [5.0, 0.0, 15.0, 10.0]  # Overlap 5x10 = 50, union = 150
        self.assertAlmostEqual(ObjectDetectionMetrics.calculate_iou(box_a, box_d), 50.0 / 150.0)

    def test_object_detection_ap(self):
        """Verify Average Precision computation."""
        recalls = [0.2, 0.4, 0.6, 0.8, 1.0]
        precisions = [1.0, 1.0, 1.0, 1.0, 1.0]
        ap = ObjectDetectionMetrics.compute_ap(recalls, precisions)
        self.assertAlmostEqual(ap, 1.0)


if __name__ == "__main__":
    unittest.main()
