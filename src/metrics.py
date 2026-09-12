"""
Evaluation Metrics Engine.
Computes classification metrics (Sensitivity, Specificity, Precision, F1, AUC-ROC)
and object detection metrics (mAP@0.5, mAP@[0.5:0.95]).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score 95% confidence interval for a proportion."""
    if trials == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% CI
    p = successes / trials
    denominator = 1 + (z**2) / trials
    centre = (p + (z**2) / (2 * trials)) / denominator
    spread = (z * np.sqrt((p * (1 - p) + (z**2) / (4 * trials)) / trials)) / denominator
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return float(lower), float(upper)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Helper wrapper around ClassificationMetrics.compute_binary_metrics."""
    return ClassificationMetrics.compute_binary_metrics(y_true, y_pred, y_prob)



class ClassificationMetrics:
    """Computes standard medical diagnostic metrics."""

    @staticmethod
    def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Computes Sensitivity (Recall), Specificity, Precision, F1-Score, and AUC-ROC.
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        
        auc = 0.0
        if y_prob is not None and len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))

        return {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "precision": float(precision),
            "f1_score": float(f1),
            "auc_roc": float(auc),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }


class ObjectDetectionMetrics:
    """Computes mean Average Precision (mAP) for object detection models."""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculates Intersection over Union (IoU) between two bounding boxes [xmin, ymin, xmax, ymax]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
