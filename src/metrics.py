"""
Clinical Evaluation Metrics Engine for Malaria Diagnostic Benchmark.
Computes diagnostic accuracy, sensitivity, specificity with 95% Wilson Score Confidence Intervals,
and object detection metrics (IoU, average precision).
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes two-sided Wilson score confidence interval for a binomial proportion.
    
    Robust for small sample sizes and proportions near 0 or 1.
    Recommended by CLSI EP12-A2 for medical diagnostic evaluation.
    """
    if trials <= 0:
        return 0.0, 0.0

    # Z-scores: 1.95996 for 95%, 2.57583 for 99%
    z = 1.95996 if abs(confidence - 0.95) < 1e-4 else float(np.abs(np.percentile(np.random.normal(0, 1, 100000), 100 * (1 - (1 - confidence) / 2))))
    p = successes / trials
    denominator = 1.0 + (z**2) / trials
    centre = (p + (z**2) / (2.0 * trials)) / denominator
    spread = (z * np.sqrt((p * (1.0 - p) + (z**2) / (4.0 * trials)) / trials)) / denominator
    
    lower = max(0.0, centre - spread)
    upper = min(1.0, centre + spread)
    return float(lower), float(upper)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Union[float, int, str]]:
    """Convenience wrapper around ClassificationMetrics.compute_binary_metrics."""
    return ClassificationMetrics.compute_binary_metrics(y_true, y_pred, y_prob)


class ClassificationMetrics:
    """Computes comprehensive clinical diagnostic metrics and epidemiological rates."""

    @staticmethod
    def compute_binary_metrics(
        y_true: Union[np.ndarray, List[int]],
        y_pred: Union[np.ndarray, List[int]],
        y_prob: Optional[Union[np.ndarray, List[float]]] = None
    ) -> Dict[str, Union[float, int, str]]:
        """
        Computes clinical diagnostic metrics:
            - Sensitivity (Recall, True Positive Rate) + 95% Wilson CI
            - Specificity (True Negative Rate) + 95% Wilson CI
            - Precision (Positive Predictive Value)
            - Negative Predictive Value (NPV)
            - False Positive Rate (1 - Specificity)
            - False Negative Rate (1 - Sensitivity)
            - F1-Score (Harmonic mean of precision and sensitivity)
            - Diagnostic Accuracy and Balanced Accuracy
            - Diagnostic Odds Ratio (DOR)
            - AUC-ROC (if prediction probabilities are provided)
            - Contingency counts (TP, FP, TN, FN)
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        n_total = len(y_true)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
        accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
        balanced_accuracy = 0.5 * (sensitivity + specificity)

        # Confidence intervals
        sens_ci_low, sens_ci_high = wilson_score_interval(int(tp), int(tp + fn))
        spec_ci_low, spec_ci_high = wilson_score_interval(int(tn), int(tn + fp))

        # Diagnostic Odds Ratio (with Haldane-Anscombe 0.5 continuity correction)
        dor = ((tp + 0.5) * (tn + 0.5)) / ((fp + 0.5) * (fn + 0.5))

        auc = 0.0
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                y_prob = np.asarray(y_prob, dtype=float)
                auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc = 0.0

        return {
            "n_samples": int(n_total),
            "sensitivity": round(float(sensitivity), 4),
            "sensitivity_ci_95": [round(sens_ci_low, 4), round(sens_ci_high, 4)],
            "specificity": round(float(specificity), 4),
            "specificity_ci_95": [round(spec_ci_low, 4), round(spec_ci_high, 4)],
            "precision": round(float(precision), 4),
            "npv": round(float(npv), 4),
            "fpr": round(float(1.0 - specificity), 4),
            "fnr": round(float(1.0 - sensitivity), 4),
            "f1_score": round(float(f1), 4),
            "accuracy": round(float(accuracy), 4),
            "balanced_accuracy": round(float(balanced_accuracy), 4),
            "diagnostic_odds_ratio": round(float(dor), 2),
            "auc_roc": round(float(auc), 4),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }


class ObjectDetectionMetrics:
    """Computes bounding-box overlap and precision-recall metrics."""

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """
        Calculates Intersection over Union (IoU) between two bounding boxes.
        Format: [xmin, ymin, xmax, ymax]
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = area1 + area2 - intersection

        return float(intersection / union) if union > 0 else 0.0

    @staticmethod
    def compute_ap(recalls: List[float], precisions: List[float]) -> float:
        """Computes Average Precision (AP) using standard 11-point or AUC interpolation."""
        if not recalls or not precisions:
            return 0.0

        r = np.array([0.0] + list(recalls) + [1.0])
        p = np.array([0.0] + list(precisions) + [0.0])

        for i in range(len(p) - 2, -1, -1):
            p[i] = max(p[i], p[i + 1])

        idx = np.where(r[1:] != r[:-1])[0]
        ap = np.sum((r[idx + 1] - r[idx]) * p[idx + 1])
        return float(ap)
