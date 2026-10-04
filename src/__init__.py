"""
Malaria Models Evaluation Benchmark Suite.
Empirical Cross-Domain Generalization and Deployment Safety Auditing Engine.
"""

from src.data_loader import MultiSourceMalariaDataset, LacunaGhanaDataset
from src.metrics import (
    ClassificationMetrics,
    ObjectDetectionMetrics,
    wilson_score_interval,
    calculate_metrics
)
from src.quality_assessment import ImageQualityAssessor
from src.deployment_safety import DeploymentSafetyEvaluator
from src.error_analysis import ErrorAnalyzer

__version__ = "1.0.0"
__author__ = "Wilfred Ayine Asumboya"

__all__ = [
    "MultiSourceMalariaDataset",
    "LacunaGhanaDataset",
    "ClassificationMetrics",
    "ObjectDetectionMetrics",
    "wilson_score_interval",
    "calculate_metrics",
    "ImageQualityAssessor",
    "DeploymentSafetyEvaluator",
    "ErrorAnalyzer",
]
