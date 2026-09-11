"""
Stratified Error Analysis Engine (Answers RQ2).
Stratifies failure modes across:
1. Object class (parasite vs. WBC / artifact)
2. Image quality strata (low/medium/high focus blur & contrast)
3. Smear type (thick vs. thin smear)
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class ErrorAnalyzer:
    """Stratifies evaluation errors to pinpoint failure mechanics."""

    def __init__(self, predictions_df: pd.DataFrame):
        self.df = predictions_df

    def stratify_by_quality(self, metric_col: str = "blur_laplacian", bins: int = 3) -> pd.DataFrame:
        """Stratifies model performance & detection confidence across image quality strata (Low, Medium, High)."""
        if self.df.empty or metric_col not in self.df.columns:
            return pd.DataFrame()

        try:
            self.df["quality_strata"] = pd.qcut(self.df[metric_col], q=bins, labels=["Low", "Medium", "High"], duplicates="drop")
            
            group_cols = ["quality_strata"]
            if "model_name" in self.df.columns:
                group_cols = ["model_name", "quality_strata"]

            grouped = self.df.groupby(group_cols, observed=False).apply(
                lambda g: pd.Series({
                    "sample_count": len(g),
                    "mean_blur_laplacian": round(g["blur_laplacian"].mean(), 2) if "blur_laplacian" in g.columns else 0.0,
                    "mean_michelson_contrast": round(g["michelson_contrast"].mean(), 4) if "michelson_contrast" in g.columns else 0.0,
                    "mean_confidence": round(g["confidence"].mean(), 4) if "confidence" in g.columns else 0.0,
                    "positive_detections": (g["predicted_class"] == 1).sum() if "predicted_class" in g.columns else 0,
                    "mean_detected_objects": round(g["detected_objects"].mean(), 2) if "detected_objects" in g.columns else 0.0
                })
            ).reset_index()
            return grouped
        except Exception as e:
            print(f"[Warning] Stratification error: {e}")
            return pd.DataFrame()
