"""
Stratified Error Analysis Engine (Answers RQ2).
Stratifies failure modes across:
1. Object class (parasite vs. WBC / artifact; ring vs. trophozoite vs. gametocyte)
2. Image quality strata (low/medium/high focus blur & contrast)
3. Smear type (thick vs. thin smear)
"""

import pandas as pd
import numpy as np
from typing import Dict, List


class ErrorAnalyzer:
    """Stratifies evaluation errors to pinpoint failure mechanics."""

    def __init__(self, predictions_df: pd.DataFrame):
        """
        Args:
            predictions_df: DataFrame containing columns:
                ['image_id', 'smear_type', 'y_true', 'y_pred', 'blur_laplacian', 'michelson_contrast', 'class_name']
        """
        self.df = predictions_df

    def stratify_by_quality(self, metric_col: str = "blur_laplacian", bins: int = 3) -> pd.DataFrame:
        """Stratifies model accuracy/sensitivity across image quality bins."""
        if self.df.empty or metric_col not in self.df.columns:
            return pd.DataFrame()

        self.df["quality_strata"] = pd.qcut(self.df[metric_col], q=bins, labels=["Low", "Medium", "High"])
        
        grouped = self.df.groupby("quality_strata").apply(
            lambda g: pd.Series({
                "count": len(g),
                "accuracy": (g["y_true"] == g["y_pred"]).mean(),
                "false_negatives": ((g["y_true"] == 1) & (g["y_pred"] == 0)).sum(),
                "false_positives": ((g["y_true"] == 0) & (g["y_pred"] == 1)).sum()
            })
        )
        return grouped

    def stratify_by_class(self) -> pd.DataFrame:
        """Stratifies performance by specific target object class."""
        if "class_name" not in self.df.columns:
            return pd.DataFrame()

        grouped = self.df.groupby("class_name").apply(
            lambda g: pd.Series({
                "total_instances": len(g),
                "correct": (g["y_true"] == g["y_pred"]).sum(),
                "recall": (g["y_true"] == g["y_pred"]).mean()
            })
        )
        return grouped
