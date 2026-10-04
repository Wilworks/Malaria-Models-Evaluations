"""
Stratified Error Analysis Engine for Malaria Diagnostic Models.
Analyzes failure modes across optical quality strata (Laplacian focus blur, contrast),
staining modalities (thick vs. thin smears), and object classes.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ErrorAnalyzer:
    """Stratifies evaluation errors to identify physical and biological failure mechanics."""

    def __init__(self, predictions_df: pd.DataFrame):
        self.df = predictions_df.copy()

    def stratify_by_quality(
        self,
        metric_col: str = "blur_laplacian",
        bins: int = 3,
        labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Stratifies model diagnostic metrics across image quality strata.
        
        Default splits into tertiles: ['Low Quality', 'Medium Quality', 'High Quality'].
        """
        if self.df.empty or metric_col not in self.df.columns:
            logger.warning("Predictions DataFrame is empty or missing column '%s'", metric_col)
            return pd.DataFrame()

        labels = labels or ["Low Quality", "Medium Quality", "High Quality"]
        df_strat = self.df.copy()

        try:
            df_strat["quality_strata"] = pd.qcut(
                df_strat[metric_col],
                q=bins,
                labels=labels,
                duplicates="drop"
            )

            group_cols = ["quality_strata"]
            if "model_name" in df_strat.columns:
                group_cols = ["model_name", "quality_strata"]
            if "smear_type" in df_strat.columns:
                group_cols = ["smear_type"] + group_cols

            rows = []
            for group_keys, grp in df_strat.groupby(group_cols, observed=False):
                if not isinstance(group_keys, tuple):
                    group_keys = (group_keys,)

                gt = grp["gt"].values if "gt" in grp.columns else grp["ground_truth"].values if "ground_truth" in grp.columns else None
                pred = grp["predicted_class"].values if "predicted_class" in grp.columns else None

                record = dict(zip(group_cols, group_keys))
                record["n_samples"] = len(grp)
                record[f"mean_{metric_col}"] = round(float(grp[metric_col].mean()), 2)

                if "michelson_contrast" in grp.columns:
                    record["mean_contrast"] = round(float(grp["michelson_contrast"].mean()), 4)
                if "confidence" in grp.columns:
                    record["mean_confidence"] = round(float(grp["confidence"].mean()), 4)

                if gt is not None and pred is not None:
                    tp = int(((pred == 1) & (gt == 1)).sum())
                    fn = int(((pred == 0) & (gt == 1)).sum())
                    tn = int(((pred == 0) & (gt == 0)).sum())
                    fp = int(((pred == 1) & (gt == 0)).sum())

                    sens = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

                    record["sensitivity"] = round(float(sens * 100), 2)
                    record["specificity"] = round(float(spec * 100), 2) if (tn + fp) > 0 else None
                    record["tp"] = tp
                    record["fn"] = fn
                    record["fp"] = fp
                    record["tn"] = tn

                rows.append(record)

            return pd.DataFrame(rows)

        except Exception as e:
            logger.error("Failed to stratify predictions by quality: %s", e)
            return pd.DataFrame()
