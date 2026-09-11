"""
Clinical Deployment-Safety Evaluator (Answers RQ3).
Translates zero-shot performance drop and error patterns into clinical risk scores
and minimum evidence thresholds for Ghanaian healthcare deployment.
"""

from typing import Dict, List, Union


class DeploymentSafetyEvaluator:
    """Translates model performance into clinical deployment safety thresholds."""

    DEFAULT_MIN_SENSITIVITY = 0.95  # WHO recommended diagnostic benchmark
    DEFAULT_MIN_SPECIFICITY = 0.90

    def evaluate_safety_profile(self, metrics: Dict[str, float]) -> Dict[str, Union[str, float, bool]]:
        """
        Evaluates model metrics against WHO clinical safety thresholds.
        """
        sens = metrics.get("sensitivity", 0.0)
        spec = metrics.get("specificity", 0.0)

        sens_pass = sens >= self.DEFAULT_MIN_SENSITIVITY
        spec_pass = spec >= self.DEFAULT_MIN_SPECIFICITY

        safety_verdict = "UNSAFE FOR CLINICAL DEPLOYMENT"
        if sens_pass and spec_pass:
            safety_verdict = "PASSES MINIMUM CLINICAL THRESHOLD"
        elif not sens_pass and spec_pass:
            safety_verdict = "HIGH RISK: SILENT FALSE NEGATIVES (Unacceptable missed infections)"
        elif sens_pass and not spec_pass:
            safety_verdict = "MODERATE RISK: HIGH FALSE POSITIVES (Triggers unnecessary antimalarial treatment)"

        return {
            "verdict": safety_verdict,
            "sensitivity_measured": sens,
            "sensitivity_required": self.DEFAULT_MIN_SENSITIVITY,
            "sensitivity_pass": sens_pass,
            "specificity_measured": spec,
            "specificity_required": self.DEFAULT_MIN_SPECIFICITY,
            "specificity_pass": spec_pass
        }
