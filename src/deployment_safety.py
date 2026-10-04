"""
Clinical Deployment-Safety Evaluator.
Benchmarks zero-shot model performance against World Health Organization (WHO)
and CLSI clinical readiness standards for diagnostic malaria screening.
"""

from typing import Dict, Union, Any
import logging

logger = logging.getLogger(__name__)


class DeploymentSafetyEvaluator:
    """
    Evaluates empirical diagnostic metrics against international healthcare standards.
    WHO Target Product Profile (TPP) for Malaria Microscopy requires:
      - Diagnostic Sensitivity: >= 95.0%
      - Diagnostic Specificity: >= 90.0%
    """

    DEFAULT_MIN_SENSITIVITY: float = 0.95
    DEFAULT_MIN_SPECIFICITY: float = 0.90

    def __init__(
        self,
        min_sensitivity: float = DEFAULT_MIN_SENSITIVITY,
        min_specificity: float = DEFAULT_MIN_SPECIFICITY
    ):
        self.min_sensitivity = min_sensitivity
        self.min_specificity = min_specificity

    def evaluate_safety_profile(self, metrics: Dict[str, Any]) -> Dict[str, Union[str, float, bool]]:
        """
        Evaluates model metrics against WHO clinical safety benchmarks.

        Returns a structured assessment with risk categorization:
            - PASSES: Meets or exceeds both sensitivity and specificity standards.
            - HIGH RISK: Fails sensitivity (silent false negatives leave infected patients untreated).
            - MODERATE RISK: Meets sensitivity but fails specificity (over-treatment and drug waste).
            - UNSAFE: Fails both criteria.
        """
        sens = float(metrics.get("sensitivity", 0.0))
        spec = float(metrics.get("specificity", 0.0))

        sens_pass = sens >= self.min_sensitivity
        spec_pass = spec >= self.min_specificity

        if sens_pass and spec_pass:
            verdict = "PASS: Meets WHO Clinical Screening Standards"
            risk_level = "LOW"
            clinical_implication = "Safe for point-of-care assisted screening under medical supervision."
        elif not sens_pass and spec_pass:
            verdict = "FAIL: Critical False-Negative Risk"
            risk_level = "CRITICAL"
            clinical_implication = "Unacceptable missed infections; patients with active parasitemia will go untreated."
        elif sens_pass and not spec_pass:
            verdict = "FAIL: High False-Positive Rate"
            risk_level = "MODERATE"
            clinical_implication = "Triggers unnecessary antimalarial treatment, increasing toxicity risk and drug resistance."
        else:
            verdict = "FAIL: Unsafe for Autonomous Clinical Deployment"
            risk_level = "SEVERE"
            clinical_implication = "Model exhibits pervasive cross-domain diagnostic degradation across both positive and negative cases."

        return {
            "verdict": verdict,
            "risk_level": risk_level,
            "clinical_implication": clinical_implication,
            "sensitivity_measured": round(sens, 4),
            "sensitivity_required": self.min_sensitivity,
            "sensitivity_pass": sens_pass,
            "specificity_measured": round(spec, 4),
            "specificity_required": self.min_specificity,
            "specificity_pass": spec_pass
        }
