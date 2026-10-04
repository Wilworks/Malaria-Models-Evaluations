"""
Unit tests for WHO clinical deployment safety evaluator.
"""

import unittest
from src.deployment_safety import DeploymentSafetyEvaluator


class TestDeploymentSafety(unittest.TestCase):
    def test_deployment_safety_pass(self):
        """Verify passes when meeting WHO standards (Sensitivity >= 95%, Specificity >= 90%)."""
        evaluator = DeploymentSafetyEvaluator()
        metrics = {"sensitivity": 0.96, "specificity": 0.92}

        res = evaluator.evaluate_safety_profile(metrics)
        self.assertEqual(res["risk_level"], "LOW")
        self.assertTrue(res["sensitivity_pass"])
        self.assertTrue(res["specificity_pass"])

    def test_deployment_safety_false_negative_risk(self):
        """Failing sensitivity must trigger CRITICAL risk warning."""
        evaluator = DeploymentSafetyEvaluator()
        metrics = {"sensitivity": 0.70, "specificity": 0.95}

        res = evaluator.evaluate_safety_profile(metrics)
        self.assertEqual(res["risk_level"], "CRITICAL")
        self.assertFalse(res["sensitivity_pass"])
        self.assertTrue(res["specificity_pass"])

    def test_deployment_safety_severe_failure(self):
        """Failing both sensitivity and specificity must be marked SEVERE."""
        evaluator = DeploymentSafetyEvaluator()
        metrics = {"sensitivity": 0.65, "specificity": 0.33}

        res = evaluator.evaluate_safety_profile(metrics)
        self.assertEqual(res["risk_level"], "SEVERE")
        self.assertFalse(res["sensitivity_pass"])
        self.assertFalse(res["specificity_pass"])


if __name__ == "__main__":
    unittest.main()
