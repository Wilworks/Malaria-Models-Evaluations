# Deterministic Reproducibility Audit Report

**Date:** 2026-09-12 09:55:32Z  
**Dataset:** Ghanaian Subset of Lacuna Malaria Dataset (Princess Marie Louise Children's Hospital, Accra)  
**Total Cohort Size:** 4,056 blood smear micrographs (3,045 thick + 1,011 thin)  
**Evaluated Models:** 4 zero-shot deep learning architectures (`MalariaScreener_Sudan`, `MalariaScreener_Thick`, `MalariaScreener_Thin`, `fbononibelloepoch_YOLOv8`)  
**Total Inferences per Run:** 16,224 model predictions  

---

## 1. Executive Summary

| Verification Gate | Result | Status |
| :--- | :--- | :--- |
| **3-Run Metric Equivalence** | 100.000% Numerical Identity | **PASS** |
| **Cryptographic Hash Parity** | SHA-256 Bit-for-Bit Identity | **PASS** |
| **Execution Sequence** | Thin Smears $\to$ Thick Smears | **PASS** |
| **Deterministic Overall** | Identical across all runs | **CONFIRMED** |

---

## 2. Master Diagnostic Performance Across All 3 Runs

All 3 independent runs produced identical diagnostic metrics:

| Modality   | Model                    |    N |   TP |   FP |   TN |   FN |   Sensitivity | Sens_95CI   |   Specificity | Spec_95CI   |   F1_Score |   Accuracy |   Mean_Conf |
|:-----------|:-------------------------|-----:|-----:|-----:|-----:|-----:|--------------:|:------------|--------------:|:------------|-----------:|-----------:|------------:|
| Thick      | MalariaScreener_Sudan    | 3043 | 2029 |   10 |    1 | 1003 |         66.92 | [65.2-68.6] |          9.09 | [1.6-37.7]  |      80.02 |      66.71 |       64.17 |
| Thick      | MalariaScreener_Thick    | 3043 | 2021 |    5 |    6 | 1011 |         66.66 | [65.0-68.3] |         54.55 | [28.0-78.7] |      79.91 |      66.61 |       65.57 |
| Thick      | MalariaScreener_Thin     | 3043 | 2296 |    8 |    3 |  736 |         75.73 | [74.2-77.2] |         27.27 | [9.7-56.6]  |      86.06 |      75.55 |       74.82 |
| Thick      | fbononibelloepoch_YOLOv8 | 3043 | 2651 |    7 |    4 |  381 |         87.43 | [86.2-88.6] |         36.36 | [15.2-64.6] |      93.18 |      87.25 |       58.35 |
| Thin       | MalariaScreener_Sudan    | 1011 |  705 |   21 |   30 |  255 |         73.44 | [70.6-76.1] |         58.82 | [45.2-71.2] |      83.63 |      72.7  |       68.11 |
| Thin       | MalariaScreener_Thick    | 1011 |  676 |   35 |   16 |  284 |         70.42 | [67.5-73.2] |         31.37 | [20.3-45.0] |      80.91 |      68.45 |       70.41 |
| Thin       | MalariaScreener_Thin     | 1011 |  709 |   45 |    6 |  251 |         73.85 | [71.0-76.5] |         11.76 | [5.5-23.4]  |      82.73 |      70.72 |       74.28 |
| Thin       | fbononibelloepoch_YOLOv8 | 1011 |  615 |   45 |    6 |  345 |         64.06 | [61.0-67.0] |         11.76 | [5.5-23.4]  |      75.93 |      61.42 |       40.86 |

---

## 3. Cryptographic Artifact SHA-256 Checksums

| Output Artifact | Run 1 SHA-256 | Run 2 SHA-256 | Run 3 SHA-256 | Bit-for-Bit Match |
| :--- | :--- | :--- | :--- | :--- |
| `predictions_manifest.csv` | `f22891c19e79...` | `f22891c19e79...` | `f22891c19e79...` | **PASSED (Identical)** |
| `table1_master_diagnostic_performance.csv` | `5b1eb94518be...` | `5b1eb94518be...` | `5b1eb94518be...` | **PASSED (Identical)** |
| `table3_quality_stratified_sensitivity.csv` | `4638f9f6fde1...` | `4638f9f6fde1...` | `4638f9f6fde1...` | **PASSED (Identical)** |

---

## 4. Methodological Conclusion
The evaluation pipeline achieves complete deterministic reproducibility. All random seeds, inference loops, optical quality strata merges, confidence intervals, and publication tables yield identical results across repeated runs.
