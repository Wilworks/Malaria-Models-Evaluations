# Cross-Domain Generalization & Deployment Safety of Externally-Trained Malaria Detection Models on Ghanaian Blood Smear Data

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Determinism-SHA--256%20Verified-00C853?style=for-the-badge)](results/reproducibility/)

<p align="center">
  <b>The First Large-Scale Empirical Zero-Shot Audit of External Deep Learning Malaria Models on West African Soil</b>
</p>

<p align="center">
  <b>Author:</b> <a href="https://github.com/Wilworks">Wilfred Ayine Asumboya</a><br>
  <i>Department of Biomedical Engineering, University of Ghana, Legon</i><br>
  <i>Clinical Cohort: Princess Marie Louise Children's Hospital, Accra, Ghana</i>
</p>

</div>

---

## 📌 Executive Summary

Deep learning architectures for automated malaria microscopy frequently report diagnostic sensitivities and accuracies exceeding **95% to 99%** in laboratory settings. However, virtually all published evidence is derived from homogeneous laboratory cohorts collected outside Africa—most prominently the United States National Institutes of Health (NIH) Chittagong benchmark from Bangladesh.

Prior to this study, claims in the AI for healthcare literature that diagnostic models degrade catastrophically under African domain shift circulated largely as narrative citations rather than empirical, data-driven audits.

This repository provides the **complete, self-contained, reproducible evaluation pipeline** executing **16,216 zero-shot inference evaluations** across the complete Ghanaian clinical cohort of the **Lacuna Malaria Dataset** ($N = 4,056$ patient micrographs):
* **Thick Blood Smears**: $N = 3,045$ micrographs (Chemical RBC lysis; parasite triage)
* **Thin Blood Smears**: $N = 1,011$ micrographs (Intact RBC monolayer; species differentiation)

---

## 🔬 Key Empirical Discoveries

```
+--------------------------------------------------------------------------------------------------+
|                                    THREE CORE CLINICAL FINDINGS                                   |
+--------------------------------------------------------------------------------------------------+
| 1. African Regional Advantage (Specificity & False-Positive Suppression)                         |
|    The East African-trained model (MS_Sudan) delivered a 5.0x higher thin-smear specificity       |
|    (58.82% vs. 11.76%) and reduced the false-positive diagnostic error rate by 2.14x             |
|    (p = 9.50 x 10^-7) compared to Asian-trained MobileNetV2 architectures.                       |
+--------------------------------------------------------------------------------------------------+
| 2. Architectural Divergence & The Modality Transfer Cliff                                        |
|    Spatial object detectors (YOLOv8) excel on thick smears (87.43% sensitivity) but collapse     |
|    by 23.37 percentage points on thin smears (64.06%, p = 5.45 x 10^-54) due to intact RBC       |
|    membrane occlusions. Conversely, whole-slide MobileNetV2 classifiers remain resilient.        |
+--------------------------------------------------------------------------------------------------+
| 3. Optical Blur Vulnerability & The Pre-Inference Safety Gate                                    |
|    Optical focus blur (quantified via circular FOV-masked Laplacian variance, sigma_Lap^2)       |
|    induces an acute ~25.37 percentage-point collapse in thick-smear diagnostic sensitivity,      |
|    mandating automated pre-inference hardware quality gating in clinical deployments.            |
+--------------------------------------------------------------------------------------------------+
```

---

## 🏛 System Architecture & Pipeline Flow

```
                             GHANAIAN CLINICAL COHORT
                     Princess Marie Louise Children's Hospital
                          [ N = 4,056 Patient Micrographs ]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
           Thick Blood Smears                        Thin Blood Smears
             ( n = 3,045 )                             ( n = 1,011 )
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                        [ src/quality_assessment.py ]
                        Otsu Circular FOV Extraction
                   Laplacian Focus Variance (sigma_Lap^2)
                     Stratification: Low | Med | High
                                         │
                                         ▼
                           [ models/wrappers/ ]
                       Unified Zero-Shot Model Zoo
       ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
       ▼                  ▼                  ▼                  ▼                  ▼
  MS_Sudan            MS_Thick           MS_Thin            YOLOv8            Detection
(MobileNetV2)       (MobileNetV2)      (MobileNetV2)       (Ultralytics)       Thresholds
 [Sudan cohort]     [Chittagong Asian] [Chittagong Asian]  [Spatial BBox]     (tau = 0.15)
       │                  │                  │                  │
       └──────────────────┴─────────┬────────┴──────────────────┘
                                    ▼
                          [ src/metrics.py ]
                     Slide-Level Clinical Metrics
              • Sensitivity (Recall)   • Specificity
              • Precision & F1-Score   • Wilson Score 95% CIs
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
       [ results/baseline/ ]            [ scripts/run_reproducibility_audit.py ]
       CSV Prediction Manifests         Multi-Run Determinism Engine
       Stratified Quality Logs          Cryptographic SHA-256 Checksums
```

---

## 📦 Evaluated Model Zoo

All 4 models are packaged directly within `models/external/` for out-of-the-box local execution:

| Model Identifier | Architecture | Training Domain / Source | Task / Modality | Checkpoint Format | File Size |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`MalariaScreener_Sudan`** | MobileNetV2 | NIH LHNCBC (Sudan, East Africa) | Thick Smear Binary Triage | TensorFlow Protobuf (`.pb`) | 1.58 MB |
| **`MalariaScreener_Thick`** | MobileNetV2 | NIH LHNCBC (Chittagong, Bangladesh) | Thick Smear Binary Triage | TensorFlow Lite (`.tflite`) | 8.53 MB |
| **`MalariaScreener_Thin`** | MobileNetV2 | NIH LHNCBC (Chittagong, Bangladesh) | Thin Smear Binary Triage | TensorFlow Lite (`.tflite`) | 1.58 MB |
| **`fbononi_YOLOv8`** | YOLOv8 Nano | Field Parasite Bounding Boxes | Spatial Object Detection | PyTorch Weights (`.pt`) | 19.29 MB |

---

## 📂 Codebase Organization

```text
Malaria-Models-Evaluations/
├── README.md                      # Primary project documentation & architecture guide
├── requirements.txt               # Locked, cross-platform dependencies
├── .gitignore                     # Excludes raw data, results cache, and scratch files
│
├── src/                           # CORE REUSABLE PYTHON MODULES
│   ├── __init__.py                # Package initializer
│   ├── data_loader.py             # Lacuna Ghana clinical dataset parser (thick/thin)
│   ├── metrics.py                 # Medical diagnostics engine & Wilson 95% CIs
│   ├── quality_assessment.py      # Circular FOV masking & Laplacian blur physics
│   ├── error_analysis.py          # Morphological error stratification engine
│   ├── deployment_safety.py       # Clinical risk scoring & hardware safety gates
│   └── visualizer.py              # Publication-grade vector figure renderer
│
├── models/                        # MODEL REPOSITORY & ADAPTERS
│   ├── wrappers/                  # Framework-agnostic prediction interfaces
│   │   ├── base_wrapper.py        # Abstract BaseModelWrapper base class
│   │   ├── malariascreener_wrapper.py # TensorFlow/TFLite adapter for NIH models
│   │   └── yolo_wrapper.py        # Ultralytics PyTorch adapter for YOLOv8
│   └── external/                  # Pre-trained authentic model checkpoints
│       ├── MalariaScreener/assets # Official NIH .tflite & .pb checkpoints
│       └── fbononibelloepoch/     # YOLOv8 best_yolo.pt checkpoint
│
├── scripts/                       # BENCHMARK & REPRODUCIBILITY ORCHESTRATION
│   ├── 01_clone_models.py         # Checkpoint integrity validator
│   ├── 02_verify_dataset.py       # Clinical cohort label & integrity checker
│   ├── 03_compute_quality.py      # Optical Laplacian sharpness batch processor
│   ├── 04_run_zero_shot.py        # Zero-shot inference for NIH MobileNetV2 models
│   ├── 04b_run_fbononi.py         # Zero-shot inference for YOLOv8 detector
│   ├── 05_stratified_errors.py    # Cross-quality performance stratification
│   ├── 06_export_paper_assets.py  # Statistical LaTeX & CSV table exporter
│   ├── 07_generate_figures.py     # Publication figures generator
│   ├── run_master_benchmark.py    # Master end-to-end reproducible pipeline runner
│   └── run_reproducibility_audit.py # Multi-run deterministic SHA-256 auditor
│
├── data/                          # DATASET MOUNT POINT (Provided locally / via USB)
│   ├── raw/                       # Untouched clinical blood smear images
│   │   ├── thick_smear/           # 3,045 thick smear micrographs (.jpg)
│   │   └── thin_smear/            # 1,011 thin smear micrographs (.jpg)
│   ├── processed/                 # Cleaned clinical labels (annotations.csv)
│   └── LACUNA_GHANA_DATASHEET.md  # Comprehensive ethical & clinical datasheet
│
└── results/                       # OUTPUT DESTINATIONS (Generated upon execution)
    ├── baseline/                  # Raw inference prediction manifests (.csv)
    └── reproducibility/           # Determinism logs & cryptographic hashes
```

---

## 🚀 Quickstart & Reproducibility Guide

Follow these instructions to reproduce the complete benchmark from scratch on any new machine.

### 1. Prerequisites & Environment Setup

* **Python**: `3.10` or higher
* **OS**: Linux, macOS, or Windows (10/11)
* **Hardware**: CPU-capable; NVIDIA GPU (CUDA) recommended for fast inference.

```bash
# Clone the repository
git clone https://github.com/Wilworks/Malaria-Models-Evaluations.git
cd Malaria-Models-Evaluations

# Create and activate a clean virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install all locked dependencies
pip install -r requirements.txt
```

### 2. Dataset Setup (Via USB / Pendrive)

Copy the Ghanaian Lacuna clinical dataset into `data/raw/`:
```text
Malaria-Models-Evaluations/
└── data/
    └── raw/
        ├── thick_smear/
        │   ├── *.jpg              (3,045 micrographs)
        │   └── labels_yolo/*.txt  (YOLO annotations)
        └── thin_smear/
            ├── *.jpg              (1,011 micrographs)
            └── labels_yolo/*.txt  (YOLO annotations)
```

Verify dataset integrity:
```bash
python scripts/02_verify_dataset.py
```

### 3. Executing the Master Benchmark

To run the complete zero-shot evaluation pipeline end-to-end:

```bash
python scripts/run_master_benchmark.py --run-id 1
```

This single command:
1. Verifies dataset image count ($3,045$ thick, $1,011$ thin).
2. Executes **16,216 zero-shot inference evaluations** across the 4 candidate models.
3. Computes slide-level Sensitivity, Specificity, Precision, $F_1$, and **Wilson Score 95% CIs**.
4. Computes FOV-masked Laplacian blur and stratifies performance across quality tertiles.
5. Emits structured CSV tables into `results/run_1/`.
6. Computes SHA-256 cryptographic checksums for verification.

### 4. Running the Multi-Run Determinism Audit

To verify bit-for-bit numerical determinism across 3 independent, isolated executions:

```bash
python scripts/run_reproducibility_audit.py
```

---

## 📊 Summary of Baseline Diagnostic Findings

### Thick Smear Diagnostic Performance ($N = 3,045$)

| Model Architecture | Training Cohort | Sensitivity (Recall) | Specificity | $F_1$-Score | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`MalariaScreener_Sudan`** | East Africa (Sudan) | **66.92%** [65.22, 68.58] | 33.33% [13.34, 61.25] | 0.8000 | 66.62% |
| **`MalariaScreener_Thick`** | South Asia (Bangladesh) | **66.66%** [64.96, 68.32] | 33.33% [13.34, 61.25] | 0.7984 | 66.36% |
| **`MalariaScreener_Thin`** | South Asia (Bangladesh) | **69.83%** [68.17, 71.45] | 33.33% [13.34, 61.25] | 0.8209 | 69.52% |
| **`fbononi_YOLOv8`** ($\tau = 0.15$) | Object Detection | **87.43%** [86.20, 88.58] | 26.67% [9.45, 55.08] | 0.9317 | **86.91%** |

*Values in brackets denote two-sided 95% Wilson Score Confidence Intervals.*

### Thin Smear Diagnostic Performance ($N = 1,011$)

| Model Architecture | Training Cohort | Sensitivity (Recall) | Specificity | False-Positive Rate | $F_1$-Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`MalariaScreener_Sudan`** | East Africa (Sudan) | 68.80% [65.88, 71.59] | **58.82%** [42.22, 73.63] | **41.18%** (2.14x reduction) | 0.8062 |
| **`MalariaScreener_Thick`** | South Asia (Bangladesh) | 68.50% [65.57, 71.29] | 11.76% [3.28, 34.33] | 88.24% | 0.8080 |
| **`MalariaScreener_Thin`** | South Asia (Bangladesh) | 71.86% [69.01, 74.55] | 11.76% [3.28, 34.33] | 88.24% | 0.8315 |
| **`fbononi_YOLOv8`** ($\tau = 0.15$) | Object Detection | 64.06% [61.03, 66.98] | 17.65% [5.71, 41.01] | 82.35% | 0.7770 |

---

## ⚖️ Ethics, Biosafety, and Dataset Provenance

The evaluation micrographs analyzed in this study are derived from the Ghanaian subset of the **Lacuna Malaria Dataset** (Harvard Dataverse DOI: `10.7910/DVN/VEADSE`), collected under local institutional ethics approvals at **Princess Marie Louise Children's Hospital in Accra, Ghana**, in partnership with **minoHealth AI Labs** and the **Makerere AI Lab**. Micrographs depict authentic Giemsa-stained pediatric blood smears captured using optical smartphones under real-world clinical microscopy conditions in West Africa.

---

## 📝 Citation

If you use this benchmark suite, models, or evaluation methodology in your research, please cite:

```bibtex
@article{asumboya2026crossdomain,
  author    = {Asumboya, Wilfred Ayine},
  title     = {Cross-Domain Generalization and Deployment Safety of Externally-Trained Deep Learning Malaria Models on Ghanaian Pediatric Blood Smears},
  journal   = {Manuscript Draft},
  volume    = {1},
  number    = {12},
  year      = {2026},
  publisher = {Department of Biomedical Engineering, University of Ghana, Legon}
}
```

---

## 📜 License

This codebase is open-sourced under the **MIT License**. Pre-trained model weights are distributed under their respective original upstream licenses (NIH LHNCBC Open Access / Creative Commons).
