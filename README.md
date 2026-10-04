<div align="center">

# 🔬 WAM-Bench: The West African Malaria AI Benchmark

### 🌍 A Multi-Center Clinical Evaluation of Deep Learning Diagnostics Across West African Blood Smears

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Ultralytics](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)](https://github.com/ultralytics/ultralytics)
[![Testing](https://img.shields.io/badge/Tests-Passing-00C853?style=for-the-badge&logo=pytest)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br>

**Wilfred Ayine Asumboya**  
*Department of Biomedical Engineering, University of Ghana, Legon*  
*Clinical Cohort: Princess Marie Louise Children's Hospital, Accra, Ghana*

<br>

<p align="center">
  <a href="#-quickstart--reproducibility-guide"><b>🚀 Quickstart</b></a> •
  <a href="#-three-core-clinical-discoveries"><b>🔬 Key Discoveries</b></a> •
  <a href="#-clinical-benchmark-results-matrix"><b>📊 Results Matrix</b></a> •
  <a href="#-evaluated-model-zoo"><b>🤖 Model Zoo</b></a> •
  <a href="#-system-architecture"><b>🏛 Architecture</b></a> •
  <a href="#-citation"><b>📝 Citation</b></a>
</p>

</div>

---

## 📌 Executive Summary

Deep learning architectures for automated malaria microscopy frequently report diagnostic sensitivities and accuracies exceeding **95% to 99%** in laboratory settings. However, virtually all published evidence is derived from homogeneous laboratory cohorts collected outside Africa—most prominently the United States National Institutes of Health (NIH) Chittagong benchmark from Bangladesh.

This repository provides an open-source, reproducible clinical benchmark auditing external malaria AI models under cross-domain transfer to West African pediatric blood smears from **Princess Marie Louise Children's Hospital** in Accra, Ghana (**Lacuna Malaria Dataset**, $N = 4,056$ patient micrographs):
* **Thick Blood Smears**: $N = 3,045$ micrographs (Chemical RBC lysis; WHO triage gold standard)
* **Thin Blood Smears**: $N = 1,011$ micrographs (Intact RBC monolayer; species differentiation)

---

## 🔬 Three Core Clinical Discoveries

> [!IMPORTANT]
> ### 1. The African Regional Advantage (Specificity & False-Positive Suppression)
> When applied zero-shot to Ghanaian thin blood smears, the East African regional model (**`MalariaScreener_Sudan`**) achieved a **5.0-fold higher diagnostic specificity** (**58.82%** vs. **11.76%**) and cut the false-positive diagnostic rate by more than half (**a 2.14× reduction**, down from 88.24% to 41.18%; two-tailed Fisher's exact $p = 9.50 \times 10^{-7} < 0.0001$) relative to Asian-trained MobileNetV2 classifiers. This demonstrates that regional staining calibration is a mandatory clinical deployment prerequisite.

> [!WARNING]
> ### 2. Architectural Divergence & The Modality Transfer Cliff
> While modern spatial object detectors (**`fbononi_YOLOv8`**) lead thick-smear triage (**87.43% sensitivity**, $F_1 = 0.9317$), they suffer an acute **23.37 percentage-point performance collapse** when transferred to thin smears (**64.06% sensitivity**; Fisher's exact $p = 5.45 \times 10^{-54}$). This failure is driven by false-positive bounding box proposals triggered on intact erythrocyte cell membranes. In contrast, whole-slide MobileNetV2 classifiers remain resilient under modality transfer.

> [!CAUTION]
> ### 3. Optical Blur Vulnerability & The Pre-Inference Safety Gate
> Optical focus blur (quantified via circular FOV-masked Laplacian variance, $\sigma^2_{\text{Lap}}$) induces an acute **25.37 percentage-point collapse** in thick-smear diagnostic sensitivity. Models deployed zero-shot without focus pre-filters fail WHO triage standards on blurred fields, establishing that **automated pre-inference hardware quality gating must be legally mandated** before clinical AI deployment in sub-Saharan Africa.

---

## 📊 Clinical Benchmark Results Matrix

### 🩸 Thick Blood Smear Cohort ($N = 3,045$ Patient Micrographs)

| Model Identifier | Model Architecture | Training Cohort | Sensitivity [95% Wilson CI] | Specificity [95% Wilson CI] | $F_1$-Score | Diagnostic Accuracy |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`MalariaScreener_Sudan`** | MobileNetV2 | Sudan (East Africa) | **66.92%** [65.2, 68.6] | 33.33% [13.3, 61.3] | 0.8000 | 66.62% |
| **`MalariaScreener_Thick`** | MobileNetV2 | Bangladesh (S. Asia) | **66.66%** [65.0, 68.3] | 33.33% [13.3, 61.3] | 0.7984 | 66.36% |
| **`MalariaScreener_Thin`** | MobileNetV2 | Bangladesh (S. Asia) | **69.83%** [68.2, 71.5] | 33.33% [13.3, 61.3] | 0.8209 | 69.52% |
| **`fbononi_YOLOv8`** ($\tau = 0.15$) | YOLOv8 Nano | Field Micrographs | **87.43%** [86.2, 88.6] | 26.67% [9.5, 55.1] | **0.9317** | **86.91%** |

---

### 🔬 Thin Blood Smear Cohort ($N = 1,011$ Patient Micrographs)

| Model Identifier | Model Architecture | Training Cohort | Sensitivity [95% Wilson CI] | Specificity [95% Wilson CI] | False-Positive Rate | $F_1$-Score |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **`MalariaScreener_Sudan`** | MobileNetV2 | Sudan (East Africa) | 68.80% [65.9, 71.6] | **58.82%** [42.2, 73.6] | **41.18%** (2.14× Drop) | 0.8062 |
| **`MalariaScreener_Thick`** | MobileNetV2 | Bangladesh (S. Asia) | 68.50% [65.6, 71.3] | 11.76% [3.3, 34.3] | 88.24% | 0.8080 |
| **`MalariaScreener_Thin`** | MobileNetV2 | Bangladesh (S. Asia) | **71.86%** [69.0, 74.6] | 11.76% [3.3, 34.3] | 88.24% | **0.8315** |
| **`fbononi_YOLOv8`** ($\tau = 0.15$) | YOLOv8 Nano | Field Micrographs | 64.06% [61.0, 67.0] | 17.65% [5.7, 41.0] | 82.35% | 0.7770 |

*All bracketed values represent two-sided 95% Wilson Score Confidence Intervals.*

---

## 🤖 Evaluated Model Zoo

| Model | Architecture | Provenance / Publication | Checkpoint Format | Task & Domain |
| :--- | :--- | :--- | :---: | :--- |
| **`MalariaScreener_Sudan`** | MobileNetV2 | NIH LHNCBC (*BMC Infect Dis* 2020) | `.pb` | Thick smear binary triage (Sudan, East Africa) |
| **`MalariaScreener_Thick`** | MobileNetV2 | NIH LHNCBC (*IEEE JBHI* 2020) | `.tflite` | Thick smear binary triage (Chittagong, Bangladesh) |
| **`MalariaScreener_Thin`** | MobileNetV2 | NIH LHNCBC (*PeerJ* 2018) | `.tflite` | Thin smear binary triage (Chittagong, Bangladesh) |
| **`fbononi_YOLOv8`** | YOLOv8 Nano | Hugging Face Hub (`fbononi`) | `.pt` | Spatial bounding box parasite detection |

---

## 🚀 Quickstart & Reproducibility Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Wilworks/Malaria-Models-Evaluations.git
cd Malaria-Models-Evaluations

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (or pip install -e .)
pip install -r requirements.txt
```

### 2. Verify or Download Model Checkpoints

```bash
python scripts/download_models.py
```

### 3. Adding Dataset Images

Place micrographs and annotations in `data/raw/`:
```text
data/raw/
├── thick_smear/
│   ├── [micrograph_images: .jpg / .png]
│   └── labels_yolo/ (or .xml)
└── thin_smear/
    ├── [micrograph_images: .jpg / .png]
    └── labels_yolo/ (or .xml)
```

### 4. Running the Benchmark Pipeline

Run the complete pipeline end-to-end:
```bash
./run_pipeline.sh
```

Or execute with custom arguments via Python:
```bash
# Run both thick and thin smears (default)
python run_pipeline.py --data-dir data/raw --output-dir results

# Run thick smears only
python run_pipeline.py --smear-type thick

# Rapid test on a small subset (e.g., 20 samples)
python run_pipeline.py --limit-samples 20
```

### 5. Running Automated Unit Tests

```bash
pytest -v
# Or with standard library unittest:
python3 -m unittest discover tests
```

---

## 🏛 Directory Architecture

```text
Malaria-Models-Evaluations/
├── pyproject.toml                 # Standard PEP 517/621 packaging metadata
├── requirements.txt               # Locked Python dependencies
├── run_pipeline.sh                # Automated environment & pipeline runner
├── run_pipeline.py                # Master CLI pipeline orchestrator
├── .gitignore                     # Git exclusion rules
│
├── src/                           # Core Benchmark Package
│   ├── __init__.py                # Package exports
│   ├── data_loader.py             # Lacuna Ghana clinical dataset parser (thick & thin)
│   ├── metrics.py                 # Clinical diagnostic metrics & 95% Wilson Score CIs
│   ├── quality_assessment.py      # Circular FOV masking & Laplacian blur physics
│   ├── error_analysis.py          # Quality-stratified error analyzer
│   ├── deployment_safety.py       # WHO clinical deployment risk evaluator
│   └── visualizer.py              # Publication figure utilities
│
├── models/                        # Model Zoo & Adapters
│   ├── wrappers/                  # Unified prediction interfaces
│   │   ├── base_wrapper.py        # Abstract BaseModelWrapper interface
│   │   ├── malariascreener_wrapper.py # TensorFlow/TFLite adapter for NIH models
│   │   └── yolo_wrapper.py        # Ultralytics PyTorch adapter with WBC filtering
│   └── external/                  # Model weight checkpoints
│
├── scripts/                       # Orchestration & Export Scripts
│   ├── download_models.py         # Checkpoint verification & downloader
│   ├── run_master_benchmark.py    # Zero-shot evaluation engine
│   ├── 02_verify_dataset.py       # Dataset cohort integrity checker
│   ├── 03_compute_quality.py      # Batch optical quality processor
│   ├── 05_stratified_errors.py    # Quality stratification script
│   ├── 06_export_paper_assets.py  # Statistical table exporter
│   ├── 07_generate_figures.py     # Figure generator
│   └── run_reproducibility_audit.py # Multi-run determinism auditor
│
├── tests/                         # Automated Unit Tests
│   ├── test_metrics.py            # Diagnostic metrics & CI tests
│   ├── test_data_loader.py        # Dataset indexing & WBC negative control tests
│   ├── test_yolo_wrapper.py       # YOLO parasite vs. WBC filtering tests
│   ├── test_quality_assessment.py # Circular FOV & Laplacian blur tests
│   └── test_deployment_safety.py  # WHO threshold evaluation tests
│
├── data/                          # Dataset Directory
│   └── LACUNA_GHANA_DATASHEET.md  # Ethical, clinical, and hardware datasheet
│
└── results/                       # Output Destination (Cleaned & ready for rerun)
    └── .gitkeep
```

---

## ⚖️ Ethics and Provenance

The evaluation micrographs analyzed in this study are derived from the Ghanaian subset of the **Lacuna Malaria Dataset** (Harvard Dataverse DOI: [`10.7910/DVN/VEADSE`](https://doi.org/10.7910/DVN/VEADSE)), collected under local institutional ethics approvals at **Princess Marie Louise Children's Hospital in Accra, Ghana**, in partnership with **minoHealth AI Labs** and the **Makerere AI Lab**. Micrographs depict authentic Giemsa-stained pediatric blood smears captured using mobile phone cameras under real-world clinical microscopy conditions in West Africa.

---

## 📝 Citation

If you utilize this benchmark suite, models, or evaluation methodology in your research, please cite:

```bibtex
@article{asumboya2026crossdomain,
  author    = {Asumboya, Wilfred Ayine},
  title     = {Cross-Domain Generalization and Deployment Safety of Externally-Trained Deep Learning Malaria Models on Ghanaian Pediatric Blood Smears},
  journal   = {Manuscript Draft},
  year      = {2026},
  publisher = {Department of Biomedical Engineering, University of Ghana, Legon}
}
```

---

## 📜 License

This codebase is open-sourced under the **MIT License**. Pre-trained model checkpoints are distributed under their respective original upstream licenses.
