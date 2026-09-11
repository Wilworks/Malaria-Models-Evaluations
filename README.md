# Cross-Domain Generalization & Deployment Safety of Externally-Trained Malaria Detection Models on Ghanaian Blood Smear Data

**Author**: Asumboya Wilfred Ayine  
**Affiliation**: Lab Lead, Innogen AI Department, Department of Biomedical Engineering, University of Ghana, Legon  
**Dataset Context**: Lacuna Malaria Dataset (minoHealth AI Labs collection, Princess Marie Louise Hospital, Accra, Ghana)

---

## Abstract

Deep learning models for automated malaria diagnosis routinely report accuracies above 95%, but this evidence is drawn almost entirely from datasets collected outside Africa (e.g., the NIH Bangladesh cell dataset). This repository contains the empirical benchmark code, evaluation pipeline, and manuscript source evaluating the zero-shot cross-domain generalization performance of externally-trained malaria detection models against the Ghanaian subset of the Lacuna Malaria Dataset without retraining.

---

## Research Questions

- **RQ1**: What is the measured drop in sensitivity, specificity, precision, F1-score, and mAP when externally-trained models are applied zero-shot to Ghanaian blood smear data relative to published performance?
- **RQ2**: What specific failure modes do these models exhibit (species/stage confusion, parasite vs. WBC misclassification, or sensitivity to smartphone camera artifacts)?
- **RQ3**: What do these failure patterns imply for clinical deployment-safety standards in Ghanaian medical settings?

---

## Repository Structure

```
Malaria-Models-Evaluations/
├── manuscript/                    # Paper draft source (main.tex, references.bib, figures, tables)
├── data/                          # Dataset directory (raw Ghanaian Lacuna data & quality strata)
├── models/                        # Pre-trained model zoo, wrappers, and cloned repositories
│   ├── external/                  # External cloned repos (e.g., LHNCBC/MalariaScreener, kossisoroyce)
│   ├── weights/                   # Pre-trained checkpoint files (.pt, .tflite, .pb)
│   └── wrappers/                  # Unified Python evaluation wrappers
├── src/                           # Core implementation module
│   ├── data_loader.py             # Lacuna Ghana parser & PyTorch/TF adapters
│   ├── quality_assessment.py      # Blur, contrast, and SNR calculators
│   ├── metrics.py                 # Classification & object detection metric suite
│   ├── error_analysis.py          # Failure mode stratification engine
│   └── deployment_safety.py       # Clinical risk scoring & safety threshold evaluator
├── scripts/                       # Executable CLI execution pipeline
├── notebooks/                     # Exploratory analysis & visual error inspection
└── tests/                         # Unit tests
```

---

## Benchmark Models

| Model | Provenance / Publication | Modality / Task | Weight Format | Repository |
| :--- | :--- | :--- | :--- | :--- |
| **MalariaScreener** | **Peer-Reviewed** (*BMC Infect Dis 2020*, NIH / Rajaraman et al.) | Binary Thin & Thick Smear | `.tflite` / `.pb` | `LHNCBC/MalariaScreener` |
| **kossisoroyce** | Marketed as "Clinical-Grade" (99.14% mAP50 claim) | Thin Smear YOLOv8 Detection | PyTorch (`.pt`) | `kossisoroyce/malaria-detection` |
| **Mmpk** | Roboflow Universe (1,328 images) | 7-Class Life-Cycle Stage Detection | PyTorch / YOLOv8 | `Mmpk/malaria-detection-4dn5i` |
| **fbononibelloepoch** | Hugging Face Hub | Binary YOLOv8n Detection | PyTorch / HF | `fbononibelloepoch/malaria-detection` |

---

## Getting Started

### 1. Environment Setup

```bash
python -m venv .venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the Ghanaian subset of the Lacuna Malaria Dataset (DOI: `10.7910/DVN/VEADSE`) into `data/raw/`:
- `data/raw/thick_smear/` (3,000 thick smear images + annotations)
- `data/raw/thin_smear/` (1,000 thin smear images + annotations)

### 3. Pipeline Execution

```bash
python scripts/01_clone_models.py
python scripts/02_verify_dataset.py
python scripts/03_compute_quality.py
python scripts/04_run_zero_shot.py
python scripts/05_stratified_errors.py
python scripts/06_export_paper_assets.py
```
