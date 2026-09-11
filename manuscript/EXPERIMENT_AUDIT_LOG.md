# Experiment, Pipeline & Audit Log

**Project**: Cross-Domain Generalization and Deployment Safety of Externally-Trained Malaria Detection Models on Ghanaian Blood Smear Data  
**Lead Investigator**: Asumboya Wilfred Ayine (Lab Lead, Innogen AI Department, University of Ghana)  
**Target Collection**: Lacuna Malaria Dataset (Princess Marie Louise Hospital, Accra — minoHealth AI Labs collection)

---

## 1. Executive Summary & Audit Trail

| Date / Timestamp | Component | Action / Event | Outcome / Resolution |
| :--- | :--- | :--- | :--- |
| **2026-09-11 18:20** | **Candidate Models** | Repository weight audit across 5 original candidate paper repos. | **Found**: 4 repos (Rajaraman 2018/2020, YOLO-SPAM, MalariAI) only ship training notebooks without weights. **Action**: Swapped to confirmed weighted models (`LHNCBC/MalariaScreener`, `kossisoroyce`, `Mmpk`). |
| **2026-09-11 18:32** | **Project Repository** | Git initialized (`e29b2b1`), folder structure & manuscript template created. | `.gitignore`, `README.md`, `requirements.txt`, `manuscript/main.tex`, `manuscript/references.bib`, `src/` core package committed. |
| **2026-09-11 18:45** | **Model Cloning** | Standard `git clone` hung due to large binary history. | **Correction**: Switched `scripts/01_clone_models.py` to `git clone --depth 1` shallow cloning for high speed & reliability. |
| **2026-09-11 19:01** | **Dataset Ingestion** | Ingested pilot batch `Thick_Ghana.part3.rar` (~695 MB) from Downloads. | Extracted 264 thick-smear `.jpg` images directly into `data/raw/thick_smear/`. |
| **2026-09-11 19:03** | **Data Sheet Audit** | Ingested official minoHealth AI Labs Lacuna Ghana Data Sheet. | Saved as `data/LACUNA_GHANA_DATASHEET.md`. Aligned `src/data_loader.py` target classes (`Parasite`, `White Blood Cells`). |
| **2026-09-11 19:04** | **Environment Setup** | Fast-tracked core data science dependencies (`numpy`, `pandas`, `opencv-python`, `pillow`, `scikit-learn`, `tqdm`). | Enabled instant dataset verification and focus-blur quality scoring while heavy PyTorch/TF wheels finalize. |

---

## 2. Model Zoo & Weight Provenance Audit

| Model | Provenance / Publication | Task / Modality | Weight Status | Format |
| :--- | :--- | :--- | :--- | :--- |
| **MalariaScreener** | **Peer-Reviewed** (*BMC Infect Dis* 2020, Rajaraman / NIH / Mahidol-Oxford) | Binary Thin & Thick Smear | **Confirmed & Checked In** | `.tflite` / `.pb` (8.5 MB thick / 1.5 MB thin) |
| **kossisoroyce** | Informal (Marketed as "Clinical-Grade", 99.14% mAP50) | Thin Smear YOLOv8 Detection | **Confirmed & Checked In** | PyTorch (`best.pt`) |
| **Mmpk** | Roboflow Universe (1,328 thin-smear images) | 7-Class Life-Cycle Stage Detection | **Confirmed & Checked In** | PyTorch / YOLOv8 |
| **fbononibelloepoch** | Hugging Face Hub | Binary YOLOv8n Detection | **Confirmed & Checked In** | PyTorch / Safetensors |

> **Key Meta-Finding Documented for Manuscript**:
> Out of 5 published peer-reviewed malaria detection papers surveyed, 80% withhold loadable model weight artifacts. `MalariaScreener` is the rare exception that provides peer-reviewed validation alongside loadable weight binaries.

---

## 3. Dataset Ingestion Log (Lacuna Ghana Subset)

- **Collection Site**: Princess Marie Louise Hospital, Accra, Ghana.
- **Collection Method**: Smartphone camera attached to microscope lenses at varying angles (8-month collection period).
- **Annotation Tool**: `makesense.ai` (Pascal VOC XML / JSON bounding box annotations).
- **Quality Note from Data Sheet**: Mobile device capture introduces focus blur and contrast variation.
- **Current Pilot Dataset**:
  - `data/raw/thick_smear/`: **264 thick smear images** loaded (`61.jpg` through `847.jpg`).
  - `data/raw/thin_smear/`: Awaiting thin smear download.

---

## 4. Errors, Bugs & Resolution Registry

### Bug #001: Git Clone Network Disconnect
- **Symptom**: `curl 56 schannel: server closed abruptly (fatal: early EOF)` during full repository clone.
- **Root Cause**: Downloading full git history for repos with binary assets over standard HTTP buffers.
- **Resolution**: Updated `scripts/01_clone_models.py` to use `--depth 1` shallow clones.

### Bug #002: System Python vs. Virtual Environment Mismatch
- **Symptom**: `ModuleNotFoundError: No module named 'cv2'` / `numpy`.
- **Root Cause**: `python` command defaulting to global system Python rather than `.venv\Scripts\python.exe`.
- **Resolution**: Updated script execution commands to explicitly target `.venv\Scripts\python.exe`. Fast-tracked pip installation of core processing dependencies.

---

## 5. Next Planned Action Items

1. Complete pip dependency installation in `.venv`.
2. Run `scripts/02_verify_dataset.py` to index the 264 thick smear pilot images.
3. Run `scripts/03_compute_quality.py` to generate Laplacian focus blur, contrast, and SNR metrics for all pilot images.
4. Execute `scripts/04_run_zero_shot.py --pilot-fraction 0.33` to run zero-shot inference across the pilot thick smear images.
5. Export initial pilot performance table & quality stratification matrix to `manuscript/tables/`.
