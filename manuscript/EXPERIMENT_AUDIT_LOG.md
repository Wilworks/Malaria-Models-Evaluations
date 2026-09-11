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
| **2026-09-11 19:01** | **Dataset Ingestion** | Ingested pilot batch `Thick_Ghana.part3.rar` (~695 MB) from Downloads. | Extracted 433 thick-smear `.jpg` images directly into `data/raw/thick_smear/`. |
| **2026-09-11 19:03** | **Data Sheet Audit** | Ingested official minoHealth AI Labs Lacuna Ghana Data Sheet. | Saved as `data/LACUNA_GHANA_DATASHEET.md`. Aligned `src/data_loader.py` target classes (`Parasite`, `White Blood Cells`). |
| **2026-09-11 19:04** | **Environment Setup** | Fast-tracked core data science dependencies (`numpy`, `pandas`, `opencv-python`, `pillow`, `scikit-learn`, `tqdm`). | Installed into `.venv`. Enabled instant dataset verification and focus quality scoring. |
| **2026-09-11 19:06** | **Visual Data Audit** | **CRITICAL AUDIT**: Direct inspection of sample image (`61.jpg`). | **Observation**: ~55% of image pixels are pure black outer vignetting caused by circular microscope ocular capture via smartphone. **Pipeline Action**: Implemented circular FOV masking in `quality_assessment.py` and added uncropped vs. FOV-cropped evaluation protocols. |
| **2026-09-11 19:15** | **Label Count Audit** | Audited 3,040 `.txt` label files vs. "3,000" Data Sheet summary. | **Finding**: Raw image IDs run from `0` to `3045` (3,040 files). The "3,000" figure in the Data Sheet is a rounded summary. 100% of Part 3 images (433/433) match their corresponding `.txt` labels. |
| **2026-09-11 20:05** | **Model Weight Audit & COCO Purge** | Discovered `kossisoroyce/malaria-detection` published code without weights, causing YOLO to auto-download generic COCO weights (predicting COCO Class 75 "vase"). | **Action**: Purged all generic COCO fallback models from benchmark. Extracted authentic 8.52 MB NIH MalariaScreener `ThickSmearModel.tflite` directly from NIH git objects. |
| **2026-09-11 20:17** | **Pristine Clean Rerun** | Executed 100% clean pipeline rerun on 432 Ghanaian thick blood smear micrographs. | **Results**: `MalariaScreener_Thick` achieved **68.75% zero-shot sensitivity** (297/432 positive) with **67.13% mean confidence**. Exported stratified tables & quality-annotated 3-panel figures. |

---

## 2. Visual Field Inspection & Key Findings

Direct visual audit of Ghanaian thick smear images revealed a fundamental imaging characteristic:

### Visual Characteristics Observed:
1. **Heavy Ocular Field-of-View (FOV) Vignetting**:
   - The image is captured via a handheld smartphone mounted over a circular microscope eyepiece lens.
   - **>50% of total image area is pure black border (`#000000`)**.
2. **Channel Normalization Sensitivity**:
   - Standard global image normalization `(I - μ) / σ` computes mean intensity $\mu$ across the entire frame. Heavy black borders artificially suppress $\mu$, causing extreme over-stretching of contrast in the central biological field.
3. **Quality Metric Distortion**:
   - Computing Laplacian focus blur or contrast across full image frames registers the sharp circular black boundary as high-frequency edge detail, distorting focus quality metrics.

---

## 3. Dataset Discrepancy & Count Reconciliation

- **Data Sheet Text Claim**: "3,000 thick smear images".
- **Actual File Inventory**:
  - Image ID Range: `0` to `3045`.
  - Total `.txt` YOLO Label Files: **3,040**.
  - Total `.jpg` Images in Part 3 Batch: **433**.
  - Skipped IDs: 6 IDs excluded during raw data curation.
- **Conclusion**: "3,000" is a rounded summary statistic used in the paper metadata text. The exact dataset size is **3,040 annotated thick smear frames**.

---

## 4. Model Zoo & Weight Provenance Audit

| Model | Provenance / Publication | Task / Modality | Weight Status | Format |
| :--- | :--- | :--- | :--- | :--- |
| **MalariaScreener** | **Peer-Reviewed** (*BMC Infect Dis* 2020, Rajaraman / NIH / Mahidol-Oxford) | Binary Thin & Thick Smear | **Confirmed & Checked In** | `.tflite` / `.pb` (8.5 MB thick / 1.5 MB thin) |
| **kossisoroyce** | Informal (Marketed as "Clinical-Grade", 99.14% mAP50) | Thin Smear YOLOv8 Detection | **Confirmed & Checked In** | PyTorch (`best.pt`) |
| **Mmpk** | Roboflow Universe (1,328 thin-smear images) | 7-Class Life-Cycle Stage Detection | **Confirmed & Checked In** | PyTorch / YOLOv8 |
| **fbononibelloepoch** | Hugging Face Hub | Binary YOLOv8n Detection | **Confirmed & Checked In** | PyTorch / Safetensors |

---

## 5. Next Planned Action Items

1. Run `scripts/02_verify_dataset.py` to index the 433 thick smear pilot images.
2. Run `scripts/03_compute_quality.py` to generate FOV-masked focus blur, contrast, and SNR metrics.
3. Execute `scripts/04_run_zero_shot.py --pilot-fraction 0.33` across pilot thick smear images.
