# Standalone Scientific Captions for Manuscript Figures

*Formatted in accordance with Nature Medicine / The Lancet Digital Health / IEEE JBHI submission guidelines.*  
*Each caption is completely self-contained, defining all abbreviations, cohort sizes ($n$), thresholds, and statistical interpretations.*

---

## 1. Cross-Modality & Domain Transfer Figures

### Figure 1: Cross-Domain × Cross-Modality Sensitivity Matrix
* **File Location**: `figures/01_cross_modality/fig_cross_modality_heatmap.[png|pdf|svg]`
* **Caption**:  
  **Figure 1 | Cross-Domain and Cross-Modality Zero-Shot Performance Matrix of External Deep Learning Models on Ghanaian Blood Smears.** Heatmap illustrating diagnostic sensitivity across four candidate architectures evaluated zero-shot on blood smear micrographs collected at Princess Marie Louise Children's Hospital, Accra, Ghana. Rows denote model architectures, their training geographical provenance, and primary training smear modality: *MalariaScreener_Sudan* (NIH/LHNCBC; MobileNetV2; thick smear, Sudan), *MalariaScreener_Thick* (NIH/LHNCBC; MobileNetV2; thick smear, Chittagong, Bangladesh), *MalariaScreener_Thin* (NIH/LHNCBC; MobileNetV2; thin smear, Chittagong, Bangladesh), and *fbononibelloepoch_YOLOv8* (modern YOLOv8n object detector). Columns denote the evaluation modality: thick blood smear cohort ($n=3,043$) and thin blood smear cohort ($n=1,011$). Cell annotations denote true positive sensitivity (percentage of parasitized slides correctly flagged positive). Cells are shaded along a normalized colorblind-safe gradient from high sensitivity (emerald) to poor generalizability (terracotta).

---

## 2. Thick Smear Cohort Figures

### Figure 2: Zero-Shot Diagnostic Performance on Ghanaian Thick Blood Smears
* **File Location**: `figures/02_thick_smears/fig_thick_sensitivity_comparison.[png|pdf|svg]`
* **Caption**:  
  **Figure 2 | Zero-Shot Diagnostic Sensitivity on Ghanaian Thick Blood Smear Micrographs ($n=3,043$).** Bar chart comparing the slide-level diagnostic sensitivity across all four candidate models against the World Health Organization (WHO) recommended clinical triage sensitivity threshold of 90% (dashed crimson line). Error bars denote 95% Clopper-Pearson binomial confidence intervals. Bars are styled using certified Okabe-Ito colorblind-safe palettes with distinct luminance profiles to ensure grayscale print legibility. All micrographs were captured via smartphone cameras attached to light microscopes at Princess Marie Louise Hospital, Accra.

### Figure 3: Focus Quality vs. Diagnostic Sensitivity on Thick Smears
* **File Location**: `figures/02_thick_smears/fig_thick_quality_sensitivity_lines.[png|pdf|svg]`
* **Caption**:  
  **Figure 3 | Quality-Stratified Robustness of Thick Smear Models across Focus Degradation Tertiles.** Line plot displaying diagnostic sensitivity as a function of optical focus quality, quantified via circular ocular Field-of-View (FOV) masked Laplacian variance. Smear micrographs ($n=3,043$) are partitioned into tertiles: Low focus quality ($\le 9.14$), Medium focus quality ($9.14 - 15.65$), and High focus quality ($> 15.65$). Distinct geometric markers and Okabe-Ito hues denote individual models. The red dashed line denotes the WHO 90% clinical triage threshold.

### Figure 4: Five-Panel Cross-Model Visual Comparison on a Ghanaian Thick Smear
* **File Location**: `figures/02_thick_smears/fig_thick_crossmodel_panel.[png|pdf|svg]`
* **Caption**:  
  **Figure 4 | Cross-Model Diagnostic Field Inspection on a Representative Ghanaian Thick Blood Smear Micrograph.** Multi-panel comparison demonstrating model inference behavior on the same microscopic field-of-view. Panel 1 shows the ground truth reference annotated by minoHealth AI Labs expert microscopists with green bounding boxes surrounding confirmed *Plasmodium* parasites. Panels 2–4 display the slide-level classification decisions of the three NIH MalariaScreener architectures (Sudan, Thick, and Thin) with binary confidence scores; these models produce no spatial bounding boxes. Panel 5 displays the spatial detections produced by the YOLOv8 object detector with predicted bounding boxes and class-specific confidence values.

---

## 3. Thin Smear Cohort Figures

### Figure 5: Zero-Shot Sensitivity and Specificity on Ghanaian Thin Blood Smears
* **File Location**: `figures/03_thin_smears/fig_thin_sensitivity_comparison.[png|pdf|svg]`
* **Caption**:  
  **Figure 5 | Dual Diagnostic Performance (Sensitivity and Specificity) on Ghanaian Thin Blood Smears ($n=1,011$).** Paired bar chart displaying slide-level diagnostic sensitivity on confirmed parasitized slides ($n=960$; solid colored bars) and diagnostic specificity on uninfected negative control slides containing only white blood cells and staining artifacts ($n=51$; hatched neutral bars). The crimson dashed line indicates the WHO 90% clinical sensitivity benchmark. Demonstrates the marked false-positive bias observed when Asian-trained thin smear models encounter West African clinical microscopy.

### Figure 6: Focus Quality vs. Diagnostic Sensitivity on Thin Smears
* **File Location**: `figures/03_thin_smears/fig_thin_quality_sensitivity_lines.[png|pdf|svg]`
* **Caption**:  
  **Figure 6 | Optical Blur Robustness on Thin Blood Smears across Focus Quality Strata.** Diagnostic sensitivity across focus blur tertiles on the Ghanaian thin-smear cohort ($n=1,011$). Demonstrates differential sensitivity collapse under optical blurring when individual red blood cell borders and intracellular ring-stage trophozoites lose morphological definition.
