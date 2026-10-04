# Data Sheet for Kano Nigeria Thin Blood Smear Dataset

*Documented following the Datasheets for Datasets framework (Gebru et al., 2021) for the West African Malaria AI Benchmark (WAM-Bench).*

---

## 1. Motivation

### For what purpose was the dataset created?
The dataset was collected to support automated morphological evaluation of thin blood smear microscopy images, specifically examining erythrocyte morphology variations under malaria infection. In clinical hematology, severe *Plasmodium falciparum* parasitemia frequently induces erythrocyte aggregation and **rouleaux formation** (stacked-coin red blood cell morphology) driven by elevated plasma fibrinogen and cytoadherence.

### Who created this dataset and on behalf of which entity?
Created by **Muhammad et al.** (2024), Department of Mathematical Sciences / Faculty of Science, Bayero University Kano, Nigeria, in collaboration with hematology laboratory units in Kano State, Nigeria.  
- **Persistent Identifier**: [Zenodo Record 13763939](https://zenodo.org/records/13763939)
- **License**: Creative Commons Attribution 4.0 International (CC-BY 4.0).

---

## 2. Composition

### What do the instances comprise?
- **Thin blood smear micrographs** captured under 100× oil immersion optical microscopy using mobile optical sensor adapters.
- **Positive Cohort (`positive/`)**: Micrographs exhibiting prominent **Rouleaux morphology** and acute *Plasmodium falciparum* malaria infection ($N = 772$ full-field micrographs).
- **Negative Cohort (`negative/`)**: Micrographs exhibiting normal, uninfected, dispersed discocyte RBC morphology ($N = 772$ full-field micrographs).
- **Format**: High-resolution digital micrographs ($2992 \times 4000$ pixels, 24-bit RGB JPEG).
- **Cropped Tiles**: 24,717 standardized $750 \times 750$ image patches (12,361 normal, 12,356 rouleaux).

### Target Classes & Ground Truth
- Binary whole-slide diagnostic classification:
  - `1` (Positive / Infected / Rouleaux RBC morphology)
  - `0` (Negative / Control / Normal RBC morphology)

### Provenance & Clinical Context
- **Clinical Site**: Health facility pathology and medical laboratory centers, Kano Metropolis, Kano State, Northern Nigeria.
- **Geographic Context**: Sahelian/Sudan Savanna eco-epidemiological zone, characterized by hyper-endemic seasonal malaria transmission dominated by *P. falciparum*.

---

## 3. Integration into WAM-Bench

In WAM-Bench, the Kano cohort forms the third multi-center West African clinical evaluation node:
1. **Princess Marie Louise Children's Hospital**, Accra, Ghana (Lacuna Dataset: 3,045 thick, 1,011 thin)
2. **Osun State Healthcare Centers**, Nigeria (Adeleke Dataset: 859 thick)
3. **Kano State Health Facilities**, Northern Nigeria (Muhammad Dataset: 1,544 thin)

This enables comprehensive geographic cross-validation (Coastal Gulf of Guinea vs. Inland Forest vs. Sahelian Savanna) and multi-device sensor robustness testing across deep learning architectures.
