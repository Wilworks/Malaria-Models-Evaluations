# Data Sheet for Malaria Image Data Collected in Ghana Under the Lacuna Project

*Created by minoHealth AI Labs following the Datasheets for Datasets framework (Gebru et al., 2021).*

---

## 1. Motivation

### For what purpose was the dataset created? Was there a specific task in mind?
The dataset was created with the specific purpose of advancing research and development in the field of malaria diagnosis and detection. By collecting the dataset from Ghana, a diverse and well-annotated collection of malaria thin and thick blood smear images will be made available to support the development of accurate, efficient, and scalable methods for malaria diagnosis.

### Who created this dataset and on behalf of which entity?
The dataset was created by a team of data scientists from **minoHealth AI Labs**, with support from medical officers, medical laboratory scientists, and officers.

---

## 2. Composition

### What do the instances that comprise the dataset represent?
Each instance in the dataset includes:
- **Thick and thin blood smear images** captured through the lens of a microscope (`JPEG`).
- **Thick smear annotations**: `Parasite`, `White Blood Cells`.
- **Thin smear annotations**: `Gametocytes`, `Trophozoites`, `Ring stage`, `White Blood Cells`, `Artifacts`.
- File type: Images and bounding box annotations (`makesense.ai` output).
- Location metadata: Present in schema but without recorded values.

### How many instances are there in total?
- **Thick smear images**: 3,000 instances
- **Thin smear images**: 1,000 instances

### Collection Site
Captured at **Princess Marie Louise Children’s Hospital** in Accra, Greater Accra Region, Ghana.

### Target Classes & Labels
- **Thick Smear**: `Parasite`, `White Blood Cells` (WBC)
- **Thin Smear**: `Gametocytes`, `Trophozoites`, `Ring stage`, `White Blood Cells`, `Artifacts`

### Sources of Noise / Quality Limitations
> *"Due to the quality of the mobile devices used to collect the data (capture the images), some instances have low quality making it difficult to observe classes in the data."*

---

## 3. Collection Process

- **Collection Method**: Manually acquired using mobile phone cameras attached to microscope lenses at various angles to introduce realistic field variation.
- **Timeframe**: Collected over an 8-month period.
- **Supervision**: Collected by the minoHealth AI team under supervision from lab technicians and medical laboratory scientists.

---

## 4. Preprocessing, Cleaning & Labeling

- **Annotation Tool**: Bounding boxes annotated using `makesense.ai`.
- **Pre-trained Assistance**: Semi-supervised object detection models were used during annotation to assist human annotators.

---

## 5. Maintenance & Contact

- **Curators**: minoHealth AI Labs & Makerere University AI Lab.
- **Lead Contact**: Darlington Akogo (`darlington@gudra-studio.com`).

---

## References

1. Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86-92.
