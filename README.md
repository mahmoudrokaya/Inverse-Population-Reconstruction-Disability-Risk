# Inverse Population Reconstruction of Disability Risk

## Overview

This repository contains the official implementation of the study:

**Inverse Population Reconstruction of Disability Risk Under Partial Observability: A Structural Learning Framework for Longitudinal Health Data**

This project presents a reproducible structural learning framework for disability risk prediction using longitudinal health data under heterogeneous and partially observable conditions.

The framework integrates:

- latent structural reconstruction
- dependency-aware feature modeling
- longitudinal validation
- partial observability analysis
- external cross-dataset harmonization

The repository includes all experiment scripts, documentation, and reproducibility materials required to reproduce the computational results reported in the manuscript.

---

# Repository Structure

```text
Inverse-Population-Reconstruction-Disability-Risk/
│
├── code/
│   ├── Experiment_0_Latent_Structure_Audit.py
│   ├── Experiment_1_Partial_Observability_Reconstruction.py
│   ├── Experiment_2_Internal_CV.py
│   ├── Experiment_3_Temporal_Cross_Cycle_Validation.py
│   ├── Experiment_4A_HRS_Structural_Audit_and_Harmonization.py
│   └── generate_folder_structure.py
│
├── data/
│   ├── data_raw/
│   └── HRS/
│
├── results/
│
├── docs/
│   ├── README_Experiment_0.md
│   ├── README_Experiment_1.md
│   ├── README_Experiment_2.md
│   ├── README_Experiment_3.md
│   └── README_Experiment_4A.md
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Dataset Information

## NHANES Dataset

This work uses data derived from the **National Health and Nutrition Examination Survey (NHANES)**.

Official source:

https://www.cdc.gov/nchs/nhanes/

Cycles included:

- NHANES 2011–2012
- NHANES 2013–2014
- NHANES 2015–2016
- NHANES 2017–2018

Required files:

- `NHANES_2011-2012_final_processed.csv`
- `NHANES_2013-2014_final_processed.csv`
- `NHANES_2015-2016_final_processed.csv`
- `NHANES_2017-2018_final_processed.csv`
- `NHANES_pooled_filled_corrected_blue.xlsx`

Store these files in:

```text
data/data_raw/
```

---

## HRS Dataset

External validation uses the:

**RAND HRS Longitudinal File 2022 (V1)**

Official source:

https://hrsdata.isr.umich.edu/

Required files:

- `randhrs1992_2022v1.dta`
- `randhrs1992_2022v1.pdf`

Store these files in:

```text
data/HRS/
```

---

# Included Experiments

## Experiment 0 — Latent Structure Audit

Evaluates:

- latent manifold geometry
- PCA structure
- correlation patterns
- subgroup consistency
- structural dependency preservation

---

## Experiment 1 — Partial Observability Reconstruction

Evaluates model robustness under incomplete and partially observed data.

Scenarios:

- 5% observability
- 10% observability
- 20% observability
- 30% observability

Outputs include:

- reconstruction fidelity
- predictive utility
- uncertainty estimates
- fairness evaluation
- runtime analysis

---

## Experiment 2 — Internal Cross-Validation

Performs:

- 5-fold internal validation
- fold-wise performance analysis
- stability assessment

Metrics include:

- ROC-AUC
- PR-AUC
- F1-score
- calibration
- variance across folds

---

## Experiment 3 — Temporal Cross-Cycle Validation

Training:

- NHANES 2011–2016

Testing:

- NHANES 2017–2018

Used to evaluate temporal generalization and cross-cycle robustness.

---

## Experiment 4A — HRS Structural Audit and Harmonization

External validation using HRS data.

Includes:

- NHANES–HRS harmonization
- structural variable alignment
- external domain consistency
- cross-dataset validation

---

# Installation

Clone repository:

```bash
git clone https://github.com/mahmoudrokaya/Inverse-Population-Reconstruction-Disability-Risk.git
cd Inverse-Population-Reconstruction-Disability-Risk
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Requirements

Recommended environment:

- Python 3.10+
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- catboost
- statsmodels
- networkx
- shap
- openpyxl
- pyreadstat

---

# Running the Code

Example:

```bash
python code/Experiment_0_Latent_Structure_Audit.py
```

or

```bash
python code/Experiment_1_Partial_Observability_Reconstruction.py
```

Outputs are automatically saved to:

```text
results/
```

including:

- figures
- tables
- logs
- performance summaries
- reconstructed outputs

---

# Reproducibility

This repository supports full computational reproducibility.

Included:

- public dataset references
- full experiment scripts
- experiment documentation
- fixed folder structure
- reproducible preprocessing workflow
- hyperparameter reporting
- external validation pipeline

Recommended execution order:

```text
Experiment 0 → Experiment 1 → Experiment 2 → Experiment 3 → Experiment 4A
```

---

# Citation

If you use this repository, please cite:

**Mahmoud Rokaya et al.**  
*Inverse Population Reconstruction of Disability Risk Under Partial Observability: A Structural Learning Framework for Longitudinal Health Data.*

---

# License

MIT License

---

# Contact

**Mahmoud Rokaya**  
Associate Professor of Information Science  
Taif University  
Saudi Arabia

GitHub:  
https://github.com/mahmoudrokaya

---

# Acknowledgments

NHANES data are publicly available through the U.S. Centers for Disease Control and Prevention.

The Health and Retirement Study (HRS) is sponsored by the National Institute on Aging and conducted by the University of Michigan.

RAND HRS data are produced by the RAND Center for the Study of Aging.
