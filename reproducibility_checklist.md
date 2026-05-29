# Reproducibility Checklist

## Project Information

**Repository:** Inverse-Population-Reconstruction-Disability-Risk

**Purpose:** Reproducible implementation of inverse population reconstruction and structural learning for disability risk prediction using NHANES and HRS longitudinal datasets.

---

# Data Availability

| Item | Status | Details |
|--------|--------|--------|
| Public Dataset | Yes | NHANES public-use data |
| External Validation Dataset | Yes | RAND HRS Longitudinal File 2022 (V1) |
| Dataset Source Provided | Yes | Included in README and manuscript |
| Dataset URLs Provided | Yes | Included in repository documentation |
| Dataset Documentation Available | Yes | NHANES and HRS documentation included |
| Raw Data Included in Repository | No | Downloaded separately from official sources |
| Processed Data Included | No | Users must generate from official sources |

---

# Code Availability

| Item | Status |
|--------|--------|
| Full Source Code Available | Yes |
| Public Repository | Yes |
| Repository Version Controlled | Yes |
| README Included | Yes |
| Experiment Documentation Included | Yes |
| Installation Instructions Included | Yes |
| Execution Instructions Included | Yes |
| License Included | Yes |

---

# Computational Environment

| Item | Status |
|--------|--------|
| Python Version Specified | Yes |
| Package Dependencies Listed | Yes |
| requirements.txt Included | Yes |
| environment.yml Included | Yes |
| Operating System Independent | Yes |
| Relative Paths Supported | Yes |

Recommended environment:

- Python 3.10+
- Windows, Linux, or macOS

---

# Experimental Design

| Item | Status |
|--------|--------|
| Dataset Description Provided | Yes |
| Feature Definitions Provided | Yes |
| Experimental Workflow Described | Yes |
| Training Procedure Documented | Yes |
| Validation Procedure Documented | Yes |
| External Validation Included | Yes |
| Reproducible Folder Structure Provided | Yes |

---

# Data Splitting and Validation

| Item | Status |
|--------|--------|
| Internal Validation | Yes |
| 5-Fold Cross-Validation | Yes |
| Temporal Validation | Yes |
| External Validation | Yes |
| Participant-Level Separation | Yes |
| Leakage Prevention Strategy Documented | Yes |

Validation experiments:

1. Internal Cross-Validation
2. Partial Observability Reconstruction
3. Temporal Cross-Cycle Validation
4. External HRS Validation

---

# Hyperparameter Reporting

| Item | Status |
|--------|--------|
| Hyperparameters Reported | Yes |
| Default Parameters Documented | Yes |
| Experiment-Specific Settings Available | Yes |
| Random Seeds Documented | Yes |

See:

- `hyperparameters.md`
- Experiment-specific documentation files

---

# Statistical Reporting

| Item | Status |
|--------|--------|
| Mean Performance Reported | Yes |
| Standard Deviation Reported | Yes |
| Cross-Validation Statistics Reported | Yes |
| Confidence Estimates Reported | Yes |
| Comparative Evaluation Reported | Yes |

---

# Fairness and Robustness

| Item | Status |
|--------|--------|
| Fairness Evaluation Included | Yes |
| Demographic Analysis Included | Yes |
| Partial Observability Analysis Included | Yes |
| Missingness Robustness Evaluated | Yes |
| Structural Consistency Evaluated | Yes |
| Uncertainty Assessment Included | Yes |

---

# Reproducible Outputs

| Item | Status |
|--------|--------|
| Figures Automatically Generated | Yes |
| Tables Automatically Generated | Yes |
| Logs Automatically Generated | Yes |
| Reconstruction Outputs Saved | Yes |
| Intermediate Results Saved | Yes |
| Final Results Saved | Yes |

Output directory:

```text
results/
```

---

# Experiment Execution Order

Recommended execution order:

```text
Experiment 0 → Experiment 1 → Experiment 2 → Experiment 3 → Experiment 4A
```

| Experiment | Description |
|------------|------------|
| Experiment 0 | Latent Structure Audit |
| Experiment 1 | Partial Observability Reconstruction |
| Experiment 2 | Internal Cross-Validation |
| Experiment 3 | Temporal Cross-Cycle Validation |
| Experiment 4A | HRS Structural Audit and Harmonization |

---

# Third-Party Resources

## NHANES

National Health and Nutrition Examination Survey (NHANES)

Source:

https://www.cdc.gov/nchs/nhanes/

---

## HRS

Health and Retirement Study (HRS)

Source:

https://hrsdata.isr.umich.edu/

RAND HRS Longitudinal File 2022 (V1)

---

# Reproduction Steps

1. Clone repository.
2. Install dependencies from `requirements.txt` or `environment.yml`.
3. Download NHANES and HRS datasets from official sources.
4. Place datasets in the required folders.
5. Execute experiments in the recommended order.
6. Review generated outputs in the `results/` directory.

---

# Contact

**Mahmoud Rokaya**  
Associate Professor of Information Science  
Taif University  
Saudi Arabia

GitHub:  
https://github.com/mahmoudrokaya

---

# Reproducibility Statement

All experiments reported in the associated manuscript were designed to be computationally reproducible using publicly accessible datasets, openly available source code, documented preprocessing procedures, fixed experimental workflows, and externally validated evaluation protocols.
