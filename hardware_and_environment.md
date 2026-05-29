# Hardware and Environment Configuration

## Purpose

This document describes the hardware and software environment used to execute the experiments contained in this repository.

The purpose of providing these details is to support computational reproducibility and facilitate independent verification of the reported results.

---

# Operating System

The experiments were developed and tested primarily on Microsoft Windows.

| Component | Specification |
|------------|------------|
| Operating System | Windows 10 / Windows 11 |
| Architecture | 64-bit |
| File System | NTFS |

The code is platform-independent and can be executed on Linux and macOS systems provided the required dependencies are installed.

---

# Python Environment

| Component | Version |
|------------|------------|
| Python | 3.10+ |
| Package Manager | pip |
| Alternative Environment Manager | Conda |

Recommended installation:

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f environment.yml
```

---

# Core Software Libraries

The following packages are required by the experiments:

| Package |
|------------|
| pandas |
| numpy |
| scipy |
| scikit-learn |
| matplotlib |
| seaborn |
| statsmodels |
| networkx |
| openpyxl |
| pyreadstat |
| xgboost |
| lightgbm |
| catboost |
| shap |
| joblib |
| tqdm |

The exact versions are specified in:

```text
requirements.txt
```

and

```text
environment.yml
```

---

# Hardware Requirements

## Minimum Recommended Configuration

| Component | Specification |
|------------|------------|
| CPU | 4-Core Processor |
| RAM | 8 GB |
| Storage | 10 GB Free Space |
| GPU | Not Required |

---

## Recommended Configuration

| Component | Specification |
|------------|------------|
| CPU | 8-Core or Higher |
| RAM | 16 GB or More |
| Storage | SSD Recommended |
| GPU | Optional |

---

# Computational Characteristics

The proposed framework is designed for tabular health data and does not require deep neural network training or GPU acceleration.

Key characteristics:

- CPU-based execution
- Moderate memory usage
- No distributed computing required
- No cloud infrastructure required
- Suitable for desktop and workstation environments

---

# Data Storage Requirements

The repository expects the following structure:

```text
Inverse-Population-Reconstruction-Disability-Risk/
│
├── data/
│   ├── data_raw/
│   └── HRS/
│
├── code/
│
├── docs/
│
└── results/
```

Input datasets include:

- NHANES processed datasets
- RAND HRS Longitudinal File 2022 (V1)

Output files include:

- CSV tables
- Excel reports
- PNG figures
- PDF summaries
- Reconstruction outputs

---

# Randomness Control

To improve reproducibility, experiments use fixed random seeds whenever applicable.

Default seed:

```text
42
```

Applied to:

- NumPy random generators
- Scikit-learn estimators
- Cross-validation procedures
- Clustering algorithms
- Reconstruction simulations

---

# Reproducibility Notes

The experiments are deterministic under fixed random seeds and identical software environments.

Minor numerical differences may occur due to:

- operating system differences
- package version differences
- processor architecture differences
- floating-point implementation differences

These differences are expected to be negligible and should not materially affect the reported conclusions.

---

# Performance Notes

Approximate execution characteristics:

| Experiment | Relative Runtime |
|------------|------------|
| Experiment 0 | Low |
| Experiment 1 | Moderate |
| Experiment 2 | Moderate |
| Experiment 3 | Moderate |
| Experiment 4A | Moderate |

Actual execution time depends on:

- CPU speed
- available RAM
- dataset size
- number of validation runs

---

# Repository Support Files

The computational environment is fully documented through:

```text
README.md
requirements.txt
environment.yml
hyperparameters.md
reproducibility_checklist.md
```

Together, these files provide the information required to reproduce the experiments and verify the reported results.
