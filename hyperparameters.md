# Hyperparameter Configuration

## Overview

This document summarizes the principal hyperparameters and experimental settings used throughout the repository.

The objective of providing these settings is to support computational reproducibility, independent verification, and transparent reporting of all experiments.

---

# Global Configuration

| Parameter | Value |
|------------|------------|
| Random Seed | 42 |
| Data Type | Tabular Longitudinal Health Data |
| Missing Value Strategy | Structural Reconstruction |
| Validation Strategy | Internal, Temporal, and External Validation |
| Scaling Method | StandardScaler |
| Numerical Precision | Float64 |
| Output Format | CSV, XLSX, PNG, PDF |

---

# Experiment 0 — Latent Structure Audit

## Principal Component Analysis (PCA)

| Parameter | Value |
|------------|------------|
| PCA Solver | Auto |
| Standardization | Enabled |
| Number of Components | Determined Automatically |
| Explained Variance Threshold | 95% |
| Random State | 42 |

## Clustering Analysis

| Parameter | Value |
|------------|------------|
| KMeans Initialization | k-means++ |
| Number of Initializations | 10 |
| Maximum Iterations | 300 |
| Random State | 42 |

## Mutual Information

| Parameter | Value |
|------------|------------|
| Random State | 42 |
| Estimator | sklearn default |

---

# Experiment 1 — Partial Observability Reconstruction

## Observability Scenarios

| Scenario | Visible Features |
|------------|------------|
| Scenario 1 | 5% |
| Scenario 2 | 10% |
| Scenario 3 | 20% |
| Scenario 4 | 30% |

## Reconstruction Settings

| Parameter | Value |
|------------|------------|
| Random Seed | 42 |
| Missing Feature Selection | Random Masking |
| Reconstruction Mode | Structural Dependency Recovery |
| Number of Repeated Runs | 5 |
| Output Samples Saved | Yes |
| Fairness Evaluation | Enabled |
| Uncertainty Evaluation | Enabled |

---

# Experiment 2 — Internal Cross-Validation

## Cross-Validation

| Parameter | Value |
|------------|------------|
| Number of Folds | 5 |
| Stratification | Enabled |
| Shuffle | Enabled |
| Random State | 42 |

## Evaluation Metrics

| Metric |
|------------|
| ROC-AUC |
| PR-AUC |
| Accuracy |
| Precision |
| Recall |
| F1-Score |
| Calibration Error |

---

# Experiment 3 — Temporal Cross-Cycle Validation

## Dataset Split

| Partition | Cycles |
|------------|------------|
| Training | NHANES 2011–2012, 2013–2014, 2015–2016 |
| Testing | NHANES 2017–2018 |

## Configuration

| Parameter | Value |
|------------|------------|
| Temporal Leakage Prevention | Enabled |
| Feature Harmonization | Enabled |
| Standardization Based on Training Data Only | Yes |
| Random State | 42 |

---

# Experiment 4A — HRS Structural Audit and Harmonization

## Harmonization Settings

| Parameter | Value |
|------------|------------|
| Variable Mapping Strategy | Manual Harmonization |
| Structural Consistency Analysis | Enabled |
| Missing Data Handling | Structural Reconstruction |
| External Validation Dataset | RAND HRS Longitudinal File 2022 (V1) |
| Random State | 42 |

---

# Statistical Reporting

## Internal Validation

| Parameter | Value |
|------------|------------|
| Validation Method | 5-Fold Cross-Validation |
| Number of Repeated Runs | 5 |
| Mean Reporting | Yes |
| Standard Deviation Reporting | Yes |
| Confidence Interval Reporting | Yes |

## Fairness Assessment

| Parameter | Value |
|------------|------------|
| Demographic Group Analysis | Enabled |
| Bias Evaluation | Enabled |
| Structural Consistency Evaluation | Enabled |

## Uncertainty Assessment

| Parameter | Value |
|------------|------------|
| Reconstruction Uncertainty | Enabled |
| Variance Analysis | Enabled |
| Robustness Analysis | Enabled |

---

# Hardware Configuration

The experiments were originally executed using a standard workstation environment.

| Parameter | Value |
|------------|------------|
| CPU | Multi-Core x64 Processor |
| RAM | ≥16 GB Recommended |
| GPU | Not Required |
| Operating System | Windows 10/11 Compatible |
| Python Version | 3.10+ |

---

# Reproducibility Notes

- All experiments use a fixed random seed where applicable.
- Data preprocessing is performed using training data only to prevent information leakage.
- Temporal validation strictly separates earlier NHANES cycles from later cycles.
- External validation is conducted on an independent HRS dataset.
- Results, figures, and tables are automatically saved in the corresponding experiment output directories.

---

# Important Note

Some experiments automatically adapt parameters to the dimensionality of the dataset and available features. Therefore, the exact number of latent dimensions, selected principal components, or harmonized variables may vary slightly depending on the version of the input data used.
