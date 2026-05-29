# Random Seeds Configuration

## Purpose

This document describes the random seed settings used throughout the repository.

Random seed control is included to improve reproducibility, stability of reported results, and consistency across repeated executions.

---

# Global Random Seed

The default seed used throughout the project is:

```text
42
```

This seed was selected and fixed across all experiments wherever stochastic processes were involved.

---

# Where Random Seeds Are Applied

The seed value is applied to all components involving randomness, including:

- dataset shuffling
- train/test splitting
- fold generation in cross-validation
- feature masking under partial observability
- PCA initialization where applicable
- KMeans clustering
- random feature sampling
- repeated simulation runs
- bootstrap or repeated evaluation procedures
- model initialization when supported by the estimator

---

# Python Random Module

```python
import random
random.seed(42)
```

---

# NumPy

```python
import numpy as np
np.random.seed(42)
```

---

# Scikit-learn

Applied in components such as:

- `train_test_split()`
- `StratifiedKFold()`
- `KFold()`
- `PCA()`
- `KMeans()`

Example:

```python
random_state=42
```

---

# Cross-Validation

All internal validation folds use:

```text
shuffle = True
random_state = 42
```

Example:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

---

# Partial Observability Reconstruction

For Experiment 1, the masking process that hides a percentage of variables before reconstruction is generated using the fixed seed:

```text
42
```

This ensures the same missingness patterns can be reproduced across runs.

Observed settings include:

- 5% observability
- 10% observability
- 20% observability
- 30% observability

---

# Clustering and Latent Space Analysis

For Experiment 0:

KMeans clustering and related latent-space exploratory procedures use:

```text
random_state = 42
```

to maintain consistent clustering assignments across repeated runs.

---

# External Validation

Experiments using HRS validation also preserve the same random seed for:

- feature alignment
- harmonization sampling where applicable
- validation reproducibility

---

# Why This Matters

Fixing random seeds helps ensure:

- reproducible performance metrics
- stable cross-validation folds
- repeatable reconstruction outputs
- comparable experimental results
- easier verification by reviewers and external researchers

---

# Reproducibility Note

Although random seeds are fixed, very small numerical differences may still occur across systems due to:

- operating system differences
- Python version differences
- package version differences
- floating-point arithmetic differences
- processor architecture differences

These differences are expected to be minimal and should not materially affect the reported findings.

---

# Summary

| Component | Seed |
|---|---:|
| Python `random` | 42 |
| NumPy | 42 |
| Scikit-learn | 42 |
| Cross-validation | 42 |
| Partial observability masking | 42 |
| Clustering | 42 |
| Repeated simulations | 42 |

---

# Recommendation

When reproducing the experiments, users are encouraged to preserve:

```text
random_seed = 42
```

to obtain results as close as possible to those reported in the manuscript.
