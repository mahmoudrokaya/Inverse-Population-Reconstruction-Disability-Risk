# -*- coding: utf-8 -*-
r"""
Experiment 2: Internal Cross-Validation within NHANES
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from scipy.stats import wasserstein_distance
import time

# ==============================
# PATHS
# ==============================
BASE_DIR = Path(r"D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments")
DATA_RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "Experiment2_Internal_CV"

OUTPUT_DIR.mkdir(exist_ok=True)

# ==============================
# CONFIG
# ==============================
OBS_FRACS = [0.05, 0.1, 0.2, 0.3]
LATENT_DIMS = [12, 16, 20, 23]
N_SPLITS = 5
TARGET = "disability_stage"

# ==============================
# LOAD DATA
# ==============================
files = list(DATA_RAW_DIR.glob("*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df = df.dropna(subset=[TARGET])

# ==============================
# PREPROCESS
# ==============================
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(str)

# simple numeric conversion
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

X = X.dropna(axis=1, how="all")

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ==============================
# RESULTS STORAGE
# ==============================
results = []

# ==============================
# CROSS VALIDATION
# ==============================
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

for frac in OBS_FRACS:
    print(f"\n=== Observed Fraction: {frac} ===")

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{N_SPLITS}")

        X_train_full = X.iloc[train_idx]
        y_train_full = y.iloc[train_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # simulate partial observation
        obs_size = int(len(X_train_full) * frac)
        obs_idx = np.random.choice(len(X_train_full), obs_size, replace=False)

        X_obs = X_train_full.iloc[obs_idx]
        y_obs = y_train_full.iloc[obs_idx]

        X_hidden = X_train_full.drop(X_train_full.index[obs_idx])
        y_hidden = y_train_full.drop(y_train_full.index[obs_idx])

        for latent_dim in LATENT_DIMS:

            # ======================
            # RECONSTRUCTION MODEL
            # ======================
            pca = PCA(n_components=min(latent_dim, X_obs.shape[1]))
            Z = pca.fit_transform(X_obs)

            # Gaussian approximation
            mean = Z.mean(axis=0)
            cov = np.cov(Z, rowvar=False) + np.eye(Z.shape[1]) * 1e-5

            Z_hat = np.random.multivariate_normal(mean, cov, size=len(X_hidden))
            X_hat = pca.inverse_transform(Z_hat)

            X_hat = pd.DataFrame(X_hat, columns=X.columns)

            # ======================
            # METRICS
            # ======================

            # Fidelity
            wass = np.mean([
                wasserstein_distance(X_hidden[col], X_hat[col])
                for col in X.columns
            ])

            # Utility
            clf = RandomForestClassifier(n_estimators=100)
            clf.fit(X_hat, y_obs.sample(len(X_hat), replace=True).values)

            y_pred = clf.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="macro")

            results.append({
                "fraction": frac,
                "fold": fold,
                "latent_dim": latent_dim,
                "wasserstein": wass,
                "macro_f1": f1
            })

# ==============================
# SAVE RESULTS
# ==============================
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / "experiment2_cv_results.csv", index=False)

# ==============================
# AGGREGATION
# ==============================
summary = results_df.groupby(["fraction", "latent_dim"]).agg({
    "wasserstein": ["mean", "std"],
    "macro_f1": ["mean", "std"]
}).reset_index()

summary.to_csv(OUTPUT_DIR / "experiment2_cv_summary.csv", index=False)

# ==============================
# PLOT
# ==============================
for metric in ["wasserstein", "macro_f1"]:
    plt.figure()

    for dim in LATENT_DIMS:
        subset = summary[summary["latent_dim"] == dim]
        plt.plot(subset["fraction"], subset[(metric, "mean")], marker='o', label=f"dim={dim}")

    plt.xlabel("Observed Fraction")
    plt.ylabel(metric)
    plt.title(f"{metric} vs Observed Fraction (Cross-Validation)")
    plt.legend()
    plt.savefig(OUTPUT_DIR / f"{metric}_cv_plot.png")
    plt.close()

print("\n✅ Experiment 2 Completed")