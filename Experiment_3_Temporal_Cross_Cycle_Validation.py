# -*- coding: utf-8 -*-
r"""
Experiment 3: Temporal / Cross-Cycle NHANES Validation

Root folder:
D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments

Temporal design:
Train on 2011-2012, 2013-2014, 2015-2016
Validate on 2017-2018
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments")
DATA_RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "Experiment3_Temporal_Cross_Cycle_Validation"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
RECON_DIR = OUTPUT_DIR / "reconstructed_samples"

for d in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, RECON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TRAIN_CYCLES = ["2011-2012", "2013-2014", "2015-2016"]
TEST_CYCLE = "2017-2018"
OBSERVED_FRACTIONS = [0.05, 0.10, 0.20, 0.30, 1.00]
LATENT_DIMS = [12, 16, 20, 23]
N_RECONSTRUCTIONS = 5
DEFAULT_LATENT_DIM = 20

TARGET_CANDIDATES = ["disability_stage", "merge01_vs_23", "disability_binary", "target", "label", "outcome"]
SENSITIVE_CANDIDATES = ["sex", "gender", "race_ethnicity", "race", "education", "income_poverty_ratio", "age_group"]
DIRECT_LEAKAGE_CANDIDATES = [
    "disability_stage", "merge01_vs_23", "disability_binary",
    "mobility_difficulty", "self_care_difficulty", "usual_activities_difficulty",
    "pain_discomfort", "anxiety_depression"
]


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_").replace("/", "_") for c in df.columns]
    return df


def infer_cycle_from_name(path: Path) -> str:
    for cycle in ["2011-2012", "2013-2014", "2015-2016", "2017-2018"]:
        if cycle in path.name:
            return cycle
    return "unknown"


def load_cycle_files(data_dir: Path) -> pd.DataFrame:
    patterns = [
        "NHANES_2011-2012_final_processed*.csv",
        "NHANES_2013-2014_final_processed*.csv",
        "NHANES_2015-2016_final_processed*.csv",
        "NHANES_2017-2018_final_processed*.csv",
    ]
    files, seen = [], set()
    for pat in patterns:
        for p in data_dir.glob(pat):
            if p.name not in seen:
                files.append(p)
                seen.add(p.name)

    if not files:
        raise FileNotFoundError(f"No explicit NHANES cycle CSV files found in {data_dir}")

    frames, manifest = [], []
    for p in files:
        df = pd.read_csv(p)
        df = normalize_column_names(df)
        cyc = infer_cycle_from_name(p)
        df["cycle"] = cyc
        df["source_file"] = p.name
        frames.append(df)
        manifest.append({"file": p.name, "cycle": cyc, "rows": int(df.shape[0]), "columns": int(df.shape[1])})

    pd.DataFrame(manifest).to_csv(TABLE_DIR / "input_cycle_file_manifest.csv", index=False)
    return normalize_column_names(pd.concat(frames, ignore_index=True, sort=False))


def select_target(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError("Target column not found.")


def select_sensitive(df: pd.DataFrame) -> Optional[str]:
    for c in SENSITIVE_CANDIDATES:
        if c in df.columns:
            return c
    lower = {c.lower(): c for c in df.columns}
    for c in SENSITIVE_CANDIDATES:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def plot_and_save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


class Preprocessor:
    def __init__(self, target_col: str, sensitive_col: Optional[str] = None):
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.feature_cols = []
        self.categorical_cols = []
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def choose_features(self, df: pd.DataFrame) -> List[str]:
        exclude = {"source_file", self.target_col}
        for c in DIRECT_LEAKAGE_CANDIDATES:
            if c in df.columns:
                exclude.add(c)
        for c in df.columns:
            if c.lower() in {"seqn", "id", "patient_id", "participant_id"}:
                exclude.add(c)
        return [c for c in df.columns if c not in exclude]

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        df = df.copy()
        self.feature_cols = self.choose_features(df)
        X = df[self.feature_cols].copy().dropna(axis=1, how="all")
        self.feature_cols = list(X.columns)

        for col in X.columns:
            if X[col].dtype == "object":
                if X[col].nunique(dropna=True) <= 80:
                    X[col] = X[col].astype("category")
                else:
                    X[col] = pd.to_numeric(X[col], errors="coerce")

        self.categorical_cols = [c for c in X.columns if str(X[c].dtype) == "category" or X[c].dtype == "object"]

        for col in self.categorical_cols:
            X[col] = X[col].astype(str).replace("nan", np.nan).fillna("Missing")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        for col in X.columns:
            if col not in self.categorical_cols:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        keep = X.nunique(dropna=True)
        keep = keep[keep > 1].index.tolist()
        X = X[keep]
        self.feature_cols = keep
        self.categorical_cols = [c for c in self.categorical_cols if c in keep]

        X_imp = pd.DataFrame(self.imputer.fit_transform(X), columns=self.feature_cols, index=X.index)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X_imp), columns=self.feature_cols, index=X.index)

        y = df[self.target_col].copy()
        s = df[self.sensitive_col].copy() if self.sensitive_col and self.sensitive_col in df.columns else None
        return X_scaled, y, s

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        df = df.copy()
        X = df[self.feature_cols].copy()

        for col in self.categorical_cols:
            X[col] = X[col].astype(str).replace("nan", np.nan).fillna("Missing")
            le = self.label_encoders[col]
            known = set(le.classes_)
            fallback = "Missing" if "Missing" in known else le.classes_[0]
            X[col] = X[col].apply(lambda v: v if v in known else fallback)
            X[col] = le.transform(X[col].astype(str))

        for col in X.columns:
            if col not in self.categorical_cols:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        X_imp = pd.DataFrame(self.imputer.transform(X), columns=self.feature_cols, index=X.index)
        X_scaled = pd.DataFrame(self.scaler.transform(X_imp), columns=self.feature_cols, index=X.index)

        y = df[self.target_col].copy()
        s = df[self.sensitive_col].copy() if self.sensitive_col and self.sensitive_col in df.columns else None
        return X_scaled, y, s


class ConditionalPCAGaussianReconstructor:
    def __init__(self, latent_dim: int, random_state: int):
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.pca = None
        self.class_stats = {}
        self.class_probs = {}
        self.classes_ = []
        self.feature_cols = []

    def fit(self, X_obs: pd.DataFrame, y_obs: pd.Series):
        self.feature_cols = list(X_obs.columns)
        n_comp = min(self.latent_dim, X_obs.shape[1], max(1, X_obs.shape[0] - 1))
        self.pca = PCA(n_components=n_comp, random_state=self.random_state)
        Z = self.pca.fit_transform(X_obs.values)
        y = y_obs.astype(str).fillna("Missing").values

        vals, counts = np.unique(y, return_counts=True)
        self.classes_ = list(vals)
        probs = counts / counts.sum()
        self.class_probs = {v: float(p) for v, p in zip(vals, probs)}

        for cls in self.classes_:
            Zc = Z[y == cls]
            mean = Zc.mean(axis=0)
            if len(Zc) <= 2:
                cov = np.eye(Z.shape[1]) * 0.05
            else:
                cov = np.cov(Zc, rowvar=False)
                if cov.ndim == 0:
                    cov = np.eye(Z.shape[1]) * float(cov)
                cov = np.asarray(cov) + np.eye(np.asarray(cov).shape[0]) * 1e-5
            self.class_stats[cls] = {"mean": mean, "cov": cov}
        return self

    def sample(self, n_samples: int, seed: int):
        rng = np.random.default_rng(seed)
        classes = np.array(self.classes_)
        probs = np.array([self.class_probs[c] for c in classes])
        probs = probs / probs.sum()
        sampled_y = rng.choice(classes, size=n_samples, replace=True, p=probs)
        Zs = [rng.multivariate_normal(self.class_stats[c]["mean"], self.class_stats[c]["cov"]) for c in sampled_y]
        Xh = self.pca.inverse_transform(np.vstack(Zs))
        return pd.DataFrame(Xh, columns=self.feature_cols), pd.Series(sampled_y, name="reconstructed_target")


def avg_ks(X_true, X_hat):
    return float(np.mean([ks_2samp(X_true[c].values, X_hat[c].values).statistic for c in X_true.columns]))


def avg_wass(X_true, X_hat):
    return float(np.mean([wasserstein_distance(X_true[c].values, X_hat[c].values) for c in X_true.columns]))


def corr_dist(X_true, X_hat):
    return float(np.linalg.norm(X_true.corr().fillna(0).values - X_hat.corr().fillna(0).values, ord="fro"))


def cov_dist(X_true, X_hat):
    return float(np.linalg.norm(np.cov(X_true.values, rowvar=False) - np.cov(X_hat.values, rowvar=False), ord="fro"))


def target_l1(y_true, y_hat):
    p1 = y_true.astype(str).value_counts(normalize=True)
    p2 = y_hat.astype(str).value_counts(normalize=True)
    idx = sorted(set(p1.index).union(set(p2.index)))
    return float(np.sum(np.abs(p1.reindex(idx, fill_value=0) - p2.reindex(idx, fill_value=0))))


def utility(X_train, y_train, X_test, y_test):
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        return {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan, "roc_auc_ovr": np.nan}

    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced_subsample")
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    out = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "roc_auc_ovr": np.nan,
    }
    try:
        proba = clf.predict_proba(X_test)
        if len(clf.classes_) == 2:
            out["roc_auc_ovr"] = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            out["roc_auc_ovr"] = float(roc_auc_score(y_test, proba, multi_class="ovr", average="macro"))
    except Exception:
        pass
    return out


def fairness_spd(y, s):
    if s is None:
        return np.nan
    yy = y.astype(str).reset_index(drop=True)
    ss = s.astype(str).fillna("Missing").reset_index(drop=True)
    if yy.nunique() < 2 or ss.nunique() < 2:
        return np.nan
    positive = sorted(yy.unique())[-1]
    rates = [float((yy[ss == g] == positive).mean()) for g in sorted(ss.unique()) if (ss == g).sum() > 0]
    return float(max(rates) - min(rates)) if len(rates) >= 2 else np.nan


def sample_sensitive(s_obs, n, seed):
    if s_obs is None:
        return None
    rng = np.random.default_rng(seed)
    dist = s_obs.astype(str).fillna("Missing").value_counts(normalize=True)
    return pd.Series(rng.choice(dist.index.values, size=n, replace=True, p=dist.values), name=s_obs.name)


def run_fraction(train_df, test_df, target_col, sensitive_col, frac):
    df_train = train_df.dropna(subset=[target_col]).copy()
    df_test = test_df.dropna(subset=[target_col]).copy()
    strat = df_train[target_col].astype(str) if df_train[target_col].nunique() > 1 else None

    if frac < 1:
        df_obs, _ = train_test_split(df_train, train_size=frac, random_state=RANDOM_STATE, stratify=strat)
    else:
        df_obs = df_train.copy()

    pre = Preprocessor(target_col, sensitive_col)
    X_obs, y_obs, s_obs = pre.fit_transform(df_obs)
    X_test, y_test, s_test = pre.transform(df_test)

    rows = []
    for dim in LATENT_DIMS:
        for k in range(N_RECONSTRUCTIONS):
            seed = RANDOM_STATE + int(frac * 1000) + dim + k
            st = time.time()
            model = ConditionalPCAGaussianReconstructor(dim, seed).fit(X_obs, y_obs)
            X_hat, y_hat = model.sample(len(X_test), seed + 1000)
            run_t = time.time() - st
            X_hat = X_hat[X_test.columns]
            s_hat = sample_sensitive(s_obs, len(X_hat), seed + 2000)

            row = {
                "observed_fraction": frac,
                "observed_rows": int(len(df_obs)),
                "temporal_test_rows": int(len(df_test)),
                "latent_dim": dim,
                "reconstruction_id": k + 1,
                "avg_ks_statistic": avg_ks(X_test, X_hat),
                "avg_wasserstein_distance": avg_wass(X_test, X_hat),
                "correlation_structure_distance": corr_dist(X_test, X_hat),
                "covariance_structure_distance": cov_dist(X_test, X_hat),
                "target_distribution_l1_difference": target_l1(y_test, y_hat),
                "test_spd": fairness_spd(y_test, s_test),
                "reconstructed_spd": fairness_spd(y_hat, s_hat),
                "runtime_seconds": run_t,
            }
            row["spd_absolute_gap"] = abs(row["test_spd"] - row["reconstructed_spd"]) if np.isfinite(row["test_spd"]) and np.isfinite(row["reconstructed_spd"]) else np.nan
            row.update({f"utility_{k2}": v for k2, v in utility(X_hat, y_hat, X_test, y_test).items()})
            rows.append(row)

            if frac in [0.10, 0.30, 1.00] and dim == DEFAULT_LATENT_DIM and k == 0:
                out = X_hat.copy()
                out[target_col] = y_hat.values
                if s_hat is not None:
                    out[f"sampled_{sensitive_col}"] = s_hat.values
                out.to_csv(RECON_DIR / f"temporal_reconstructed_obs{int(frac*100)}pct_latent{dim}.csv", index=False)

    return pd.DataFrame(rows)


def aggregate(metrics, baseline):
    metrics.to_csv(TABLE_DIR / "experiment3_temporal_all_reconstruction_metrics.csv", index=False)
    summary = (
        metrics.groupby(["observed_fraction", "latent_dim"])
        .agg(
            avg_ks_statistic_mean=("avg_ks_statistic", "mean"),
            avg_ks_statistic_std=("avg_ks_statistic", "std"),
            avg_wasserstein_distance_mean=("avg_wasserstein_distance", "mean"),
            avg_wasserstein_distance_std=("avg_wasserstein_distance", "std"),
            correlation_structure_distance_mean=("correlation_structure_distance", "mean"),
            correlation_structure_distance_std=("correlation_structure_distance", "std"),
            covariance_structure_distance_mean=("covariance_structure_distance", "mean"),
            covariance_structure_distance_std=("covariance_structure_distance", "std"),
            utility_macro_f1_mean=("utility_macro_f1", "mean"),
            utility_macro_f1_std=("utility_macro_f1", "std"),
            utility_balanced_accuracy_mean=("utility_balanced_accuracy", "mean"),
            utility_balanced_accuracy_std=("utility_balanced_accuracy", "std"),
            spd_absolute_gap_mean=("spd_absolute_gap", "mean"),
            spd_absolute_gap_std=("spd_absolute_gap", "std"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(TABLE_DIR / "experiment3_temporal_summary_by_fraction_and_latent_dim.csv", index=False)
    save_json(baseline, TABLE_DIR / "experiment3_temporal_real_training_baseline.json")

    main = summary[summary["latent_dim"] == DEFAULT_LATENT_DIM].copy()

    plt.figure(figsize=(8, 5))
    plt.plot(main["observed_fraction"], main["avg_wasserstein_distance_mean"], marker="o")
    plt.xlabel("Observed fraction from 2011-2016")
    plt.ylabel("Average Wasserstein distance to 2017-2018")
    plt.title("Temporal Fidelity under Cross-Cycle Validation")
    plot_and_save(FIGURE_DIR / "fig01_temporal_fidelity_wasserstein.png")

    plt.figure(figsize=(8, 5))
    plt.plot(main["observed_fraction"], main["correlation_structure_distance_mean"], marker="o")
    plt.xlabel("Observed fraction from 2011-2016")
    plt.ylabel("Correlation structure distance to 2017-2018")
    plt.title("Temporal Structural Consistency")
    plot_and_save(FIGURE_DIR / "fig02_temporal_structure_distance.png")

    plt.figure(figsize=(8, 5))
    plt.plot(main["observed_fraction"], main["utility_macro_f1_mean"], marker="o")
    plt.axhline(y=baseline.get("macro_f1", np.nan), linestyle="--")
    plt.xlabel("Observed fraction from 2011-2016")
    plt.ylabel("Macro-F1 on 2017-2018")
    plt.title("Temporal Predictive Utility")
    plot_and_save(FIGURE_DIR / "fig03_temporal_utility_macro_f1.png")

    plt.figure(figsize=(8, 5))
    plt.plot(main["observed_fraction"], main["spd_absolute_gap_mean"], marker="o")
    plt.xlabel("Observed fraction from 2011-2016")
    plt.ylabel("Absolute SPD gap")
    plt.title("Temporal Fairness Alignment")
    plot_and_save(FIGURE_DIR / "fig04_temporal_fairness_spd_gap.png")

    pivot = summary.pivot(index="observed_fraction", columns="latent_dim", values="utility_macro_f1_mean")
    plt.figure(figsize=(9, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=f"latent_dim={col}")
    plt.axhline(y=baseline.get("macro_f1", np.nan), linestyle="--", label="real-data baseline")
    plt.xlabel("Observed fraction from 2011-2016")
    plt.ylabel("Macro-F1 on 2017-2018")
    plt.title("Temporal Latent Dimension Sensitivity")
    plt.legend()
    plot_and_save(FIGURE_DIR / "fig05_temporal_latent_dimension_sensitivity.png")

    report = f"""# Experiment 3: Temporal / Cross-Cycle NHANES Validation Report

Training cycles: {TRAIN_CYCLES}
Temporal test cycle: {TEST_CYCLE}

Observed fractions: {OBSERVED_FRACTIONS}
Latent dimensions: {LATENT_DIMS}
Stochastic reconstructions per setting: {N_RECONSTRUCTIONS}

Real-data temporal baseline:
{json.dumps(baseline, indent=4)}
"""
    (OUTPUT_DIR / "Experiment3_Temporal_Cross_Cycle_Validation_Report.md").write_text(report, encoding="utf-8")


def main():
    start = time.time()
    print("=" * 90)
    print("Experiment 3: Temporal / Cross-Cycle NHANES Validation")
    print("=" * 90)
    print(f"Input folder : {DATA_RAW_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    df = load_cycle_files(DATA_RAW_DIR)
    print(f"Loaded data shape: {df.shape}")

    target_col = select_target(df)
    sensitive_col = select_sensitive(df)
    print(f"Detected target column   : {target_col}")
    print(f"Detected sensitive column: {sensitive_col}")

    cycle_counts = df["cycle"].value_counts().reset_index()
    cycle_counts.columns = ["cycle", "rows"]
    cycle_counts.to_csv(TABLE_DIR / "experiment3_cycle_counts.csv", index=False)

    train_df = df[df["cycle"].isin(TRAIN_CYCLES)].copy()
    test_df = df[df["cycle"] == TEST_CYCLE].copy()
    print(f"Training rows: {len(train_df)}")
    print(f"Temporal test rows: {len(test_df)}")

    if train_df.empty or test_df.empty:
        raise ValueError("Temporal split failed. Check cycle file names and cycle labels.")

    pre = Preprocessor(target_col, sensitive_col)
    X_train, y_train, _ = pre.fit_transform(train_df.dropna(subset=[target_col]).copy())
    X_test, y_test, _ = pre.transform(test_df.dropna(subset=[target_col]).copy())
    baseline = utility(X_train, y_train, X_test, y_test)
    print(f"Real-data temporal baseline: {baseline}")

    all_metrics = []
    for frac in OBSERVED_FRACTIONS:
        print("-" * 90)
        print(f"Running temporal observed fraction: {frac:.0%}")
        all_metrics.append(run_fraction(train_df, test_df, target_col, sensitive_col, frac))

    metrics = pd.concat(all_metrics, ignore_index=True)
    aggregate(metrics, baseline)

    elapsed = time.time() - start
    save_json(
        {
            "runtime_seconds": elapsed,
            "runtime_minutes": elapsed / 60,
            "output_dir": str(OUTPUT_DIR),
            "train_cycles": TRAIN_CYCLES,
            "test_cycle": TEST_CYCLE,
            "observed_fractions": OBSERVED_FRACTIONS,
            "latent_dims": LATENT_DIMS,
            "n_reconstructions": N_RECONSTRUCTIONS,
        },
        LOG_DIR / "runtime_summary.json",
    )

    print("=" * 90)
    print("Completed successfully.")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
