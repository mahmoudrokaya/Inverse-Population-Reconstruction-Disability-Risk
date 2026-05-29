# -*- coding: utf-8 -*-
r"""
Experiment 1: Partial Observability Reconstruction
Project: Inverse Population Reconstruction of Disability Risk NHANES

Root folder:
D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments

Input folder:
D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments\data_raw

Output folder:
D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments\Experiment1_Partial_Observability_Reconstruction
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments")
DATA_RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "Experiment1_Partial_Observability_Reconstruction"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"
RECON_DIR = OUTPUT_DIR / "reconstructed_samples"
for folder in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR, RECON_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OBSERVED_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
N_RECONSTRUCTIONS = 5
LATENT_DIMENSIONS = [12, 16, 20, 23]
DEFAULT_LATENT_DIM = 20

TARGET_CANDIDATES = ["disability_stage", "merge01_vs_23", "disability_binary", "target", "label", "outcome"]
SENSITIVE_CANDIDATES = ["sex", "gender", "race_ethnicity", "race", "education", "income_poverty_ratio", "age_group"]
DIRECT_LEAKAGE_CANDIDATES = [
    "disability_stage", "merge01_vs_23", "disability_binary", "mobility_difficulty",
    "self_care_difficulty", "usual_activities_difficulty", "pain_discomfort", "anxiety_depression"
]


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace(" ", "_").replace("-", "_").replace("/", "_") for c in df.columns]
    return df


def infer_cycle_from_name(path: Path) -> str:
    name = path.name
    for cycle in ["2011-2012", "2013-2014", "2015-2016", "2017-2018"]:
        if cycle in name:
            return cycle
    if "pooled" in name.lower():
        return "pooled"
    return "unknown"


def find_existing_files(data_dir: Path) -> List[Path]:
    patterns = [
        "NHANES_2011-2012_final_processed*.csv", "NHANES_2013-2014_final_processed*.csv",
        "NHANES_2015-2016_final_processed*.csv", "NHANES_2017-2018_final_processed*.csv",
        "NHANES_pooled_filled_corrected_blue*.xlsx", "NHANES_pooled_filled_corrected*.csv",
        "*.csv", "*.xlsx"
    ]
    files, seen = [], set()
    for pattern in patterns:
        for path in data_dir.glob(pattern):
            if path.name not in seen:
                files.append(path)
                seen.add(path.name)
    return files


def load_nhanes_files(data_dir: Path) -> pd.DataFrame:
    files = find_existing_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {data_dir}")
    frames, manifest = [], []
    for path in files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            elif path.suffix.lower() in [".xlsx", ".xls"]:
                df = pd.read_excel(path)
            else:
                continue
            df = normalize_column_names(df)
            cycle = infer_cycle_from_name(path)
            if "cycle" not in df.columns:
                df["cycle"] = cycle
            df["source_file"] = path.name
            frames.append(df)
            manifest.append({"file": path.name, "cycle": cycle, "rows": int(df.shape[0]), "columns": int(df.shape[1])})
        except Exception as e:
            manifest.append({"file": path.name, "cycle": infer_cycle_from_name(path), "error": str(e)})
    if not frames:
        raise RuntimeError("Files were found but none could be loaded.")
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = normalize_column_names(all_df)
    pd.DataFrame(manifest).to_csv(TABLE_DIR / "input_file_manifest.csv", index=False)
    return all_df


def select_target(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("No target column found.")


def select_sensitive_column(df: pd.DataFrame) -> Optional[str]:
    for col in SENSITIVE_CANDIDATES:
        if col in df.columns:
            return col
    lower_map = {c.lower(): c for c in df.columns}
    for col in SENSITIVE_CANDIDATES:
        if col.lower() in lower_map:
            return lower_map[col.lower()]
    return None


def plot_and_save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


class Preprocessor:
    def __init__(self, target_col: str, sensitive_col: Optional[str] = None):
        self.target_col = target_col
        self.sensitive_col = sensitive_col
        self.feature_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def _choose_features(self, df: pd.DataFrame) -> List[str]:
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
        self.feature_cols = self._choose_features(df)
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
        keep_cols = X.nunique(dropna=True)[lambda s: s > 1].index.tolist()
        X = X[keep_cols]
        self.feature_cols = keep_cols
        self.categorical_cols = [c for c in self.categorical_cols if c in self.feature_cols]
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
    def __init__(self, latent_dim: int = DEFAULT_LATENT_DIM, random_state: int = RANDOM_STATE):
        self.latent_dim = latent_dim
        self.random_state = random_state
        self.pca: Optional[PCA] = None
        self.class_stats: Dict[str, Dict[str, np.ndarray]] = {}
        self.class_probs: Dict[str, float] = {}
        self.classes_: List[str] = []
        self.feature_cols: List[str] = []

    def fit(self, X_obs: pd.DataFrame, y_obs: pd.Series) -> "ConditionalPCAGaussianReconstructor":
        self.feature_cols = list(X_obs.columns)
        n_components = min(self.latent_dim, X_obs.shape[1], max(1, X_obs.shape[0] - 1))
        self.pca = PCA(n_components=n_components, random_state=self.random_state)
        Z = self.pca.fit_transform(X_obs.values)
        y_str = y_obs.astype(str).fillna("Missing").values
        values, counts = np.unique(y_str, return_counts=True)
        self.classes_ = list(values)
        probs = counts / counts.sum()
        self.class_probs = {cls: float(prob) for cls, prob in zip(values, probs)}
        for cls in self.classes_:
            Zc = Z[y_str == cls]
            mean = Zc.mean(axis=0)
            if len(Zc) <= 2:
                cov = np.eye(Z.shape[1]) * 0.05
            else:
                cov = np.cov(Zc, rowvar=False)
                if cov.ndim == 0:
                    cov = np.eye(Z.shape[1]) * float(cov)
                cov = np.asarray(cov)
                cov += np.eye(cov.shape[0]) * 1e-5
            self.class_stats[cls] = {"mean": mean, "cov": cov}
        return self

    def sample(self, n_samples: int, seed: int) -> Tuple[pd.DataFrame, pd.Series]:
        if self.pca is None:
            raise RuntimeError("Model must be fitted before sampling.")
        rng = np.random.default_rng(seed)
        classes = np.array(self.classes_)
        probs = np.array([self.class_probs[c] for c in classes])
        probs = probs / probs.sum()
        sampled_classes = rng.choice(classes, size=n_samples, replace=True, p=probs)
        Z_samples = []
        for cls in sampled_classes:
            stats = self.class_stats[cls]
            Z_samples.append(rng.multivariate_normal(stats["mean"], stats["cov"]))
        X_samples = self.pca.inverse_transform(np.vstack(Z_samples))
        X_hat = pd.DataFrame(X_samples, columns=self.feature_cols)
        y_hat = pd.Series(sampled_classes, name="reconstructed_target")
        return X_hat, y_hat


def average_ks_statistic(X_true: pd.DataFrame, X_hat: pd.DataFrame) -> float:
    vals = []
    for col in X_true.columns:
        try:
            vals.append(ks_2samp(X_true[col].values, X_hat[col].values).statistic)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan


def average_wasserstein_distance(X_true: pd.DataFrame, X_hat: pd.DataFrame) -> float:
    vals = []
    for col in X_true.columns:
        try:
            vals.append(wasserstein_distance(X_true[col].values, X_hat[col].values))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else np.nan


def correlation_structure_distance(X_true: pd.DataFrame, X_hat: pd.DataFrame) -> float:
    c1 = X_true.corr().fillna(0).values
    c2 = X_hat.corr().fillna(0).values
    return float(np.linalg.norm(c1 - c2, ord="fro"))


def covariance_structure_distance(X_true: pd.DataFrame, X_hat: pd.DataFrame) -> float:
    c1 = np.cov(X_true.values, rowvar=False)
    c2 = np.cov(X_hat.values, rowvar=False)
    return float(np.linalg.norm(c1 - c2, ord="fro"))


def distribution_difference_by_target(y_true: pd.Series, y_hat: pd.Series) -> float:
    p_true = y_true.astype(str).value_counts(normalize=True)
    p_hat = y_hat.astype(str).value_counts(normalize=True)
    idx = sorted(set(p_true.index).union(set(p_hat.index)))
    return float(np.sum(np.abs(p_true.reindex(idx, fill_value=0) - p_hat.reindex(idx, fill_value=0))))


def predictive_utility(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
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


def fairness_spd(y: pd.Series, s: Optional[pd.Series]) -> float:
    if s is None:
        return np.nan
    yy = y.astype(str).reset_index(drop=True)
    ss = s.astype(str).fillna("Missing").reset_index(drop=True)
    if yy.nunique() < 2 or ss.nunique() < 2:
        return np.nan
    positive = sorted(yy.unique())[-1]
    rates = []
    for group in sorted(ss.unique()):
        mask = ss == group
        if mask.sum() > 0:
            rates.append(float((yy[mask] == positive).mean()))
    return float(max(rates) - min(rates)) if len(rates) >= 2 else np.nan


def impute_sensitive_for_reconstructed(s_obs: Optional[pd.Series], n_samples: int, seed: int) -> Optional[pd.Series]:
    if s_obs is None:
        return None
    rng = np.random.default_rng(seed)
    s_clean = s_obs.astype(str).fillna("Missing")
    values = s_clean.value_counts(normalize=True)
    sampled = rng.choice(values.index.values, size=n_samples, replace=True, p=values.values)
    return pd.Series(sampled, name=s_obs.name)


def run_single_fraction(df: pd.DataFrame, target_col: str, sensitive_col: Optional[str], observed_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start = time.time()
    df_work = df.dropna(subset=[target_col]).copy()
    stratify = df_work[target_col].astype(str) if df_work[target_col].nunique() > 1 else None
    df_obs, df_hidden = train_test_split(df_work, train_size=observed_fraction, random_state=RANDOM_STATE, stratify=stratify)
    hidden_stratify = df_hidden[target_col].astype(str) if df_hidden[target_col].nunique() > 1 else None
    df_val, df_test = train_test_split(df_hidden, test_size=0.50, random_state=RANDOM_STATE, stratify=hidden_stratify)
    pre = Preprocessor(target_col=target_col, sensitive_col=sensitive_col)
    X_obs, y_obs, s_obs = pre.fit_transform(df_obs)
    X_hidden, y_hidden, s_hidden = pre.transform(df_hidden)
    X_test, y_test, s_test = pre.transform(df_test)
    all_recon_metrics = []
    for latent_dim in LATENT_DIMENSIONS:
        for k in range(N_RECONSTRUCTIONS):
            seed = RANDOM_STATE + k + int(observed_fraction * 1000) + latent_dim
            reconstructor = ConditionalPCAGaussianReconstructor(latent_dim=latent_dim, random_state=seed)
            fit_start = time.time()
            reconstructor.fit(X_obs, y_obs)
            X_hat, y_hat = reconstructor.sample(n_samples=len(df_hidden), seed=seed + 1000)
            runtime = time.time() - fit_start
            s_hat = impute_sensitive_for_reconstructed(s_obs, len(X_hat), seed=seed + 2000)
            X_hat = X_hat[X_hidden.columns]
            utility = predictive_utility(X_hat, y_hat, X_test, y_test)
            spd_hidden = fairness_spd(y_hidden, s_hidden)
            spd_recon = fairness_spd(y_hat, s_hat)
            spd_gap = abs(spd_hidden - spd_recon) if np.isfinite(spd_hidden) and np.isfinite(spd_recon) else np.nan
            row = {
                "observed_fraction": observed_fraction,
                "observed_rows": int(len(df_obs)),
                "hidden_rows": int(len(df_hidden)),
                "test_rows": int(len(df_test)),
                "latent_dim": latent_dim,
                "reconstruction_id": k + 1,
                "avg_ks_statistic": average_ks_statistic(X_hidden, X_hat),
                "avg_wasserstein_distance": average_wasserstein_distance(X_hidden, X_hat),
                "correlation_structure_distance": correlation_structure_distance(X_hidden, X_hat),
                "covariance_structure_distance": covariance_structure_distance(X_hidden, X_hat),
                "target_distribution_l1_difference": distribution_difference_by_target(y_hidden, y_hat),
                "hidden_spd": spd_hidden,
                "reconstructed_spd": spd_recon,
                "spd_absolute_gap": spd_gap,
                "runtime_seconds": runtime,
            }
            row.update({f"utility_{key}": val for key, val in utility.items()})
            all_recon_metrics.append(row)
            if latent_dim == DEFAULT_LATENT_DIM and k == 0:
                sample_out = X_hat.copy()
                sample_out[target_col] = y_hat.values
                if s_hat is not None:
                    sample_out[f"sampled_{sensitive_col}"] = s_hat.values
                sample_out.to_csv(RECON_DIR / f"reconstructed_observed_{int(observed_fraction*100)}pct_latent{latent_dim}.csv", index=False)
    metrics_df = pd.DataFrame(all_recon_metrics)
    uncertainty_rows = []
    metric_cols = [
        "avg_ks_statistic", "avg_wasserstein_distance", "correlation_structure_distance",
        "covariance_structure_distance", "target_distribution_l1_difference",
        "utility_macro_f1", "utility_balanced_accuracy", "spd_absolute_gap"
    ]
    for latent_dim, group in metrics_df.groupby("latent_dim"):
        row = {"observed_fraction": observed_fraction, "latent_dim": latent_dim}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(group[col].std(ddof=0))
        uncertainty_rows.append(row)
    save_json({"observed_fraction": observed_fraction, "observed_rows": int(len(df_obs)), "hidden_rows": int(len(df_hidden)), "test_rows": int(len(df_test)), "runtime_seconds": time.time() - start}, LOG_DIR / f"runtime_observed_{int(observed_fraction*100)}pct.json")
    return metrics_df, pd.DataFrame(uncertainty_rows)


def aggregate_and_plot(metrics: pd.DataFrame, uncertainty: pd.DataFrame) -> None:
    metrics.to_csv(TABLE_DIR / "experiment1_all_reconstruction_metrics.csv", index=False)
    uncertainty.to_csv(TABLE_DIR / "experiment1_uncertainty_by_fraction_and_latent_dim.csv", index=False)
    summary = metrics.groupby(["observed_fraction", "latent_dim"]).agg(
        avg_ks_statistic_mean=("avg_ks_statistic", "mean"),
        avg_wasserstein_distance_mean=("avg_wasserstein_distance", "mean"),
        correlation_structure_distance_mean=("correlation_structure_distance", "mean"),
        covariance_structure_distance_mean=("covariance_structure_distance", "mean"),
        utility_macro_f1_mean=("utility_macro_f1", "mean"),
        utility_balanced_accuracy_mean=("utility_balanced_accuracy", "mean"),
        spd_absolute_gap_mean=("spd_absolute_gap", "mean"),
        runtime_seconds_mean=("runtime_seconds", "mean"),
    ).reset_index()
    summary.to_csv(TABLE_DIR / "experiment1_summary_by_fraction_and_latent_dim.csv", index=False)
    main = summary[summary["latent_dim"] == DEFAULT_LATENT_DIM].copy()
    fig_specs = [
        ("avg_wasserstein_distance_mean", "Average Wasserstein distance", "Fidelity Under Partial Observability", "fig01_fidelity_wasserstein_by_observed_fraction.png"),
        ("correlation_structure_distance_mean", "Correlation structure distance", "Structural Consistency Under Partial Observability", "fig02_structure_distance_by_observed_fraction.png"),
        ("utility_macro_f1_mean", "Macro-F1 on held-out data", "Predictive Utility of Reconstructed Data", "fig03_utility_macro_f1_by_observed_fraction.png"),
        ("spd_absolute_gap_mean", "Absolute SPD gap", "Fairness Alignment Under Partial Observability", "fig04_fairness_spd_gap_by_observed_fraction.png"),
    ]
    for col, ylabel, title, fname in fig_specs:
        plt.figure(figsize=(8, 5))
        plt.plot(main["observed_fraction"], main[col], marker="o")
        plt.xlabel("Observed fraction")
        plt.ylabel(ylabel)
        plt.title(title)
        plot_and_save(FIGURE_DIR / fname)
    pivot = summary.pivot(index="observed_fraction", columns="latent_dim", values="utility_macro_f1_mean")
    plt.figure(figsize=(9, 5))
    for col in pivot.columns:
        plt.plot(pivot.index, pivot[col], marker="o", label=f"latent_dim={col}")
    plt.xlabel("Observed fraction")
    plt.ylabel("Macro-F1")
    plt.title("Latent Dimension Sensitivity")
    plt.legend()
    plot_and_save(FIGURE_DIR / "fig05_latent_dimension_sensitivity_macro_f1.png")
    report = f"""# Experiment 1: Partial Observability Reconstruction Report

This experiment validates the core methodological claim that latent population structure can be inferred from partial observations and used to reconstruct plausible population extensions.

Observed fractions: {OBSERVED_FRACTIONS}
Latent dimensions tested: {LATENT_DIMENSIONS}
Stochastic reconstructions per setting: {N_RECONSTRUCTIONS}

Main outputs are saved in the tables and figures folders.
"""
    (OUTPUT_DIR / "Experiment1_Partial_Observability_Reconstruction_Report.md").write_text(report, encoding="utf-8")


def main() -> None:
    start = time.time()
    print("=" * 90)
    print("Experiment 1: Partial Observability Reconstruction")
    print("=" * 90)
    print(f"Input folder : {DATA_RAW_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    df = load_nhanes_files(DATA_RAW_DIR)
    print(f"Loaded data shape: {df.shape}")
    target_col = select_target(df)
    sensitive_col = select_sensitive_column(df)
    print(f"Detected target column   : {target_col}")
    print(f"Detected sensitive column: {sensitive_col}")
    all_metrics, all_uncertainty = [], []
    for frac in OBSERVED_FRACTIONS:
        print("-" * 90)
        print(f"Running observed fraction: {frac:.0%}")
        metrics_df, uncertainty_df = run_single_fraction(df, target_col, sensitive_col, frac)
        all_metrics.append(metrics_df)
        all_uncertainty.append(uncertainty_df)
    metrics = pd.concat(all_metrics, ignore_index=True)
    uncertainty = pd.concat(all_uncertainty, ignore_index=True)
    aggregate_and_plot(metrics, uncertainty)
    elapsed = time.time() - start
    save_json({"runtime_seconds": elapsed, "runtime_minutes": elapsed / 60, "output_dir": str(OUTPUT_DIR), "observed_fractions": OBSERVED_FRACTIONS, "latent_dimensions": LATENT_DIMENSIONS, "n_reconstructions": N_RECONSTRUCTIONS}, LOG_DIR / "runtime_summary.json")
    print("=" * 90)
    print("Completed successfully.")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
