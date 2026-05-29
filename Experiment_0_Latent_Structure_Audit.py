# -*- coding: utf-8 -*-
r"""
Experiment 0: Latent Structure Audit for NHANES Disability Risk Data

Input and output root:
D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments

Raw data folder:
D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments\data_raw

Output folder:
D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments\Experiment0_Latent_Structure_Audit
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


BASE_DIR = Path(r"D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments")
DATA_RAW_DIR = BASE_DIR / "data_raw"
OUTPUT_DIR = BASE_DIR / "Experiment0_Latent_Structure_Audit"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
LOG_DIR = OUTPUT_DIR / "logs"

for folder in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
PCA_VARIANCE_THRESHOLDS = [0.70, 0.80, 0.90, 0.95]
MAX_MI_FEATURES = 40
MAX_LOADING_FEATURES = 15
MAX_CLUSTER_K = 8

TARGET_CANDIDATES = [
    "disability_stage",
    "merge01_vs_23",
    "disability_binary",
    "target",
    "label",
    "outcome",
]

SENSITIVE_CANDIDATES = [
    "sex",
    "gender",
    "race_ethnicity",
    "race",
    "education",
    "income",
    "income_poverty_ratio",
    "age_group",
]

DIRECT_LEAKAGE_CANDIDATES = [
    "disability_stage",
    "merge01_vs_23",
    "disability_binary",
    "mobility_difficulty",
    "self_care_difficulty",
    "usual_activities_difficulty",
    "pain_discomfort",
    "anxiety_depression",
]


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(c).strip().replace(" ", "_").replace("-", "_").replace("/", "_")
        for c in df.columns
    ]
    return df


def find_existing_files(data_dir: Path) -> List[Path]:
    patterns = [
        "NHANES_2011-2012_final_processed*.csv",
        "NHANES_2013-2014_final_processed*.csv",
        "NHANES_2015-2016_final_processed*.csv",
        "NHANES_2017-2018_final_processed*.csv",
        "NHANES_pooled_filled_corrected_blue*.xlsx",
        "NHANES_pooled_filled_corrected*.csv",
        "*.csv",
        "*.xlsx",
    ]

    files: List[Path] = []
    seen = set()

    for pattern in patterns:
        for path in data_dir.glob(pattern):
            if path.name not in seen:
                files.append(path)
                seen.add(path.name)

    return files


def infer_cycle_from_name(path: Path) -> str:
    name = path.name
    for cycle in ["2011-2012", "2013-2014", "2015-2016", "2017-2018"]:
        if cycle in name:
            return cycle
    if "pooled" in name.lower():
        return "pooled"
    return "unknown"


def load_nhanes_files(data_dir: Path) -> pd.DataFrame:
    files = find_existing_files(data_dir)

    if not files:
        raise FileNotFoundError(f"No CSV/XLSX files found in {data_dir}")

    frames = []
    manifest = []

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

            manifest.append(
                {
                    "file": path.name,
                    "cycle": cycle,
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                }
            )

        except Exception as e:
            manifest.append(
                {
                    "file": path.name,
                    "cycle": infer_cycle_from_name(path),
                    "error": str(e),
                }
            )

    if not frames:
        raise RuntimeError("Files were found but none could be loaded.")

    all_df = pd.concat(frames, ignore_index=True, sort=False)
    all_df = normalize_column_names(all_df)

    pd.DataFrame(manifest).to_csv(TABLE_DIR / "input_file_manifest.csv", index=False)
    return all_df


def select_target(df: pd.DataFrame) -> str | None:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    return None


def get_sensitive_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    lower_map = {c.lower(): c for c in df.columns}

    for c in SENSITIVE_CANDIDATES:
        if c in df.columns:
            cols.append(c)
        elif c.lower() in lower_map:
            cols.append(lower_map[c.lower()])

    return sorted(set(cols))


def prepare_model_matrix(
    df: pd.DataFrame,
    target_col: str | None,
    exclude_leakage: bool = True,
) -> Tuple[pd.DataFrame, pd.Series | None, List[str]]:
    df = df.copy()

    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].copy()

    exclude_cols = {"source_file"}

    if target_col:
        exclude_cols.add(target_col)

    if exclude_leakage:
        for c in DIRECT_LEAKAGE_CANDIDATES:
            if c in df.columns:
                exclude_cols.add(c)

    for c in df.columns:
        if c.lower() in {"seqn", "id", "patient_id", "participant_id"}:
            exclude_cols.add(c)

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    X = X.dropna(axis=1, how="all")

    for col in X.columns:
        if X[col].dtype == "object":
            nunique = X[col].nunique(dropna=True)
            if nunique <= 50:
                X[col] = X[col].astype("category")
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")

    categorical_cols = [
        c for c in X.columns if str(X[c].dtype) == "category" or X[c].dtype == "object"
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    for col in categorical_cols:
        X[col] = X[col].astype(str).replace("nan", np.nan)
        X[col] = X[col].fillna("Missing")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    nunique = X.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    X = X[keep_cols]

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns, index=X.index)

    return X_scaled, y, X_scaled.columns.tolist()


def plot_and_save(fig_path: Path) -> None:
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_data_quality_audit(df: pd.DataFrame, target_col: str | None) -> None:
    quality = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
            "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )

    quality.to_csv(TABLE_DIR / "data_quality_column_summary.csv", index=False)

    summary = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "target_column": target_col,
        "overall_missing_rate": float(df.isna().mean().mean()),
        "duplicated_rows": int(df.duplicated().sum()),
    }

    save_json(summary, LOG_DIR / "data_quality_summary.json")

    top_missing = quality.sort_values("missing_rate", ascending=False).head(30)

    plt.figure(figsize=(12, 7))
    plt.barh(top_missing["column"], top_missing["missing_rate"])
    plt.xlabel("Missing rate")
    plt.ylabel("Feature")
    plt.title("Top Missingness Rates by Feature")
    plt.gca().invert_yaxis()
    plot_and_save(FIGURE_DIR / "fig01_top_missingness_rates.png")

    if target_col and target_col in df.columns:
        target_counts = df[target_col].value_counts(dropna=False).reset_index()
        target_counts.columns = [target_col, "count"]
        target_counts.to_csv(TABLE_DIR / "target_distribution.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.bar(target_counts[target_col].astype(str), target_counts["count"])
        plt.xlabel(target_col)
        plt.ylabel("Count")
        plt.title("Target Distribution")
        plot_and_save(FIGURE_DIR / "fig02_target_distribution.png")

    if "cycle" in df.columns:
        cycle_counts = df["cycle"].value_counts(dropna=False).reset_index()
        cycle_counts.columns = ["cycle", "count"]
        cycle_counts.to_csv(TABLE_DIR / "cycle_distribution.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.bar(cycle_counts["cycle"].astype(str), cycle_counts["count"])
        plt.xlabel("NHANES cycle")
        plt.ylabel("Count")
        plt.title("Records by NHANES Cycle")
        plot_and_save(FIGURE_DIR / "fig03_cycle_distribution.png")


def run_variance_and_correlation_audit(X: pd.DataFrame) -> None:
    variances = X.var(axis=0).sort_values(ascending=False)

    pd.DataFrame(
        {
            "feature": variances.index,
            "standardized_variance": variances.values,
        }
    ).to_csv(TABLE_DIR / "feature_variance_summary.csv", index=False)

    plt.figure(figsize=(12, 5))
    clean_var = np.asarray(variances.values, dtype=float)
    clean_var = clean_var[np.isfinite(clean_var)]

    if len(clean_var) == 0:
        plt.text(0.5, 0.5, "No finite variance values available", ha="center", va="center")
        plt.axis("off")
    else:
        plt.bar(range(len(clean_var)), clean_var)
        plt.xlabel("Feature index")
        plt.ylabel("Standardized variance")
        plt.title("Feature Variance by Feature After Standardization")

    plot_and_save(FIGURE_DIR / "fig04_feature_variance_distribution.png")

    corr = X.corr()
    corr.to_csv(TABLE_DIR / "feature_correlation_matrix.csv")

    plt.figure(figsize=(12, 10))
    plt.imshow(corr.values, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.title("Feature Correlation Matrix")
    plt.xticks([])
    plt.yticks([])
    plot_and_save(FIGURE_DIR / "fig05_correlation_heatmap.png")

    abs_corr = corr.abs()
    upper = abs_corr.where(np.triu(np.ones(abs_corr.shape), k=1).astype(bool))

    pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_correlation"})
        .sort_values("abs_correlation", ascending=False)
    )

    pairs.to_csv(TABLE_DIR / "feature_redundancy_pairs_by_correlation.csv", index=False)


def run_pca_analysis(X: pd.DataFrame) -> Dict:
    pca = PCA(random_state=RANDOM_STATE)
    pcs = pca.fit_transform(X)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    pca_table = pd.DataFrame(
        {
            "component": np.arange(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )

    pca_table.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)

    intrinsic_dims = {}
    for threshold in PCA_VARIANCE_THRESHOLDS:
        intrinsic_dims[f"components_for_{int(threshold * 100)}pct"] = int(
            np.argmax(cumulative >= threshold) + 1
        )

    save_json(intrinsic_dims, LOG_DIR / "pca_intrinsic_dimensionality.json")

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(explained) + 1), explained, marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA Scree Plot")
    plot_and_save(FIGURE_DIR / "fig06_pca_scree_plot.png")

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o")

    for threshold in PCA_VARIANCE_THRESHOLDS:
        plt.axhline(y=threshold, linestyle="--")

    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Cumulative Explained Variance")
    plot_and_save(FIGURE_DIR / "fig07_pca_cumulative_variance.png")

    if pcs.shape[1] >= 2:
        pca_projection = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1]})
        pca_projection.to_csv(TABLE_DIR / "pca_2d_projection.csv", index=False)

        plt.figure(figsize=(8, 6))
        plt.scatter(pcs[:, 0], pcs[:, 1], s=10, alpha=0.5)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Two-Dimensional PCA Projection")
        plot_and_save(FIGURE_DIR / "fig08_pca_2d_projection.png")

    loading_count = min(10, pca.components_.shape[0])

    loadings = pd.DataFrame(
        pca.components_[:loading_count].T,
        index=X.columns,
        columns=[f"PC{i + 1}" for i in range(loading_count)],
    )

    loadings.to_csv(TABLE_DIR / "pca_loadings_first_10_components.csv")

    top_rows = []

    for pc in loadings.columns:
        top_features = loadings[pc].abs().sort_values(ascending=False).head(MAX_LOADING_FEATURES)

        for feature, abs_loading in top_features.items():
            top_rows.append(
                {
                    "component": pc,
                    "feature": feature,
                    "loading": float(loadings.loc[feature, pc]),
                    "abs_loading": float(abs_loading),
                }
            )

    top_loading_df = pd.DataFrame(top_rows)
    top_loading_df.to_csv(TABLE_DIR / "pca_top_loading_features.csv", index=False)

    if not top_loading_df.empty:
        top_feature_names = (
            top_loading_df.groupby("feature")["abs_loading"]
            .max()
            .sort_values(ascending=False)
            .head(30)
            .index
            .tolist()
        )

        heat = loadings.loc[top_feature_names]

        plt.figure(figsize=(10, 8))
        plt.imshow(heat.values, aspect="auto")
        plt.colorbar(label="Loading")
        plt.xticks(range(len(heat.columns)), heat.columns, rotation=45)
        plt.yticks(range(len(heat.index)), heat.index)
        plt.title("PCA Loading Heatmap for Top Features")
        plot_and_save(FIGURE_DIR / "fig09_pca_loading_heatmap.png")

    return {
        "pca": pca,
        "pcs": pcs,
        "explained": explained,
        "cumulative": cumulative,
        "intrinsic_dims": intrinsic_dims,
    }


def run_pca_stability_analysis(X: pd.DataFrame) -> None:
    X_train, X_test = train_test_split(X, test_size=0.30, random_state=RANDOM_STATE)

    n_components = min(20, X.shape[1], X_train.shape[0], X_test.shape[0])

    pca_train = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_test = PCA(n_components=n_components, random_state=RANDOM_STATE)

    pca_train.fit(X_train)
    pca_test.fit(X_test)

    similarities = []

    for i in range(n_components):
        v1 = pca_train.components_[i]
        v2 = pca_test.components_[i]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)

        if denom == 0:
            cosine_abs = np.nan
        else:
            cosine_abs = abs(np.dot(v1, v2) / denom)

        similarities.append(cosine_abs)

    stability = pd.DataFrame(
        {
            "component": np.arange(1, n_components + 1),
            "absolute_cosine_similarity": similarities,
        }
    )

    stability.to_csv(TABLE_DIR / "pca_stability_train_test.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(stability["component"], stability["absolute_cosine_similarity"], marker="o")
    plt.ylim(0, 1.05)
    plt.xlabel("Principal component")
    plt.ylabel("Absolute cosine similarity")
    plt.title("PCA Component Stability Across Train/Test Subsamples")
    plot_and_save(FIGURE_DIR / "fig10_pca_stability.png")


def run_clustering_audit(pcs: np.ndarray, target: pd.Series | None) -> None:
    n_latent = min(10, pcs.shape[1])
    Z = pcs[:, :n_latent]

    rows = []
    max_k = min(MAX_CLUSTER_K, max(2, len(Z) - 1))

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(Z)
        sil = silhouette_score(Z, labels)

        row = {
            "k": k,
            "silhouette_score": float(sil),
            "inertia": float(kmeans.inertia_),
        }

        if target is not None:
            y_clean = pd.Series(target).astype(str).fillna("Missing")

            if y_clean.nunique() > 1:
                row["adjusted_rand_with_target"] = float(adjusted_rand_score(y_clean, labels))
                row["normalized_mutual_info_with_target"] = float(
                    normalized_mutual_info_score(y_clean, labels)
                )

        rows.append(row)

    cluster_df = pd.DataFrame(rows)
    cluster_df.to_csv(TABLE_DIR / "latent_space_clustering_scores.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(cluster_df["k"], cluster_df["silhouette_score"], marker="o")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.title("Cluster Tendency in PCA Latent Space")
    plot_and_save(FIGURE_DIR / "fig11_latent_cluster_silhouette.png")


def run_target_dependency_audit(X: pd.DataFrame, target: pd.Series | None) -> None:
    if target is None:
        return

    y = pd.Series(target).copy()
    valid = ~y.isna()

    Xv = X.loc[valid].copy()
    yv = y.loc[valid].astype(str)

    if yv.nunique() < 2:
        return

    if Xv.shape[1] > MAX_MI_FEATURES:
        le = LabelEncoder()
        y_enc = le.fit_transform(yv)

        clf = ExtraTreesClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        clf.fit(Xv, y_enc)

        imp = pd.Series(clf.feature_importances_, index=Xv.columns).sort_values(ascending=False)
        selected = imp.head(MAX_MI_FEATURES).index.tolist()

        pd.DataFrame({"feature": imp.index, "importance": imp.values}).to_csv(
            TABLE_DIR / "target_extratrees_feature_importance.csv",
            index=False,
        )

        Xmi = Xv[selected]
    else:
        Xmi = Xv

    le = LabelEncoder()
    y_enc = le.fit_transform(yv)

    mi = mutual_info_classif(Xmi, y_enc, random_state=RANDOM_STATE)

    mi_df = pd.DataFrame(
        {
            "feature": Xmi.columns,
            "mutual_information_with_target": mi,
        }
    ).sort_values("mutual_information_with_target", ascending=False)

    mi_df.to_csv(TABLE_DIR / "mutual_information_with_target.csv", index=False)

    plt.figure(figsize=(10, 7))
    top = mi_df.head(25)
    plt.barh(top["feature"], top["mutual_information_with_target"])
    plt.xlabel("Mutual information")
    plt.ylabel("Feature")
    plt.title("Top Mutual Information Features with Target")
    plt.gca().invert_yaxis()
    plot_and_save(FIGURE_DIR / "fig12_mutual_information_target.png")


def run_subgroup_audit(df: pd.DataFrame, target_col: str | None) -> None:
    sens_cols = get_sensitive_columns(df)

    pd.DataFrame({"sensitive_column": sens_cols}).to_csv(
        TABLE_DIR / "detected_sensitive_columns.csv",
        index=False,
    )

    if not target_col or not sens_cols:
        return

    rows = []

    for s_col in sens_cols:
        if s_col not in df.columns:
            continue

        temp = df[[s_col, target_col]].copy()
        temp[s_col] = temp[s_col].astype(str).fillna("Missing")
        temp[target_col] = temp[target_col].astype(str).fillna("Missing")

        tab = pd.crosstab(temp[s_col], temp[target_col], normalize="index")
        tab.to_csv(TABLE_DIR / f"subgroup_target_distribution_{s_col}.csv")

        for subgroup in tab.index:
            for outcome in tab.columns:
                rows.append(
                    {
                        "sensitive_attribute": s_col,
                        "subgroup": subgroup,
                        "target_value": outcome,
                        "within_subgroup_rate": float(tab.loc[subgroup, outcome]),
                    }
                )

    if rows:
        pd.DataFrame(rows).to_csv(
            TABLE_DIR / "subgroup_target_distribution_long.csv",
            index=False,
        )


def write_interpretation_report(
    df: pd.DataFrame,
    X: pd.DataFrame,
    target_col: str | None,
    pca_result: Dict,
) -> None:
    dims = pca_result["intrinsic_dims"]
    explained = pca_result["explained"]

    first_pc = float(explained[0]) if len(explained) else np.nan
    first_5 = float(np.sum(explained[:5])) if len(explained) >= 5 else np.nan
    first_10 = float(np.sum(explained[:10])) if len(explained) >= 10 else np.nan

    report = f"""# Experiment 0: Latent Structure Audit Report

## Data overview

Rows: {df.shape[0]}
Columns before modeling: {df.shape[1]}
Modeling features after cleaning: {X.shape[1]}
Target column detected: {target_col}

## PCA latent structure evidence

Variance explained by PC1: {first_pc:.4f}
Variance explained by first 5 PCs: {first_5:.4f}
Variance explained by first 10 PCs: {first_10:.4f}

Components needed for 70% variance: {dims.get("components_for_70pct")}
Components needed for 80% variance: {dims.get("components_for_80pct")}
Components needed for 90% variance: {dims.get("components_for_90pct")}
Components needed for 95% variance: {dims.get("components_for_95pct")}

## Interpretation guide

If a relatively small number of components explains a large fraction of variance, this supports the claim that the dataset contains a compressed latent structure. If many components are required, the population structure is more distributed and the reconstruction model should use a richer latent space.

The PCA stability report evaluates whether the principal directions remain similar across train/test subsamples. Higher cosine similarity indicates that the detected latent directions are not merely sampling noise.

The correlation, redundancy, and mutual-information reports define which variables carry implicit relationships. These outputs should guide the structural loss terms in later inverse population reconstruction experiments.
"""

    (OUTPUT_DIR / "Experiment0_Latent_Structure_Audit_Report.md").write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    start = time.time()

    print("=" * 80)
    print("Experiment 0: Latent Structure Audit")
    print("=" * 80)
    print(f"Input folder : {DATA_RAW_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    df = load_nhanes_files(DATA_RAW_DIR)
    print(f"Loaded data shape: {df.shape}")

    target_col = select_target(df)
    print(f"Detected target column: {target_col}")

    run_data_quality_audit(df, target_col)

    X, y, feature_cols = prepare_model_matrix(
        df,
        target_col=target_col,
        exclude_leakage=True,
    )

    print(f"Modeling matrix shape after cleaning: {X.shape}")

    pd.DataFrame({"feature": feature_cols}).to_csv(
        TABLE_DIR / "modeling_feature_list.csv",
        index=False,
    )

    run_variance_and_correlation_audit(X)
    pca_result = run_pca_analysis(X)
    run_pca_stability_analysis(X)
    run_clustering_audit(pca_result["pcs"], y)
    run_target_dependency_audit(X, y)
    run_subgroup_audit(df, target_col)
    write_interpretation_report(df, X, target_col, pca_result)

    elapsed = time.time() - start

    runtime = {
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60,
        "output_dir": str(OUTPUT_DIR),
    }

    save_json(runtime, LOG_DIR / "runtime_summary.json")

    print("Completed successfully.")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()