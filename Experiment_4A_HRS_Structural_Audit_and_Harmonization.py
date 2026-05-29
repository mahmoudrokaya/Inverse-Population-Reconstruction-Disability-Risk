# -*- coding: utf-8 -*-
r"""
Experiment 4A: HRS Structural Audit and Harmonization
Root: D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments
"""

from __future__ import annotations

import json
import re
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(r"D:\47\472\New-Papers\InverseP-R-D-NHANES\Experiments")
NHANES_DIR = BASE_DIR / "data_raw"
HRS_DIR = BASE_DIR / "HRS"
OUTPUT_DIR = BASE_DIR / "Experiment4A_HRS_Structural_Audit_Harmonization"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "harmonized_data"
LOG_DIR = OUTPUT_DIR / "logs"

for folder in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, DATA_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MIN_NON_MISSING_RATE = 0.50
MAX_UNIQUE_FOR_CATEGORICAL = 60
PCA_COMPONENTS_TO_COMPARE = 20

CONCEPT_KEYWORDS = {
    "age": ["age", "r1age", "r2age", "ragey", "agey"],
    "sex": ["sex", "gender", "male", "female", "ragender"],
    "education": ["educ", "education", "degree", "schlyrs", "raedyrs"],
    "race_ethnicity": ["race", "ethnic", "hispan", "racem", "rahispan"],
    "marital_status": ["marital", "married", "divorce", "widow"],
    "income": ["income", "earn", "wealth", "asset", "hhinc", "hhincome"],
    "bmi": ["bmi", "body_mass", "bodymass"],
    "weight": ["weight", "body_weight"],
    "height": ["height"],
    "smoking": ["smoke", "smoking", "cigar"],
    "diabetes": ["diabetes", "diab"],
    "hypertension": ["hypertension", "hibp", "blood_pressure", "bp"],
    "heart_disease": ["heart", "cardiac", "angina", "stroke"],
    "arthritis": ["arthritis", "arthr"],
    "cancer": ["cancer"],
    "depression": ["depress", "cesd", "phq"],
    "physical_activity": ["activity", "exercise", "walk", "vigorous", "moderate"],
    "adl_disability": ["adl", "dress", "bathe", "eat", "bed", "toilet"],
    "iadl_disability": ["iadl", "money", "phone", "meal", "shop", "medication"],
    "mobility": ["mobility", "walk", "stairs", "climb", "stoop", "kneel", "lift"],
    "grip_strength": ["grip", "strength"],
    "self_rated_health": ["self_rated", "srh", "health_status", "shlt"],
    "memory_cognition": ["memory", "cog", "word", "recall", "mental"],
}


def save_json(obj: Dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)


def clean_col_name(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r"[^a-z0-9_]+", "_", c)
    c = re.sub(r"_+", "_", c)
    return c.strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [clean_col_name(c) for c in df.columns]
    return df


def infer_cycle(path: Path) -> str:
    for cycle in ["2011-2012", "2013-2014", "2015-2016", "2017-2018"]:
        if cycle in path.name:
            return cycle
    if "pooled" in path.name.lower():
        return "pooled"
    return "unknown"


def plot_and_save(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def concept_for_column(col: str) -> Optional[str]:
    c = clean_col_name(col)
    for concept, kws in CONCEPT_KEYWORDS.items():
        for kw in kws:
            if clean_col_name(kw) in c:
                return concept
    return None


def choose_best_column(df: pd.DataFrame, concept: str) -> Optional[str]:
    candidates = []
    for col in df.columns:
        if concept_for_column(col) == concept:
            non_missing = 1.0 - float(df[col].isna().mean())
            unique_count = int(df[col].nunique(dropna=True))
            candidates.append((col, non_missing, unique_count))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (x[1], x[2]), reverse=True)[0][0]


def load_nhanes() -> pd.DataFrame:
    patterns = [
        "NHANES_2011-2012_final_processed*.csv",
        "NHANES_2013-2014_final_processed*.csv",
        "NHANES_2015-2016_final_processed*.csv",
        "NHANES_2017-2018_final_processed*.csv",
        "*.csv",
    ]
    files, seen = [], set()
    for pat in patterns:
        for p in NHANES_DIR.glob(pat):
            if p.name not in seen:
                files.append(p)
                seen.add(p.name)
    if not files:
        raise FileNotFoundError(f"No NHANES CSV files found in {NHANES_DIR}")

    frames, manifest = [], []
    for p in files:
        df = normalize_columns(pd.read_csv(p))
        df["cycle"] = infer_cycle(p)
        df["source_file"] = p.name
        frames.append(df)
        manifest.append({"file": p.name, "cycle": infer_cycle(p), "rows": int(df.shape[0]), "columns": int(df.shape[1])})
    pd.DataFrame(manifest).to_csv(TABLE_DIR / "nhanes_input_manifest.csv", index=False)
    return normalize_columns(pd.concat(frames, ignore_index=True, sort=False))


def load_hrs() -> pd.DataFrame:
    files = list(HRS_DIR.glob("*.dta"))
    if not files:
        raise FileNotFoundError(f"No .dta HRS file found in {HRS_DIR}")
    path = files[0]
    df = pd.read_stata(path, convert_categoricals=False)
    df = normalize_columns(df)
    df["source_file"] = path.name
    pd.DataFrame([{"file": path.name, "rows": int(df.shape[0]), "columns": int(df.shape[1])}]).to_csv(
        TABLE_DIR / "hrs_input_manifest.csv", index=False
    )
    return df


def data_quality(df: pd.DataFrame, name: str) -> pd.DataFrame:
    q = pd.DataFrame({
        "dataset": name,
        "column": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "missing_count": [int(df[c].isna().sum()) for c in df.columns],
        "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
        "non_missing_rate": [float(1.0 - df[c].isna().mean()) for c in df.columns],
        "unique_count": [int(df[c].nunique(dropna=True)) for c in df.columns],
    })
    q.to_csv(TABLE_DIR / f"{name.lower()}_data_quality_summary.csv", index=False)

    top = q.sort_values("missing_rate", ascending=False).head(40)
    plt.figure(figsize=(12, 8))
    plt.barh(top["column"], top["missing_rate"])
    plt.xlabel("Missing rate")
    plt.ylabel("Variable")
    plt.title(f"{name}: Top Missingness Rates")
    plt.gca().invert_yaxis()
    plot_and_save(FIGURE_DIR / f"fig_{name.lower()}_top_missingness.png")
    return q


def build_overlap(nhanes: pd.DataFrame, hrs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for concept in CONCEPT_KEYWORDS:
        nh = choose_best_column(nhanes, concept)
        hr = choose_best_column(hrs, concept)
        rows.append({
            "concept": concept,
            "nhanes_column": nh,
            "hrs_column": hr,
            "available_in_nhanes": nh is not None,
            "available_in_hrs": hr is not None,
            "matched": nh is not None and hr is not None,
            "nhanes_missing_rate": float(nhanes[nh].isna().mean()) if nh else np.nan,
            "hrs_missing_rate": float(hrs[hr].isna().mean()) if hr else np.nan,
            "nhanes_unique_count": int(nhanes[nh].nunique(dropna=True)) if nh else np.nan,
            "hrs_unique_count": int(hrs[hr].nunique(dropna=True)) if hr else np.nan,
        })
    overlap = pd.DataFrame(rows)
    overlap.to_csv(TABLE_DIR / "nhanes_hrs_concept_overlap.csv", index=False)
    overlap[overlap["matched"]].to_csv(TABLE_DIR / "matched_variables.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.bar(
        ["NHANES concepts", "HRS concepts", "Matched concepts"],
        [int(overlap["available_in_nhanes"].sum()), int(overlap["available_in_hrs"].sum()), int(overlap["matched"].sum())]
    )
    plt.ylabel("Count")
    plt.title("NHANES-HRS Concept Overlap")
    plot_and_save(FIGURE_DIR / "fig01_concept_overlap_counts.png")
    return overlap


def compatibility(overlap: pd.DataFrame) -> pd.DataFrame:
    comp = overlap.copy()
    comp["missing_compatible"] = (
        (comp["nhanes_missing_rate"].fillna(1) <= (1 - MIN_NON_MISSING_RATE)) &
        (comp["hrs_missing_rate"].fillna(1) <= (1 - MIN_NON_MISSING_RATE))
    )
    comp["unique_count_ratio"] = comp.apply(
        lambda r: min(r["nhanes_unique_count"], r["hrs_unique_count"]) / max(r["nhanes_unique_count"], r["hrs_unique_count"])
        if pd.notna(r["nhanes_unique_count"]) and pd.notna(r["hrs_unique_count"]) and max(r["nhanes_unique_count"], r["hrs_unique_count"]) > 0
        else np.nan,
        axis=1,
    )
    comp["compatible"] = comp["matched"] & comp["missing_compatible"]
    comp.to_csv(TABLE_DIR / "feature_compatibility_matrix.csv", index=False)
    return comp


def make_harmonized(nhanes: pd.DataFrame, hrs: pd.DataFrame, comp: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    selected = comp[comp["compatible"]].copy()
    if selected.empty:
        selected = comp[comp["matched"]].copy()
    if selected.empty:
        raise RuntimeError("No matched variables found.")

    nh_h, hr_h = pd.DataFrame(index=nhanes.index), pd.DataFrame(index=hrs.index)
    concepts = []
    rows = []
    for _, r in selected.iterrows():
        concept, nh_col, hr_col = r["concept"], r["nhanes_column"], r["hrs_column"]
        if pd.isna(nh_col) or pd.isna(hr_col):
            continue
        nh_h[concept] = nhanes[nh_col]
        hr_h[concept] = hrs[hr_col]
        concepts.append(concept)
        rows.append({"concept": concept, "nhanes_column": nh_col, "hrs_column": hr_col, "compatible": bool(r["compatible"])})

    nh_h["dataset"] = "NHANES"
    hr_h["dataset"] = "HRS"
    nh_h.to_csv(DATA_DIR / "nhanes_harmonized_raw.csv", index=False)
    hr_h.to_csv(DATA_DIR / "hrs_harmonized_raw.csv", index=False)
    pd.DataFrame(rows).to_csv(TABLE_DIR / "harmonization_map.csv", index=False)
    return nh_h, hr_h, concepts


def numeric_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for col in X.columns:
        if X[col].dtype == "object" or str(X[col].dtype).startswith("category"):
            if X[col].nunique(dropna=True) <= MAX_UNIQUE_FOR_CATEGORICAL:
                vals = X[col].astype(str).replace("nan", np.nan).fillna("Missing")
                X[col] = LabelEncoder().fit_transform(vals.astype(str))
            else:
                X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.dropna(axis=1, how="all")
    X = X[X.nunique(dropna=True)[lambda s: s > 1].index.tolist()]
    return X


def scale(X: pd.DataFrame) -> pd.DataFrame:
    X_imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=X.columns, index=X.index)
    return pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=X.columns, index=X.index)


def run_pca_and_structure(nh_h: pd.DataFrame, hr_h: pd.DataFrame, concepts: List[str]) -> Dict[str, float]:
    Xn = numeric_matrix(nh_h, concepts)
    Xh = numeric_matrix(hr_h, concepts)
    common = [c for c in Xn.columns if c in Xh.columns]
    Xn, Xh = scale(Xn[common]), scale(Xh[common])
    Xn.to_csv(DATA_DIR / "nhanes_harmonized_scaled.csv", index=False)
    Xh.to_csv(DATA_DIR / "hrs_harmonized_scaled.csv", index=False)

    n_comp = min(PCA_COMPONENTS_TO_COMPARE, Xn.shape[1], Xh.shape[1], Xn.shape[0]-1, Xh.shape[0]-1)
    if n_comp < 2:
        raise RuntimeError("Not enough harmonized features for PCA comparison.")

    pn, ph = PCA(n_components=n_comp, random_state=RANDOM_STATE), PCA(n_components=n_comp, random_state=RANDOM_STATE)
    Zn, Zh = pn.fit_transform(Xn), ph.fit_transform(Xh)

    evr = pd.DataFrame({
        "component": np.arange(1, n_comp + 1),
        "nhanes_explained_variance_ratio": pn.explained_variance_ratio_,
        "hrs_explained_variance_ratio": ph.explained_variance_ratio_,
        "nhanes_cumulative_variance": np.cumsum(pn.explained_variance_ratio_),
        "hrs_cumulative_variance": np.cumsum(ph.explained_variance_ratio_),
    })
    evr.to_csv(TABLE_DIR / "pca_comparison_explained_variance.csv", index=False)

    sims = []
    for i in range(n_comp):
        v1, v2 = pn.components_[i], ph.components_[i]
        sims.append(float(abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))))
    sim_df = pd.DataFrame({"component": np.arange(1, n_comp + 1), "absolute_cosine_similarity": sims})
    sim_df.to_csv(TABLE_DIR / "pca_component_similarity_nhanes_hrs.csv", index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(evr["component"], evr["nhanes_cumulative_variance"], marker="o", label="NHANES")
    plt.plot(evr["component"], evr["hrs_cumulative_variance"], marker="o", label="HRS")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Cumulative Variance: NHANES vs HRS")
    plt.legend()
    plot_and_save(FIGURE_DIR / "fig02_pca_cumulative_variance_nhanes_hrs.png")

    plt.figure(figsize=(9, 5))
    plt.plot(sim_df["component"], sim_df["absolute_cosine_similarity"], marker="o")
    plt.xlabel("Principal component")
    plt.ylabel("Absolute cosine similarity")
    plt.ylim(0, 1.05)
    plt.title("Latent Direction Similarity: NHANES vs HRS")
    plot_and_save(FIGURE_DIR / "fig03_pca_component_similarity.png")

    plt.figure(figsize=(8, 6))
    plt.scatter(Zn[:, 0], Zn[:, 1], s=5, alpha=0.35, label="NHANES")
    sample_n = min(len(Zh), 20000)
    idx = np.random.default_rng(RANDOM_STATE).choice(len(Zh), size=sample_n, replace=False) if len(Zh) > sample_n else np.arange(len(Zh))
    plt.scatter(Zh[idx, 0], Zh[idx, 1], s=5, alpha=0.35, label="HRS")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Latent Geometry Projection: NHANES vs HRS")
    plt.legend()
    plot_and_save(FIGURE_DIR / "fig04_latent_geometry_projection.png")

    Cn, Ch = Xn.corr().fillna(0), Xh.corr().fillna(0)
    diff = Cn - Ch
    diff.to_csv(TABLE_DIR / "correlation_difference_nhanes_minus_hrs.csv")

    plt.figure(figsize=(10, 8))
    plt.imshow(diff.values, aspect="auto")
    plt.colorbar(label="Correlation difference")
    plt.xticks(range(len(diff.columns)), diff.columns, rotation=90)
    plt.yticks(range(len(diff.index)), diff.index)
    plt.title("Correlation Difference Matrix: NHANES - HRS")
    plot_and_save(FIGURE_DIR / "fig05_correlation_difference_heatmap.png")

    Covn, Covh = np.cov(Xn.values, rowvar=False), np.cov(Xh.values, rowvar=False)
    flatn = Cn.values[np.triu_indices_from(Cn.values, k=1)]
    flath = Ch.values[np.triu_indices_from(Ch.values, k=1)]
    corr_vec_sim = float(np.corrcoef(flatn, flath)[0, 1]) if np.std(flatn) > 0 and np.std(flath) > 0 else np.nan

    struct = {
        "n_harmonized_features": int(len(common)),
        "correlation_structure_distance": float(np.linalg.norm(Cn.values - Ch.values, ord="fro")),
        "covariance_structure_distance": float(np.linalg.norm(Covn - Covh, ord="fro")),
        "correlation_vector_similarity": corr_vec_sim,
        "mean_pca_component_similarity": float(np.mean(sims)),
        "pc1_similarity": float(sims[0]),
        "pc2_similarity": float(sims[1]) if len(sims) > 1 else np.nan,
        "nhanes_pc1_variance": float(pn.explained_variance_ratio_[0]),
        "hrs_pc1_variance": float(ph.explained_variance_ratio_[0]),
    }
    save_json(struct, LOG_DIR / "cross_dataset_structural_similarity_summary.json")
    return struct


def write_report(nhanes: pd.DataFrame, hrs: pd.DataFrame, overlap: pd.DataFrame, comp: pd.DataFrame, struct: Dict[str, float]) -> None:
    report = f"""# Experiment 4A: HRS Structural Audit and Harmonization Report

NHANES rows: {nhanes.shape[0]}
NHANES columns: {nhanes.shape[1]}

HRS rows: {hrs.shape[0]}
HRS columns: {hrs.shape[1]}

Candidate concepts evaluated: {len(CONCEPT_KEYWORDS)}
Matched concepts: {int(overlap["matched"].sum())}
Compatible concepts after missingness screening: {int(comp["compatible"].sum())}

Harmonized features used: {struct.get("n_harmonized_features")}
Correlation structure distance: {struct.get("correlation_structure_distance")}
Covariance structure distance: {struct.get("covariance_structure_distance")}
Correlation vector similarity: {struct.get("correlation_vector_similarity")}
Mean PCA component similarity: {struct.get("mean_pca_component_similarity")}
PC1 similarity: {struct.get("pc1_similarity")}
PC2 similarity: {struct.get("pc2_similarity")}

This audit determines whether HRS can support external validation of the NHANES-trained inverse population reconstruction framework.
"""
    (OUTPUT_DIR / "Experiment4A_HRS_Structural_Audit_Harmonization_Report.md").write_text(report, encoding="utf-8")


def main() -> None:
    start = time.time()
    print("=" * 90)
    print("Experiment 4A: HRS Structural Audit and Harmonization")
    print("=" * 90)
    print(f"NHANES folder: {NHANES_DIR}")
    print(f"HRS folder   : {HRS_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")

    nhanes = load_nhanes()
    hrs = load_hrs()

    print(f"NHANES shape: {nhanes.shape}")
    print(f"HRS shape   : {hrs.shape}")

    qn = data_quality(nhanes, "NHANES")
    qh = data_quality(hrs, "HRS")
    pd.concat([qn, qh], ignore_index=True).to_csv(TABLE_DIR / "combined_data_quality_summary.csv", index=False)

    overlap = build_overlap(nhanes, hrs)
    comp = compatibility(overlap)
    nh_h, hr_h, concepts = make_harmonized(nhanes, hrs, comp)
    struct = run_pca_and_structure(nh_h, hr_h, concepts)
    write_report(nhanes, hrs, overlap, comp, struct)

    elapsed = time.time() - start
    save_json({"runtime_seconds": elapsed, "runtime_minutes": elapsed / 60, "output_dir": str(OUTPUT_DIR)}, LOG_DIR / "runtime_summary.json")

    print("=" * 90)
    print("Completed successfully.")
    print(f"Runtime: {elapsed:.2f} seconds")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
