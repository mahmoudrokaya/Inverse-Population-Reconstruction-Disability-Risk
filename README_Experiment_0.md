# Experiment 0: Latent Structure Audit

## Purpose
This script audits the NHANES raw data before inverse population reconstruction. It checks whether the dataset contains meaningful latent structure through data quality analysis, correlation analysis, PCA, PCA stability, clustering, mutual information, and subgroup distribution checks.

## Expected folders
Input:
`D:\47\472\New-Papers\Inverse Population Reconstruction of Disability Risk NHANES\Experiments\data_raw`

Output:
`D:\47\472\New-Papers\Inverse Population Reconstruction of Disability Risk NHANES\Experiments\Experiment0_Latent_Structure_Audit`

## Run
Open PowerShell or CMD and run:

```powershell
cd "D:\47\472\New-Papers\Inverse Population Reconstruction of Disability Risk NHANES\Experiments"
python Experiment_0_Latent_Structure_Audit.py
```

## Main outputs
- `tables/data_quality_column_summary.csv`
- `tables/feature_correlation_matrix.csv`
- `tables/feature_redundancy_pairs_by_correlation.csv`
- `tables/pca_explained_variance.csv`
- `tables/pca_top_loading_features.csv`
- `tables/pca_stability_train_test.csv`
- `tables/latent_space_clustering_scores.csv`
- `tables/mutual_information_with_target.csv`
- `figures/fig06_pca_scree_plot.png`
- `figures/fig07_pca_cumulative_variance.png`
- `figures/fig09_pca_loading_heatmap.png`
- `Experiment0_Latent_Structure_Audit_Report.md`

## Interpretation
This experiment supports the paper's latent reconstruction claim by showing whether the data has:
1. compressible latent structure,
2. stable PCA directions,
3. meaningful variable dependencies,
4. cluster tendency in latent space,
5. subgroup and target relationships.
