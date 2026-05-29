# Experiment 0: Latent Structure Audit Report

## Data overview

Rows: 20998
Columns before modeling: 35
Modeling features after cleaning: 30
Target column detected: disability_stage

## PCA latent structure evidence

Variance explained by PC1: 0.1113
Variance explained by first 5 PCs: 0.4102
Variance explained by first 10 PCs: 0.6339

Components needed for 70% variance: 12
Components needed for 80% variance: 16
Components needed for 90% variance: 20
Components needed for 95% variance: 23

## Interpretation guide

If a relatively small number of components explains a large fraction of variance, this supports the claim that the dataset contains a compressed latent structure. If many components are required, the population structure is more distributed and the reconstruction model should use a richer latent space.

The PCA stability report evaluates whether the principal directions remain similar across train/test subsamples. Higher cosine similarity indicates that the detected latent directions are not merely sampling noise.

The correlation, redundancy, and mutual-information reports define which variables carry implicit relationships. These outputs should guide the structural loss terms in later inverse population reconstruction experiments.
