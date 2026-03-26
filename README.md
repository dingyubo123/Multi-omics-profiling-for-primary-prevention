# Multi-omics-profiling-for-primary-prevention
Multi-omics Profiling for Primary Prevention: Integrating Clinical, Proteomic, and polygenic scores to Predict Cardiovascular Disease
CVD Multi-Omics & Clinical Risk Assessment Toolkit

1. Project Overview
This repository contains a comprehensive computational pipeline for Cardiovascular Disease (CVD) risk stratification. By integrating traditional clinical scores (SCORE2, FRS) with advanced proteomic signatures (Lasso-Cox modeling) and multiple imputation techniques (MICE), this project aims to enhance the predictive accuracy of 10-year CVD events using the UK Biobank (UKB) and other clinical datasets.

2. Key Modules
Clinical Risk Calculators
PREVENT™ 2023: Implementation of the AHA PREVENT™ equations for 10-year Total CVD and ASCVD risk, including renal function (eGFR) and metabolic parameters.
SCORE2 & SCORE2-OP: Calibrated for European low-risk regions, providing detailed diagnostic feedback on patient eligibility and LDL targets.
Framingham Risk Score (FRS): Batch processing of the official FRS model with automated unit conversion (mmol/L to mg/dL) and robust column mapping.

Proteomic Risk Scoring (PRS)
Lasso-Cox Modeling: A high-dimensional feature selection pipeline to identify stable protein biomarkers.
Stability Analysis: Includes 1,000x bootstrap resampling to ensure the consistency of selected protein features (Selection Frequency ≥ 80%).
Quantile Normalization: Pre-processing of proteomic data to ensure Gaussian distribution for improved model fitting.

Bioinformatics & Pathway Analysis
Enrichment Pipeline: R-based scripts for Reactome, GO (BP, CC, MF), and DisGeNET enrichment.
Network Visualization: Generation of Edge/Node files for Cytoscape to visualize Gene-Pathway interactions.

3. Workflow & Data Processing
Data Imputation: Uses MICE (Multivariate Imputation by Chained Equations) via miceforest or IterativeImputer to handle missing clinical values.
Feature Selection: Employs Lasso-penalized Cox regression to condense ~1,460 proteins into a stable "Protein Risk Score."
Model Evaluation: Comparison of incremental predictive value (C-index, NRI, IDI) when adding proteins to the base clinical models (SCORE2/PREVENT).
