# RDKit-Cheminformatics-Assignment-Data-Curation-Fingerprint-Generation-Preprocessing
A comprehensive Jupyter Notebook-based assignment demonstrating molecular descriptor and fingerprint generation using RDKit, with data preprocessing and standardization techniques.

## 📋 Project Overview

This project implements a cheminformatics workflow covering:
- **Part A**: Data curation and fingerprint generation
- **Part B**: Generation of multiple fingerprint types and 2D molecular descriptors
- **Part C**: Data cleaning and statistical preprocessing (scaling & standardization)
- **Part D**: Reusable functions for descriptor computation and standardization

## 🎯 Objectives

- Practice molecular data curation and validation.
- Generate various types of molecular fingerprints (MACCS, Morgan/ECFP, Atom Pair, Topological Torsion, Pharmacophore).
- Compute RDKit 2D numerical descriptors.
- Apply min-max normalization and Z-score standardization.

## 📊 Assignment Breakdown

### Part A: Data Curation & MACCS Fingerprints
- **Input**: `dataset_1.csv` (contains SMILES and TOXICITY_LABEL)
- **Process**:
  - Remove rows with missing SMILES or LABELS
  - Validate SMILES strings using RDKit
  - Generate MACCS fingerprints (167 bits)
- **Output**: `macckeys_output.csv`

### Part B: Multiple Fingerprint Types & 2D Descriptors
- **Fingerprints Generated**:
  1. **Morgan Fingerprints (ECFP)**: Circular fingerprints, radius=2, 2048 bits
  2. **Atom Pair Fingerprints**: Topological atom pair descriptors, 2048 bits
  3. **Topological Torsion Fingerprints**: 4-atom torsion angles, 2048 bits
  4. **Pharmacophore Fingerprints**: Gobbi Pharm2D features
  5. **2D Descriptors**: 217 RDKit molecular descriptors

- **Output Files**: Individual CSVs for each fingerprint type + combined descriptor file

### Part C: Data Scaling & Standardization
- **Input**: `dataset_2.csv` (numerical descriptor data)
- **Cleaning Steps**:
  - Replace infinite values (±∞) with NaN
  - Remove rows with all missing values
  - Remove columns with any missing values
- **Transformations**:
  - **Min-Max Normalization**: X_scaled = (X - X_min) / (X_max - X_min)
  - **Z-score Standardization**: X_std = (X - μ) / σ

### Part D: Reusable Functions
Two key functions implemented:

```python
def compute_rdkit_2d_descriptors(csv_file_path)
    # Takes SMILES CSV (path_file)→ Returns DataFrame with 2D descriptors for given SMILES

def standardize_descriptors(raw_descriptors_df_csv_file_path)
    # Takes raw descriptors (path_file) → Returns normalized & standardized descriptors in the form of a DataFrame
```

## 💻 Code Structure

### Notebook Sections

| Section | Description | Input | Output |
|---------|-------------|-------|--------|
| **Part A** | Data Curation & MACCS FP | dataset_1.csv | macckeys_output.csv |
| **Part B** | Fingerprints & Descriptors | Cleaned molecules | FP CSVs + descriptor CSV |
| **Part C** | Scaling & Standardization | dataset_2.csv | Standardized DataFrame |
| **Part D** | Reusable Functions | CSV files | DataFrames |

## 🛠️ Setup

### Dependencies:
  - python=3.14.2
  - rdkit
  - jupyter
  - conda-forge::pubchempy
  - conda-forge::scikit-learn

## 📝 Key Assumptions & Workflow

### Assumptions Made

1. **Data Format**: 
   - `dataset_1.csv` contains "SMILES" and "LABELS" (as in toxicity labels) columns.
   - `dataset_2.csv` contains numeric columns (which is checked in the code as well).

2. **SMILES Validation**:
   - Invalid SMILES are identified using RDKit's `SanitizeMol()` function.
   - Problematic molecules are removed entirely from the dataset and their SMILES IDs can be saved in a separate csv file for troubleshooting.

3. **Fingerprint Parameters**:
   - All count fingerprints converted to bit vectors for consistency.
   - Fingerprints stored as bit strings in CSV for readability.

4. **Descriptor Scope**:
   - All 217 RDKit Descriptors_v2. 
   - Descriptors computed for valid molecules only.

5. **Scaling Strategy**:
   - Min-Max normalization applied BEFORE Z-score standardization
   - This ensures standardized values are centered at 0 with unit variance
   - Both transformations prevent data leakage (fit on training set only)

6. **Missing Values**:
   - Part A: Rows with missing SMILES or LABELS are dropped.
   - Part C: Columns and Rows with ANY missing or infinite values are dropped.

## 🔍 Notes on Data

- **Dataset_1**: 643 molecules after cleaning (0 invalid SMILES in given data).
- **Dataset_2**: 217 molecular descriptors computed; 205 remain after cleaning.
- **Fingerprints**: All 2048-bit representations for consistency (except MACCS).

## 📚 References


### Official Documentation
- [RDKit Documentation](https://www.rdkit.org/) - Comprehensive guide to RDKit cheminformatics toolkit
- [RDKit Descriptors](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors) - List of available molecular descriptors
- [RDKit Fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprints-and-molecular-similarity) - Guide to fingerprint generation and similarity calculations
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html) - Documentation for scaling and standardization methods

### Educational Resources
- [Molecular Fingerprints with Python and RDKit for AI Models](https://zoehlerbz.medium.com/representation-of-molecular-fingerprints-with-python-and-rdkit-for-ai-models-8b146bcf3230) - Guide to understanding fingerprint representations and their applications in machine learning models. Provides practical examples of different fingerprint types and their use in molecular similarity calculations.

- [Extracting 200+ RDKit Features for Machine Learning](https://medium.com/@hamidhadipour74/unlocking-molecular-insights-a-comprehensive-guide-to-extracting-200-rdkit-features-for-machine-e43c619bec46) - Detailed guide on how to normalize and standardize molecular descriptors in the context of descriptor extraction and data preprocessing. Covers best practices for preparing molecular data for machine learning pipelines.

- [RDKit Cheatsheet](https://xinhaoli74.github.io/blog/rdkit/2021/01/06/rdkit.html#RDKit-2D-Descriptors) - Breif explaination on how different functions are utilized with RDKit, particularly 2D Descriptors.

### Research References
- [Gobbi, A. and Poppinger, D. (2000).](https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z) - "Genetic optimization of combinatorial libraries" : Feature definitions for 2D pharmacophore fingerprints used in this assignment.

## 📝 License

This assignment is done for educational purposes.

## 🙏 Acknowledgments

This repository structure and documentation were organized and created with assistance from **[Perplexity AI](https://www.perplexity.ai/)**.

## 👤 Author

- **Anirudh** / [an17rudh](https://github.com/an17rudh)
- **Date**: January 6, 2026


**Last Updated**: January 6, 2026

