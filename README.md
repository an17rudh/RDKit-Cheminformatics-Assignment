# RDKit-Cheminformatics-Assignment-Data-Curation-Fingerprint-Generation-Preprocessing
A comprehensive Jupyter Notebook-based assignment demonstrating molecular descriptor and fingerprint generation using RDKit, with data preprocessing and standardization techniques.

## 📋 Project Overview

This project implements a complete cheminformatics pipeline covering:
- **Part A**: Data curation and fingerprint generation
- **Part B**: Generation of multiple fingerprint types and 2D molecular descriptors
- **Part C**: Data cleaning and statistical preprocessing (scaling & standardization)
- **Part D**: Reusable functions for descriptor computation and standardization

## 🎯 Objectives

- Practice molecular data curation and validation
- Generate various types of molecular fingerprints (MACCS, Morgan/ECFP, Atom Pair, Topological Torsion, Pharmacophore)
- Compute RDKit 2D numerical descriptors
- Apply min-max normalization and Z-score standardization

## 📁 Repository Structure

```
assignment-1-rdkit/
├── README.md                          # This file
├── Assignment_1.ipynb                 # Main Jupyter Notebook
├── descriptor_functions.py            # Reusable helper functions
├── .gitignore                         # Git ignore rules
├── data/                              # Input datasets (add locally)
│   ├── dataset_1.csv                  # Molecular data with SMILES
│   └── dataset_2.csv                  # Numerical descriptor data
└── outputs/                           # Generated output files
    ├── macckeys_output.csv
    ├── ecfp_output.csv
    ├── atom_pair_output.csv
    ├── topological_torsion_output.csv
    ├── pharmacophore_fp_output.csv
    ├── descriptors_2d_output.csv
    └── rdkit_2d_descriptors_standardized_output.csv
```

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
    # Takes SMILES CSV → Returns DataFrame with 2D descriptors for given SMILWS

def standardize_descriptors(raw_descriptors_df_csv_file_path)
    # Takes raw descriptors → Returns normalized & standardized descriptors in the form of a DataFrame
```

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
   - All 217 RDKit Descriptors_v2 
   - Descriptors computed for valid molecules only.

5. **Scaling Strategy**:
   - Min-Max normalization applied BEFORE Z-score standardization
   - This ensures standardized values are centered at 0 with unit variance
   - Both transformations prevent data leakage (fit on training set only)

6. **Missing Values**:
   - Part A: Rows with missing SMILES or LABELS are dropped.
   - Part C: Columns and Rows with ANY missing or infinite values are dropped.

### Workflow

```
Load Dataset
    ↓
Data Curation (Remove NaN, validate SMILES)
    ↓
Generate Fingerprints (5 types)
    ↓
Calculate 2D Descriptors
    ↓
Save Outputs
    ↓
[Separate Flow for Dataset 2]
    ↓
Clean Descriptors (remove inf, drop missing)
    ↓
Apply Min-Max Normalization
    ↓
Apply Z-score Standardization
    ↓
Save Standardized Data
```

## 💻 Code Structure

### Notebook Sections

| Section | Description | Input | Output |
|---------|-------------|-------|--------|
| **Part A** | Data Curation & MACCS FP | dataset_1.csv | macckeys_output.csv |
| **Part B** | Fingerprints & Descriptors | Cleaned molecules | 5 FP CSVs + descriptor CSV |
| **Part C** | Scaling & Standardization | dataset_2.csv | Standardized CSV |
| **Part D** | Reusable Functions | CSV files | DataFrames |



## 🔍 Data Quality Notes

- **Dataset_1**: 643 molecules after cleaning (0 invalid SMILES in given data).
- ***Dataset_2**: 217 molecular descriptors computed; 205 remain after cleaning.
- **Fingerprints**: All 2048-bit representations for consistency (except MACCs.


## 📚 References

- [RDKit Documentation](https://www.rdkit.org/)
- Descriptors: [RDKit Descriptors](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors)
- Fingerprints: [RDKit Fingerprints](https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprints-and-molecular-similarity)
- Scikit-learn: [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

## 📝 License

This assignment is provided for educational purposes.

## 👤 Author

- **Your Name** / Your GitHub Handle
- **Date**: January 2026
- **Course**: [Your Course Name]

## ❓ FAQ

**Q: Do I need the dataset files?**  
A: Yes, place `dataset_1.csv` and `dataset_2.csv` in the `data/` folder before running.

**Q: Can I modify the fingerprint parameters?**  
A: Yes! Edit the radius, nBits, or fpSize parameters in Part B cells.

**Q: What if I have invalid SMILES?**  
A: The notebook will print their indices and remove them. Check console output.

**Q: How do I use the reusable functions?**  
A: Import from `descriptor_functions.py` and call with your CSV paths.

---

**Last Updated**: January 6, 2026

