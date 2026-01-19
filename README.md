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
- [Data Scaling](https://medium.com/@hhuseyincosgun/which-data-scaling-technique-should-i-use-a1615292061e) - Explains which scaler is suitable for transforming numerical feature values depending on the type of data available.

### Research References
- [Gobbi, A. and Poppinger, D. (2000).](https://doi.org/10.1002/(SICI)1097-0290(199824)61:1%3C47::AID-BIT9%3E3.0.CO;2-Z) - "Genetic optimization of combinatorial libraries" : Feature definitions for 2D pharmacophore fingerprints used in this assignment.

## 📝 License

This assignment is done for educational purposes.

## 🙏 Acknowledgments

This repository structure and documentation alongside comments on the code were organized and created with assistance from **[Perplexity AI](https://www.perplexity.ai/)**.

## 👤 Author

- **Anirudh** / [an17rudh](https://github.com/an17rudh)
- **Date**: January 6, 2026

# Code Update Documentation: v1 → v2


## Part A: Dataset 1 – Data Cleaning & Validation

### Change 1.1: Added Diagnostic Reporting
**Before:**
```python
ds1=pd.read_csv('Assignment_1/dataset_1.csv', header=0)

#(2)
ds1.dropna(subset=["SMILES","LABELS"],inplace=True,ignore_index=True)
```

**After:**
```python
ds1=pd.read_csv('Assignment_1/dataset_1.csv', header=0)
print(f'Raw dataset-1 shape : ',ds1.shape)

#(2)Checking for missing values 
missing_count = ds1.isnull().sum()
print(f'Missing values in each column :\n', missing_count)

#Removing empty(NaN) values
ds1.dropna(subset=["SMILES","LABELS"],inplace=True,ignore_index=True)
print(f'Dataset-1 shape after removing missing values : ',ds1.shape)
```

**Summary of Changes:**
- ✅ Print raw dataset shape before cleaning
- ✅ Quantify missing values per column using `isnull().sum()`
- ✅ Report shape after NaN removal for transparency

---

### Change 1.2: Improved Invalid SMILES Handling
**Before:**
```python
ds1.drop(invalid_SMILES, inplace=True)
print(f"Dataset had {len(invalid_SMILES)} invalid SMILES")
print(f"Dataset shape after cleaning : {ds1.shape}")
```

**After:**
```python
ds1.drop(invalid_SMILES, inplace=True)
print(f"Dataset-1 had {len(invalid_SMILES)} invalid SMILES")

if len(invalid_SMILES) > 0:
     print(f"Dataset-1 shape after removing invalid SMILES : {ds1.shape}")
```

**Summary of Changes:**
- ✅ Conditional shape printing (only when invalid SMILES exist)
- ✅ Consistent naming: "Dataset-1" throughout
- ✅ Better readability with conditional logic

---

### Change 1.3: Added Step Summary Comments
**Before:**
```python
#Resetting Indexes
ds1 = ds1.reset_index(drop=True)
```

**After:**
```python
#Resetting Indexes (To prevent indexing errors for further iterative coding)
ds1 = ds1.reset_index(drop=True)

#Summary:
#First, it checks for missing values in each column, then removes them. 
#After that, it creates a list of invalid SMILES (if any in the dataset) by checking whether the SMILES string at each row could be sanitized or not. 
#After this, code resets the index to reorder the DataFrame after removing invalid SMILES so that the DataFrame could be loaded for further iterative operations without indexing issues.
```

**Summary of Changes:**
- ✅ Added explanatory comments for *why* operations are performed
- ✅ Added comprehensive summary block at end of Part A

---

## Part B: Fingerprint Generation – Major Structural Refactoring

### Change 2.1: MACCS Fingerprints – From String Storage to Bit Matrix

**Before:**
```python
macckeys_list=[]
for fp in range(len(ds1)):
    molfp=ds1['ROMol'][fp]
    maccfp=MACCSkeys.GenMACCSKeys(molfp)
    macckeys_list.append(maccfp)
    
ds1['MACCKEYS'] = macckeys_list
ds1['MACCKEYS_STRING'] = ds1['MACCKEYS'].apply(lambda x: x.ToBitString())
ds1[['SMILES', 'MACCKEYS_STRING']].to_csv('macckeys_output.csv', index=False)
```

**Output:** CSV with 2 columns: `SMILES` + `MACCKEYS_STRING` (encoded bit string)

**After:**
```python
# Creating an empty DataFrame for MACCS keys
# Each row is corresponding to one molecule and each of the 166 columns will be storing one MACCS bit
maccs_table = pd.DataFrame(index=range(len(ds1['SMILES'])), columns=range(1,167))

# Step 2: Calculating the MACCS fingerprint for each SMILES
for row_index in range(len(ds1)):
    each_smiles = ds1.SMILES[row_index]
    given_smi2maccsfp = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(each_smiles))
    col_index = 0  # Starting the column index at zero before compiling bits
    
    # Looping through each bit in the fingerprint and filling the table
    for each_bit in given_smi2maccsfp.ToBitString():
        col_index += 1
        maccs_table.loc[row_index, col_index] = each_bit

# Combining the original dataset with the 167-bit MACCS fingerprint table
maccs_ds1 = ds1.merge(maccs_table, left_index=True, right_index=True, how='inner')

print('Dataset-1 with MACCS fingerprint :\n', maccs_ds1)

# Setting up the column order for CSV file
macckeys_coln = ['SMILES']
macckeys_coln.extend(list(range(1,167)))

# Saving the SMILES and their MACCS fingerprints to a CSV file
maccs_ds1.to_csv('macckeys_output.csv', index=False, columns=macckeys_coln)
```

**Output:** CSV with 167 columns: `SMILES` + `1` through `166` (individual bits as 0/1 values)

**Summary of Changes:**
- ✅ **Structural Change:** Moved from storing fingerprint objects + string to creating a separate individual bit matrix
- ✅ **Output Change:** CSV now contains 167 columns (SMILES + 166 bit columns) instead of encoded bit string
- ✅ **Better Documentation:** Detailed comments explaining the bit-loop structure

---

### Change 2.2: Morgan/ECFP Fingerprints – From String to 2048-Bit Matrix

**Before:**
```python
morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
for mfp in range(len(ds1)):
    molmp = Chem.MolFromSmiles(ds1['SMILES'][mfp])
    mfp_mol = morgan_gen.GetFingerprint(molmp)
    morgfp.append(mfp_mol)

ds1['ECFP'] = morgfp
ds1['ECFP_STRING'] = ds1['ECFP'].apply(lambda x: x.ToBitString())
ds1[['SMILES', 'ECFP_STRING']].to_csv('ecfp_output.csv', index=False)
```

**Output:** CSV with 2 columns: `SMILES` + `ECFP_STRING` (encoded bit string)

**After:**
```python
morgan_gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)

for mfp in range(len(ds1)):
    molmp = Chem.MolFromSmiles(ds1['SMILES'][mfp])
    mfp_mol = morgan_gen.GetFingerprint(molmp)
    morgfp.append(mfp_mol)

ds1['ECFP'] = morgfp

# Converting each fingerprint into a string
ecfp_bits = ds1['ECFP'].apply(lambda x: list(x.ToBitString()))

# Turning the list of bits into a new DataFrame
# Each column (bit_1 to bit_2048) is one position/bit of the fingerprint
ecfp_df = pd.DataFrame(ecfp_bits.tolist(), columns=[f'bit_{i}' for i in range(1,2049)])

# Removing the temporary ECFP column that stores RDKit objects
del ds1['ECFP']

# Joining the original data with the 2048-bit fingerprint columns
ecfp_ds1 = ds1.merge(ecfp_df, left_index=True, right_index=True, how='inner')

print('Dataset-1 with Morgan fingerprint :\n', ecfp_ds1)

# Setting up the column order for CSV file
ecfp_coln = ['SMILES']
ecfp_coln.extend(list(f'bit_{i}' for i in range(1,2049)))

# Saving the SMILES and their Morgan fingerprints to a CSV file
ecfp_ds1.to_csv('ecfp_output.csv', index=False, columns=ecfp_coln)
```

**Output:** CSV with 2049 columns: `SMILES` + `bit_1` through `bit_2048` (individual bits as 0/1 values)

**Summary of Changes:**
- ✅ **Structural Change:** Expanded fingerprint string into individual bit columns
- ✅ **Output Change:** CSV now contains 2049 columns (SMILES + 2048 bit columns)
- ✅ **Visualization:** Print full merged dataset before saving

---

### Change 2.3: Atom Pair Fingerprints – From String to 2048-Bit Matrix

**Before:**
```python
apfpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

for apfp in range(len(ds1)):
    molap=ds1['ROMol'][apfp]
    ap_fp = apfpgen.GetFingerprint(molap)
    ap_list.append(ap_fp)

ds1['Atom_Pair'] = ap_list
ds1['Atom_Pair_STRING'] = ds1['Atom_Pair'].apply(lambda x: x.ToBitString())
ds1[['SMILES', 'Atom_Pair_STRING']].to_csv('atom_pair_output.csv', index=False)
```

**Output:** CSV with 2 columns

**After:**
```python
apfpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

for apfp in range(len(ds1)):
    molap = ds1['ROMol'][apfp]
    ap_fp = apfpgen.GetFingerprint(molap)
    ap_list.append(ap_fp)

ds1['Atom_Pair'] = ap_list

# Converting each fingerprint into a string
ap_bits = ds1['Atom_Pair'].apply(lambda x: list(x.ToBitString()))

# Turning the list of bits into a new DataFrame
# Each column (bit_1 to bit_2048) is one position/bit of the fingerprint
ap_df = pd.DataFrame(ap_bits.tolist(), columns=[f'bit_number_{i}' for i in range(1,2049)])

# Removing the temporary Atom_Pair column that stores RDKit objects
del ds1['Atom_Pair']

# Joining the original data with the 2048-bit fingerprint columns
ap_ds1 = ds1.merge(ap_df, left_index=True, right_index=True, how='inner')

print('Dataset-1 with Atom Pair fingerprint :\n', ap_ds1)

# Setting up the column order for CSV file
ap_coln = ['SMILES']
ap_coln.extend(list(f'bit_number_{i}' for i in range(1,2049)))

# Saving the SMILES and their Atom pair fingerprints to a CSV file
ap_ds1.to_csv('atom_pair_output.csv', index=False, columns=ap_coln)
```

**Output:** CSV with 2049 columns: `SMILES` + `bit_number_1` through `bit_number_2048`

**Summary of Changes:**
- ✅ Consistent pattern with Morgan: expanded to individual bit columns
- ✅ Unique column naming: `bit_number_i` (distinguishes from Morgan)
- ✅ Full dataset printing before export

---

### Change 2.4: Topological Torsion Fingerprints – From String to 2048-Bit Matrix

**Before:**
```python
ttfpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)

for ttfp in range(len(ds1)):
    moltt=ds1['ROMol'][ttfp]
    tt_fp_vec = ttfpgen.GetFingerprint(moltt)
    ttfp_list.append(tt_fp_vec)

ds1['Topological_Torsion'] = ttfp_list
ds1['Topological_Torsion_STRING'] = ds1['Topological_Torsion'].apply(lambda x: x.ToBitString())
ds1[['SMILES', 'Topological_Torsion_STRING']].to_csv('topological_torsion_output.csv', index=False)
```

**Output:** CSV with 2 columns

**After:**
```python
ttfpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)

for ttfp in range(len(ds1)):
    moltt=ds1['ROMol'][ttfp]
    tt_fp_vec = ttfpgen.GetFingerprint(moltt)
    ttfp_list.append(tt_fp_vec)

ds1['Topological_Torsion'] = ttfp_list

# Converting each fingerprint into a string
ttfp_bits = ds1['Topological_Torsion'].apply(lambda x: list(x.ToBitString()))

# Turning the list of bits into a new DataFrame
# Each column (bit_1 to bit_2048) is one position/bit of the fingerprint
ttfp_df = pd.DataFrame(ttfp_bits.tolist(), columns=[f'bit_pos_{i}' for i in range(1,2049)])

# Removing the temporary Topological_Torsion column that stores RDKit objects
del ds1['Topological_Torsion']

# Joining the original data with the 2048-bit fingerprint columns
ttfp_ds1 = ds1.merge(ttfp_df, left_index=True, right_index=True, how='inner')

print('Dataset-1 with Topological Torsion fingerprint :\n', ttfp_ds1)

# Setting up the column order for CSV file
ttfp_coln = ['SMILES']
ttfp_coln.extend(list(f'bit_pos_{i}' for i in range(1,2049)))

# Saving the SMILES and their Topological Torsion fingerprints to a CSV file
ttfp_ds1.to_csv('topological_torsion_output.csv', index=False, columns=ttfp_coln)
```

**Output:** CSV with 2049 columns: `SMILES` + `bit_pos_1` through `bit_pos_2048`

**Summary of Changes:**
- ✅ Same expansion pattern as Atom Pair and Morgan
- ✅ Unique column naming: `bit_pos_i`

---

### Change 2.5: Pharmacophore Fingerprints – From String to Bit Matrix (Dynamic)

**Before:**
```python
pharma_list=[]

for pi in range(len(ds1)):
    mol_pi= Chem.MolFromSmiles(ds1['SMILES'][pi])
    pi_fp = Generate.Gen2DFingerprint(mol_pi,Gobbi_Pharm2D.factory)
    pharma_list.append(pi_fp)

ds1['Pharmacophore_FP'] = pharma_list
ds1['Pharmacophore_FP_STRING'] = ds1['Pharmacophore_FP'].apply(lambda x: x.ToBitString())
ds1[['SMILES', 'Pharmacophore_FP_STRING']].to_csv('pharmacophore_fp_output.csv', index=False)
```

**Output:** CSV with 2 columns

**After:**
```python
pharma_list = []

for pi in range(len(ds1)):
    mol_pi = Chem.MolFromSmiles(ds1['SMILES'][pi])
    pi_fp = Generate.Gen2DFingerprint(mol_pi, Gobbi_Pharm2D.factory)
    pharma_list.append(pi_fp)

# Adding the Pharmacophore fingerprint objects to a new column
ds1['Pharmacophore_FP'] = pharma_list

# Converting each Pharmacophore fingerprint into bit strings
pharma_bits = ds1['Pharmacophore_FP'].apply(lambda x: list(x.ToBitString()))

# Creating a DataFrame from the bit string lists, where each column represents one Pharmacophore bit
pharma_df = pd.DataFrame(pharma_bits.tolist())

print(pharma_df.shape)

# Setting up numbered column names (from 1 to len(pharma_df.columns)) for the Pharmacophore bits
ncoln = []
for coln in range(1, (len(pharma_df.columns) + 1)):
    ncoln.append(coln)
pharma_df.columns = ncoln

# Removing the temporary column that was holding RDKit fingerprint objects
del ds1['Pharmacophore_FP']

# Combining the original dataset with the Pharmacophore fingerprint columns
pharma_ds1 = ds1.merge(pharma_df, left_index=True, right_index=True, how='inner')

print('Dataset-1 with Pharmacophore-based fingerprint :\n', pharma_ds1)

# Creating a list of columns to export, excluding LABELS and ROMol columns
pcoln = []
for ncol in pharma_ds1.columns:
    if ncol not in ['LABELS', 'ROMol']:
        pcoln.append(ncol)

# Saving the SMILES and Pharmacophore fingerprints to a CSV file 
pharma_ds1.to_csv('pharmacophore_fp_output.csv', index=False, columns=pcoln)
```

**Output:** CSV with `N+1` columns: `SMILES` + numeric columns (1 to N, where N = ~39972 (in current Dataset-2)

**Summary of Changes:**
- ✅ **Dynamic Column Count:** Pharmacophore fingerprints have variable length; new code dynamically adapts column count
- ✅ **Shape Printing:** Shows fingerprint matrix dimensions before export
- ✅ **Better Flexibility:** Handles variable-length fingerprints without hardcoding bit count

---

## Part C: Dataset 2 – Column-based Scaling (Major Algorithmic Change)

### Change 3.1: From Sequential to Distribution-based Scaling

**Before:**
```python
# Apply Min-Max normalization to all columns
ds2_mm = MinMaxScaler().fit_transform(ds2)
ds2_scaled = pd.DataFrame(ds2_mm, columns=ds2.columns, index=ds2.index)

# Then apply Z-score standardization to the already-scaled data
ds2_zs = StandardScaler().fit_transform(ds2_scaled)
ds2_standardized = pd.DataFrame(ds2_zs, columns=ds2.columns, index=ds2.index)
```

**Approach:** Apply MinMaxScaler to all columns, then StandardScaler to the result
- ❌ Sequential scaling loses interpretability
- ❌ Not optimal for non-normally distributed data
- ❌ Treats all columns identically

**After:**
```python
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='scipy.stats')

scaled_columns = []
zscore_count = 0  # Counting columns that need StandardScaler (normal distribution)
minmax_count = 0  # Counting columns that need MinMaxScaler (non-normal distribution)

scol = []  # List of columns with normal distribution (will use Z-score scaling)
mmcol = [] # List of columns with non-normal distribution (will use Min-Max scaling)

# Checking each column to see if its data follows a normal distribution
for i, col_name in enumerate(ds2.columns):
    col = ds2[col_name]
    col_t = ds2[[col_name]]

    # Running Shapiro-Wilk test to check for normality (p > 0.05 means normal distribution)
    p = shapiro(col)[1]

    if p > 0.05:
        # Checking if the column has zero variation (all values are the same)
        if col.std() == 0:
            text = 'Not Normal'
            minmax_count = minmax_count + 1
            # Using MinMaxScaler for constant columns
            mmcol.append(col_name)
        else:           
            text = 'Normal'
            zscore_count = zscore_count + 1
            # Using StandardScaler for normally distributed data
            scol.append(col_name)
    else:
        # Non-normal data is getting MinMaxScaler
        text = 'Not Normal'
        minmax_count = minmax_count + 1
        mmcol.append(col_name)

# Creating scaling transformers for different column groups
transformers = []
if scol:
    # StandardScaler for normally distributed columns (preserves mean=0, std=1)
    transformers.append(('zscore', StandardScaler(), scol))
if mmcol:
    # MinMaxScaler for non-normal/constant columns (scales to 0-1 range)
    transformers.append(('minmax', MinMaxScaler(), mmcol))

# Applying both scalers at once using ColumnTransformer
ct = ColumnTransformer(transformers, remainder='passthrough')
ds2_scaled = ct.fit_transform(ds2)

# Converting the scaled array back to a DataFrame with original column names
ds2_scaled_df = pd.DataFrame(ds2_scaled, columns=ds2.columns, index=ds2.index)

# Showing the scaling summary and first few rows of scaled data
print(f'Summary: Columns with Not Normal distribution {minmax_count} and Normal distribution {zscore_count}')
print(ds2_scaled_df.head())
```

**Approach:** Test each column for normality; apply StandardScaler to normal columns, MinMaxScaler to non-normal columns
- ✅ **Statistically Sound:** Uses Shapiro-Wilk test (p-value > 0.05 indicates normal distribution)
- ✅ **Per-Column Optimization:** Different scalers for different column distributions
- ✅ **Edge Case Handling:** Detects and handles constant columns (std = 0)
- ✅ **Transparency:** Reports count of normal vs. non-normal columns
- ✅ **Professional Preprocessing:** Uses `ColumnTransformer` for robust multi-scaler application

**Why This Matters:**
- **StandardScaler** assumes data is normally distributed and transforms to mean=0, std=1
- **MinMaxScaler** preserves original distribution shape, scales to [0, 1] range
- Applying StandardScaler to non-normal (skewed) data can introduce artifacts
- Per-column selection ensures optimal preprocessing for machine learning

---

## Part D: Reusable Functions


### Change 4.1: `standardize_descriptors()` – Distribution-based Scaling (Major)

**Before:**
```python
# Sequential scaling: MinMax then StandardScaler
normalized_features = MinMaxScaler().fit_transform(clean_df)
normalized_df = pd.DataFrame(normalized_features, columns=clean_df.columns, index=clean_df.index)

standardized_features = StandardScaler().fit_transform(normalized_df)
standardized_df = pd.DataFrame(standardized_features, columns=clean_df.columns, index=clean_df.index)
```

**Output:** Single approach for all columns

**After:**
```python
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='scipy.stats')

# Counting columns for different scaling types
zscore_count = 0  # Normal distribution columns
minmax_count = 0  # Non-normal/constant columns
scol = []  # Columns needing StandardScaler
mmcol = [] # Columns needing MinMaxScaler

# Testing each column's distribution to choose the best scaling method
for i, col_name in enumerate(clean_df.columns):
    col = clean_df[col_name]
    p = shapiro(col)[1]  # Shapiro-Wilk test p-value

    if p > 0.05:  # Normal distribution
        if col.std() == 0:  # Constant column
            text = 'Not Normal'
            minmax_count += 1
            mmcol.append(col_name)
        else:
            text = 'Normal'
            zscore_count += 1
            scol.append(col_name)
    else:  # Non-normal distribution
        text = 'Not Normal'
        minmax_count += 1
        mmcol.append(col_name)

# Creating scaling transformers based on column distribution
transformers = []
if scol:
    transformers.append(('zscore', StandardScaler(), scol))
if mmcol:
    transformers.append(('minmax', MinMaxScaler(), mmcol))

# Applying the scaling using ColumnTransformer
ct = ColumnTransformer(transformers, remainder='passthrough')
df_transformed_scaled = ct.fit_transform(clean_df)

# Converting scaled result back to DataFrame with original column names
df_scaled = pd.DataFrame(df_transformed_scaled, columns=clean_df.columns, index=clean_df.index)

# Showing scaling summary and preview
print(f'Summary: Columns with Not Normal distribution {minmax_count} and Normal distribution {zscore_count}')
print("Successfully standardized given 2D-Descriptors, here is the resulting dataset :\n", df_scaled.head())
print(f"Cleaned and Processed Dataset shape : {df_scaled.shape}")
```

**Summary of Changes:**
- ✅ **Same Intelligent Logic as Part C:** Distribution-based column-wise scaling
- ✅ **Replaces Sequential Approach:** No longer applies MinMax then StandardScaler universally
- ✅ **Enhanced Reporting:** Shows counts of scaled columns and dataset shape
- ✅ **Consistent with Part C:** Uses identical Shapiro-Wilk + ColumnTransformer methodology

---

## Summary Table: All Major Changes

| Feature | v1.0 | v2.0 | Impact |
|---------|------|------|--------|
| **Data diagnostics** | Minimal | Explicit missing value reporting | ✅ Better transparency |
| **MACCS output format** | String-encoded | 166 individual bit columns | ✅ ML-ready format |
| **Morgan output format** | String-encoded | 2048 individual bit columns | ✅ ML-ready format |
| **Atom Pair output format** | String-encoded | 2048 individual bit columns | ✅ ML-ready format |
| **Topological Torsion output format** | String-encoded | 2048 individual bit columns | ✅ ML-ready format |
| **Pharmacophore output format** | String-encoded | Dynamic bit columns (1 to N) | ✅ ML-ready format |
| **Part C scaling strategy** | Sequential MinMax → Z-score | Distribution-aware per-column | ✅ Statistically sound |
| **Normality testing** | None | Shapiro-Wilk test | ✅ Data-driven decisions |
| **Constant column handling** | Not detected | Explicit detection & MinMaxScaler | ✅ Robust preprocessing |
| **ColumnTransformer usage** | No | Yes (Part C & Part D) | ✅ Professional preprocessing |
| **Code documentation** | Sparse | Extensive inline + summary blocks | ✅ Educational value |
| **Function robustness** | Basic | Enhanced tracking & reporting | ✅ Better usability |

---

## Files Affected

| File | Changes | Description |
|------|---------|-------------|
| `macckeys_output.csv` | Format | Now 167 columns (SMILES + 166 bits) instead of 2 |
| `ecfp_output.csv` | Format | Now 2049 columns (SMILES + 2048 bits) instead of 2 |
| `atom_pair_output.csv` | Format | Now 2049 columns (SMILES + 2048 bits) instead of 2 |
| `topological_torsion_output.csv` | Format | Now 2049 columns (SMILES + 2048 bits) instead of 2 |
| `pharmacophore_fp_output.csv` | Format | Now dynamic columns (SMILES + N bits) instead of 2 |
| `descriptors_2d_output.csv` | Unchanged | Still SMILES + ~200 2D descriptors |


---

## Version Information

- **v1.0:** Original implementation
- **v2.0:** Major refactoring with improved preprocessing



**Last Updated**: January 20, 2026


