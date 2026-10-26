import nbformat as nbf
from nbclient import NotebookClient
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 📊 Telco Customer Churn - Data Cleaning & Preprocessing
**Author**: Data Science Team  
**Objective**: Prepare the raw Telco customer churn dataset for exploratory data analysis (EDA) and predictive modeling.

---

### Pipeline Steps:
1. **Data Ingestion & Initial Inspection**: Understand shapes, schemas, and missing values.
2. **Handle Anomalies in `TotalCharges`**: Resolve whitespace formatting for new customers (`tenure == 0`).
3. **Categorical Standardizations**: Rationalize redundant categories across phone and internet service addons.
4. **Target Encoding**: Encode binary target `Churn` for downstream ML pipelines.
5. **Feature Engineering**: Derive tenure cohorts, addon service counts, and family flags.
6. **Data Validation & Export**: Validate schema integrity and save the cleaned dataset.
"""))

# Cell 1: Imports & Setup
cells.append(nbf.v4.new_code_cell("""import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Styling
sns.set_theme(style="whitegrid", palette="muted")
pd.set_option('display.max_columns', None)
print("Libraries imported successfully!")
"""))

# Cell 2: Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Load Raw Dataset"""))
cells.append(nbf.v4.new_code_cell("""# Define path to raw data
data_path = os.path.join("..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not os.path.exists(data_path):
    data_path = os.path.join("data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(data_path)
print(f"Dataset Dimensions: {df.shape[0]:,} rows, {df.shape[1]} columns")
df.head(5)
"""))

# Cell 3: Initial Data Info & Schema
cells.append(nbf.v4.new_markdown_cell("""## 2. Inspect Column Types and Missing Values"""))
cells.append(nbf.v4.new_code_cell("""df.info()
"""))

# Cell 4: Duplicate & Null Check
cells.append(nbf.v4.new_code_cell("""# Check duplicate customer records
dup_count = df['customerID'].duplicated().sum()
print(f"Duplicate customerID count: {dup_count}")

# Check standard null values
null_summary = df.isna().sum()
print("Standard missing values (NaN) count:")
print(null_summary[null_summary > 0] if (null_summary > 0).any() else "No NaN values found.")
"""))

# Cell 5: Investigating TotalCharges
cells.append(nbf.v4.new_markdown_cell("""## 3. Resolve the `TotalCharges` Data Quality Issue
Notice that `TotalCharges` is stored as `object` (string) despite containing monetary values. Let's inspect rows with non-numeric or whitespace characters.
"""))
cells.append(nbf.v4.new_code_cell("""# Detect blank whitespace strings in TotalCharges
whitespace_mask = df['TotalCharges'].str.strip() == ''
blank_rows = df[whitespace_mask]

print(f"Number of rows with blank 'TotalCharges': {len(blank_rows)}")
blank_rows[['customerID', 'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Churn']]
"""))

cells.append(nbf.v4.new_markdown_cell("""**Observation**: All 11 records with blank `TotalCharges` have `tenure == 0`. These represent newly onboarded customers who have not yet received their first month's invoice.

**Solution**: Convert `TotalCharges` to `float64` and set these 11 rows to `0.0`.
"""))
cells.append(nbf.v4.new_code_cell("""# Convert TotalCharges to float, setting whitespace to NaN then filling with 0.0
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

print(f"TotalCharges data type: {df['TotalCharges'].dtype}")
print(f"Remaining nulls in TotalCharges: {df['TotalCharges'].isna().sum()}")
"""))

# Cell 6: Categorical Standardization
cells.append(nbf.v4.new_markdown_cell("""## 4. Standardize Categorical Variables
Several columns have redundant levels (e.g., `'No internet service'` for security/backup features, or `'No phone service'` for multiple lines). Standardizing these simplifies downstream one-hot encoding and tree-based splits.
"""))
cells.append(nbf.v4.new_code_cell("""# Add-on internet services
addon_cols = [
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
    'TechSupport', 'StreamingTV', 'StreamingMovies'
]

print("Unique values before standardization:")
for col in addon_cols:
    print(f"  {col}: {df[col].unique().tolist()}")

# Standardize 'No internet service' to 'No'
for col in addon_cols:
    df[col] = df[col].replace({'No internet service': 'No'})

# Standardize 'No phone service' to 'No' in MultipleLines
df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

print("\\nUnique values after standardization:")
for col in addon_cols + ['MultipleLines']:
    print(f"  {col}: {df[col].unique().tolist()}")
"""))

# Cell 7: Encode Target Variable
cells.append(nbf.v4.new_markdown_cell("""## 5. Target Variable (`Churn`) Analysis & Encoding"""))
cells.append(nbf.v4.new_code_cell("""# Class distribution
churn_counts = df['Churn'].value_counts()
churn_pct = df['Churn'].value_counts(normalize=True) * 100

print(f"Churn Counts:\\n{churn_counts}\\n")
print(f"Churn Percentages:\\n{churn_pct.round(2)}")

# Create numerical target variable
df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

# Quick visual of target distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x='Churn', hue='Churn', palette=['#3b82f6', '#ef4444'], legend=False, ax=ax)
ax.set_title("Customer Churn Distribution", fontsize=14, fontweight='bold')
ax.set_xlabel("Churn Status")
ax.set_ylabel("Count")
for p in ax.patches:
    ax.annotate(f'{p.get_height():,} ({p.get_height()/len(df)*100:.1f}%)', 
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, xytext=(0, 4), textcoords='offset points')
plt.tight_layout()
plt.show()
"""))

# Cell 8: Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 6. Feature Engineering
We create three high-value business features:
1. **`TotalAddonServices`**: Number of optional service add-ons active (0 to 6).
2. **`TenureGroup`**: Cohort bins (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `60m+`).
3. **`HasFamily`**: Binary flag indicating whether the customer has a partner or dependents.
"""))
cells.append(nbf.v4.new_code_cell("""# 1. Number of add-on services
df['TotalAddonServices'] = df[addon_cols].apply(lambda row: sum(val == 'Yes' for val in row), axis=1)

# 2. Tenure cohorts
tenure_bins = [-1, 12, 24, 48, 60, 100]
tenure_labels = ['0-12m', '13-24m', '25-48m', '49-60m', '60m+']
df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels)

# 3. Family account flag
df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)

# Preview engineered features
df[['tenure', 'TenureGroup', 'TotalAddonServices', 'HasFamily', 'MonthlyCharges', 'TotalCharges', 'Churn']].head(8)
"""))

# Cell 9: Churn by Tenure Cohort Visual
cells.append(nbf.v4.new_code_cell("""fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Churn Rate by Tenure Group
tenure_churn = df.groupby('TenureGroup', observed=False)['Churn_Numeric'].mean().reset_index()
sns.barplot(data=tenure_churn, x='TenureGroup', y='Churn_Numeric', color='#f97316', ax=ax1)
ax1.set_title("Churn Rate by Tenure Cohort", fontsize=13, fontweight='bold')
ax1.set_ylabel("Churn Rate")
ax1.set_ylim(0, 0.6)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

# Churn Rate by Total Add-on Services
addon_churn = df.groupby('TotalAddonServices')['Churn_Numeric'].mean().reset_index()
sns.barplot(data=addon_churn, x='TotalAddonServices', y='Churn_Numeric', color='#06b6d4', ax=ax2)
ax2.set_title("Churn Rate by Number of Add-on Services", fontsize=13, fontweight='bold')
ax2.set_ylabel("Churn Rate")
ax2.set_ylim(0, 0.6)
for p in ax2.patches:
    ax2.annotate(f'{p.get_height()*100:.1f}%', (p.get_x() + p.get_width()/2., p.get_height()),
                 ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')

plt.tight_layout()
plt.show()
"""))

# Cell 10: Export Clean Data
cells.append(nbf.v4.new_markdown_cell("""## 7. Export Cleaned Dataset"""))
cells.append(nbf.v4.new_code_cell("""output_path = os.path.join("..", "data", "telco_churn_cleaned.csv")
if not os.path.exists(os.path.dirname(output_path)):
    output_path = os.path.join("data", "telco_churn_cleaned.csv")

df.to_csv(output_path, index=False)
print(f"Cleaned dataset successfully exported to: {os.path.abspath(output_path)}")
print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
"""))

nb.cells = cells
nb.metadata.kernelspec = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3"
}

target_nb_path = os.path.join("notebooks", "01_cleaning.ipynb")
with open(target_nb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generated successfully at:", target_nb_path)

# Execute the notebook
try:
    print("Executing notebook to populate all outputs and charts...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    with open(target_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("Notebook executed and saved with all cell outputs!")
except Exception as e:
    print(f"Warning during execution: {e}")
