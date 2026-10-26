# 📊 Telco Customer Churn Prediction Project

## 1. Project Overview & Business Objective
This repository implements an end-to-end Machine Learning pipeline to predict customer churn for a telecommunications provider based on the standard **IBM Telco Customer Churn dataset**.

- **Goal**: Identify high-risk churn customers early and uncover key business drivers influencing customer attrition to enable targeted retention strategies.
- **Dataset Size**: 7,043 customer records, 21 initial attributes.
- **Target Variable**: `Churn` (Binary: `"Yes"` / `"No"` -> encoded as `1` / `0`).
- **Target Class Distribution**: ~26.5% Churn (`1,869`), ~73.5% Non-Churn (`5,174`) — moderately imbalanced classification problem.

---

## 2. Repository Structure

```
churn-project/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 rows x 21 cols)
│   ├── telco_churn_cleaned.csv                # Primary cleaned & feature-engineered dataset
│   ├── clean.csv                              # Cleaned data copy
│   └── churn.db                               # SQLite database export
├── src/
│   └── clean.py                               # Modular Python cleaning & feature engineering pipeline
├── scripts/
│   └── build_notebook.py                      # Automated script generating & running Jupyter notebooks
├── notebooks/
│   ├── 01_cleaning.ipynb                      # [COMPLETED] Data cleaning, validation & initial checks
│   ├── 02_eda.ipynb                           # [PENDING] In-depth Exploratory Data Analysis (EDA)
│   └── 03_model.ipynb                         # [PENDING] Model training, evaluation & feature importance
└── README.md                                  # Project overview and context handoff
```

---

## 3. Data Cleaning & Preprocessing Summary (`src/clean.py` & `01_cleaning.ipynb`)

1. **`TotalCharges` Formatting & Missing Value Imputation**:
   - The raw `TotalCharges` column was stored as an `object` (string) with 11 blank whitespace values (`' '`).
   - Investigation proved that all 11 records had `tenure == 0` (brand new accounts who have not yet received their first month's invoice).
   - Solution: Converted to `float64` and filled these 11 rows with `0.0`. No rows dropped.

2. **Categorical Normalization**:
   - Six internet add-on columns (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) had redundant values: `"No internet service"` was normalized to `"No"`.
   - `MultipleLines` had `"No phone service"` normalized to `"No"`.
   - Simplifies one-hot encoding and tree-based decision splits.

3. **Target Encoding**:
   - Created `Churn_Numeric`: mapped `'Yes'` -> `1`, `'No'` -> `0`.

4. **Engineered Domain Features**:
   - **`TenureGroup`**: Cohort bins (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `60m+`).
   - **`TotalAddonServices`**: Integer count (0 to 6) representing the total number of subscribed add-on protection/streaming services.
   - **`HasFamily`**: Binary flag (`1` / `0`) combining `Partner == 'Yes'` or `Dependents == 'Yes'`.

5. **Cleaned Dataset Output**:
   - Exported to `data/telco_churn_cleaned.csv` with dimensions: **7,043 rows × 24 columns** (all original rows retained, schema validated, zero remaining nulls).

---

## 4. Key Initial Findings
- **Tenure Cohort Effect**: Customers in their first 12 months (`0-12m`) have the highest churn rate (~47.4%). Churn drops dramatically to ~6.6% for customers staying over 60 months.
- **Product Stickiness**: Customers with 0 add-on services churn at ~43%, whereas customers utilizing 4+ services have churn rates below 15%.
- **Contract Type & Payment**: Month-to-month contracts and paperless/electronic check billing are prime indicators of elevated churn risk.

---

## 5. Current Project Status & Roadmap

| Phase | Component | Status | Description |
|---|---|---|---|
| **Phase 1** | Data Cleaning & Preprocessing | ✅ **Completed** | Clean script (`src/clean.py`), notebook (`01_cleaning.ipynb`), clean CSV generated. |
| **Phase 2** | Exploratory Data Analysis | ⏳ **Pending** | `notebooks/02_eda.ipynb`: Bivariate/multivariate analysis, correlation heatmap, demographic breakdowns. |
| **Phase 3** | Predictive ML Modeling | ⏳ **Pending** | `notebooks/03_model.ipynb`: Baseline & advanced models (Logistic Regression, Random Forest, XGBoost/LightGBM), SMOTE/class weighting, ROC-AUC / PR-AUC evaluation, and SHAP/feature importance. |

---

## 6. How to Run Existing Code

### Run the Cleaning Script:
```powershell
python src/clean.py
```

### Rebuild and Execute Cleaning Notebook:
```powershell
python scripts/build_notebook.py
```
