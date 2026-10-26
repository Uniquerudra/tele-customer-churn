# 📊 Telco Customer Churn Analysis & Prediction

An end-to-end churn project on the IBM Telco Customer Churn dataset: a Python data-cleaning pipeline, a Power BI dashboard for business insights, and a planned machine learning model to predict which customers are likely to leave.

![Power BI Dashboard](powerbi/dashboard.png)

---

## 1. Project Overview & Business Objective

- **Goal:** Identify customers at high risk of churning early, and uncover the main drivers of attrition so retention efforts can be targeted.
- **Dataset:** IBM Telco Customer Churn, 7,043 customer records, 21 original columns.
- **Target variable:** `Churn` (`"Yes"` / `"No"`, encoded as `1` / `0`).
- **Class distribution:** about 26.5% churn (1,869) and 73.5% non-churn (5,174), a moderately imbalanced classification problem.

---

## 2. Tech Stack

Python (pandas, Jupyter), SQLite, Power BI (DAX, Power Query), scikit-learn, XGBoost, imbalanced-learn, SHAP.

---

## 3. Repository Structure

```
telco-customer-churn/
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (7,043 rows x 21 cols)
│   ├── telco_churn_cleaned.csv                # Cleaned & feature-engineered dataset (7,043 x 24)
│   ├── clean.csv                              # Copy of the cleaned data
│   └── churn.db                               # SQLite database export
├── src/
│   └── clean.py                               # Cleaning & feature engineering pipeline
├── scripts/
│   └── build_notebook.py                      # Builds and executes the Jupyter notebooks
├── notebooks/
│   ├── 01_cleaning.ipynb                      # [DONE] Cleaning, validation, initial checks
│   ├── 02_eda.ipynb                           # [DONE] Exploratory data analysis
│   └── 03_model.ipynb                         # [DONE] Modeling & evaluation
├── powerbi/
│   ├── Telco_Churn_Dashboard.pbix             # [DONE] Interactive dashboard
│   └── dashboard.png                          # Dashboard screenshot
└── README.md
```

---

## 4. Data Cleaning & Feature Engineering (`src/clean.py`, `01_cleaning.ipynb`)

**TotalCharges fix**
- The raw column was stored as a string and had 11 blank values (`' '`).
- All 11 rows have `tenure == 0` (new accounts that have not been billed yet).
- Converted to `float64` and filled these rows with `0.0`. No rows were dropped.

**Categorical normalization**
- For `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV` and `StreamingMovies`, `"No internet service"` was mapped to `"No"`.
- For `MultipleLines`, `"No phone service"` was mapped to `"No"`.
- This simplifies one-hot encoding and tree-based splits.

**Target encoding**
- `Churn_Numeric`: `Yes` → 1, `No` → 0.

**Engineered features**
- `TenureGroup`: cohort bins (`0-12m`, `13-24m`, `25-48m`, `49-60m`, `60m+`).
- `TotalAddonServices`: count (0 to 6) of subscribed add-on services.
- `HasFamily`: 1 if `Partner == 'Yes'` or `Dependents == 'Yes'`, else 0.

**Output:** `data/telco_churn_cleaned.csv` with 7,043 rows × 24 columns, all original rows kept and zero nulls.

---

## 5. Power BI Dashboard

An interactive dashboard built on the Telco data.

**KPI cards:** Churn Rate, Total Customers, Revenue at Risk.

**Visuals:** Churn Rate by Contract, Churn Rate by Tenure Group, Churned Customers by Payment Method, and a Stayed vs Churned donut chart.

**Slicers:** gender, InternetService, SeniorCitizen. Every card and chart updates when a slicer is used.

**DAX measures**
- `Churn Rate`: share of customers who churned.
- `Total Customers`: count of customers.
- `Churned Customers`: count of customers with Churn Label = Churned.
- `Revenue at Risk`: sum of `MonthlyCharges` for churned customers.

> **Note:** The Power BI model has 7,032 customers because the 11 rows with blank `TotalCharges` were excluded there. The Python pipeline keeps all 7,043 rows (filled with 0.0). The churn rate is about 26.5% in both. Tenure groups also differ: the dashboard uses 3 groups (0-12, 13-36, 37+ months), the Python pipeline uses 5 bins.

---

## 6. Key Findings

| Finding | Evidence |
|---|---|
| Overall churn is about 26.6%, with about **$139K monthly revenue at risk** | Dashboard KPI cards |
| **Month-to-month** contracts churn the most; two-year contracts churn the least | Churn Rate by Contract |
| **New customers (0-12 months)** churn the most, about 47%. Churn falls to about 6.6% after 60+ months | Tenure cohorts |
| **Electronic check** users make up the largest group of churned customers | Churned Customers by Payment Method |
| **Fiber optic** customers churn at about 41.9%, well above average | InternetService slicer |
| Customers with **0 add-on services** churn at about 43%; with 4+ add-ons, under 15% | `TotalAddonServices` |

**Suggested actions:** push new customers toward longer contracts, bundle add-on services to increase stickiness, and encourage a move from electronic check to automatic payments.

---

## 7. Project Status & Roadmap

| Phase | Component | Status | Details |
|---|---|---|---|
| 1 | Data cleaning & feature engineering | ✅ Done | `src/clean.py`, `01_cleaning.ipynb`, cleaned CSV |
| 2 | Business dashboard (Power BI) | ✅ Done | KPIs, 4 charts, 3 slicers, DAX measures |
| 3 | Exploratory analysis in Python | ✅ Done | `02_eda.ipynb`: churn vs Contract, PaymentMethod, InternetService and TenureGroup; correlation heatmap; MonthlyCharges and add-on analysis. Goal: confirm the dashboard insights in code |
| 4 | Predictive modeling | ✅ Done | `03_model.ipynb`: stratified train/test split, encoding and scaling, class weights or SMOTE, Logistic Regression (baseline), Random Forest, XGBoost; evaluate with ROC-AUC, PR-AUC and **recall** on the churn class; feature importance with SHAP |

Recall on the churn class is the priority metric, since missing a customer who is about to leave costs more than a false alarm.

---

## 8. How to Run

```bash
# Install dependencies
pip install pandas jupyter

# Run the cleaning pipeline
python src/clean.py

# Rebuild and execute the cleaning notebook
python scripts/build_notebook.py
```

To explore the dashboard, open `powerbi/Telco_Churn_Dashboard.pbix` in Power BI Desktop.

---

## 9. Author

**Rudra Pratap Pal**
