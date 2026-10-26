import os
import pandas as pd
import numpy as np

def clean_telco_churn_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Cleans and preprocesses the Telco Customer Churn dataset.
    
    Key cleaning steps:
    1. Trims whitespace and converts 'TotalCharges' to numeric float.
       Fills the 11 blank entries (tenure == 0) with 0.0.
    2. Maps binary target 'Churn' ('Yes'/'No') to (1/0).
    3. Standardizes redundant categorical levels ('No internet service' / 'No phone service' -> 'No').
    4. Adds engineered features:
       - 'TenureGroup': binned tenure cohorts (0-12, 13-24, 25-48, 49-60, 60+ months)
       - 'NumServices': count of optional add-on services
       - 'HasPartnerOrDependents': binary flag for family account
    5. Saves cleaned dataset to output_path if provided.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at: {input_path}")
        
    print(f"[1/5] Loading raw data from {input_path}...")
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    print(f"      Initial shape: {initial_shape[0]} rows, {initial_shape[1]} columns")
    
    # 1. TotalCharges cleaning
    print("[2/5] Cleaning 'TotalCharges' column...")
    blank_mask = df['TotalCharges'].astype(str).str.strip() == ''
    blank_count = blank_mask.sum()
    print(f"      Found {blank_count} rows with blank/whitespace TotalCharges (all have tenure == 0).")
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
    print("      'TotalCharges' converted to float64 and nulls filled with 0.0.")
    
    # 2. SeniorCitizen as categorical string or maintain int
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)
    
    # 3. Standardize redundant categorical values
    print("[3/5] Standardizing service category representations...")
    service_cols = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        df[col] = df[col].replace({'No internet service': 'No'})
        
    df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})
    
    # 4. Target variable encoding
    print("[4/5] Encoding target variable 'Churn'...")
    df['Churn_Numeric'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)
    
    # 5. Feature Engineering
    print("[5/5] Generating engineered features...")
    # Count of additional services subscribed
    df['TotalAddonServices'] = df[service_cols].apply(lambda row: sum(val == 'Yes' for val in row), axis=1)
    
    # Tenure Cohorts
    tenure_bins = [-1, 12, 24, 48, 60, 100]
    tenure_labels = ['0-12m', '13-24m', '25-48m', '49-60m', '60m+']
    df['TenureGroup'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels)
    
    # Family flag
    df['HasFamily'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
    
    # Validation
    assert df['TotalCharges'].isna().sum() == 0, "Nulls remain in TotalCharges!"
    assert df.shape[0] == initial_shape[0], "Row count changed unexpectedly!"
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[Done] Cleaned dataset saved to: {output_path}")
        print(f"       Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
    return df

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    output_file = os.path.join(base_dir, "data", "telco_churn_cleaned.csv")
    
    clean_telco_churn_data(input_file, output_file)
