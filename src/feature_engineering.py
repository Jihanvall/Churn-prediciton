import numpy as np
import pandas as pd

def build_advanced_features(df):
    """
    Engineers advanced domain-specific features using row-by-row logic
    on the raw text dataset before any categorical encoding occurs.
    """
    # Create a copy to protect the original dataframe from in-place mutations
    df_enhanced = df.copy()
    
    # 1. Standardize text columns if needed (replacing specific sub-service 'No internet service' to 'No')
    internet_sub_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in internet_sub_services:
        if col in df_enhanced.columns:
            df_enhanced[col] = df_enhanced[col].replace('No internet service', 'No')
            
    if 'MultipleLines' in df_enhanced.columns:
        df_enhanced['MultipleLines'] = df_enhanced['MultipleLines'].replace('No phone service', 'No')

    # 2. Feature 1: Is_Alone
    # Check if the customer has NO partner AND NO dependents
    if 'Partner' in df_enhanced.columns and 'Dependents' in df_enhanced.columns:
        df_enhanced['Is_Alone'] = ((df_enhanced['Partner'] == 'No') & (df_enhanced['Dependents'] == 'No')).astype(int)

    # 3. Feature 2: TotalServices
    # Sum up the total active digital services for each customer
    if all(col in df_enhanced.columns for col in internet_sub_services):
        df_enhanced['TotalServices'] = (df_enhanced[internet_sub_services] == 'Yes').sum(axis=1)

    # 4. Feature 3: HighCost_ShortContract
    # Capture the financial risk: High monthly charges combined with a Month-to-month contract
    if 'Contract' in df_enhanced.columns and 'MonthlyCharges' in df_enhanced.columns:
        month_to_month_logic = (df_enhanced['Contract'] == 'Month-to-month').astype(int)
        df_enhanced['HighCost_ShortContract'] = df_enhanced['MonthlyCharges'] * month_to_month_logic

    return df_enhanced