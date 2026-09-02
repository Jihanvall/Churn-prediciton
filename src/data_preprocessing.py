import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTENC

# Import the feature engineering module
from feature_engineering import build_advanced_features

def clean_and_encode(df):
    """
    Applies feature engineering, encodes categorical variables,
    and handles basic data types.
    """
    # 1. Apply advanced features
    df = build_advanced_features(df)
    
    # 2. Handle missing TotalCharges
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])
    
    # 3. Binary Encoding
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'PaperlessBilling', 
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies'
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
        
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # 4. One-Hot Encoding
    multi_category_cols = ['InternetService', 'Contract', 'PaymentMethod']
    df_encoded = pd.get_dummies(df, columns=multi_category_cols, drop_first=True, dtype=int)
    
    # Drop irrelevant columns (e.g., customerID)
    if 'customerID' in df_encoded.columns:
        df_encoded = df_encoded.drop(columns=['customerID'])
        
    return df_encoded
def train_test_split_encoded(df_encoded):
    """
    Splits encoded data into train/test sets using the project's
    standard split parameters. Shared by split_scale_and_balance()
    and evaluate.py, so the two never drift out of sync.
    """
    X = df_encoded.drop(columns=['Churn'])
    y = df_encoded['Churn']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def split_scale_and_balance(df_encoded):

    X_train, X_test, y_train, y_test = train_test_split_encoded(df_encoded)
    
    # 2. Strict Scaling
    scaler = StandardScaler()
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # 3. Balance with SMOTENC
    categorical_features_indices = [
        idx for idx, col in enumerate(X_train.columns) 
        if X_train[col].nunique() <= 2
    ]
    
    smote_nc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
    X_train_resampled, y_train_resampled = smote_nc.fit_resample(X_train, y_train)
    
    return X_train_resampled, X_test, y_train_resampled, y_test, scaler