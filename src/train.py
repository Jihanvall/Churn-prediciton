import os
import pickle
import pandas as pd
from xgboost import XGBClassifier

# Import the preprocessing pipeline
from data_preprocessing import clean_and_encode, split_scale_and_balance

def train_and_save_model(raw_data_path):
    """
    Executes the full training pipeline and serializes the model artifacts.
    """
    # 1. Load raw data
    print("Loading raw dataset...")
    df = pd.read_csv(raw_data_path)
    
    # 2. Execute Preprocessing and Balancing Pipeline
    print("Applying feature engineering, encoding, scaling, and SMOTENC...")
    df_encoded = clean_and_encode(df)
    X_train_resampled, X_test, y_train_resampled, y_test, scaler = split_scale_and_balance(df_encoded)
    
    # 3. Initialize and Train the Winning XGBoost Model
    # Note: scale_pos_weight is removed because SMOTENC already balanced the classes perfectly
    print("Training the XGBoost model...")
    xgb_final = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=(1033 / 374), # Intentionally kept for maximum Class 1 sensitivity
        random_state=42,
        eval_metric='logloss'
    )
    
    xgb_final.fit(X_train_resampled, y_train_resampled)
    
    # 4. Serialize and Save Artifacts
    print("Saving model and scaler artifacts...")
    os.makedirs('../models', exist_ok=True)
    
    model_filename = '../models/final_xgboost_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(xgb_final, model_file)
        
    scaler_filename = '../models/fitted_scaler.pkl'
    with open(scaler_filename, 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
        
    print("Success: Training complete and artifacts saved in the 'models' directory.")

if __name__ == "__main__":
    # Define the path to the raw dataset
    # Adjust this path if your data folder structure is different
    DATA_PATH = '../data/raw/customer_churn_raw.csv'
    train_and_save_model(DATA_PATH)