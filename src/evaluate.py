import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import clean_and_encode, train_test_split_encoded

def load_and_evaluate(raw_data_path, model_path, scaler_path):
    """
    Loads the frozen model and scaler, prepares the test set, 
    and outputs the final evaluation metrics.
    """
    print("1. Loading raw data and preparing the exact Test Set...")
    df = pd.read_csv(raw_data_path)
    df_encoded = clean_and_encode(df)
    # Reuse the shared split helper so this can never drift out of sync
    # with the split used during training
    _, X_test_raw, _, y_test = train_test_split_encoded(df_encoded)
    print("2. Loading the saved Scaler and Model artifacts...")
    with open(scaler_path, 'rb') as scaler_file:
        saved_scaler = pickle.load(scaler_file)
        
    with open(model_path, 'rb') as model_file:
        saved_model = pickle.load(model_file)
        
    print("3. Applying the saved mathematical translation (Scaling) to the test set...")
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # CRITICAL: We strictly use .transform() here with the saved scaler!
    X_test_scaled = X_test_raw.copy()
    X_test_scaled[numerical_cols] = saved_scaler.transform(X_test_raw[numerical_cols])
    
    print("4. Executing Predictions...\n")
    y_pred = saved_model.predict(X_test_scaled)
    
    # Output the final business metrics
    print("="*50)
    print(f"Final Production Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("="*50)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    # Define paths
    DATA_PATH = '../data/raw/customer_churn_raw.csv'
    MODEL_PATH = '../models/final_xgboost_model.pkl'
    SCALER_PATH = '../models/fitted_scaler.pkl'
    
    load_and_evaluate(DATA_PATH, MODEL_PATH, SCALER_PATH)