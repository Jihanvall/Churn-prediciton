import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from data_preprocessing import clean_and_encode


def test_drops_rows_with_invalid_total_charges():
    df = pd.DataFrame({
        "TotalCharges": ["100.5", " ", "200.0"],
        "gender": ["Male", "Female", "Male"],
        "Partner": ["Yes", "No", "No"],
        "Dependents": ["No", "No", "Yes"],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "No", "No"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "OnlineSecurity": ["Yes", "No", "No"],
        "OnlineBackup": ["No", "No", "Yes"],
        "DeviceProtection": ["No", "No", "No"],
        "TechSupport": ["No", "No", "No"],
        "StreamingTV": ["No", "No", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "InternetService": ["DSL", "DSL", "Fiber optic"],
        "Contract": ["Month-to-month", "Month-to-month", "Two year"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Credit card"],
        "Churn": ["No", "Yes", "No"],
    })
    result = clean_and_encode(df)
    assert len(result) == 2  # the row with " " should be dropped


def test_binary_columns_are_encoded_as_integers():
    df = pd.DataFrame({
        "TotalCharges": ["100.5"],
        "gender": ["Male"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "PaperlessBilling": ["Yes"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "InternetService": ["DSL"],
        "Contract": ["Month-to-month"],
        "PaymentMethod": ["Electronic check"],
        "Churn": ["Yes"],
    })
    result = clean_and_encode(df)
    assert result["Partner"].iloc[0] == 1
    assert result["Dependents"].iloc[0] == 0
    assert result["Churn"].iloc[0] == 1


def test_customer_id_column_is_dropped():
    df = pd.DataFrame({
        "customerID": ["1234-ABCD"],
        "TotalCharges": ["100.5"],
        "gender": ["Male"],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "PaperlessBilling": ["Yes"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "InternetService": ["DSL"],
        "Contract": ["Month-to-month"],
        "PaymentMethod": ["Electronic check"],
        "Churn": ["Yes"],
    })
    result = clean_and_encode(df)
    assert "customerID" not in result.columns