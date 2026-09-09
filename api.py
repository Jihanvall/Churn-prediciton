import pickle
import os
import sys

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.join(os.getcwd(), 'src'))
from feature_engineering import build_advanced_features

app = FastAPI(title="Churn Prediction API")

with open(os.path.join('models', 'final_xgboost_model.pkl'), 'rb') as f:
    model = pickle.load(f)
with open(os.path.join('models', 'fitted_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)


@app.get("/")
def health_check():
    return {"status": "ok"}

class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def predict(customer: CustomerData):
    input_dict = customer.model_dump()
    input_df = pd.DataFrame([input_dict])

    input_df = build_advanced_features(input_df)

    is_alone = input_df["Is_Alone"].iloc[0]
    total_services = input_df["TotalServices"].iloc[0]
    cost_per_service = (
        input_dict["MonthlyCharges"] / total_services if total_services > 0 else 0
    )

    encoded = {
        "tenure": input_dict["tenure"],
        "MonthlyCharges": input_dict["MonthlyCharges"],
        "TotalCharges": input_dict["TotalCharges"],
        "gender": 1 if input_dict["gender"] == "Male" else 0,
        "Partner": 1 if input_dict["Partner"] == "Yes" else 0,
        "Dependents": 1 if input_dict["Dependents"] == "Yes" else 0,
        "PhoneService": 1 if input_dict["PhoneService"] == "Yes" else 0,
        "MultipleLines": 1 if input_dict["MultipleLines"] == "Yes" else 0,
        "OnlineSecurity": 1 if input_dict["OnlineSecurity"] == "Yes" else 0,
        "OnlineBackup": 1 if input_dict["OnlineBackup"] == "Yes" else 0,
        "DeviceProtection": 1 if input_dict["DeviceProtection"] == "Yes" else 0,
        "TechSupport": 1 if input_dict["TechSupport"] == "Yes" else 0,
        "StreamingTV": 1 if input_dict["StreamingTV"] == "Yes" else 0,
        "StreamingMovies": 1 if input_dict["StreamingMovies"] == "Yes" else 0,
        "PaperlessBilling": 1 if input_dict["PaperlessBilling"] == "Yes" else 0,
        "InternetService_Fiber optic": 1 if input_dict["InternetService"] == "Fiber optic" else 0,
        "InternetService_No": 1 if input_dict["InternetService"] == "No" else 0,
        "Contract_One year": 1 if input_dict["Contract"] == "One year" else 0,
        "Contract_Two year": 1 if input_dict["Contract"] == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if input_dict["PaymentMethod"] == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if input_dict["PaymentMethod"] == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if input_dict["PaymentMethod"] == "Mailed check" else 0,
        "Is_Alone": is_alone,
        "TotalServices": total_services,
        "HighCost_ShortContract": input_dict["MonthlyCharges"] if input_dict["Contract"] == "Month-to-month" else 0.0,
        "Cost_Per_Service": cost_per_service,
    }

    final_df = pd.DataFrame([encoded])
    expected_columns = model.get_booster().feature_names
    for col in expected_columns:
        if col not in final_df.columns:
            final_df[col] = 0
    final_df = final_df[expected_columns]

    numerical_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])

    probability = float(model.predict_proba(final_df)[0][1])
    prediction = int(probability > 0.5)

    return {
        "churn_prediction": prediction,
        "churn_probability": probability,
    }