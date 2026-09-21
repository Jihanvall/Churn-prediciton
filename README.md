# Churn Prediction Intelligence: Advanced Customer Attrition Diagnostic System

Churn Prediction Intelligence is an end-to-end Machine Learning solution designed to predict customer attrition and transform predictive insights into actionable business intelligence. Unlike traditional churn prediction systems that only identify whether a customer is likely to leave, this project combines predictive analytics, explainable AI, and root cause analysis to help organizations understand why customers are at risk.

The system provides both individual customer diagnostics and large-scale batch analysis through a React interface backed by a FastAPI service.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)

## Installation

Follow these steps to set up the environment and install the necessary dependencies.

### 1. Clone the Repository

```bash
git clone https://github.com/Jihanvall/Churn-prediciton.git
cd Churn-prediciton
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Verify Model Artifacts

Ensure that the trained model and scaler are available inside the `models/` directory:

- `models/final_xgboost_model.pkl`
- `models/fitted_scaler.pkl`

### 2. Launch the Application

**Terminal 1 — start the backend:**
```bash
uvicorn api:app --reload
```

**Terminal 2 — start the frontend:**
```bash
cd frontend
npm install
npm run dev
```

Then open `http://localhost:5173` in your browser.

### 3. Usage Modes

- **Single Customer Prediction:** Enter the customer attributes in the "Single Customer" tab, and execute the diagnostic to get real-time churn probability along with a SHAP-based breakdown of the top factors driving that prediction.
- **Batch Prediction:** Upload a CSV dataset in the "Batch Prediction" tab to generate bulk predictions, view Root Cause Analysis charts, and see the top factors driving churn across all high-risk customers.

## System Architecture

The project follows a highly modular software architecture divided into distinct layers:

### 1. Data Preprocessing & Feature Engineering

- **Data Preprocessing:** Handles missing values, data cleaning, and categorical encoding.
- **Feature Engineering:** Creates custom domain-specific indicators such as `Is_Alone`, `TotalServices`, and `Cost_Per_Service`.
- **Handling Class Imbalance:** Employs advanced over-sampling techniques (SMOTENC) to balance training distributions.

### 2. Machine Learning Engine

- Powered by an optimized **XGBoost Classifier**.
- Pipeline integration ensures consistent feature scaling via a fitted `StandardScaler`.

### 3. Explainability & Frontend Layers

- **Explainability Layer:** Uses SHAP to break down each prediction into the specific factors driving it, both for individual customers and across the high-risk group as a whole.
- **Frontend Layer:** Built via React, providing a clean, light-themed UI with interactive charts and SHAP-based explanations.

## Key Features

- **Predictive Analytics:** FastAPI backend serving an optimized XGBoost model, for both single-customer and batch predictions.
- **Explainable AI (XAI):** Every prediction includes a SHAP-based breakdown of the top factors driving that specific customer's risk score, not just a global feature importance list.
- **Root Cause Analysis (RCA):** Batch predictions automatically surface contract type and internet service distributions among high-risk customers.
- **Modern Frontend:** A React interface with interactive charts for exploring batch results.

## Technologies Used

- **Machine Learning:** XGBoost, Scikit-learn, Imbalanced-learn
- **Explainability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Backend:** FastAPI
- **Frontend:** React, Recharts
- **Containerization:** Docker
- **Programming Language:** Python 3.10+

## Project Structure

```text
.
├── api.py                         # FastAPI backend (single + batch prediction, SHAP)
├── Dockerfile                     # Container definition
├── requirements.txt               # Project dependencies
├── MODEL_CARD.md                  # Model performance, limitations, intended use
├── .gitignore                     # Git exclusion configuration
├── README.md                      # System documentation
├── .github/workflows/             # CI (runs tests on every push)
├── frontend/                      # React frontend
│   └── src/
│       ├── App.jsx                # UI: single + batch prediction, charts, explanations
│       └── App.css                # Styling
├── data/
│   └── raw/
│       └── customer_churn_raw.csv # Baseline customer churn dataset
├── models/
│   ├── final_xgboost_model.pkl    # Trained XGBoost model
│   └── fitted_scaler.pkl          # Fitted feature scaler
├── notebooks/
│   └── Customer.ipynb             # Exploratory data analysis
├── tests/                         # Test suite
└── src/
    ├── data_preprocessing.py      # Data cleaning and transformation
    ├── evaluate.py                # Model evaluation
    ├── feature_engineering.py     # Feature extraction
    └── train.py                   # Model training pipeline
```

## Model Performance

The predictive power of the core engine is continuously evaluated against historical test partitions using standard classification metrics:

- **Accuracy & Precision:** Measures overall correctness and minimization of false positives.
- **Recall & F1-Score:** Ensures maximum capture of actual high-risk customers.

See `MODEL_CARD.md` for detailed performance figures and known limitations.

## Future Improvements

- **Data Validation:** Reject uploaded CSV files that don't match the expected schema, instead of silently filling missing columns with zeros.
- **Model Monitoring:** Track prediction accuracy over time in production and alert on data or performance drift.
- **Cloud Infrastructure Deployment:** Migrating the interface to secure cloud services to handle enterprise-level request volumes.