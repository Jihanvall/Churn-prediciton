# Churn Prediction Intelligence: Advanced Customer Attrition Diagnostic System

Churn Prediction Intelligence is an end-to-end Machine Learning solution designed to predict customer attrition and transform predictive insights into actionable business intelligence. Unlike traditional churn prediction systems that only identify whether a customer is likely to leave, this project combines predictive analytics, explainable AI, and root cause analysis to help organizations understand why customers are at risk.

The system provides both individual customer diagnostics and large-scale batch analysis through an interactive Streamlit dashboard.

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

A brief guide to getting the application up and running locally.

### 1. Verify Model Artifacts

Ensure that the trained model and scaler are available inside the `models/` directory:
- `models/final_xgboost_model.pkl`
- `models/fitted_scaler.pkl`

### 2. Launch the Dashboard

Run the following command in your terminal to start the Streamlit application:
```bash
streamlit run app.py
```

### 3. Usage Modes

- **Single Customer Prediction:** Navigate to the "Single Diagnostic" section, input the customer attributes, and execute the diagnostic to get real-time churn probability.
- **Batch Prediction:** Upload a CSV dataset (without the target `Churn` column) in the "Batch Prediction" section to generate bulk predictions and view the automated Root Cause Analysis.

## System Architecture

The project follows a highly modular software architecture divided into distinct layers:

### 1. Data Preprocessing & Feature Engineering
- **Data Preprocessing:** Handles missing values, data cleaning, and categorical encoding.
- **Feature Engineering:** Creates custom domain-specific indicators such as `Is_Alone` and `TotalServices`.
- **Handling Class Imbalance:** Employs advanced over-sampling techniques to balance training distributions.

### 2. Machine Learning Engine
- Powered by an optimized **XGBoost Classifier** with hyperparameter tuning.
- Pipeline integration ensures consistent feature scaling via a fitted `StandardScaler`.

### 3. Explainability & Dashboard Layers
- **Explainability Layer:** Extracts and visualizes global feature importances directly from the trained tree ensemble.
- **Dashboard Layer:** Built via Streamlit, providing high-contrast UI rendering using a custom Navy Blue to White linear gradient theme.

## Key Features

- **Predictive Analytics:** High-performance classification pipeline tailored for commercial customer data.
- **Explainable AI (XAI):** Built-in model transparency to display the top features influencing the engine's decisions.
- **Root Cause Analysis (RCA):** Automatically isolates high-risk customer groups to perform statistical normalization and identify core churn drivers.
- **Data Export:** Supports immediate down-stream actions by enabling users to download full batch prediction results as a CSV file.

## Technologies Used

- **Machine Learning:** XGBoost, Scikit-learn, Imbalanced-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Programming Language:** Python 3.9+

## Project Structure

```text
.
├── app.py                         # Main Streamlit Application UI
├── requirements.txt               # Project Dependency Affirmation
├── .gitignore                     # Git Exclusion Configuration
├── README.md                      # System Documentation
├── data/                          # Runtime Data Storage
│   └── raw/                       # Immutable Raw Data Ingestion
│       └── customer_churn_raw.csv # Baseline Customer Churn Dataset
├── models/                        # Serialized Model Objects
│   ├── final_xgboost_model.pkl    # Trained XGBoost Binary
│   └── fitted_scaler.pkl          # Serialized Feature Scaler
├── notebooks/                     # Exploratory Data Analysis
│   └── Customer.ipynb             # Research and Model Development Sandbox
└── src/                           # Source Code Modules
    ├── data_preprocessing.py      # Data Cleaning and Transformation Pipeline
    ├── evaluate.py                # Model Evaluation and Metrics Compute
    ├── feature_engineering.py     # Mathematical Extraction of Predictors
    └── train.py                   # Automated Model Training Pipeline
```

## Model Performance

The predictive power of the core engine is continuously monitored and evaluated against historical test partitions using standard classification metrics:

- **Accuracy & Precision:** Measures overall correctness and minimization of false positives.
- **Recall & F1-Score:** Ensures maximum capture of actual high-risk customers.

*Note: Visual evaluation graphs and precise operational thresholds are generated interactively within the dashboard layers during execution.*

## Future Improvements

- **SHAP Integration:** Implementing SHapley Additive Explanations for local, per-customer prediction breakdown.
- **CI/CD Automation:** Setting up automated model retraining and deployment pipelines upon data drift detection.
- **Cloud Infrastructure Deployment:** Migrating the interface to secure cloud services to handle enterprise-level request volumes.
