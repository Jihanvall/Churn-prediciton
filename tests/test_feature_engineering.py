import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
from feature_engineering import build_advanced_features


def test_is_alone_true_when_no_partner_and_no_dependents():
    df = pd.DataFrame({"Partner": ["No"], "Dependents": ["No"]})
    result = build_advanced_features(df)
    assert result["Is_Alone"].iloc[0] == 1


def test_is_alone_false_when_has_partner():
    df = pd.DataFrame({"Partner": ["Yes"], "Dependents": ["No"]})
    result = build_advanced_features(df)
    assert result["Is_Alone"].iloc[0] == 0


def test_total_services_counts_yes_values():
    df = pd.DataFrame({
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["No"],
    })
    result = build_advanced_features(df)
    assert result["TotalServices"].iloc[0] == 3


def test_high_cost_short_contract_applies_only_for_month_to_month():
    df = pd.DataFrame({
        "Contract": ["Month-to-month"],
        "MonthlyCharges": [100.0],
    })
    result = build_advanced_features(df)
    assert result["HighCost_ShortContract"].iloc[0] == 100.0


def test_high_cost_short_contract_zero_for_long_contract():
    df = pd.DataFrame({
        "Contract": ["Two year"],
        "MonthlyCharges": [100.0],
    })
    result = build_advanced_features(df)
    assert result["HighCost_ShortContract"].iloc[0] == 0.0


def test_no_internet_service_replaced_with_no():
    df = pd.DataFrame({
        "Partner": ["No"],
        "Dependents": ["No"],
        "OnlineSecurity": ["No internet service"],
        "OnlineBackup": ["No internet service"],
        "DeviceProtection": ["No internet service"],
        "TechSupport": ["No internet service"],
        "StreamingTV": ["No internet service"],
        "StreamingMovies": ["No internet service"],
    })
    result = build_advanced_features(df)
    assert result["TotalServices"].iloc[0] == 0