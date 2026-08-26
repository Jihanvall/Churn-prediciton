import streamlit as st
import pandas as pd
import pickle
import os
import sys
import matplotlib.pyplot as plt

# 1. Robust Path Configuration
current_directory = os.getcwd()
src_directory = os.path.join(current_directory, 'src')

if src_directory not in sys.path:
    sys.path.append(src_directory)

try:
    from data_preprocessing import clean_and_encode
except ImportError:
    st.error("Error: Could not import preprocessing module. Ensure 'src' folder exists and contains 'data_preprocessing.py'.")

# 2. Page Configuration & Theme Styling
st.set_page_config(page_title="Churn Prediction Intelligence", layout="wide")

# Custom CSS for Dark Blue Theme
st.markdown("""
    <style>
    /* Main Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom, #000080, #FFFFFF);
        color: #000000;
    }
    /* Headers and Text */
    h1, h2, h3, label, p, div {
        color: #F8F8FF !important;
    }
    /* Sidebar styling if needed */
    [data-testid="stSidebar"] {
        background-color: #00004B;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load Frozen Artifacts
@st.cache_resource
def load_artifacts():
    model_path = os.path.join('models', 'final_xgboost_model.pkl')
    scaler_path = os.path.join('models', 'fitted_scaler.pkl')

    if not os.path.exists(model_path):
        st.error(f"Model file not found at '{model_path}'. Please ensure the model artifact is present before running the app.")
        st.stop()

    if not os.path.exists(scaler_path):
        st.error(f"Scaler file not found at '{scaler_path}'. Please ensure the scaler artifact is present before running the app.")
        st.stop()

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

    return model, scaler

xgb_model, scaler = load_artifacts()
# 4. Main Header
st.title("Customer Attrition Diagnostic Panel")

# 5. Create Tabs
tab_single, tab_batch = st.tabs(["Single Customer Diagnostic", "Batch Prediction (File Upload)"])

# --- TAB 1: Single Customer ---
with tab_single:
    col_demo, col_services, col_financial = st.columns(3)

    with col_demo:
        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents?", ["No", "Yes"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

    with col_services:
        st.subheader("Active Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])

    with col_financial:
        st.subheader("Contract & Billing")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)

    if st.button("Execute Single Diagnostic", type="primary"):
        is_alone = 1 if (partner == "No" and dependents == "No") else 0
        
        if internet_service == "No":
            total_services = 0
        else:
            services_list = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
            total_services = sum(1 for s in services_list if s == "Yes")
            
        high_cost_short_contract = monthly_charges if contract == "Month-to-month" else 0.0

        input_dict = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'gender': 1 if gender == "Male" else 0,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'PhoneService': 1 if phone_service == "Yes" else 0,
            'MultipleLines': 1 if multiple_lines == "Yes" else 0,
            'OnlineSecurity': 1 if online_security == "Yes" and internet_service != "No" else 0,
            'OnlineBackup': 1 if online_backup == "Yes" and internet_service != "No" else 0,
            'DeviceProtection': 1 if device_protection == "Yes" and internet_service != "No" else 0,
            'TechSupport': 1 if tech_support == "Yes" and internet_service != "No" else 0,
            'StreamingTV': 1 if streaming_tv == "Yes" and internet_service != "No" else 0,
            'StreamingMovies': 1 if streaming_movies == "Yes" and internet_service != "No" else 0,
            'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
            'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
            'InternetService_No': 1 if internet_service == "No" else 0,
            'Contract_One year': 1 if contract == "One year" else 0,
            'Contract_Two year': 1 if contract == "Two year" else 0,
            'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
            'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
            'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
            'Is_Alone': is_alone,
            'TotalServices': total_services,
            'HighCost_ShortContract': high_cost_short_contract
        }
        
        input_df = pd.DataFrame([input_dict])
        
        expected_columns = xgb_model.get_booster().feature_names
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[expected_columns]
        
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
        
        probability = xgb_model.predict_proba(input_df)[0][1]
        prediction = int(probability > 0.5)
        
        st.markdown("---")
        st.subheader("Diagnostic Results")
        if prediction == 1:
            st.error("HIGH RISK: Customer is likely to churn.")
        else:
            st.success("STABLE: Customer is likely to stay.")
        st.write(f"Calculated Attrition Probability: {probability * 100:.2f}%")
        st.progress(float(probability))

# --- TAB 2: Batch Prediction ---
with tab_batch:
    st.subheader("Customer Diagnostic & Root Cause Analysis")
    uploaded_file = st.file_uploader("Upload Customer Data (CSV Format)", type="csv")
    
    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        st.write("Raw Data Preview:", raw_df.head())
        
        
        if st.button("Execute  Diagnostic & Analyze Root Causes", type="primary"):
                try:
                    processed_df = clean_and_encode(raw_df)
                    
                    expected_columns = xgb_model.get_booster().feature_names
                    for col in expected_columns:
                        if col not in processed_df.columns:
                            processed_df[col] = 0
                            
                    features_df = processed_df[expected_columns]
                    
                    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
                    features_df_scaled = features_df.copy()
                    features_df_scaled[numerical_cols] = scaler.transform(features_df[numerical_cols])
                    
                    probabilities = xgb_model.predict_proba(features_df_scaled)[:, 1]
                    
                    results_df = raw_df.copy()
                    results_df['Churn_Probability'] = probabilities
                    results_df['Churn_Prediction'] = (probabilities > 0.5).astype(int)
                    
                    st.success("diagnostic completed successfully.")
                    
                    st.markdown("---")
                    st.subheader("Predicted Attrition Overview")
                    
                    total_analyzed = len(results_df)
                    total_predicted_churn = results_df['Churn_Prediction'].sum()
                    overall_churn_rate = (total_predicted_churn / total_analyzed) * 100
                    
                    col_metrics, col_pie = st.columns(2)
                    
                    with col_metrics:
                        st.write("") 
                        st.write("")
                        st.metric("Total Customers Evaluated", total_analyzed)
                        st.metric("Predicted Attrition Rate", f"{overall_churn_rate:.2f}%", f"{total_predicted_churn} High Risk Customers", delta_color="inverse")
                        
                    with col_pie:
                        churn_counts = results_df['Churn_Prediction'].value_counts().rename(index={0: 'Stable', 1: 'High Risk'})
                        pie_colors = ['#ff4b4b' if label == 'High Risk' else '#28a745' for label in churn_counts.index]
                        
                        fig, ax = plt.subplots(figsize=(4, 3))
                        fig.patch.set_alpha(0.0) 
                        ax.patch.set_alpha(0.0)
                        
                        ax.pie(
                            churn_counts, 
                            labels=churn_counts.index, 
                            autopct='%1.1f%%', 
                            startangle=90, 
                            colors=pie_colors,
                            wedgeprops={'edgecolor': 'white', 'linewidth': 1},
                            textprops={'color': "#F8F8FF"}
                        )
                        ax.axis('equal')
                        st.pyplot(fig)
                        
                    # --- ROOT CAUSE ANALYSIS ---
                    st.markdown("---")
                    st.subheader("Root Cause Analysis (High Risk Customers Only)")
                    high_risk_df = results_df[results_df['Churn_Prediction'] == 1]
                    
                    if len(high_risk_df) > 0:
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            st.write("Distribution by Contract Type")
                            if 'Contract' in high_risk_df.columns:
                                contract_dist = high_risk_df['Contract'].value_counts(normalize=True) * 100
                                st.bar_chart(contract_dist)
                                
                        with col_chart2:
                            st.write("Distribution by Internet Service")
                            if 'InternetService' in high_risk_df.columns:
                                internet_dist = high_risk_df['InternetService'].value_counts(normalize=True) * 100
                                st.bar_chart(internet_dist)
                                
                        # --- MODEL ATTRIBUTION ---
                        st.markdown("---")
                        st.subheader("Model Decision Logic")
                        importances = xgb_model.feature_importances_
                        feature_names = xgb_model.get_booster().feature_names
                        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
                        st.bar_chart(importance_df.set_index('Feature'))

                    else:
                        st.info("Zero customers predicted to churn in this dataset.")
                    
                    # 6. Export
                    st.markdown("---")
                    st.subheader("Raw Predictions Export")
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Results", data=csv, file_name='results.csv', mime='text/csv')
                except Exception as e:
                    st.error(f"An error occurred: {e}")