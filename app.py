import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set page config
st.set_page_config(page_title="Credit Card Default Predictor", layout="wide")

# Title
st.title("🏦 Credit Card Default Prediction")
st.write("Predict whether a customer will default on their credit card payment next month")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Sidebar for input
st.sidebar.header("📋 Customer Information")

# Create input fields for all 23 features
col1, col2 = st.columns(2)

with col1:
    limit_bal = st.number_input("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=50000, step=1000)
    age = st.slider("Age", 21, 80, 40)
    sex = st.selectbox("Gender", [1, 2],
                       format_func=lambda x: {1: "Male", 2: "Female"}[x])
    education = st.selectbox("Education Level", [1, 2, 3, 4], 
                            format_func=lambda x: {1: "Graduate School", 2: "University", 
                                                   3: "High School", 4: "Other"}[x])
    marriage = st.selectbox("Marital Status", [1, 2, 3],
                           format_func=lambda x: {1: "Married", 2: "Single", 3: "Other"}[x])

with col2:
    pay_0 = st.selectbox("Most Recent Repayment Status (PAY_0)", 
                        [-1, 0, 1, 2, 3, 4, 5, 6],
                        format_func=lambda x: {-1: "Paid in full", 0: "Revolving", 
                                              1: "1 month delay", 2: "2 month delay",
                                              3: "3 month delay", 4: "4 month delay",
                                              5: "5 month delay", 6: "6+ month delay"}[x])
    
    # Previous repayment status
    pay_2 = st.selectbox("Repayment Status 2 Months Ago (PAY_2)", [-1, 0, 1, 2, 3, 4, 5, 6], index=1)
    pay_3 = st.selectbox("Repayment Status 3 Months Ago (PAY_3)", [-1, 0, 1, 2, 3, 4, 5, 6], index=1)
    pay_4 = st.selectbox("Repayment Status 4 Months Ago (PAY_4)", [-1, 0, 1, 2, 3, 4, 5, 6], index=1)

# Bill amounts
st.sidebar.subheader("💳 Bill Amounts (NT$)")
bill_amt1 = st.sidebar.number_input("Bill Amount (Month 1)", min_value=0, value=5000, step=100)
bill_amt2 = st.sidebar.number_input("Bill Amount (Month 2)", min_value=0, value=5000, step=100)
bill_amt3 = st.sidebar.number_input("Bill Amount (Month 3)", min_value=0, value=5000, step=100)

# Payment amounts
st.sidebar.subheader("💰 Payment Amounts (NT$)")
pay_amt1 = st.sidebar.number_input("Payment Amount (Month 1)", min_value=0, value=2000, step=100)
pay_amt2 = st.sidebar.number_input("Payment Amount (Month 2)", min_value=0, value=2000, step=100)
pay_amt3 = st.sidebar.number_input("Payment Amount (Month 3)", min_value=0, value=2000, step=100)

# Additional repayment status fields
pay_5 = st.sidebar.selectbox("Repayment Status 5 Months Ago (PAY_5)", [-1, 0, 1, 2, 3, 4, 5, 6], index=1)
pay_6 = st.sidebar.selectbox("Repayment Status 6 Months Ago (PAY_6)", [-1, 0, 1, 2, 3, 4, 5, 6], index=1)

# More bill and payment amounts
bill_amt4 = st.sidebar.number_input("Bill Amount (Month 4)", min_value=0, value=5000, step=100)
bill_amt5 = st.sidebar.number_input("Bill Amount (Month 5)", min_value=0, value=5000, step=100)
bill_amt6 = st.sidebar.number_input("Bill Amount (Month 6)", min_value=0, value=5000, step=100)

pay_amt4 = st.sidebar.number_input("Payment Amount (Month 4)", min_value=0, value=2000, step=100)
pay_amt5 = st.sidebar.number_input("Payment Amount (Month 5)", min_value=0, value=2000, step=100)
pay_amt6 = st.sidebar.number_input("Payment Amount (Month 6)", min_value=0, value=2000, step=100)

# Button to predict
if st.button("🔮 Predict Default Risk", use_container_width=True):
    # Prepare input features matching exact training and scaler schema
    feature_order = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    input_data = pd.DataFrame([{
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0,
        'PAY_2': pay_2,
        'PAY_3': pay_3,
        'PAY_4': pay_4,
        'PAY_5': pay_5,
        'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1,
        'BILL_AMT2': bill_amt2,
        'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4,
        'BILL_AMT5': bill_amt5,
        'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1,
        'PAY_AMT2': pay_amt2,
        'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4,
        'PAY_AMT5': pay_amt5,
        'PAY_AMT6': pay_amt6,
    }])[feature_order]
    
    # Scale features with fitted scaler
    input_scaled = scaler.transform(input_data)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_order)
    
    # Make prediction
    prediction = model.predict(input_scaled_df)[0]
    probability = model.predict_proba(input_scaled_df)[0]
    
    # Display results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if prediction == 1:
            st.error("⚠️ **HIGH RISK - LIKELY TO DEFAULT**")
            risk_level = f"Default Probability: {probability[1]*100:.1f}%"
        else:
            st.success("✅ **LOW RISK - UNLIKELY TO DEFAULT**")
            risk_level = f"Default Probability: {probability[1]*100:.1f}%"
        
        st.metric("Prediction Result", "DEFAULT" if prediction == 1 else "NO DEFAULT")
        st.write(risk_level)
    
    with col2:
        st.metric("Non-Default Probability", f"{probability[0]*100:.1f}%")
        st.metric("Default Probability", f"{probability[1]*100:.1f}%")
    
    # Show a visual risk gauge
    st.progress(probability[1])
    st.caption("Risk Probability (0% = Safe, 100% = High Risk)")

st.markdown("---")
st.caption("🏦 Credit Card Default Prediction System | Built with Streamlit & Scikit-Learn | Made with ❤️ by Harsh Mishra & Sarthak Tajane")