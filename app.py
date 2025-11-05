import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Load model and scaler
model = joblib.load("loan_prediction_model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🏦 Loan Approval Prediction App")

Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0.0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=0.0)
Loan_Amount_Term = st.number_input("Loan Amount Term (in months)", min_value=0.0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])

# 🔹 Convert categorical inputs to numeric (as per LabelEncoder mappings)
Gender = 1 if Gender == "Male" else 0
Married = 1 if Married == "Yes" else 0
Education = 1 if Education == "Graduate" else 0  # ✅ Corrected mapping
Self_Employed = 1 if Self_Employed == "Yes" else 0
Property_Area_Semiurban = 1 if Property_Area == "Semiurban" else 0
Property_Area_Urban = 1 if Property_Area == "Urban" else 0

Dependents = 3 if Dependents == "3+" else int(Dependents)

# 🔹 Prepare final DataFrame in same order as training
input_data = pd.DataFrame([[
    Gender, Married, Dependents, Education, Self_Employed,
    ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
    Credit_History, Property_Area_Semiurban, Property_Area_Urban
]], columns=[
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area_Semiurban', 'Property_Area_Urban'
])

# Ensure numeric types (important for scaler)
numeric_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
input_data[numeric_cols] = input_data[numeric_cols].astype(float)

# -----------------------
# Scale numeric fields
# -----------------------
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

# -----------------------
# Predict Button
# -----------------------
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]  # confidence score
        if prediction == 1:
            st.success(f" Loan Approved (Confidence: {prob*100:.2f}%)")
        else:
            st.error(f" Loan Not Approved (Confidence: {(1-prob)*100:.2f}%)")
    except Exception as e:
        st.error(f" Something went wrong: {e}")

st.markdown("---")
st.caption("Model built using Logistic Regression | Streamlit App © 2025")
