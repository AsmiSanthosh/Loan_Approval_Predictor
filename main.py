import streamlit as st
import pandas as pd
import joblib
import os

# ------------------------------
# Load Model
# ------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model", "best_xgb.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="💳 Loan Approval Predictor", page_icon="💳", layout="wide")
st.title("💳 Loan Approval Prediction")
st.write("Fill in the details below to predict whether the loan will be approved.")

# ------------------------------
# Two-column layout
# ------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Info")
    person_age = st.number_input("Person Age", min_value=18, max_value=75, value=30)
    person_income = st.number_input("Person Income", min_value=10000, step=100, value=50000)
    person_emp_length = st.number_input("Employment Length (years)", min_value=0, max_value=50, value=5)
    person_home_ownership = st.selectbox("Home Ownership", ["OWN", "RENT", "MORTGAGE", "OTHER"])
    cb_person_default_on_file = st.selectbox("Default on File", ["Y", "N"])
    cb_person_cred_hist_length = st.number_input("Credit History Length (years)", min_value=0, value=10)

with col2:
    st.subheader("💰 Loan Info")
    loan_intent = st.selectbox("Loan Intent", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount", min_value=5000, step=1000, value=10000)
    loan_int_rate = st.number_input("Loan Interest Rate (%)", min_value=5.0, max_value=25.0, value=12.5)
    loan_percent_income = st.number_input("Loan Percent Income", min_value=0.0, step=0.01, value=0.2)

# Warnings for unrealistic values
if not (20 <= person_age <= 75):
    st.warning("⚠️ Age should be between 20 and 75 years.")

if not (person_income >= 10000):
    st.warning("⚠️ Income should be atleast 10000.")

if not (0 <= person_emp_length <= 50):
    st.warning("⚠️ Employment length should be between 0 and 50 years.")

if not (loan_amnt >= 5000):
    st.warning("⚠️ Loan amount should be atleast 5000.")

if not (5 <= loan_int_rate <= 25):
    st.warning("⚠️ Interest rate should be between 5% and 25%.")

# ------------------------------
# Predict button
# ------------------------------
if st.button("Predict"):
    try:
        # Preprocess input
        cb_person_default_on_file_bin = 1 if cb_person_default_on_file == "Y" else 0
        debt_to_income_ratio = loan_amnt / max(person_income, 1)  # avoid division by zero

        # Numeric features
        numeric_df = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "cb_person_default_on_file": cb_person_default_on_file_bin,
            "debt_to_income_ratio": debt_to_income_ratio
        }])

        # Categorical features
        cat_df = pd.DataFrame([{
            "person_home_ownership": person_home_ownership,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade
        }])

        # One-hot encode
        cat_df_encoded = pd.get_dummies(cat_df)

        # Combine numeric + categorical
        input_df = pd.concat([numeric_df, cat_df_encoded], axis=1)

        # Ensure all model columns exist
        model_columns = model.feature_names_in_  # sklearn models
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match model
        input_df = input_df[model_columns]

        # Predict
        prediction = model.predict(input_df)[0]

        # Predict probability (if supported)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]  # probability of loan approved
        else:
            proba = None

        # Show result
        if prediction == 1:
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Not Approved!")

        if proba is not None:
            st.info(f"Approval Probability: {proba * 100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")