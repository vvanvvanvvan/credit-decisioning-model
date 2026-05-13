import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# ── Load model and config ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model  = joblib.load(os.path.join(BASE_DIR, "credit_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))

with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    config = json.load(f)

features          = config["features"]
APPROVE_THRESHOLD = config["approve_threshold"]
DECLINE_THRESHOLD = config["decline_threshold"]

st.set_page_config(
    page_title="Credit Decisioning Model",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Decisioning Model")
st.markdown("Enter applicant details below to receive an instant credit decision.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Financial Details")
    income         = st.number_input("Annual Income (€)", min_value=0, max_value=200000, value=25000, step=1000)
    loan_amount    = st.number_input("Loan Amount (€)", min_value=0, max_value=100000, value=15000, step=500)
    term_length    = st.number_input("Term Length (months)", min_value=6, max_value=240, value=60, step=6)
    install_to_inc = st.number_input("Instalment to Income (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1)

with col2:
    st.subheader("Credit Profile")
    schufa      = st.number_input("SCHUFA Score", min_value=0, max_value=20000, value=7500, step=100)
    num_applic  = st.selectbox("Number of Applicants", [1, 2])
    obs_year    = st.number_input("Observation Year", min_value=2008, max_value=2030, value=2024, step=1)
    obs_quarter = st.selectbox("Observation Quarter", [1, 2, 3, 4])

with col3:
    st.subheader("Personal Details")
    occupation = st.selectbox("Occupation", ["Employee", "Student", "Unknown", "Worker"])
    marital    = st.selectbox("Marital Status", ["Married", "Living together", "Single", "Separated", "Divorced", "Unknown"])

st.divider()

occup_map   = {"Employee": 1, "Student": 2, "Unknown": 3, "Worker": 4}
marital_map = {"Married": 1, "Living together": 2, "Single": 3, "Separated": 4, "Divorced": 5, "Unknown": 6}

occup_encoded   = occup_map[occupation]
marital_encoded = marital_map[marital]

income_x_schufa = income * schufa
loan_x_term     = loan_amount * term_length
loan_x_install  = loan_amount * install_to_inc
install_x_term  = install_to_inc * term_length

input_data = pd.DataFrame([{
    "schufa":          schufa,
    "income":          income,
    "term_length":     term_length,
    "install_to_inc":  install_to_inc,
    "occup_encoded":   occup_encoded,
    "marital_encoded": marital_encoded,
    "loan_amount":     loan_amount,
    "num_applic":      num_applic,
    "obs_year":        obs_year,
    "obs_quarter":     obs_quarter,
    "income_x_schufa": income_x_schufa,
    "loan_x_term":     loan_x_term,
    "loan_x_install":  loan_x_install,
    "install_x_term":  install_x_term
}])

if st.button("Run Credit Decision", type="primary", use_container_width=True):

    probability = model.predict_proba(input_data)[0][1]

    if probability < APPROVE_THRESHOLD:
        decision = "APPROVE"
        icon     = "✅"
        message  = "Low default risk — auto approved"
    elif probability < DECLINE_THRESHOLD:
        decision = "REFER"
        icon     = "⚠️"
        message  = "Moderate risk — manual review required"
    else:
        decision = "DECLINE"
        icon     = "❌"
        message  = "High default risk — auto declined"

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Decision", f"{icon} {decision}")
    with col2:
        st.metric("Default Probability", f"{probability:.1%}")
    with col3:
        st.metric("Risk Band",
                  "Low" if decision=="APPROVE"
                  else "Medium" if decision=="REFER"
                  else "High")

    st.markdown(f"**{message}**")
    st.divider()

    st.subheader("Why this decision was made")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(
        shap.Explanation(
            values        = shap_values[0],
            base_values   = explainer.expected_value,
            data          = input_data.iloc[0],
            feature_names = features
        ),
        show=False
    )
    st.pyplot(fig)
    plt.close()

    st.subheader("Key risk factors")
    shap_df = pd.DataFrame({
        "Feature": features,
        "Impact":  shap_values[0]
    }).sort_values("Impact")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Factors increasing risk:**")
        for _, row in shap_df.tail(3).iterrows():
            st.markdown(f"- {row['Feature']}: +{row['Impact']:.3f}")
    with col2:
        st.markdown("**Factors reducing risk:**")
        for _, row in shap_df.head(3).iterrows():
            st.markdown(f"- {row['Feature']}: {row['Impact']:.3f}")
