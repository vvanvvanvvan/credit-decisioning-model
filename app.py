import os
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Load model and config ─────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "credit_model.pkl"))

with open(os.path.join(BASE_DIR, "model_config.json")) as f:
    config = json.load(f)

features          = config["features"]
APPROVE_THRESHOLD = config["approve_threshold"]
DECLINE_THRESHOLD = config["decline_threshold"]

# ── Auto-calculate time features ──────────────────────────────
now         = datetime.datetime.now()
obs_year    = now.year
obs_quarter = (now.month - 1) // 3 + 1

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Decisioning Model",
    page_icon="🏦",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────
st.title("🏦 Credit Decisioning Model")
st.markdown(
    "Enter applicant details below to receive an instant credit decision.")
st.divider()

# ── Input form ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Financial Details")
    income      = st.number_input(
        "Monthly Income (€)",
        min_value=0,
        value=25000,
        step=1000)
    loan_amount = st.number_input(
        "Loan Amount (€)",
        min_value=0,
        value=150000,
        step=5000)
    term_length = st.number_input(
        "Term Length (months)",
        min_value=1,
        value=240,
        step=6)

with col2:
    st.subheader("Credit Profile")
    schufa_input = st.number_input(
        "SCHUFA Credit Score",
        min_value=100,
        max_value=999,
        value=587,
        step=1,
        help="Enter the applicant's SCHUFA credit score (100-999). "
             "Higher scores indicate better creditworthiness.")

    # Convert real SCHUFA to model scale (×15)
    schufa = schufa_input * 15

    num_applic = st.selectbox(
        "Number of Applicants", [1, 2])

with col3:
    st.subheader("Personal Details")
    occupation = st.selectbox(
        "Occupation",
        ["Employee", "Student", "Unknown", "Worker"])
    marital    = st.selectbox(
        "Marital Status",
        ["Married", "Living together", "Single",
         "Separated", "Divorced", "Unknown"])

st.divider()

# ── Encode categoricals ───────────────────────────────────────
occup_map = {
    "Employee": 1, "Student": 2,
    "Unknown":  3, "Worker":  4
}
marital_map = {
    "Married":         1,
    "Living together": 2,
    "Single":          3,
    "Separated":       4,
    "Divorced":        5,
    "Unknown":         6
}

occup_encoded   = occup_map[occupation]
marital_encoded = marital_map[marital]

# ── Calculate instalment to income ───────────────────────────
if income > 0 and term_length > 0:
    install_to_inc = (loan_amount / term_length) / income * 100
else:
    install_to_inc = 0.0

# ── Calculate interaction features ───────────────────────────
income_x_schufa = income * schufa
loan_x_term     = loan_amount * term_length
loan_x_install  = loan_amount * install_to_inc
install_x_term  = install_to_inc * term_length

# ── Show calculated affordability metric ─────────────────────
st.markdown(
    f"**Calculated instalment-to-income ratio: "
    f"{install_to_inc:.2f}%**")
st.divider()

# ── Build input dataframe ─────────────────────────────────────
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

# ── Run decision ──────────────────────────────────────────────
if st.button("Run Credit Decision",
             type="primary",
             use_container_width=True):

    probability = model.predict_proba(input_data)[0][1]

    if probability < APPROVE_THRESHOLD:
        decision = "APPROVE"
        icon     = "✅"
        message  = "Low default risk — auto approved"
    elif probability < DECLINE_THRESHOLD:
        decision = "REFER"
        icon     = "⚠️"
        message  = "Moderate risk — refer for manual review"
    else:
        decision = "DECLINE"
        icon     = "❌"
        message  = "High default risk — auto declined"

    st.divider()

    # ── Decision metrics ──────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Decision", f"{icon} {decision}")
    with col2:
        st.metric("Default Probability", f"{probability:.1%}")
    with col3:
        st.metric("Risk Band",
                  "Low"    if decision == "APPROVE"
                  else "Medium" if decision == "REFER"
                  else "High")
    with col4:
        st.metric("Instalment Ratio", f"{install_to_inc:.2f}%")

    st.markdown(f"**{message}**")
    st.divider()

    # ── Threshold explanation ─────────────────────────────────
    st.subheader("Decision Thresholds")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            f"✅ **APPROVE**  \n"
            f"Below {APPROVE_THRESHOLD:.0%} default probability")
    with col2:
        st.warning(
            f"⚠️ **REFER**  \n"
            f"Between {APPROVE_THRESHOLD:.0%} "
            f"and {DECLINE_THRESHOLD:.0%}")
    with col3:
        st.error(
            f"❌ **DECLINE**  \n"
            f"Above {DECLINE_THRESHOLD:.0%} default probability")

    st.divider()

    # ── SHAP values ───────────────────────────────────────────
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)

    # ── Feature labels ────────────────────────────────────────
    feature_labels = {
        "schufa":          "SCHUFA Credit Score",
        "income":          "Annual Income",
        "term_length":     "Loan Term Length",
        "install_to_inc":  "Repayment Burden",
        "occup_encoded":   "Employment Type",
        "marital_encoded": "Marital Status",
        "loan_amount":     "Loan Amount",
        "num_applic":      "Number of Applicants",
        "obs_year":        "Observation Year",
        "obs_quarter":     "Observation Quarter",
        "income_x_schufa": "Income & Credit Score Combined",
        "loan_x_term":     "Loan Size vs Term",
        "loan_x_install":  "Loan Size vs Repayment Burden",
        "install_x_term":  "Repayment Burden vs Term Length"
    }

    # ── Build SHAP dataframe ──────────────────────────────────
    shap_df = pd.DataFrame({
        "Feature": [feature_labels.get(f, f) for f in features],
        "Impact":  shap_values[0]
    }).sort_values("Impact")

    total = shap_df["Impact"].abs().sum()
    shap_df["Weight"] = (
        shap_df["Impact"].abs() / total * 100
    ).round(1)

    approve_factors = shap_df[shap_df["Impact"] < 0].sort_values("Impact")
    decline_factors = shap_df[shap_df["Impact"] > 0].sort_values(
        "Impact", ascending=False)

    # ── Factors section ───────────────────────────────────────
    st.subheader("Factors Impacting the Decision")
    st.markdown(
        "The factors below show what drove this credit decision "
        "in plain terms.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        if decision == "APPROVE":
            st.success("### ✅ Reasons for Approval")
        else:
            st.success("### ✅ Factors in Applicant's Favour")

        for _, row in approve_factors.iterrows():
            weight = row["Weight"]
            if weight > 20:
                strength = "🟢 Strong positive"
            elif weight > 10:
                strength = "🟡 Moderate positive"
            else:
                strength = "⚪ Minor positive"

            st.markdown(
                f"**{row['Feature']}**  \n"
                f"{strength} — contributes "
                f"**{weight:.1f}%** of decision weight"
            )
            st.progress(min(weight / 40, 1.0))

    with col2:
        if decision == "DECLINE":
            st.error("### ❌ Reasons for Decline")
        else:
            st.warning("### ⚠️ Areas of Concern")

        for _, row in decline_factors.iterrows():
            weight = row["Weight"]
            if weight > 20:
                strength = "🔴 High concern"
            elif weight > 10:
                strength = "🟠 Moderate concern"
            else:
                strength = "🟡 Minor concern"

            st.markdown(
                f"**{row['Feature']}**  \n"
                f"{strength} — contributes "
                f"**{weight:.1f}%** of decision weight"
            )
            st.progress(min(weight / 40, 1.0))

    st.divider()

    # ── Plain English summary ─────────────────────────────────
    top_approve = (approve_factors.iloc[0]
                   if len(approve_factors) > 0 else None)
    top_decline = (decline_factors.iloc[0]
                   if len(decline_factors) > 0 else None)

    if decision == "APPROVE":
        st.success(
            f"**Decision Summary:** This application was automatically "
            f"approved. The strongest positive factor was "
            f"**{top_approve['Feature']}** which accounted for "
            f"{top_approve['Weight']:.1f}% of the model's decision. "
            f"The predicted probability of default is "
            f"**{probability:.1%}** — well below the approval "
            f"threshold of {APPROVE_THRESHOLD:.0%}."
        )
    elif decision == "REFER":
        st.warning(
            f"**Decision Summary:** This application has been referred "
            f"for manual underwriter review. The model identified mixed "
            f"signals — positive factors include "
            f"**{top_approve['Feature'] if top_approve is not None else 'N/A'}** "
            f"but concerns around "
            f"**{top_decline['Feature'] if top_decline is not None else 'N/A'}** "
            f"mean the decision requires human judgement. "
            f"Predicted default probability: **{probability:.1%}**."
        )
    else:
        st.error(
            f"**Decision Summary:** This application was automatically "
            f"declined. The primary concern was "
            f"**{top_decline['Feature']}** which accounted for "
            f"{top_decline['Weight']:.1f}% of the model's decision. "
            f"The predicted probability of default is "
            f"**{probability:.1%}** — above the decline threshold "
            f"of {DECLINE_THRESHOLD:.0%}."
        )

    st.divider()

    # ── Applicant summary ─────────────────────────────────────
    st.subheader("Applicant Summary")
    summary = pd.DataFrame({
        "Field": [
            "Monthly Income",
            "Loan Amount",
            "Term Length",
            "SCHUFA Credit Score",
            "Instalment to Income",
            "Number of Applicants",
            "Occupation",
            "Marital Status"
        ],
        "Value": [
            f"€{income:,}",
            f"€{loan_amount:,}",
            f"{term_length} months",
            f"{schufa_input}",
            f"{install_to_inc:.2f}%",
            num_applic,
            occupation,
            marital
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)
