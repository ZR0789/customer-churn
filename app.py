import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load model & scaler ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ── Exact feature columns (must match training) ───────────────────────────────
FEATURE_COLS = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'SeniorCitizen_1', 'Partner_Yes', 'Dependents_Yes',
    'PhoneService_Yes', 'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_One year', 'Contract_Two year',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

# ── Friendly risk factor labels ───────────────────────────────────────────────
RISK_LABELS = {
    'tenure':                               'Customer tenure length',
    'MonthlyCharges':                       'Monthly charges amount',
    'TotalCharges':                         'Total charges paid',
    'gender_Male':                          'Gender: Male',
    'SeniorCitizen_1':                      'Senior citizen status',
    'Partner_Yes':                          'Has a partner',
    'Dependents_Yes':                       'Has dependents',
    'PhoneService_Yes':                     'Has phone service',
    'MultipleLines_No phone service':       'No phone service',
    'MultipleLines_Yes':                    'Has multiple lines',
    'InternetService_Fiber optic':          'Uses fiber optic internet',
    'InternetService_No':                   'No internet service',
    'OnlineSecurity_No internet service':   'No internet (online security)',
    'OnlineSecurity_Yes':                   'Has online security',
    'OnlineBackup_No internet service':     'No internet (online backup)',
    'OnlineBackup_Yes':                     'Has online backup',
    'DeviceProtection_No internet service': 'No internet (device protection)',
    'DeviceProtection_Yes':                 'Has device protection',
    'TechSupport_No internet service':      'No internet (tech support)',
    'TechSupport_Yes':                      'Has tech support',
    'StreamingTV_No internet service':      'No internet (streaming TV)',
    'StreamingTV_Yes':                      'Streams TV',
    'StreamingMovies_No internet service':  'No internet (streaming movies)',
    'StreamingMovies_Yes':                  'Streams movies',
    'Contract_One year':                    'One-year contract',
    'Contract_Two year':                    'Two-year contract',
    'PaperlessBilling_Yes':                 'Uses paperless billing',
    'PaymentMethod_Credit card (automatic)':'Pays by credit card (auto)',
    'PaymentMethod_Electronic check':       'Pays by electronic check',
    'PaymentMethod_Mailed check':           'Pays by mailed check',
}

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.write(
    "Enter customer details on the left to predict whether they will churn. "
    "Built with Logistic Regression trained on the Telco Customer Churn dataset — **AUC 0.836**."
)
st.divider()

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("🧾 Customer Details")

st.sidebar.subheader("Account Info")
tenure           = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges  = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=1.0)
total_charges    = monthly_charges * tenure   # auto-calculated
st.sidebar.caption(f"Total Charges (auto): ${total_charges:.2f}")

contract         = st.sidebar.selectbox("Contract Type",
                       ["Month-to-month", "One year", "Two year"])
paperless        = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment          = st.sidebar.selectbox("Payment Method", [
                       "Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"])

st.sidebar.subheader("Demographics")
gender          = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior          = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner         = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents      = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

st.sidebar.subheader("Services")
phone_service   = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines  = st.sidebar.selectbox("Multiple Lines",
                       ["No", "Yes", "No phone service"])
internet        = st.sidebar.selectbox("Internet Service",
                       ["DSL", "Fiber optic", "No"])
online_sec      = st.sidebar.selectbox("Online Security",
                       ["No", "Yes", "No internet service"])
online_bkp      = st.sidebar.selectbox("Online Backup",
                       ["No", "Yes", "No internet service"])
device_prot     = st.sidebar.selectbox("Device Protection",
                       ["No", "Yes", "No internet service"])
tech_support    = st.sidebar.selectbox("Tech Support",
                       ["No", "Yes", "No internet service"])
streaming_tv    = st.sidebar.selectbox("Streaming TV",
                       ["No", "Yes", "No internet service"])
streaming_mov   = st.sidebar.selectbox("Streaming Movies",
                       ["No", "Yes", "No internet service"])

# ── Build input row matching training columns ─────────────────────────────────
def build_input():
    row = {col: 0 for col in FEATURE_COLS}

    # Numericals
    row['tenure']         = tenure
    row['MonthlyCharges'] = monthly_charges
    row['TotalCharges']   = total_charges

    # One-hot flags  (drop_first=True logic reproduced)
    row['gender_Male']                          = int(gender == "Male")
    row['SeniorCitizen_1']                      = int(senior == "Yes")
    row['Partner_Yes']                          = int(partner == "Yes")
    row['Dependents_Yes']                       = int(dependents == "Yes")
    row['PhoneService_Yes']                     = int(phone_service == "Yes")
    row['MultipleLines_No phone service']       = int(multiple_lines == "No phone service")
    row['MultipleLines_Yes']                    = int(multiple_lines == "Yes")
    row['InternetService_Fiber optic']          = int(internet == "Fiber optic")
    row['InternetService_No']                   = int(internet == "No")
    row['OnlineSecurity_No internet service']   = int(online_sec == "No internet service")
    row['OnlineSecurity_Yes']                   = int(online_sec == "Yes")
    row['OnlineBackup_No internet service']     = int(online_bkp == "No internet service")
    row['OnlineBackup_Yes']                     = int(online_bkp == "Yes")
    row['DeviceProtection_No internet service'] = int(device_prot == "No internet service")
    row['DeviceProtection_Yes']                 = int(device_prot == "Yes")
    row['TechSupport_No internet service']      = int(tech_support == "No internet service")
    row['TechSupport_Yes']                      = int(tech_support == "Yes")
    row['StreamingTV_No internet service']      = int(streaming_tv == "No internet service")
    row['StreamingTV_Yes']                      = int(streaming_tv == "Yes")
    row['StreamingMovies_No internet service']  = int(streaming_mov == "No internet service")
    row['StreamingMovies_Yes']                  = int(streaming_mov == "Yes")
    row['Contract_One year']                    = int(contract == "One year")
    row['Contract_Two year']                    = int(contract == "Two year")
    row['PaperlessBilling_Yes']                 = int(paperless == "Yes")
    row['PaymentMethod_Credit card (automatic)']= int(payment == "Credit card (automatic)")
    row['PaymentMethod_Electronic check']       = int(payment == "Electronic check")
    row['PaymentMethod_Mailed check']           = int(payment == "Mailed check")

    df = pd.DataFrame([row])[FEATURE_COLS]

    # Scale numericals (same scaler used in training)
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    return df

# ── Prediction section ────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("Prediction")
    predict_btn = st.button("🔍 Predict Churn", type="primary", use_container_width=True)

    if predict_btn:
        input_df   = build_input()
        prob       = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("⚠️  This customer is likely to **CHURN**")
        else:
            st.success("✅  This customer is likely to **STAY**")

        st.metric("Churn Probability", f"{prob:.1%}")
        st.progress(float(prob))

        # Risk factors via coefficients × input values
        st.subheader("Top Risk Factors")
        coefs        = pd.Series(model.coef_[0], index=FEATURE_COLS)
        input_vals   = input_df.iloc[0]
        risk_scores  = coefs * input_vals
        top_increase = risk_scores.nlargest(3)
        top_decrease = risk_scores.nsmallest(3)

        st.write("**Factors increasing churn risk:**")
        for feat, score in top_increase.items():
            label = RISK_LABELS.get(feat, feat)
            st.write(f"🔴 {label}")

        st.write("**Factors reducing churn risk:**")
        for feat, score in top_decrease.items():
            label = RISK_LABELS.get(feat, feat)
            st.write(f"🟢 {label}")

        # Business recommendation
        st.subheader("💡 Recommendation")
        if prediction == 1:
            recs = []
            if contract == "Month-to-month":
                recs.append("Offer a discount to upgrade to a one-year or two-year contract.")
            if tenure < 12:
                recs.append("Assign a dedicated onboarding specialist — new customers are highest risk.")
            if internet == "Fiber optic":
                recs.append("Check service quality for fiber-optic customers in this area.")
            if tech_support == "No":
                recs.append("Offer a free TechSupport trial — it significantly reduces churn.")
            if not recs:
                recs.append("Reach out with a personalised retention offer.")
            for r in recs:
                st.write(f"• {r}")
        else:
            st.write("No immediate action needed. Monitor at next billing cycle.")

with col2:
    st.subheader("Input Summary")
    summary = {
        "Tenure":         f"{tenure} months",
        "Monthly Charges":f"${monthly_charges:.2f}",
        "Total Charges":  f"${total_charges:.2f}",
        "Contract":       contract,
        "Internet":       internet,
        "Tech Support":   tech_support,
        "Online Security":online_sec,
        "Payment":        payment,
        "Senior Citizen": senior,
        "Partner":        partner,
    }
    for k, v in summary.items():
        st.write(f"**{k}:** {v}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built by Zainab Rasti · "
    "Model: Logistic Regression · Dataset: Telco Customer Churn (7,032 records) · "
    "AUC: 0.836 · Recall: 0.572"
)
