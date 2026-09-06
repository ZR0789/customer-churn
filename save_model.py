# ─────────────────────────────────────────────────────────────
# Run this script ONCE after running your notebook
# It saves the trained model and scaler as .pkl files
# which app.py will load
# ─────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load and clean data (same as notebook)
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]
df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
X = pd.concat([df[numerical_cols], df_encoded], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled  = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols]  = scaler.transform(X_test[numerical_cols])

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Save both files
joblib.dump(model,  'churn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ churn_model.pkl saved")
print("✅ scaler.pkl saved")
print(f"   Features: {len(model.feature_names_in_)} columns")
print(f"   Test accuracy: {model.score(X_test_scaled, y_test):.4f}")
