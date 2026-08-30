# Customer Churn Prediction 🔍

Predicts whether a Telco customer will churn using 4 machine learning models —
Logistic Regression, Decision Tree, Random Forest, and XGBoost —
with full model comparison, ROC/AUC curves, and business recommendations.

---

## What it does

- Cleans and preprocesses the Telco Customer Churn dataset (7,043 records, 19 features)
- Handles `TotalCharges` type coercion and missing value removal
- Applies One-Hot Encoding to 16 categorical features + StandardScaler on numerical ones
- Stratified 80/20 train/test split to preserve churn ratio
- Trains and compares 4 models:
  - **Logistic Regression** — baseline with coefficient interpretation
  - **Decision Tree (unconstrained vs pruned)** — overfitting analysis
  - **Random Forest** — bagging ensemble, 100 trees
  - **XGBoost** — boosting ensemble, 100 estimators
- Evaluates all models on Accuracy, Precision, Recall, F1-Score, and AUC
- Identifies highest-recall model and plots its Confusion Matrix
- Plots ROC curves for all 4 models on a single chart
- Extracts feature importances from RF and XGBoost

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 80.45% | — | **0.5722** | — | **0.8361** |
| Pruned Decision Tree | — | — | — | — | — |
| Random Forest | — | — | — | — | — |
| XGBoost | — | — | — | — | — |

> **Best model by AUC and Recall: Logistic Regression (AUC = 0.8361)**

**Key churn drivers identified:**
- Month-to-month contract customers are highest risk
- Low tenure (new customers) strongly predicts churn
- Fiber-optic internet users show higher churn rates
- Customers without TechSupport or OnlineSecurity more likely to leave

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Pandas / NumPy | Data loading, cleaning, feature engineering |
| Scikit-learn | Logistic Regression, Decision Tree, Random Forest, metrics, preprocessing |
| XGBoost | Gradient boosting classifier |
| Matplotlib / Seaborn | All visualizations — ROC curves, confusion matrix, feature coefficients |
| Jupyter Notebook | Development environment |

---

## How to run

```bash
git clone https://github.com/zainab-rasti/customer-churn
cd customer-churn
pip install -r requirements.txt
jupyter notebook 23F_0775_Task1.ipynb
```

The dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is included in the repo.

---

## Project Structure

```
customer-churn/
├── code.ipynb              # Main notebook — all 5 parts
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Telco dataset (7,043 records)
├── requirements.txt
└── README.md
```

---

## Notebook Structure

| Part | Description |
|------|-------------|
| Part 1 | Data preprocessing — cleaning, encoding, scaling, stratified split |
| Part 2 | Logistic Regression — training + coefficient visualization |
| Part 3 | Decision Trees — unconstrained vs pruned, overfitting analysis |
| Part 4 | Ensemble learning — Random Forest (Bagging) + XGBoost (Boosting) |
| Part 5 | Model evaluation — comparison table, confusion matrix, ROC/AUC curves |

---

## Business Recommendations

Based on model findings:
- Prioritize retention offers for month-to-month contract customers
- Focus onboarding programmes for new customers (low tenure)
- Improve services for fiber-optic users and promote TechSupport / OnlineSecurity
- Review paperless billing experience to reduce churn

---

