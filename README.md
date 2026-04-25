# Customer Churn Prediction
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## Overview
An end-to-end machine learning project predicting customer churn for a 
telecommunications company. Built as part of a data science portfolio to 
demonstrate a complete ML workflow — from raw data to a saved, deployable model.

**The business problem:** Customer churn is costly. Identifying at-risk customers 
*before* they leave allows companies to intervene with targeted retention strategies 
and protect revenue.

---

## Results
| Model | ROC-AUC | Churn Recall | CV Mean AUC |
|-------|---------|--------------|-------------|
| Logistic Regression (Tuned) | 0.834 | 80% | 0.846 |
| Random Forest | 0.816 | 51% | 0.827 |

**Final model:** Tuned Logistic Regression wrapped in a sklearn Pipeline
**Key win:** Catches 4 out of 5 actual churners (80% recall)

---

## Project Structure

Customer_Churn_Project/
│
├── churn_prediction.ipynb        # Full notebook — EDA to deployment
├── churn_model.pkl               # Saved trained pipeline (ready to deploy)
├── README.md                     # This file
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset

---

## Workflow
| Step | Description |
|------|-------------|
| 1. Setup | Library imports, reusable evaluation function |
| 2. Load & Inspect | Shape, dtypes, missing values, basic stats |
| 3. EDA | Distributions, churn rates by feature, correlation heatmap |
| 4. Cleaning | Fixed TotalCharges dtype, dropped 11 missing rows, one-hot encoding |
| 5. Feature Engineering | X/y split, StandardScaler on numeric columns |
| 6. Train/Test Split | 80/20 stratified split — scaler fit on train only (no leakage) |
| 7. Model Training | Logistic Regression + Random Forest |
| 7b. Cross-Validation | 5-fold stratified CV to confirm result stability |
| 7c. Hyperparameter Tuning | GridSearchCV across C and solver — best: C=100, liblinear |
| 7d. Pipeline | Full sklearn Pipeline wrapping scaler + tuned model |
| 8. Evaluation | Confusion matrix, precision, recall, F1, ROC-AUC, ROC curve |
| 9. Interpretation | LR coefficients + RF feature importances |
| 10. Conclusions | Business insights + recommendations |
| 11. Save Model | joblib export of final pipeline |

---

## Key Findings
- **Contract type** is the strongest predictor — two year contracts dramatically reduce churn
- **Tenure** is the second strongest — long-term customers are far more loyal
- **Fiber optic internet** customers churn at significantly higher rates
- **Electronic check** payment method is strongly associated with churn
- **Tech support & online security** add-ons correlate with lower churn

---

## Business Recommendations
1. **Incentivize longer contracts** — offer discounts for one or two year commitments, especially for new customers
2. **Investigate fiber optic satisfaction** — conduct surveys to understand dissatisfaction drivers
3. **Target electronic check users** — proactively reach out with retention offers
4. **Bundle protective services** — promote tech support and online security add-ons

---

## How to Run

git clone https://github.com/yourusername/Customer_Churn_Project.git
cd Customer_Churn_Project
pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib
jupyter notebook churn_prediction.ipynb

---

## Load the Saved Model

import joblib

model = joblib.load('churn_model.pkl')
predictions = model.predict(new_customer_data)
probabilities = model.predict_proba(new_customer_data)[:, 1]

---

## Dataset
- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers, 21 features
- **Target:** Churn (Yes/No → 1/0)
- **Class balance:** 73.5% No Churn / 26.5% Churn

---

## Future Improvements
- XGBoost / LightGBM for comparison
- SMOTE oversampling to handle class imbalance at data level
- SHAP values for deeper model explainability
- Decision threshold tuning
- Streamlit deployment as an interactive web app

---

## Author
**Akil Ahnaf**
