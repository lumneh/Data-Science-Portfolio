# Credit Score Classification & Monitoring System

An end-to-end machine learning system for classifying customer credit
scores into **Good, Poor, and Standard** categories, with leakage-aware
data splitting, feature engineering, XGBoost modelling, SHAP
explainability, FastAPI deployment, and post-deployment monitoring.

## Project Overview

Credit scoring is a core component of lending and financial risk
management. The objective of this project is to build a machine learning
system that can classify customers into credit-score categories using
financial, credit-history, and payment-behaviour information.

The project covers the broader machine learning lifecycle:

**Data validation → preprocessing → modelling → evaluation →
explainability → API deployment → monitoring**

### Key Results

  Metric              Final XGBoost Model
  ----------------- ---------------------
  Accuracy                     **70.79%**
  Macro Precision              **68.51%**
  Macro Recall                 **68.67%**
  Macro F1                     **68.58%**

## Architecture

![Credit Score Classification System
Architecture](assets/architecture.png)

## Business Problem

Financial institutions need reliable ways to assess customer credit
risk. A credit classification model can support customer risk
segmentation, credit assessment, lending decision support, portfolio
risk analysis, and early identification of higher-risk customers.

A useful production ML system requires more than a high-performing
model. It also needs protection against data leakage, explainability for
individual decisions, an accessible prediction interface, and monitoring
after deployment.

## Dataset

The project uses the Kaggle Credit Score Classification dataset.

-   **100,000 observations**
-   **12,500 unique customers**
-   **28 original columns**
-   Multiple observations per customer
-   Three target classes: Good, Poor, Standard

Because each customer has multiple observations, conventional random
row-level splitting can cause customer-level data leakage.

## Data Validation

The validation stage examined dataset dimensions, data types, missing
values, duplicate records, invalid numeric values, suspicious age
values, inconsistent categorical values, malformed numeric fields, and
customer-level observation structure.

The dataset contained data-quality issues typical of real-world
financial data, including malformed numeric values, missing
observations, inconsistent representations, and implausible values.
These issues were addressed before model development.

## Data Preprocessing & Feature Engineering

The preprocessing pipeline was designed to produce consistent features
for both model training and deployment.

Key steps included:

-   Cleaning malformed numeric fields
-   Converting numeric variables to appropriate types
-   Handling missing values
-   Cleaning categorical variables
-   Encoding categorical features
-   Creating derived credit-history features
-   Creating loan-type indicators
-   Maintaining a reusable preprocessing pipeline

Examples of engineered features include:

-   `Credit_History_Months`
-   `Num_Loan_Types`
-   `Has_auto_loan`
-   `Has_credit_builder_loan`
-   `Has_personal_loan`
-   `Has_home_equity_loan`
-   `Has_mortgage_loan`
-   `Has_student_loan`
-   `Has_debt_consolidation_loan`
-   `Has_payday_loan`

The final model uses **61 processed features** derived from **31 raw
model-input features**.

## Customer-Level Data Splitting

Customers were split first, with all observations belonging to a
customer assigned to the same dataset.

  Dataset          Rows   Customers
  ------------ -------- -----------
  Training       70,000       8,750
  Validation     15,000       1,875
  Test           15,000       1,875

There was **zero customer overlap** between the three datasets.

This provides a more realistic estimate of performance on previously
unseen customers.

## Model Development

Dummy classification, logistic regression, and XGBoost were evaluated.

  Model                     Accuracy     Macro F1
  --------------------- ------------ ------------
  Dummy Classifier            53.36%       23.20%
  Logistic Regression         61.38%       57.43%
  Baseline XGBoost            69.09%       66.74%
  Tuned XGBoost           **69.33%**   **67.06%**

The final tuned XGBoost model was selected based on validation
performance and evaluated on the held-out test set.

### Final test performance

  Class                 Precision     Recall         F1
  ------------------- ----------- ---------- ----------
  Good                       0.60       0.62       0.61
  Poor                       0.72       0.70       0.71
  Standard                   0.74       0.74       0.74
  **Macro Average**      **0.69**   **0.69**   **0.69**

The final test accuracy was **70.79%**, with a **68.58% macro F1**.

### Class imbalance

A weighted XGBoost experiment increased minority-class recall but
reduced overall accuracy and slightly reduced macro F1. The final
unweighted model was therefore selected based on the overall balance of
predictive performance.

## Model Explainability

The project uses **SHAP (SHapley Additive exPlanations)** to understand
model behaviour.

### Global explainability

Important model drivers included:

-   Outstanding debt
-   Credit-history length
-   Number of credit inquiries
-   Delay from due date
-   Credit mix
-   Annual income
-   Total monthly EMI
-   Payment-of-minimum-amount behaviour
-   Number of credit cards
-   Number of loan types

### Local explainability

Individual predictions can be explained using feature-level SHAP
contributions showing which characteristics push the prediction toward
or away from the predicted class.

SHAP values are treated as model-attribution information rather than
causal effects.

## API Deployment

The trained model is deployed through a **FastAPI REST API**. The API
loads the saved preprocessing pipeline and XGBoost model and applies the
same transformation used during model development.

  Endpoint     Method   Purpose
  ------------ -------- -------------------------------------------
  `/`          GET      API status
  `/health`    GET      Health check
  `/predict`   POST     Credit-score prediction
  `/docs`      GET      Interactive Swagger/OpenAPI documentation

The `/predict` endpoint returns the predicted credit-score class, class
probabilities, and feature-level SHAP explanations.

**API:** FastAPI

### Example API Request

``` json
{
  "Age": 40,
  "Annual_Income": 33751.27,
  "Monthly_Inhand_Salary": 2948.61,
  "Num_Bank_Accounts": 5,
  "Num_Credit_Card": 5,
  "Interest_Rate": 20,
  "Num_of_Loan": 3,
  "Delay_from_due_date": 16,
  "Num_of_Delayed_Payment": 20,
  "Changed_Credit_Limit": 11,
  "Num_Credit_Inquiries": 4,
  "Outstanding_Debt": 1328.93,
  "Credit_Utilization_Ratio": 37.09,
  "Total_EMI_per_month": 65.01,
  "Amount_invested_monthly": 117.31,
  "Monthly_Balance": 362.55,
  "Credit_History_Months": 230,
  "Num_Loan_Types": 3,
  "Month": "January",
  "Occupation": "Teacher",
  "Credit_Mix": "Standard",
  "Payment_of_Min_Amount": "NM",
  "Payment_Behaviour": "High_spent_Medium_value_payments"
}
```

## Streamlit Application

An interactive **Streamlit web application** was added as a user-facing
interface for the trained credit-score model.

The application loads the saved preprocessing pipeline and XGBoost model
and provides a simple form for entering the 31 raw model-input features.
It also includes demonstration profiles for quickly testing the model.

### Streamlit features

- Customer input form
- Custom customer prediction
- Good, Standard, and Poor demonstration profiles
- Predicted credit-score class
- Class probability visualization
- Model performance summary
- Optional SHAP explanation section

The Streamlit application is the interactive demonstration layer, while
FastAPI provides the REST API layer for programmatic model access.

### Running the Streamlit application

From the project root:

```bash
streamlit run app.py
```

The application opens in a local browser and sends the entered customer
features through the same saved preprocessing pipeline used during model
development before generating the prediction.

### Deployment

The Streamlit application is designed for deployment through
**Streamlit Community Cloud** using the GitHub repository as the source.

The repository contains the saved model and preprocessing artifacts
required by the application.

The current deployed demonstration focuses on reliable prediction and
probability outputs. SHAP remains an optional component because of a
dependency compatibility issue in the current local SHAP/Numba
environment. The SHAP implementation and explainability notebook remain
available in the repository for reproduction in a compatible environment.

This distinction allows the interactive application to remain usable
while preserving the project's full explainability implementation.

## 🚀 Live Demo

**Try the interactive Streamlit application:**

👉 [Credit Score Classification App](https://creditscoringclassification.streamlit.app/)

The application provides an interactive interface for entering customer
financial information and generating credit-score predictions and class
probabilities.

**Deployment:** Streamlit Community Cloud

**Model:** XGBoost

## Model Monitoring

A lightweight post-deployment monitoring framework was implemented to
detect changes in incoming data and model behaviour.

The monitoring reference population consists of the **70,000 customer
records used for model training**.

### Numerical feature drift

Numerical distributions are monitored using **Population Stability Index
(PSI)**.

            PSI Interpretation
  ------------- ----------------------
       `< 0.10` No significant drift
    `0.10–0.25` Moderate drift
      `>= 0.25` Significant drift

### Categorical feature drift

Categorical distributions are compared using **Total Variation
Distance**.

    TV Distance Interpretation
  ------------- ----------------------
       `< 0.05` No significant drift
    `0.05–0.10` Moderate drift
      `>= 0.10` Significant drift

### Missingness drift

Missing-value rates are monitored separately because changes in
missingness can indicate upstream data-quality problems.

              Absolute change Interpretation
  --------------------------- ----------------------
      `< 5 percentage points` No significant drift
     `5–10 percentage points` Moderate drift
    `>= 10 percentage points` Significant drift

### Prediction drift

The distribution of model predictions is compared against the reference
population using Total Variation Distance across Good, Poor, and
Standard predictions.

### Monitoring validation

Historical production observations were not available, so a controlled
synthetic production dataset was generated to validate the monitoring
framework.

The scenario introduced changes in outstanding debt, monthly EMI, credit
inquiries, payment delays, credit mix, payment behaviour, and
missingness.

The monitoring framework successfully detected the intentionally
introduced changes.

These results validate the monitoring implementation but **should not be
interpreted as evidence of actual production drift**.

### Production response

If meaningful drift were detected in a real deployment:

1.  Investigate affected features.
2.  Check upstream data pipelines and data-quality processes.
3.  Determine whether the shift represents a genuine population change.
4.  Monitor prediction distributions and model confidence.
5.  Evaluate model performance when ground-truth outcomes become
    available.
6.  Consider recalibration or retraining if persistent degradation is
    confirmed.

## Project Structure

``` text
CreditScoring_Kaggle/
│
├── assets/
│   └── architecture.png
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── monitoring_reference.pkl
│       ├── X_train_processed.pkl
│       ├── X_val_processed.pkl
│       ├── X_test_processed.pkl
│       ├── y_train.pkl
│       ├── y_val.pkl
│       ├── y_test.pkl
│       ├── feature_names.pkl
│       └── test_customer_ids.pkl
│
├── models/
│   ├── preprocessor.pkl
│   └── xgboost_credit_score.pkl
│
├── notebooks/
│   ├── 01_data_validation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_explainability.ipynb
│   ├── 04_deployment_testing.ipynb
│   └── 05_monitoring.ipynb
│
├── reports/
│   ├── categorical_drift_report.csv
│   ├── missingness_drift_report.csv
│   ├── missingness_drift.png
│   ├── monitoring_report.md
│   ├── monitoring_summary.csv
│   ├── numeric_drift_report.csv
│   ├── numeric_feature_drift.png
│   ├── prediction_distribution_drift.png
│   └── prediction_drift_report.csv
│
├── src/
│   ├── api.py
│   ├── explainability.py
│   ├── monitoring.py
│   └── preprocessing.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

``` bash
pip install -r requirements.txt
```

## Running the API

From the project root:

``` bash
uvicorn src.api:app --reload
```

The API is available locally at:

``` text
http://127.0.0.1:8000
```

Interactive documentation:

``` text
http://127.0.0.1:8000/docs
```

## Reproducibility

The project saves the key artifacts required to reproduce predictions
without retraining:

-   Preprocessing pipeline
-   Trained XGBoost model
-   Processed feature names
-   Processed evaluation datasets
-   Monitoring reference population

The customer-level split uses a fixed random seed and stratification.

## Limitations

1.  The dataset is a Kaggle dataset rather than a live institutional
    credit portfolio.
2.  Historical production data was not available for genuine
    post-deployment drift analysis.
3.  Monitoring validation uses controlled synthetic production data.
4.  Real production model-performance monitoring requires delayed
    ground-truth outcomes.
5.  SHAP values describe model behaviour and should not be interpreted
    as causal relationships.
6.  Further calibration and fairness analysis would be appropriate
    before using the model in a real lending decision process.

## Future Improvements

-   Automated scheduled monitoring
-   Production data ingestion
-   Automated drift alerting
-   Model-performance monitoring after labels become available
-   Probability calibration
-   Fairness and subgroup performance analysis
-   Model registry and version management
-   Automated retraining workflows
-   Containerized and cloud deployment
-   Monitoring dashboard

## Key Takeaways

This project demonstrates the ability to move beyond model training and
build a broader machine learning system combining:

-   Real-world data cleaning
-   Leakage-aware dataset design
-   Feature engineering
-   Supervised machine learning
-   Hyperparameter tuning
-   Class-imbalance analysis
-   Model evaluation
-   SHAP explainability
-   REST API deployment
-   Data-quality monitoring
-   Feature drift monitoring
-   Prediction drift monitoring

The result is an end-to-end credit-risk ML workflow designed with both
**model performance and production considerations** in mind.

## Technologies

-   Python
-   pandas
-   NumPy
-   scikit-learn
-   XGBoost
-   SHAP
-   FastAPI
-   Pydantic
-   Uvicorn
-   Streamlit
-   Joblib
-   Jupyter Notebook
-   Git / GitHub

## Portfolio Summary

**Credit Risk Classification & Monitoring System**

Developed an end-to-end credit risk classification system using XGBoost,
achieving **70.79% accuracy and 68.58% macro F1** on a customer-level
held-out test set. Implemented leakage-aware data splitting, feature
engineering, SHAP-based explainability, FastAPI deployment, and
post-deployment monitoring for feature drift, missingness, and
prediction-distribution changes.
