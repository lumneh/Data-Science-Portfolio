import joblib
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from src.explainability import CreditScoreExplainer

class CustomerInput(BaseModel):
    Age: Optional[float] = None
    Annual_Income: Optional[float] = None
    Monthly_Inhand_Salary: Optional[float] = None
    Num_Bank_Accounts: Optional[float] = None
    Num_Credit_Card: Optional[float] = None
    Interest_Rate: Optional[float] = None
    Num_of_Loan: Optional[float] = None
    Delay_from_due_date: Optional[float] = None
    Num_of_Delayed_Payment: Optional[float] = None
    Changed_Credit_Limit: Optional[float] = None
    Num_Credit_Inquiries: Optional[float] = None
    Outstanding_Debt: Optional[float] = None
    Credit_Utilization_Ratio: Optional[float] = None
    Total_EMI_per_month: Optional[float] = None
    Amount_invested_monthly: Optional[float] = None
    Monthly_Balance: Optional[float] = None
    Credit_History_Months: Optional[float] = None
    Num_Loan_Types: Optional[float] = None

    Has_auto_loan: Optional[int] = None
    Has_credit_builder_loan: Optional[int] = None
    Has_personal_loan: Optional[int] = None
    Has_home_equity_loan: Optional[int] = None
    Has_mortgage_loan: Optional[int] = None
    Has_student_loan: Optional[int] = None
    Has_debt_consolidation_loan: Optional[int] = None
    Has_payday_loan: Optional[int] = None

    Month: Optional[str] = None
    Occupation: Optional[str] = None
    Credit_Mix: Optional[str] = None
    Payment_of_Min_Amount: Optional[str] = None
    Payment_Behaviour: Optional[str] = None

# -------------------------
# Project paths
# -------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# -------------------------
# Load saved artifacts
# -------------------------

model = joblib.load(
    MODELS_DIR / "xgboost_credit_score.pkl"
)

preprocessor = joblib.load(
    MODELS_DIR / "preprocessor.pkl"
)

feature_names = joblib.load(
    PROCESSED_DIR / "feature_names.pkl"
)

# -------------------------
# Create SHAP explainer
# -------------------------

explainer = CreditScoreExplainer(
    model=model,
    feature_names=feature_names,
    numeric_features=preprocessor.transformers_[0][2],
    categorical_features=preprocessor.transformers_[1][2]
)

# -------------------------
# Create FastAPI application
# -------------------------

app = FastAPI(
    title="Credit Score Classification API",
    description="API for predicting customer credit-score categories.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Credit Score Classification API is running."
    }
# -------------------------
# Health check endpoint
# -------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "xgboost_credit_score",
        "version": app.version
    }

# -------------------------
# Prediction endpoint
# -------------------------

@app.post("/predict")
def predict_credit_score(customer_data: CustomerInput):

    import pandas as pd

    # Convert validated request to DataFrame
    customer_df = pd.DataFrame([customer_data.model_dump()])

    # Transform raw data
    processed_data = preprocessor.transform(customer_df)

    # Generate prediction
    prediction = model.predict(processed_data)[0]

    # Generate probabilities
    probabilities = model.predict_proba(processed_data)[0]

    # Generate SHAP explanation
    explanation_result = explainer.explain_observation(
        X_processed=processed_data,
        original_row=customer_df,
        observation_index=0,
        top_n=10
    )

    # Convert explanation DataFrame to JSON-compatible format
    explanation = explanation_result["explanation"].copy()

    explanation_records = []

    for _, row in explanation.iterrows():

        value = row["Value"]

        # Convert NumPy numeric values to native Python types
        if hasattr(value, "item"):
            value = value.item()

        category = row["Category"]

        # Convert missing category values to JSON null
        if pd.isna(category):
            category = None

        explanation_records.append({
            "feature": str(row["Feature"]),
            "category": category,
            "value": value,
            "shap": float(row["SHAP"]),
            "direction": str(row["Direction"])
        })

    return {
        "prediction": prediction,
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(
                model.classes_,
                probabilities
            )
        },
        "explanation": explanation_records
    }