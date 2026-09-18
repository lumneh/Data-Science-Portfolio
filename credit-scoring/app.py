import sys
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Credit Score Classifier", page_icon="💳", layout="wide")

st.markdown("""
<style>
.main-title {font-size:2.6rem;font-weight:700;margin-bottom:.2rem;}
.subtitle {font-size:1.05rem;color:#666;margin-bottom:1.5rem;}
.prediction-box {padding:1.5rem;border-radius:12px;border:1px solid #ddd;text-align:center;}
.prediction-label {font-size:.9rem;color:#666;}
.prediction-value {font-size:2.2rem;font-weight:700;}
.section-title {font-size:1.35rem;font-weight:650;margin-top:1rem;margin-bottom:.7rem;}
</style>
""", unsafe_allow_html=True)

def load_artifacts():
    model = joblib.load(MODELS_DIR / "xgboost_credit_score.pkl")
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.pkl")
    explainer = None
    try:
        from src.explainability import CreditScoreExplainer
        explainer = CreditScoreExplainer(
            model=model,
            feature_names=feature_names,
            numeric_features=list(preprocessor.transformers_[0][2]),
            categorical_features=list(preprocessor.transformers_[1][2]),
        )
    except Exception:
        pass
    return model, preprocessor, feature_names, explainer

try:
    model, preprocessor, feature_names, explainer = load_artifacts()
except Exception as e:
    st.error("The model artifacts could not be loaded.")
    st.exception(e)
    st.stop()

st.markdown('<div class="main-title">💳 Credit Score Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">An explainable machine-learning application for credit risk classification.</div>', unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Model", "XGBoost")
m2.metric("Test Accuracy", "70.79%")
m3.metric("Macro F1", "68.58%")
m4.metric("Classes", "3")
st.caption("Classifies customers as Good, Standard, or Poor using financial, credit-history, loan, and payment-behaviour features.")
st.divider()

st.sidebar.header("Demo profile")
profile = st.sidebar.selectbox("Choose a profile", ["Custom", "Standard demo", "Good demo", "Poor demo"])
st.sidebar.markdown("---")
st.sidebar.subheader("About the model")
st.sidebar.write("XGBoost multiclass classifier with customer-level splitting to reduce data leakage.")
st.sidebar.caption("Portfolio project • Explainable ML • FastAPI + Streamlit")

base = {
"Month":"January","Age":30,"Occupation":"Engineer","Annual_Income":50000.0,
"Monthly_Inhand_Salary":4000.0,"Num_Bank_Accounts":4,"Num_Credit_Card":4,
"Interest_Rate":15,"Num_of_Loan":3,"Delay_from_due_date":10,
"Num_of_Delayed_Payment":5,"Changed_Credit_Limit":10.0,"Num_Credit_Inquiries":3,
"Outstanding_Debt":5000.0,"Credit_Utilization_Ratio":30.0,"Total_EMI_per_month":300.0,
"Amount_invested_monthly":500.0,"Monthly_Balance":3000.0,"Credit_History_Months":60,
"Num_Loan_Types":3,"Has_auto_loan":0,"Has_credit_builder_loan":0,"Has_personal_loan":1,
"Has_home_equity_loan":0,"Has_mortgage_loan":0,"Has_student_loan":0,
"Has_debt_consolidation_loan":0,"Has_payday_loan":0,"Credit_Mix":"Standard",
"Payment_of_Min_Amount":"Yes","Payment_Behaviour":"High_spent_Medium_value_payments"
}
demo = base.copy()
if profile == "Good demo":
    demo.update({"Age":35,"Annual_Income":80000.0,"Num_Bank_Accounts":3,"Num_Credit_Card":3,
                 "Interest_Rate":8,"Delay_from_due_date":2,"Num_of_Delayed_Payment":1,
                 "Num_Credit_Inquiries":1,"Outstanding_Debt":1500.0,
                 "Credit_Utilization_Ratio":15.0,"Credit_Mix":"Good"})
elif profile == "Poor demo":
    demo.update({"Age":28,"Annual_Income":30000.0,"Num_Bank_Accounts":7,"Num_Credit_Card":7,
                 "Interest_Rate":28,"Delay_from_due_date":35,"Num_of_Delayed_Payment":15,
                 "Num_Credit_Inquiries":10,"Outstanding_Debt":15000.0,
                 "Credit_Utilization_Ratio":75.0,"Credit_Mix":"Bad",
                 "Payment_of_Min_Amount":"No","Payment_Behaviour":"Low_spent_Large_value_payments"})

st.markdown('<div class="section-title">1. Customer information</div>', unsafe_allow_html=True)
st.info("Choose a demo profile in the sidebar or select Custom and enter customer information.")

with st.form("credit_score_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        months=["January","February","March","April","May","June","July","August","September","October","November","December"]
        month=st.selectbox("Month",months,index=months.index(demo["Month"]))
        age=st.number_input("Age",18,100,int(demo["Age"]))
        occupations=["_______","Accountant","Architect","Developer","Doctor","Engineer","Entrepreneur","Journalist","Lawyer","Manager","Mechanic","Media_Manager","Musician","Scientist","Teacher","Writer"]
        occupation=st.selectbox("Occupation",occupations,index=occupations.index(demo["Occupation"]) if demo["Occupation"] in occupations else 0)
        annual_income=st.number_input("Annual Income",min_value=0.0,value=float(demo["Annual_Income"]),step=1000.0)
        monthly_salary=st.number_input("Monthly Inhand Salary",min_value=0.0,value=float(demo["Monthly_Inhand_Salary"]),step=100.0)
        bank=st.number_input("Number of Bank Accounts",min_value=0,value=int(demo["Num_Bank_Accounts"]))
        cards=st.number_input("Number of Credit Cards",min_value=0,value=int(demo["Num_Credit_Card"]))
        interest=st.number_input("Interest Rate",min_value=0,value=int(demo["Interest_Rate"]))
        loans=st.number_input("Number of Loans",min_value=0,value=int(demo["Num_of_Loan"]))
        delay=st.number_input("Delay From Due Date",min_value=0,value=int(demo["Delay_from_due_date"]))
    with c2:
        delayed=st.number_input("Number of Delayed Payments",min_value=0,value=int(demo["Num_of_Delayed_Payment"]))
        limit_change=st.number_input("Changed Credit Limit",min_value=0.0,value=float(demo["Changed_Credit_Limit"]),step=1.0)
        inquiries=st.number_input("Number of Credit Inquiries",min_value=0,value=int(demo["Num_Credit_Inquiries"]))
        debt=st.number_input("Outstanding Debt",min_value=0.0,value=float(demo["Outstanding_Debt"]),step=100.0)
        utilization=st.number_input("Credit Utilization Ratio",min_value=0.0,value=float(demo["Credit_Utilization_Ratio"]),step=1.0)
        emi=st.number_input("Total EMI Per Month",min_value=0.0,value=float(demo["Total_EMI_per_month"]),step=50.0)
        invested=st.number_input("Amount Invested Monthly",min_value=0.0,value=float(demo["Amount_invested_monthly"]),step=50.0)
        balance=st.number_input("Monthly Balance",min_value=0.0,value=float(demo["Monthly_Balance"]),step=100.0)
        history=st.number_input("Credit History (Months)",min_value=0,value=int(demo["Credit_History_Months"]))
        loan_types=st.number_input("Number of Loan Types",min_value=0,value=int(demo["Num_Loan_Types"]))
        mixes=["Bad","Good","Standard","_"]
        credit_mix=st.selectbox("Credit Mix",mixes,index=mixes.index(demo["Credit_Mix"]))
    with c3:
        mins=["Yes","No","NM"]
        payment_min=st.selectbox("Payment of Minimum Amount",mins,index=mins.index(demo["Payment_of_Min_Amount"]))
        behaviours=["High_spent_Large_value_payments","High_spent_Medium_value_payments","High_spent_Small_value_payments","Low_spent_Large_value_payments","Low_spent_Medium_value_payments","Low_spent_Small_value_payments"]
        behaviour=st.selectbox("Payment Behaviour",behaviours,index=behaviours.index(demo["Payment_Behaviour"]))
        st.markdown("**Loan indicators**")
        auto=st.checkbox("Auto loan",bool(demo["Has_auto_loan"]))
        builder=st.checkbox("Credit builder loan",bool(demo["Has_credit_builder_loan"]))
        personal=st.checkbox("Personal loan",bool(demo["Has_personal_loan"]))
        home=st.checkbox("Home equity loan",bool(demo["Has_home_equity_loan"]))
        mortgage=st.checkbox("Mortgage loan",bool(demo["Has_mortgage_loan"]))
        student=st.checkbox("Student loan",bool(demo["Has_student_loan"]))
        consolidation=st.checkbox("Debt consolidation loan",bool(demo["Has_debt_consolidation_loan"]))
        payday=st.checkbox("Payday loan",bool(demo["Has_payday_loan"]))
    submitted=st.form_submit_button("Predict Credit Score",type="primary",use_container_width=True)

if submitted:
    data={"Month":month,"Age":age,"Occupation":occupation,"Annual_Income":annual_income,
    "Monthly_Inhand_Salary":monthly_salary,"Num_Bank_Accounts":bank,"Num_Credit_Card":cards,
    "Interest_Rate":interest,"Num_of_Loan":loans,"Delay_from_due_date":delay,
    "Num_of_Delayed_Payment":delayed,"Changed_Credit_Limit":limit_change,
    "Num_Credit_Inquiries":inquiries,"Outstanding_Debt":debt,"Credit_Utilization_Ratio":utilization,
    "Total_EMI_per_month":emi,"Amount_invested_monthly":invested,"Monthly_Balance":balance,
    "Credit_History_Months":history,"Num_Loan_Types":loan_types,"Has_auto_loan":int(auto),
    "Has_credit_builder_loan":int(builder),"Has_personal_loan":int(personal),
    "Has_home_equity_loan":int(home),"Has_mortgage_loan":int(mortgage),
    "Has_student_loan":int(student),"Has_debt_consolidation_loan":int(consolidation),
    "Has_payday_loan":int(payday),"Credit_Mix":credit_mix,
    "Payment_of_Min_Amount":payment_min,"Payment_Behaviour":behaviour}
    customer_df=pd.DataFrame([data])
    processed=preprocessor.transform(customer_df)
    prediction=model.predict(processed)[0]
    probs=model.predict_proba(processed)[0]
    probability_dict=dict(zip(model.classes_,probs))

    st.divider()
    st.markdown('<div class="section-title">2. Prediction result</div>',unsafe_allow_html=True)
    r,p=st.columns([1,2])
    with r:
        st.markdown(f'<div class="prediction-box"><div class="prediction-label">Predicted Credit Score</div><div class="prediction-value">{prediction}</div></div>',unsafe_allow_html=True)
    with p:
        pdf=pd.DataFrame({"Credit Score":list(probability_dict.keys()),"Probability (%)":[round(x*100,2) for x in probability_dict.values()]})
        st.markdown("**Prediction probabilities**")
        st.bar_chart(pdf.set_index("Credit Score")["Probability (%)"])
        st.dataframe(pdf,hide_index=True,use_container_width=True)

    st.markdown('<div class="section-title">3. Why did the model make this prediction?</div>',unsafe_allow_html=True)
    if explainer is not None:
        try:
            result=explainer.explain_observation(X_processed=processed,X_original=customer_df)
            explanation=result["explanation"].copy()
            if "abs_shap" in explanation.columns:
                explanation=explanation.sort_values("abs_shap",ascending=False).head(10)
            st.dataframe(explanation,hide_index=True,use_container_width=True)
            st.caption("SHAP identifies the features contributing most strongly to the individual prediction.")
        except Exception:
            st.info("SHAP explanations are temporarily unavailable in this environment. The prediction and probability outputs remain available.")
    else:
        st.info("SHAP explanations are temporarily unavailable in this environment. The prediction and probability outputs remain available.")

    with st.expander("About this model"):
        st.write("""
**Model:** XGBoost multiclass classifier

**Target:** Credit Score — Good, Standard, Poor

**Test accuracy:** 70.79%

**Test macro F1:** 68.58%

**Explainability:** SHAP TreeExplainer

**Deployment:** FastAPI API + Streamlit interface

**Monitoring:** Numeric drift, categorical drift, missingness, and prediction drift using PSI and total variation distance.

The training workflow uses a customer-level train/validation/test split so records from the same customer do not appear across different datasets.
""")

st.divider()
st.markdown(
    "*Demo only: this application is for demonstrating the ML system "
    "and is not a production lending decision tool.*"
)