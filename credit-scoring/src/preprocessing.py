
import pandas as pd
import numpy as np


NUMERIC_COLUMNS = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Delay_from_due_date",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance",
]


def clean_numeric_columns(df):
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("_", "", regex=False)
                .str.strip()
            )

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    return df


def clean_categorical_columns(df):
    df = df.copy()

    df["Occupation"] = df["Occupation"].replace(
        "_______",
        np.nan
    )

    df["Credit_Mix"] = df["Credit_Mix"].replace(
        "_",
        np.nan
    )

    df["Payment_Behaviour"] = df["Payment_Behaviour"].replace(
        "!@9#%8",
        np.nan
    )

    return df


def clean_age(df):
    df = df.copy()

    df["Age"] = pd.to_numeric(
        df["Age"],
        errors="coerce"
    )

    df.loc[
        (df["Age"] < 18) | (df["Age"] > 100),
        "Age"
    ] = np.nan

    return df


def convert_credit_history_age(df):
    df = df.copy()

    pattern = (
        r"(?P<years>\d+)\s+Years?\s+and\s+"
        r"(?P<months>\d+)\s+Months?"
    )

    extracted = df["Credit_History_Age"].str.extract(pattern)

    df["Credit_History_Months"] = (
        pd.to_numeric(extracted["years"], errors="coerce") * 12
        + pd.to_numeric(extracted["months"], errors="coerce")
    )

    df = df.drop(columns=["Credit_History_Age"])

    return df


def engineer_loan_features(df):
    df = df.copy()

    loan_col = df["Type_of_Loan"].fillna("")

    df["Num_Loan_Types"] = loan_col.apply(
        lambda x: 0 if x == "" else len(
            [loan.strip() for loan in x.split(",")]
        )
    )

    loan_types = [
        "Auto Loan",
        "Credit-Builder Loan",
        "Personal Loan",
        "Home Equity Loan",
        "Mortgage Loan",
        "Student Loan",
        "Debt Consolidation Loan",
        "Payday Loan",
    ]

    for loan_type in loan_types:
        feature_name = (
            "Has_"
            + loan_type.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        df[feature_name] = (
            loan_col
            .str.contains(
                loan_type,
                case=False,
                regex=False
            )
            .astype(int)
        )

    df = df.drop(columns=["Type_of_Loan"])

    return df


def clean_raw_data(df):
    df = df.copy()

    df = clean_numeric_columns(df)
    df = clean_categorical_columns(df)
    df = clean_age(df)
    df = convert_credit_history_age(df)
    df = engineer_loan_features(df)

    return df
