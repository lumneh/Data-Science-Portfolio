import numpy as np
import pandas as pd


class CreditScoreMonitor:
    """
    Monitoring utilities for the Credit Score Classification model.

    The monitor compares a reference population against current
    production data using:

    1. Population Stability Index (PSI) for numerical features
    2. Total Variation Distance (TV) for categorical features
    3. Missingness-rate changes
    4. Prediction-distribution drift
    """

    def __init__(self, reference_data):
        """
        Parameters
        ----------
        reference_data : pandas.DataFrame
            Reference population used to establish baseline
            feature distributions.
        """
        self.reference_data = reference_data.copy()

    # ---------------------------------------------------------
    # PSI
    # ---------------------------------------------------------

    @staticmethod
    def calculate_psi(reference, current, bins=10):
        """
        Calculate Population Stability Index (PSI).

        PSI interpretation:
            < 0.10       No significant drift
            0.10 - 0.25  Moderate drift
            >= 0.25      Significant drift
        """

        reference = pd.Series(reference).dropna()
        current = pd.Series(current).dropna()

        if len(reference) == 0 or len(current) == 0:
            return np.nan

        # Constant feature
        if reference.nunique() <= 1:
            return 0.0

        # Quantile-based bins using reference population
        breakpoints = np.percentile(
            reference,
            np.linspace(0, 100, bins + 1)
        )

        breakpoints = np.unique(breakpoints)

        # Not enough unique values to construct bins
        if len(breakpoints) < 3:
            return 0.0

        # Ensure future values outside the reference range
        # are still included
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        reference_binned = pd.cut(
            reference,
            bins=breakpoints,
            include_lowest=True
        )

        current_binned = pd.cut(
            current,
            bins=breakpoints,
            include_lowest=True
        )

        reference_dist = (
            reference_binned
            .value_counts(normalize=True, sort=False)
        )

        current_dist = (
            current_binned
            .value_counts(normalize=True, sort=False)
        )

        # Avoid log(0)
        epsilon = 1e-6

        reference_dist = reference_dist + epsilon
        current_dist = current_dist + epsilon

        psi = np.sum(
            (current_dist - reference_dist)
            * np.log(current_dist / reference_dist)
        )

        return float(psi)

    @staticmethod
    def interpret_psi(psi):
        """Interpret PSI value."""

        if pd.isna(psi):
            return "Unavailable"

        if psi < 0.10:
            return "No significant drift"

        if psi < 0.25:
            return "Moderate drift"

        return "Significant drift"

    # ---------------------------------------------------------
    # Numeric drift
    # ---------------------------------------------------------

    def check_numeric_drift(
        self,
        current_data,
        numeric_features
    ):
        """
        Calculate PSI for numerical features.
        """

        results = []

        for feature in numeric_features:

            psi = self.calculate_psi(
                self.reference_data[feature],
                current_data[feature]
            )

            results.append({
                "Feature": feature,
                "PSI": psi,
                "Interpretation": self.interpret_psi(psi)
            })

        return (
            pd.DataFrame(results)
            .sort_values("PSI", ascending=False)
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # Categorical drift
    # ---------------------------------------------------------

    @staticmethod
    def calculate_tv_distance(reference, current):
        """
        Calculate Total Variation Distance between
        two categorical distributions.
        """

        reference = (
            pd.Series(reference)
            .fillna("__MISSING__")
        )

        current = (
            pd.Series(current)
            .fillna("__MISSING__")
        )

        categories = (
            set(reference.unique())
            | set(current.unique())
        )

        reference_dist = (
            reference
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
        )

        current_dist = (
            current
            .value_counts(normalize=True)
            .reindex(categories, fill_value=0)
        )

        tv_distance = 0.5 * np.abs(
            reference_dist - current_dist
        ).sum()

        return float(tv_distance)

    @staticmethod
    def interpret_tv(tv_distance):
        """Interpret categorical drift."""

        if pd.isna(tv_distance):
            return "Unavailable"

        if tv_distance < 0.05:
            return "No significant drift"

        if tv_distance < 0.10:
            return "Moderate drift"

        return "Significant drift"

    def check_categorical_drift(
        self,
        current_data,
        categorical_features
    ):
        """
        Calculate Total Variation Distance
        for categorical features.
        """

        results = []

        for feature in categorical_features:

            tv_distance = self.calculate_tv_distance(
                self.reference_data[feature],
                current_data[feature]
            )

            results.append({
                "Feature": feature,
                "TV_Distance": tv_distance,
                "Interpretation": self.interpret_tv(
                    tv_distance
                )
            })

        return (
            pd.DataFrame(results)
            .sort_values(
                "TV_Distance",
                ascending=False
            )
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # Missingness monitoring
    # ---------------------------------------------------------

    @staticmethod
    def calculate_missingness(data, features):
        """
        Calculate missing-value percentage
        for each feature.
        """

        return (
            data[features]
            .isna()
            .mean()
            .mul(100)
            .sort_values(ascending=False)
        )

    def check_missingness_drift(
        self,
        current_data,
        features
    ):
        """
        Compare missingness rates between
        reference and current data.
        """

        reference_missingness = self.calculate_missingness(
            self.reference_data,
            features
        )

        current_missingness = self.calculate_missingness(
            current_data,
            features
        )

        results = pd.DataFrame({
            "Feature": features,
            "Reference_Missing_%": [
                reference_missingness.get(
                    feature, np.nan
                )
                for feature in features
            ],
            "Current_Missing_%": [
                current_missingness.get(
                    feature, np.nan
                )
                for feature in features
            ]
        })

        results["Change_pp"] = (
            results["Current_Missing_%"]
            - results["Reference_Missing_%"]
        )

        results["Absolute_Change_pp"] = (
            results["Change_pp"].abs()
        )

        results["Interpretation"] = results[
            "Absolute_Change_pp"
        ].apply(
            lambda x:
                "No significant drift"
                if x < 5
                else (
                    "Moderate drift"
                    if x < 10
                    else "Significant drift"
                )
        )

        return (
            results
            .sort_values(
                "Absolute_Change_pp",
                ascending=False
            )
            .reset_index(drop=True)
        )

    # ---------------------------------------------------------
    # Prediction drift
    # ---------------------------------------------------------

    @staticmethod
    def check_prediction_drift(
        reference_predictions,
        current_predictions,
        classes
    ):
        """
        Calculate prediction-distribution drift
        using Total Variation Distance.
        """

        reference_dist = (
            pd.Series(reference_predictions)
            .value_counts(normalize=True)
            .reindex(classes, fill_value=0)
        )

        current_dist = (
            pd.Series(current_predictions)
            .value_counts(normalize=True)
            .reindex(classes, fill_value=0)
        )

        tv_distance = 0.5 * np.abs(
            reference_dist - current_dist
        ).sum()

        result = pd.DataFrame({
            "Class": classes,
            "Reference_%": (
                reference_dist.values * 100
            ),
            "Current_%": (
                current_dist.values * 100
            )
        })

        return {
            "tv_distance": float(tv_distance),
            "interpretation": (
                "No significant drift"
                if tv_distance < 0.05
                else (
                    "Moderate drift"
                    if tv_distance < 0.10
                    else "Significant drift"
                )
            ),
            "distribution": result
        }

    # ---------------------------------------------------------
    # Overall summary
    # ---------------------------------------------------------

    def generate_summary(
        self,
        numeric_drift,
        categorical_drift,
        missingness_drift,
        prediction_drift
    ):
        """
        Generate a high-level monitoring summary.
        """

        numeric_significant = (
            numeric_drift["Interpretation"]
            == "Significant drift"
        ).sum()

        numeric_moderate = (
            numeric_drift["Interpretation"]
            == "Moderate drift"
        ).sum()

        categorical_significant = (
            categorical_drift["Interpretation"]
            == "Significant drift"
        ).sum()

        categorical_moderate = (
            categorical_drift["Interpretation"]
            == "Moderate drift"
        ).sum()

        missingness_significant = (
            missingness_drift["Interpretation"]
            == "Significant drift"
        ).sum()

        missingness_moderate = (
            missingness_drift["Interpretation"]
            == "Moderate drift"
        ).sum()

        prediction_status = (
            prediction_drift["interpretation"]
        )

        return pd.DataFrame({
            "Monitoring_Signal": [
                "Numeric feature drift",
                "Categorical feature drift",
                "Missingness drift",
                "Prediction drift"
            ],
            "Significant_Drift_Count": [
                numeric_significant,
                categorical_significant,
                missingness_significant,
                int(
                    prediction_status
                    == "Significant drift"
                )
            ],
            "Moderate_Drift_Count": [
                numeric_moderate,
                categorical_moderate,
                missingness_moderate,
                int(
                    prediction_status
                    == "Moderate drift"
                )
            ],
            "Overall_Status": [
                (
                    "Significant drift"
                    if numeric_significant > 0
                    else (
                        "Moderate drift"
                        if numeric_moderate > 0
                        else "No significant drift"
                    )
                ),
                (
                    "Significant drift"
                    if categorical_significant > 0
                    else (
                        "Moderate drift"
                        if categorical_moderate > 0
                        else "No significant drift"
                    )
                ),
                (
                    "Significant drift"
                    if missingness_significant > 0
                    else (
                        "Moderate drift"
                        if missingness_moderate > 0
                        else "No significant drift"
                    )
                ),
                prediction_status
            ]
        })