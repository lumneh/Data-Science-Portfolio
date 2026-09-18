import numpy as np
import pandas as pd
import shap


class CreditScoreExplainer:

    def __init__(
        self,
        model,
        feature_names,
        numeric_features,
        categorical_features
    ):
        self.model = model
        self.feature_names = list(feature_names)
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

        self.class_to_index = {
            class_name: i
            for i, class_name in enumerate(model.classes_)
        }

        self.explainer = shap.TreeExplainer(model)

    def get_shap_values(self, X):

        return self.explainer.shap_values(X)

    def explain_observation(
        self,
        X_processed,
        original_row,
        observation_index=0,
        top_n=10
    ):

        prediction = self.model.predict(
            X_processed
        )[observation_index]

        probabilities = self.model.predict_proba(
            X_processed
        )[observation_index]

        confidence = probabilities.max()

        class_index = self.class_to_index[
            prediction
        ]

        shap_values = self.explainer.shap_values(
            X_processed
        )

        shap_vector = shap_values[
            class_index
        ][observation_index]

        rows = []

        # -------------------------
        # Numeric features
        # -------------------------

        for feature in self.numeric_features:

            transformed_name = (
                f"numeric__{feature}"
            )

            if transformed_name in self.feature_names:

                idx = self.feature_names.index(
                    transformed_name
                )

                rows.append({
                    "Feature": feature,
                    "Category": None,
                    "Value": original_row.iloc[observation_index][feature],
                    "SHAP": shap_vector[idx]
                })

        # -------------------------
        # Categorical features
        # -------------------------

        for feature in self.categorical_features:

            prefix = (
                f"categorical__{feature}_"
            )

            actual_value = original_row.iloc[observation_index][feature]

            for idx, transformed_name in enumerate(
                self.feature_names
            ):

                if transformed_name.startswith(
                    prefix
                ):

                    category = transformed_name.replace(
                        prefix,
                        ""
                    )

                    if str(actual_value) == str(
                        category
                    ):

                        rows.append({
                            "Feature": feature,
                            "Category": category,
                            "Value": actual_value,
                            "SHAP": shap_vector[idx]
                        })

        explanation = pd.DataFrame(rows)

        explanation["Abs_SHAP"] = (
            explanation["SHAP"].abs()
        )

        explanation["Direction"] = np.where(
            explanation["SHAP"] > 0,
            f"Toward {prediction}",
            f"Away from {prediction}"
        )

        explanation = (
            explanation
            .sort_values(
                "Abs_SHAP",
                ascending=False
            )
            .head(top_n)
        )

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": dict(
                zip(
                    self.model.classes_,
                    probabilities
                )
            ),
            "explanation": explanation
        }
    
    def interpret_shap_strength(self, shap_value):
        magnitude = abs(shap_value)

        if magnitude >= 0.30:
            return "Strong"

        elif magnitude >= 0.10:
            return "Moderate"

        elif magnitude >= 0.03:
            return "Small"

        else:
            return "Very Small"