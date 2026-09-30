"""
Symptoms Agent
----------------
Cycle regularity, hirsutism, BMI (+ related symptom flags) via XGBoost,
plus a simple rule-based dampener for cases with no androgenic signs at all.
"""

import pandas as pd
import xgboost as xgb


class SymptomsAgent:
    def __init__(self, model_path: str = None, model: xgb.XGBClassifier = None):
        if model is not None:
            self.model = model
        else:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)

    def run(self, symptoms: dict) -> dict:
        """
        symptoms: {
            "bmi": float,
            "cycle_irregular": 0 or 1,
            "hair_growth": 0 or 1,
            "weight_gain": 0 or 1,
            "skin_darkening": 0 or 1,
            "hair_loss": 0 or 1,
            "pimples": 0 or 1,
        }
        """
        features = pd.DataFrame([symptoms])
        prob = float(self.model.predict_proba(features)[0][1])

        # Rule-based dampener: with zero androgenic signs (hair growth, skin
        # darkening, weight gain all absent), cap confidence rather than trust
        # the model outright — a simple stand-in for the diagram's "+ rules" step.
        no_androgenic_signs = (
            symptoms.get("hair_growth", 0) == 0 and
            symptoms.get("skin_darkening", 0) == 0 and
            symptoms.get("weight_gain", 0) == 0
        )
        rule_applied = False
        if no_androgenic_signs and prob > 0.4:
            prob = 0.4
            rule_applied = True

        return {
            "agent": "symptoms",
            "pcos_probability": prob,
            "predicted_label": "positive" if prob >= 0.5 else "negative",
            "rule_applied": rule_applied,
        }
