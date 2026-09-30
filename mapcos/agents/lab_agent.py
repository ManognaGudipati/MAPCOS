"""
Lab Agent
-----------
FSH, LH, AMH via gradient boosting (XGBoost).
"""

import pandas as pd
import xgboost as xgb


class LabAgent:
    def __init__(self, model_path: str = None, model: xgb.XGBClassifier = None):
        if model is not None:
            self.model = model
        else:
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)

    def run(self, lab_values: dict) -> dict:
        """
        lab_values: {"fsh": float, "lh": float, "amh": float}
        """
        fsh = lab_values["fsh"]
        lh = lab_values["lh"]
        amh = lab_values["amh"]
        fsh_lh_ratio = fsh / lh if lh != 0 else 0.0

        features = pd.DataFrame([{
            "fsh": fsh, "lh": lh, "fsh_lh_ratio": fsh_lh_ratio, "amh": amh
        }])

        prob = float(self.model.predict_proba(features)[0][1])
        return {
            "agent": "lab",
            "pcos_probability": prob,
            "predicted_label": "positive" if prob >= 0.5 else "negative",
            "fsh_lh_ratio": fsh_lh_ratio,
        }
