"""
Trains the Symptoms Agent's XGBoost model on cycle regularity, hirsutism, BMI,
and related symptom flags.
Run: python -m mapcos.training.train_symptoms_agent
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from mapcos.config import (
    TABULAR_XLSX_PATH, TABULAR_SHEET_NAME, TARGET_COLUMN,
    SYMPTOMS_RAW_COLUMNS, SYMPTOMS_MODEL_PATH
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c = SYMPTOMS_RAW_COLUMNS
    cycle_irregular = (df[c["cycle_code"]] != 2).astype(int)  # 2=Regular; treat 4 and stray 5 as Irregular
    return pd.DataFrame({
        "bmi": df[c["bmi"]],
        "cycle_irregular": cycle_irregular,
        "hair_growth": df[c["hair_growth"]],
        "weight_gain": df[c["weight_gain"]],
        "skin_darkening": df[c["skin_darkening"]],
        "hair_loss": df[c["hair_loss"]],
        "pimples": df[c["pimples"]],
    })


def main():
    df = pd.read_excel(TABULAR_XLSX_PATH, sheet_name=TABULAR_SHEET_NAME)

    X = build_features(df)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"Test accuracy : {accuracy_score(y_test, preds):.3f}")
    print(f"Test AUC      : {roc_auc_score(y_test, probs):.3f}")

    model.save_model(SYMPTOMS_MODEL_PATH)
    print(f"Model saved to {SYMPTOMS_MODEL_PATH}")


if __name__ == "__main__":
    main()
