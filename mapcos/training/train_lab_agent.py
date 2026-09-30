"""
Trains the Lab Agent's XGBoost model on FSH, LH, AMH.
Run: python -m mapcos.training.train_lab_agent
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from mapcos.config import (
    TABULAR_XLSX_PATH, TABULAR_SHEET_NAME, TARGET_COLUMN,
    LAB_RAW_COLUMNS, LAB_MODEL_PATH
)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    fsh = df[LAB_RAW_COLUMNS["fsh"]]
    lh = df[LAB_RAW_COLUMNS["lh"]]
    amh = df[LAB_RAW_COLUMNS["amh"]]
    fsh_lh_ratio = fsh / lh.replace(0, 1e-6)  # avoid divide-by-zero
    return pd.DataFrame({"fsh": fsh, "lh": lh, "fsh_lh_ratio": fsh_lh_ratio, "amh": amh})


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

    model.save_model(LAB_MODEL_PATH)
    print(f"Model saved to {LAB_MODEL_PATH}")


if __name__ == "__main__":
    main()
