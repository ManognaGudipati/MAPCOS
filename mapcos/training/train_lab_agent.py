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
    fsh = pd.to_numeric(df[LAB_RAW_COLUMNS["fsh"]], errors="coerce")
    lh = pd.to_numeric(df[LAB_RAW_COLUMNS["lh"]], errors="coerce")
    amh = pd.to_numeric(df[LAB_RAW_COLUMNS["amh"]], errors="coerce")
    fsh_lh_ratio = fsh / lh.replace(0, 1e-6)
    return pd.DataFrame({"fsh": fsh, "lh": lh, "fsh_lh_ratio": fsh_lh_ratio, "amh": amh})
    
def main():
    df = pd.read_excel(TABULAR_XLSX_PATH, sheet_name=TABULAR_SHEET_NAME)

    X = build_features(df)
    y = df[TARGET_COLUMN]

    valid_rows = X.notna().all(axis=1)
    dropped = (~valid_rows).sum()
    if dropped > 0:
        print(f"Dropping {dropped} rows with non-numeric lab values")
    X, y = X[valid_rows], y[valid_rows]

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
