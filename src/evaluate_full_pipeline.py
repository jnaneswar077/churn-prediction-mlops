import os
import sys
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)


def run_full_pipeline_evaluation():
    print("[INFO] Starting Lab 7 Stage 4: End-to-End Pipeline Evaluation...")

    pipeline_path = 'models/churn_full_pipeline.pkl'
    data_path = 'data/raw/churn.csv'

    if not os.path.exists(pipeline_path):
        print(f"[ERROR] Unified pipeline not found at: {pipeline_path}")
        print("[ERROR] Run src/train_full_pipeline.py first (Stage 2).")
        return False

    if not os.path.exists(data_path):
        print(f"[ERROR] Raw dataset not found at: {data_path}")
        return False

    # 1. Load the unified pipeline (preprocessing + model, one object)
    pipeline = joblib.load(pipeline_path)

    # 2. Rebuild the SAME raw test split used in train_full_pipeline.py
    #    (identical cleaning rules and split parameters -> identical rows)
    df = pd.read_csv(data_path)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Predict directly on RAW rows -- no manual preprocessing step.
    #    This is the actual point of Stage 4: proving the full raw-to-
    #    prediction path works, not just the classifier in isolation.
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # 4. Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\n--- Lab 7: Unified Pipeline Evaluation Report (raw-input, end-to-end) ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("--------------------------------------------------------------------------\n")

    # 5. Persist the report (this is what finally gives reports/ real content)
    os.makedirs('reports', exist_ok=True)
    report = {
        "stage": "Lab 7 - Stage 4: End-to-End Pipeline Evaluation",
        "pipeline_artifact": pipeline_path,
        "evaluated_on": "raw, untransformed test rows",
        "test_rows": int(X_test.shape[0]),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc_auc), 4)
        },
        "confusion_matrix": cm.tolist()
    }
    report_path = 'reports/lab7_full_pipeline_evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"[INFO] Evaluation report saved to {report_path}")

    # 6. Error analysis export (kept separate from Lab 2's files so nothing
    #    from earlier labs gets overwritten)
    os.makedirs('outputs', exist_ok=True)
    errors_df = X_test.copy()
    errors_df['Actual_Churn'] = y_test.values
    errors_df['Predicted_Churn'] = y_pred

    false_negatives = errors_df[(errors_df['Actual_Churn'] == 1) & (errors_df['Predicted_Churn'] == 0)]
    false_positives = errors_df[(errors_df['Actual_Churn'] == 0) & (errors_df['Predicted_Churn'] == 1)]

    false_negatives.to_csv('outputs/lab7_false_negatives.csv', index=False)
    false_positives.to_csv('outputs/lab7_false_positives.csv', index=False)

    print("[SUCCESS] Stage 4 evaluation complete.")
    return True


if __name__ == "__main__":
    passed = run_full_pipeline_evaluation()
    sys.exit(0 if passed else 1)