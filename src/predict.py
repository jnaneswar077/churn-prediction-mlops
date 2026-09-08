import os
import sys
import argparse
import pandas as pd
import joblib

PIPELINE_PATH = 'models/churn_full_pipeline.pkl'


def load_pipeline(pipeline_path=PIPELINE_PATH):
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Unified pipeline not found at: {pipeline_path}. "
            f"Run src/train_full_pipeline.py first (Stage 2)."
        )
    return joblib.load(pipeline_path)


def predict_churn(raw_df, pipeline=None):
    """Runs inference on RAW, untransformed customer rows.

    raw_df must have the same input columns the model was trained on
    (i.e. the original dataset columns minus 'customerID' and 'Churn').
    No manual preprocessing is required -- the loaded pipeline object
    owns both the preprocessing and the model.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    # Defensive cleanup: apply the same light raw-data fixes used at
    # training time, in case the caller passes truly unprocessed rows
    # straight from a source system.
    df = raw_df.copy()
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    predictions = pipeline.predict(df)
    probabilities = pipeline.predict_proba(df)[:, 1]

    result = raw_df.copy()
    result['Predicted_Churn'] = predictions
    result['Churn_Probability'] = probabilities.round(4)
    return result


def run_inference_demo(input_path=None, n_sample=5):
    print("[INFO] Starting Lab 7 Stage 5: Inference...")

    try:
        pipeline = load_pipeline()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    if input_path:
        if not os.path.exists(input_path):
            print(f"[ERROR] Input file not found: {input_path}")
            return False
        new_customers = pd.read_csv(input_path)
        print(f"[INFO] Loaded {len(new_customers)} customer rows from {input_path}")
    else:
        # Demo mode: simulate "new, unseen customers" using a handful of
        # rows from the raw dataset, WITHOUT the true Churn label -- this
        # mirrors exactly what a real inference request would look like.
        data_path = 'data/raw/churn.csv'
        if not os.path.exists(data_path):
            print(f"[ERROR] Raw dataset not found at: {data_path}")
            return False
        full_df = pd.read_csv(data_path)
        new_customers = full_df.drop(columns=['Churn']).sample(n=n_sample, random_state=7)
        print(f"[INFO] No --input given. Using {n_sample} sampled rows as demo 'new customers'.")

    result = predict_churn(new_customers, pipeline=pipeline)

    print("\n--- Inference Results ---")
    display_cols = ['Predicted_Churn', 'Churn_Probability']
    print(result[display_cols].to_string())
    print("--------------------------\n")

    os.makedirs('outputs', exist_ok=True)
    output_path = 'outputs/lab7_inference_predictions.csv'
    result.to_csv(output_path, index=False)
    print(f"[SUCCESS] Predictions saved to {output_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn inference using the unified Lab 7 pipeline.")
    parser.add_argument('--input', type=str, default=None,
                         help="Path to a CSV of raw customer rows (no Churn column). "
                              "If omitted, a demo sample is drawn from data/raw/churn.csv.")
    args = parser.parse_args()

    passed = run_inference_demo(input_path=args.input)
    sys.exit(0 if passed else 1)