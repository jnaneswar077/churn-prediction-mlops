import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def build_preprocessor(X_train):
    """Builds the same preprocessing logic as preprocess_pipeline.py,
    but as a component to be embedded inside a single unified Pipeline
    rather than fitted and saved on its own."""
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline(steps=[
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    return preprocessor


def run_full_pipeline_training():
    print("[INFO] Starting Lab 7 Stage 2: Unified Pipeline Training...")

    # 1. Load and clean raw data (same cleaning rules as preprocess_pipeline.py)
    data_path = 'data/raw/churn.csv'
    df = pd.read_csv(data_path)

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = df['Churn'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0).astype(int)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # 2. Same split parameters as preprocess_pipeline.py, so the held-out
    #    test rows are identical across pipeline stages
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build ONE pipeline that owns both preprocessing and the model
    preprocessor = build_preprocessor(X_train)

    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        ))
    ])

    print("[INFO] Fitting unified pipeline (preprocessing + model) on raw training data...")
    full_pipeline.fit(X_train, y_train)

    # 4. Save as a single deployable artifact
    os.makedirs('models', exist_ok=True)
    joblib.dump(full_pipeline, 'models/churn_full_pipeline.pkl')

    print("[SUCCESS] Unified pipeline trained and saved to models/churn_full_pipeline.pkl")
    print(f"[INFO] Training rows: {X_train.shape[0]} | Held-out test rows: {X_test.shape[0]}")


if __name__ == "__main__":
    run_full_pipeline_training()