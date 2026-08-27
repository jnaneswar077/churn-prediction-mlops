import os
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

def run_training():
    print("Starting Model Training with MLflow...")
    
    # Load processed training data
    X_train = np.load('data/processed/X_train_final.npy')
    y_train = np.load('data/processed/y_train.npy')
    
    # 1. Set the MLflow Experiment Name
    mlflow.set_experiment("Telco_Churn_Prediction")
    
    # 2. Start the tracking run
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        
        # Define our hyperparameters
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }
        
        # Log parameters to MLflow
        mlflow.log_params(params)
        
        # Initialize and train the model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Save the model locally (standard practice)
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/random_forest_model.pkl')
        
        # 3. Log the trained model as an artifact in MLflow
        mlflow.sklearn.log_model(model, "model")
        
        print("Model trained and logged to MLflow successfully!")

if __name__ == "__main__":
    run_training()