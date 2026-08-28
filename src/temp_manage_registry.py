import mlflow
from mlflow.tracking import MlflowClient

def manage_model_lifecycle():
    print("[INFO] Starting Model Lifecycle Management...")
    
    client = MlflowClient()
    model_name = "Telco_Churn_Production_Model"
    
    try:
        # 1. Transition Version 1 to Staging
        print("[INFO] Transitioning Version 1 to 'Staging'...")
        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage="Staging",
            archive_existing_versions=False
        )
        
        # 2. Promote Version 2 to Production
        print("[INFO] Promoting Version 2 to 'Production' (Champion Model)...")
        client.transition_model_version_stage(
            name=model_name,
            version=2,
            stage="Production",
            archive_existing_versions=False
        )
        
        # 3. Rollback/Archive Version 1 (Retiring the older model)
        print("[INFO] Archiving Version 1 (Rollback)...")
        client.transition_model_version_stage(
            name=model_name,
            version=1,
            stage="Archived",
            archive_existing_versions=False
        )
        
        # 4. Display Current Status
        print("\n--- Current Model Registry Status ---")
        versions = client.search_model_versions(f"name='{model_name}'")
        for mv in versions:
            print(f"Version: {mv.version} | Stage: {mv.current_stage} | Status: {mv.status}")
            
        print("\n[SUCCESS] Lifecycle management complete!")
        
    except Exception as e:
        print(f"[ERROR] Failed to update lifecycle stages: {e}")

if __name__ == "__main__":
    manage_model_lifecycle()