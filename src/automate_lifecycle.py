import mlflow
from mlflow.tracking import MlflowClient


def automate_model_lifecycle():
    print("[INFO] Starting Automated Model Lifecycle Manager...")

    client = MlflowClient()
    model_name = "Telco_Churn_Production_Model"
    metric_to_optimize = "recall"

    try:
        # Find newly registered models and move them to Staging
        all_versions = client.search_model_versions(f"name='{model_name}'")
        new_models = [mv for mv in all_versions if mv.current_stage == "None"]

        if new_models:
            print(f"\n[INFO] Found {len(new_models)} new model(s). Moving them to Staging...")

            for mv in new_models:
                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Staging", archive_existing_versions=False
                )
                print(f"  -> Version {mv.version} is now in Staging.")

        # Find models currently in Staging
        all_versions = client.search_model_versions(f"name='{model_name}'")
        models_staging = [mv for mv in all_versions if mv.current_stage == "Staging"]

        if not models_staging:
            print("\n[INFO] No models in Staging to evaluate. Exiting.")
            return

        # Find the best model in the Staging area
        print("\n[INFO] Evaluating models in Staging...")

        best_model_staging = None
        best_model_staging_score = -1.0

        for mv in models_staging:
            run = client.get_run(mv.run_id)
            model_score = run.data.metrics.get(metric_to_optimize, 0.0)

            print(f"  -> Staging Model: Version {mv.version} | {metric_to_optimize}: {model_score:.4f}")

            if model_score > best_model_staging_score:
                best_model_staging_score = model_score
                best_model_staging = mv

        print(f"\n[INFO] Best Model in Staging: Version {best_model_staging.version} ({metric_to_optimize}: {best_model_staging_score:.4f})")

        # Find the model currently in Production
        models_production = [mv for mv in all_versions if mv.current_stage == "Production"]
        production_model = models_production[0] if models_production else None
        move_to_production = False

        if not production_model:
            print("[INFO] No model currently in Production. Moving the Staging model to Production.")
            move_to_production = True

        else:
            production_run = client.get_run(production_model.run_id)
            production_model_score = production_run.data.metrics.get(metric_to_optimize, 0.0)

            print(f"[INFO] Current Production Model: Version {production_model.version} | {metric_to_optimize}: {production_model_score:.4f}")

            # Compare the Staging model with the Production model
            if best_model_staging_score > production_model_score:
                print(f"[SUCCESS] Staging model has a better {metric_to_optimize} than the Production model.")
                move_to_production = True

            else:
                print(f"[INFO] Staging model did not improve the {metric_to_optimize}. Production model will remain unchanged.")

        # Move the selected model to Production and archive the previous one
        if move_to_production:
            print(f"\n[INFO] Moving Version {best_model_staging.version} to Production...")

            client.transition_model_version_stage(
                name=model_name, version=best_model_staging.version, stage="Production", archive_existing_versions=False
            )

            if production_model:
                print(f"[INFO] Archiving previous Production Model (Version {production_model.version})...")

                client.transition_model_version_stage(
                    name=model_name, version=production_model.version, stage="Archived", archive_existing_versions=False
                )

        # Archive all remaining models in Staging
        final_versions = client.search_model_versions(f"name='{model_name}'")
        remaining_staging_models = [mv for mv in final_versions if mv.current_stage == "Staging"]

        if remaining_staging_models:
            print("\n[INFO] Cleaning up remaining Staging models...")

            for mv in remaining_staging_models:
                print(f"  -> Archiving Version {mv.version} (not selected for Production)")

                client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage="Archived", archive_existing_versions=False
                )

        print("\n[SUCCESS] Automated Model Lifecycle execution complete!")

    except Exception as e:
        print(f"[ERROR] Failed to automate model lifecycle: {e}")


if __name__ == "__main__":
    automate_model_lifecycle()