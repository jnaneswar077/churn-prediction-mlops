from pathlib import Path
from typing import Any
import os
import joblib
import mlflow
import mlflow.sklearn
import yaml


class ModelLoader:
    """
    Loads the production ML model from MLflow and the preprocessing
    pipeline from the local model artifacts.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]
        self.config_path = self.project_root / config_path

        self.config: dict[str, Any] = {}

        self.model = None
        self.preprocessor = None

        self.model_uri = None
        self.model_name = None

    def load_config(self) -> None:
        """Load project configuration from config.yaml."""

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self.config_path}"
            )

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

    def configure_mlflow(self) -> None:
        """Configure MLflow using environment variable or config.yaml."""

        tracking_uri = (
            os.getenv("MLFLOW_TRACKING_URI")
            or self.config["mlflow"].get("tracking_uri")
        )

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

    def load_production_model(self) -> None:
        """Load the Production model from MLflow Model Registry."""

        self.model_name = self.config["registry"]["model_name"]

        self.model_uri = f"models:/{self.model_name}/Production"

        print(f"[INFO] Loading production model from MLflow: {self.model_uri}")

        self.model = mlflow.sklearn.load_model(self.model_uri)

        print("[SUCCESS] Production model loaded successfully.")

    def load_preprocessor(self) -> None:
        """Load the fitted preprocessing pipeline."""

        preprocessor_path = (
            self.project_root / self.config["paths"]["preprocessor"]
        )

        if not preprocessor_path.exists():
            raise FileNotFoundError(
                f"Preprocessor not found: {preprocessor_path}"
            )

        print(f"[INFO] Loading preprocessor from: {preprocessor_path}")

        self.preprocessor = joblib.load(preprocessor_path)

        print("[SUCCESS] Preprocessor loaded successfully.")

    def load(self) -> None:
        """Load configuration, MLflow model, and preprocessor."""

        print("[INFO] Initializing ModelLoader...")

        self.load_config()
        self.configure_mlflow()
        self.load_production_model()
        self.load_preprocessor()

        print("[SUCCESS] ModelLoader initialization completed.")

    def is_ready(self) -> bool:
        """Return True when both model and preprocessor are loaded."""

        return self.model is not None and self.preprocessor is not None

if __name__ == "__main__":
    loader = ModelLoader()

    try:
        loader.load()

        print("\n========== MODEL LOADER STATUS ==========")
        print(f"Model loaded       : {loader.model is not None}")
        print(f"Preprocessor loaded: {loader.preprocessor is not None}")
        print(f"Model name         : {loader.model_name}")
        print(f"Model URI          : {loader.model_uri}")
        print("Ready              :", loader.is_ready())
        print("=========================================\n")

    except Exception as exc:
        print(f"[ERROR] Model loading failed: {exc}")
        raise