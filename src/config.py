import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)

MODEL_CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "cv_folds": 5,
    "fraud_threshold": 0.5,
}

MLFLOW_TRACKING_URI = "sqlite:///mlflow_runs/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "fraud-detection"