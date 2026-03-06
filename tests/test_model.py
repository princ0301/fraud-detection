import pytest
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROC_DIR   = BASE_DIR / "data" / "processed"

@pytest.fixture(scope="module")
def model():
    return joblib.load(MODELS_DIR / "best_model.pkl")

@pytest.fixture(scope="module")
def artifacts():
    return joblib.load(MODELS_DIR / "feature_artifacts.pkl")

@pytest.fixture(scope="module")
def metadata():
    with open(MODELS_DIR / "model_metadata.json") as f:
        return json.load(f)
    
@pytest.fixture(scope="module")
def sample_data():
    X_val = pd.read_csv(PROC_DIR / "X_val.csv")
    y_val = pd.read_csv(PROC_DIR / "y_val.csv").squeeze()
    return X_val.head(100), y_val.head(100)

def test_model_loads(model):
    """Model file exists and loads correctly."""
    assert model is not None

def test_model_has_predict_proba(model):
    """Model has predict_proba method."""
    assert hasattr(model, "predict_proba")

def test_artifacts_load(artifacts):
    """Feature artifacts load correctly."""
    assert "feature_cols" in artifacts
    assert "target_encoders" in artifacts
    assert len(artifacts["feature_cols"]) > 0

def test_metadata_has_required_keys(metadata):
    """Model metadata has all required keys."""
    required = ["best_model", "pr_auc", "roc_auc", "f1", "threshold"]
    for key in required:
        assert key in metadata, f"Missing key: {key}"

def test_metadata_metrics_reasonable(metadata):
    """Model metrics are within reasonable range."""
    assert metadata["pr_auc"]  > 0.3, "PR-AUC too low"
    assert metadata["roc_auc"] > 0.7, "ROC-AUC too low"
    assert 0 < metadata["threshold"] < 1, "Threshold out of range"

def test_model_prediction_shape(model, sample_data):
    """Model outputs correct shape."""
    X, _ = sample_data
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2), "Wrong prediction shape"

def test_model_prediction_probabilities(model, sample_data):
    """Probabilities are between 0 and 1."""
    X, _ = sample_data
    proba = model.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0, "Probability below 0"
    assert proba.max() <= 1.0, "Probability above 1"

def test_model_predicts_both_classes(model, sample_data):
    """Model predicts both fraud and legit."""
    X, _ = sample_data
    proba  = model.predict_proba(X)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    assert len(np.unique(y_pred)) >= 1

def test_model_pr_auc_on_sample(model, sample_data):
    """PR-AUC on sample is above random baseline."""
    from sklearn.metrics import average_precision_score
    X, y = sample_data
    proba  = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, proba)
    assert pr_auc > 0.05, f"PR-AUC too low on sample: {pr_auc}"