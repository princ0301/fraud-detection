import pytest
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROC_DIR   = BASE_DIR / "data" / "processed"
 
needs_model = pytest.mark.skipif(
    not (MODELS_DIR / "best_model.pkl").exists(),
    reason="Model file not in CI — train model locally first"
)
needs_data = pytest.mark.skipif(
    not (PROC_DIR / "X_val.csv").exists(),
    reason="Validation data not in CI — run feature engineering locally first"
)
 
@pytest.fixture(scope="module")
def model():
    import joblib
    return joblib.load(MODELS_DIR / "best_model.pkl")

@pytest.fixture(scope="module")
def metadata():
    with open(MODELS_DIR / "model_metadata.json") as f:
        return json.load(f)

@pytest.fixture(scope="module")
def sample_data():
    X_val = pd.read_csv(PROC_DIR / "X_val.csv").head(100)
    y_val = pd.read_csv(PROC_DIR / "y_val.csv").squeeze().head(100)
    return X_val, y_val

@needs_model
def test_metadata_has_required_keys(metadata):
    required = ["best_model", "pr_auc", "roc_auc", "f1", "threshold"]
    for key in required:
        assert key in metadata, f"Missing key: {key}"

@needs_model
def test_metadata_metrics_reasonable(metadata):
    assert metadata["pr_auc"]  > 0.3,  "PR-AUC too low"
    assert metadata["roc_auc"] > 0.7,  "ROC-AUC too low"
    assert 0 < metadata["threshold"] < 1, "Threshold out of range"

@needs_model
def test_model_loads(model):
    assert model is not None

@needs_model
def test_model_has_predict_proba(model):
    assert hasattr(model, "predict_proba")

@needs_model
def test_artifacts_load():
    import joblib
    artifacts = joblib.load(MODELS_DIR / "feature_artifacts.pkl")
    assert "feature_cols"    in artifacts
    assert "target_encoders" in artifacts
    assert len(artifacts["feature_cols"]) > 0
 
@needs_model
@needs_data
def test_model_prediction_shape(model, sample_data):
    X, _ = sample_data
    proba = model.predict_proba(X)
    assert proba.shape == (100, 2)

@needs_model
@needs_data
def test_model_prediction_probabilities(model, sample_data):
    X, _  = sample_data
    proba = model.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0

@needs_model
@needs_data
def test_model_pr_auc_on_sample(model, sample_data):
    from sklearn.metrics import average_precision_score
    X, y   = sample_data
    proba  = model.predict_proba(X)[:, 1]
    pr_auc = average_precision_score(y, proba)
    assert pr_auc > 0.05, f"PR-AUC too low: {pr_auc}"
 
def test_api_predict_runs():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client   = TestClient(app)
    response = client.post("/predict", json={
        "TransactionAmt": 100.0,
        "ProductCD"     : "W",
        "card1"         : 9500,
    })
    assert response.status_code == 200

def test_api_probability_in_range():
    from fastapi.testclient import TestClient
    from src.api.main import app
    client   = TestClient(app)
    response = client.post("/predict", json={
        "TransactionAmt": 100.0,
        "ProductCD"     : "W",
        "card1"         : 9500,
    })
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0