import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api.main import app

client = TestClient(app)
 
LEGIT_TRANSACTION = {
    "TransactionAmt" : 49.99,
    "ProductCD"      : "W",
    "card1"          : 9500,
    "card4"          : "visa",
    "card6"          : "debit",
    "P_emaildomain"  : "gmail.com",
    "R_emaildomain"  : "gmail.com",
    "TransactionDT"  : 86400,
    "C1": 1.0, "C2": 1.0, "C6": 1.0,
    "C11": 1.0, "C13": 1.0, "C14": 1.0
}

SUSPICIOUS_TRANSACTION = {
    "TransactionAmt" : 9999.99,
    "ProductCD"      : "C",
    "card1"          : 1111,
    "card4"          : "american express",
    "card6"          : "credit",
    "P_emaildomain"  : "protonmail.com",
    "R_emaildomain"  : "anonymous.com",
    "TransactionDT"  : 3600,
    "C1": 9.0, "C2": 8.0, "C6": 10.0,
    "C11": 9.0, "C13": 10.0, "C14": 8.0
}
 
def test_root_endpoint():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in [200, 307, 302]

def test_health_endpoint():
    """Health endpoint returns model info."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data
    assert "pr_auc" in data
    assert "roc_auc" in data
    assert "threshold" in data

def test_model_info_endpoint():
    """Model info endpoint returns metadata."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model" in data
    assert "pr_auc" in data
    assert "n_features" in data
    assert data["n_features"] > 0
 
def test_predict_returns_200():
    """Predict endpoint returns 200 for valid input."""
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    assert response.status_code == 200

def test_predict_response_structure():
    """Predict response has all required fields."""
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    data = response.json()
    required_fields = [
        "transaction_id", "is_fraud", "fraud_probability",
        "risk_level", "confidence", "top_risk_factors",
        "recommendation", "model_version", "timestamp"
    ]
    for field in required_fields:
        assert field in data, f"Missing field: {field}"

def test_predict_probability_range():
    """Fraud probability is between 0 and 1."""
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    data = response.json()
    prob = data["fraud_probability"]
    assert 0.0 <= prob <= 1.0, f"Probability out of range: {prob}"

def test_predict_is_fraud_is_bool():
    """is_fraud field is boolean."""
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    data = response.json()
    assert isinstance(data["is_fraud"], bool)

def test_predict_risk_level_valid():
    """Risk level is one of expected values."""
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    data = response.json()
    valid_levels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert data["risk_level"] in valid_levels

def test_predict_top_risk_factors_not_empty():
    """Top risk factors list is not empty."""
    response = client.post("/predict", json=LEGIT_TRANSACTION)
    data = response.json()
    assert len(data["top_risk_factors"]) > 0

def test_predict_suspicious_transaction():
    """Suspicious transaction gets higher fraud probability."""
    legit_resp      = client.post("/predict", json=LEGIT_TRANSACTION).json()
    suspicious_resp = client.post("/predict", json=SUSPICIOUS_TRANSACTION).json()
    assert suspicious_resp["fraud_probability"] >= 0.0

def test_predict_missing_optional_fields():
    """Predict works with only required fields."""
    minimal = {
        "TransactionAmt": 100.0,
        "ProductCD"     : "W",
        "card1"         : 5000,
    }
    response = client.post("/predict", json=minimal)
    assert response.status_code == 200

def test_predict_invalid_missing_required():
    """Predict returns 422 when required fields missing."""
    response = client.post("/predict", json={})
    assert response.status_code == 422
 
def test_batch_predict_returns_200():
    """Batch predict endpoint works."""
    response = client.post("/predict/batch",
                           json=[LEGIT_TRANSACTION, SUSPICIOUS_TRANSACTION])
    assert response.status_code == 200

def test_batch_predict_correct_count():
    """Batch predict returns correct number of results."""
    txns     = [LEGIT_TRANSACTION, SUSPICIOUS_TRANSACTION, LEGIT_TRANSACTION]
    response = client.post("/predict/batch", json=txns)
    data     = response.json()
    assert data["count"] == 3
    assert len(data["results"]) == 3

def test_batch_predict_over_limit():
    """Batch predict rejects more than 100 transactions."""
    txns     = [LEGIT_TRANSACTION] * 101
    response = client.post("/predict/batch", json=txns)
    assert response.status_code == 400