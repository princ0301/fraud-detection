import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
def test_processed_files_exist():
    """All processed data files exist."""
    base    = Path(__file__).resolve().parent.parent
    proc    = base / "data" / "processed"
    files   = ["X_train.csv", "y_train.csv", "X_val.csv", "y_val.csv"]
    for f in files:
        assert (proc / f).exists(), f"Missing: {f}"

def test_train_val_shapes():
    """Train and val sets have consistent shapes."""
    base    = Path(__file__).resolve().parent.parent
    proc    = base / "data" / "processed"
    X_train = pd.read_csv(proc / "X_train.csv", nrows=100)
    X_val   = pd.read_csv(proc / "X_val.csv",   nrows=100)
    assert X_train.shape[1] == X_val.shape[1], "Feature count mismatch!"

def test_no_string_columns():
    """No string columns in processed data."""
    base    = Path(__file__).resolve().parent.parent
    proc    = base / "data" / "processed"
    X_train = pd.read_csv(proc / "X_train.csv", nrows=500)
    str_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    assert len(str_cols) == 0, f"String columns found: {str_cols}"

def test_no_infinite_values():
    """No infinite values in processed data."""
    base    = Path(__file__).resolve().parent.parent
    proc    = base / "data" / "processed"
    X_train = pd.read_csv(proc / "X_train.csv", nrows=500)
    inf_count = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
    assert inf_count == 0, f"Found {inf_count} infinite values"

def test_target_is_binary():
    """Target variable is binary (0 or 1)."""
    base  = Path(__file__).resolve().parent.parent
    proc  = base / "data" / "processed"
    y     = pd.read_csv(proc / "y_train.csv").squeeze()
    unique_vals = set(y.unique())
    assert unique_vals.issubset({0, 1}), f"Non-binary values: {unique_vals}"

def test_fraud_rate_reasonable():
    """Fraud rate is between 1% and 10%."""
    base  = Path(__file__).resolve().parent.parent
    proc  = base / "data" / "processed"
    y     = pd.read_csv(proc / "y_train.csv").squeeze()
    rate  = y.mean() * 100
    assert 1.0 <= rate <= 10.0, f"Unexpected fraud rate: {rate:.2f}%"

def test_feature_engineering_from_request():
    """Feature engineering produces correct output for API request."""
    from src.api.main import engineer_features, TransactionRequest

    tx = TransactionRequest(
        TransactionAmt = 150.0,
        ProductCD      = "W",
        card1          = 9500,
        card4          = "visa",
        card6          = "debit",
        P_emaildomain  = "gmail.com",
        TransactionDT  = 86400,
        C1=1.0, C13=1.0
    )
    X = engineer_features(tx)
    assert X.shape[0] == 1,    "Should return 1 row"
    assert X.shape[1] > 10,    "Should have multiple features"
    assert X.isnull().sum().sum() == 0, "Should have no NaN values"

def test_log_amount_positive():
    """log_amount feature is always positive."""
    from src.api.main import engineer_features, TransactionRequest
    for amt in [1.0, 10.0, 100.0, 9999.99]:
        tx = TransactionRequest(TransactionAmt=amt, ProductCD="W", card1=1000)
        X  = engineer_features(tx)
        assert X["log_amount"].iloc[0] > 0

def test_time_features_range():
    """Time features are within valid ranges."""
    from src.api.main import engineer_features, TransactionRequest
    tx = TransactionRequest(
        TransactionAmt=100.0, ProductCD="W",
        card1=1000, TransactionDT=86400
    )
    X = engineer_features(tx)
    assert 0 <= X["tx_hour"].iloc[0] <= 23
    assert 0 <= X["tx_day"].iloc[0]  <= 6
    assert X["is_night"].iloc[0] in [0, 1]
    assert X["is_weekend"].iloc[0] in [0, 1]