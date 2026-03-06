import json
import uuid
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import uvicorn
 
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
PROC_DIR   = BASE_DIR / "data" / "processed"
STATIC_DIR = BASE_DIR / "static"
TMPL_DIR   = BASE_DIR / "templates"

print("Loading model artifacts...")

model        = joblib.load(MODELS_DIR / "best_model.pkl")
artifacts    = joblib.load(MODELS_DIR / "feature_artifacts.pkl")
feature_cols = artifacts["feature_cols"]
te_encoders  = artifacts["target_encoders"]

pca_path  = MODELS_DIR / "pca_model.pkl"
pca_model = joblib.load(pca_path) if pca_path.exists() else None

with open(MODELS_DIR / "model_metadata.json") as f:
    metadata = json.load(f)

THRESHOLD = metadata.get("threshold", 0.5)
explainer = shap.TreeExplainer(model)

print(f"Model loaded : {metadata['best_model']}")
print(f"PR-AUC       : {metadata['pr_auc']}")
print(f"ROC-AUC      : {metadata['roc_auc']}")
print(f"Threshold    : {THRESHOLD}")
 
app = FastAPI(
    title       = "🛡️ Fraud Detection API",
    description = "Real-time fraud detection using XGBoost + SHAP explainability",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)
 
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TMPL_DIR))
 
class TransactionRequest(BaseModel):
    TransactionAmt : float           = Field(...,  example=150.00)
    ProductCD      : str             = Field(...,  example="W")
    card1          : int             = Field(...,  example=9500)
    card2          : Optional[float] = Field(None, example=360.0)
    card3          : Optional[float] = Field(None, example=150.0)
    card4          : Optional[str]   = Field(None, example="visa")
    card5          : Optional[float] = Field(None, example=226.0)
    card6          : Optional[str]   = Field(None, example="debit")
    addr1          : Optional[float] = Field(None, example=299.0)
    addr2          : Optional[float] = Field(None, example=87.0)
    P_emaildomain  : Optional[str]   = Field(None, example="gmail.com")
    R_emaildomain  : Optional[str]   = Field(None, example="gmail.com")
    TransactionDT  : Optional[int]   = Field(None, example=86400)
    C1  : Optional[float] = Field(None, example=1.0)
    C2  : Optional[float] = Field(None, example=1.0)
    C4  : Optional[float] = Field(None, example=0.0)
    C5  : Optional[float] = Field(None, example=0.0)
    C6  : Optional[float] = Field(None, example=1.0)
    C7  : Optional[float] = Field(None, example=0.0)
    C8  : Optional[float] = Field(None, example=1.0)
    C9  : Optional[float] = Field(None, example=0.0)
    C10 : Optional[float] = Field(None, example=0.0)
    C11 : Optional[float] = Field(None, example=2.0)
    C12 : Optional[float] = Field(None, example=0.0)
    C13 : Optional[float] = Field(None, example=1.0)
    C14 : Optional[float] = Field(None, example=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "TransactionAmt": 150.00, "ProductCD": "W",
                "card1": 9500, "card4": "visa", "card6": "debit",
                "P_emaildomain": "gmail.com", "TransactionDT": 86400,
                "C1": 1.0, "C13": 1.0, "C14": 1.0
            }
        }


class PredictionResponse(BaseModel):
    transaction_id    : str
    is_fraud          : bool
    fraud_probability : float
    risk_level        : str
    confidence        : str
    top_risk_factors  : list
    recommendation    : str
    model_version     : str
    timestamp         : str


class HealthResponse(BaseModel):
    status    : str
    model     : str
    pr_auc    : float
    roc_auc   : float
    threshold : float
    uptime    : str
 
def engineer_features(tx: TransactionRequest) -> pd.DataFrame:
    global_fraud_rate = 0.035
    row = {}

    dt = tx.TransactionDT or 86400
    row["tx_hour"]    = (dt // 3600) % 24
    row["tx_day"]     = (dt // (3600 * 24)) % 7
    row["tx_day_abs"] = dt // (3600 * 24)
    row["is_night"]   = int(0 <= row["tx_hour"] <= 6)
    row["is_weekend"] = int(row["tx_day"] in [5, 6])

    amt = tx.TransactionAmt
    row["TransactionAmt"]     = amt
    row["log_amount"]         = np.log1p(amt)
    row["amount_cents"]       = round(amt % 1, 2)
    row["is_round_amount"]    = int(row["amount_cents"] == 0)
    row["amount_card_zscore"] = 0.0

    row["card1_count"]  = 1
    row["card2_count"]  = 1
    row["pemail_count"] = 1
    row["addr1_count"]  = 1

    row["card1"] = tx.card1
    row["card2"] = tx.card2 or 0
    row["card3"] = tx.card3 or 0
    row["card5"] = tx.card5 or 0
    row["addr1"] = tx.addr1 or 0
    row["addr2"] = tx.addr2 or 0

    def te(col, val, fallback=global_fraud_rate):
        enc = te_encoders.get(col, {})
        return enc.get(str(val), enc.get(val, fallback))

    row["card1_encoded"]         = te("card1",         tx.card1)
    row["card2_encoded"]         = te("card2",         tx.card2)
    row["card3_encoded"]         = te("card3",         tx.card3)
    row["card5_encoded"]         = te("card5",         tx.card5)
    row["P_emaildomain_encoded"] = te("P_emaildomain", tx.P_emaildomain)
    row["R_emaildomain_encoded"] = te("R_emaildomain", tx.R_emaildomain)
    row["addr1_encoded"]         = te("addr1",         tx.addr1)
    row["addr2_encoded"]         = te("addr2",         tx.addr2)

    row["ProductCD"] = {"W": 0, "H": 1, "C": 2, "S": 3, "R": 4}.get(tx.ProductCD, 0)
    row["card4"]     = {"visa": 0, "mastercard": 1, "american express": 2, "discover": 3}.get(
                        str(tx.card4).lower(), 0)
    row["card6"]     = {"debit": 0, "credit": 1}.get(str(tx.card6).lower(), 0)
    row["P_emaildomain"] = hash(str(tx.P_emaildomain)) % 100
    row["R_emaildomain"] = hash(str(tx.R_emaildomain)) % 100

    for c in ["C1","C2","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14"]:
        row[c] = getattr(tx, c, None) or 0.0

    if pca_model:
        for i in range(pca_model.n_components_):
            row[f"V_pca_{i}"] = 0.0

    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    df = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    return df
 
def get_risk_level(prob: float) -> str:
    if prob < 0.2:  return "LOW"
    if prob < 0.5:  return "MEDIUM"
    if prob < 0.75: return "HIGH"
    return "CRITICAL"


def get_recommendation(prob: float, is_fraud: bool) -> str:
    if prob < 0.2:  return "Approve transaction"
    if prob < 0.5:  return "Flag for manual review"
    if prob < 0.75: return "Block and notify customer"
    return "Block immediately and escalate to fraud team"


def get_confidence(prob: float) -> str:
    dist = abs(prob - 0.5)
    if dist > 0.35: return "High"
    if dist > 0.15: return "Medium"
    return "Low"
 
START_TIME = datetime.now()


@app.get("/", tags=["Root"])
def root():
    """Redirect to dashboard."""
    return RedirectResponse(url="/dashboard")


@app.get("/dashboard", tags=["Dashboard"])
def dashboard(request: Request):
    """Serve the fraud detection dashboard."""
    info = {
        "model"      : metadata["best_model"],
        "pr_auc"     : metadata["pr_auc"],
        "roc_auc"    : metadata["roc_auc"],
        "f1"         : metadata["f1"],
        "threshold"  : metadata["threshold"],
        "n_features" : len(feature_cols),
    }
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"model_info": info}
    )

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    uptime = str(datetime.now() - START_TIME).split(".")[0]
    return HealthResponse(
        status    = "healthy",
        model     = metadata["best_model"],
        pr_auc    = metadata["pr_auc"],
        roc_auc   = metadata["roc_auc"],
        threshold = THRESHOLD,
        uptime    = uptime,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(tx: TransactionRequest):
    try:
        X        = engineer_features(tx)
        prob     = float(model.predict_proba(X)[0][1])
        is_fraud = prob >= THRESHOLD

        shap_vals   = explainer.shap_values(X)[0]
        shap_series = pd.Series(shap_vals, index=feature_cols)
        top_factors = (
            shap_series.abs()
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )

        risk_factors = []
        for feat in top_factors:
            val      = float(X[feat].iloc[0])
            shap_val = float(shap_series[feat])
            risk_factors.append({
                "feature"   : feat,
                "value"     : round(val, 4),
                "impact"    : f"{'↑ increases' if shap_val > 0 else '↓ decreases'} fraud risk",
                "shap_score": round(shap_val, 4),
            })

        return PredictionResponse(
            transaction_id    = str(uuid.uuid4())[:8],
            is_fraud          = is_fraud,
            fraud_probability = round(prob, 4),
            risk_level        = get_risk_level(prob),
            confidence        = get_confidence(prob),
            top_risk_factors  = risk_factors,
            recommendation    = get_recommendation(prob, is_fraud),
            model_version     = metadata["best_model"],
            timestamp         = datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[TransactionRequest]):
    """Predict fraud for multiple transactions at once (max 100)."""
    if len(transactions) > 100:
        raise HTTPException(status_code=400, detail="Max 100 transactions per batch")

    results = []
    for tx in transactions:
        X        = engineer_features(tx)
        prob     = float(model.predict_proba(X)[0][1])
        is_fraud = prob >= THRESHOLD
        results.append({
            "is_fraud"          : is_fraud,
            "fraud_probability" : round(prob, 4),
            "risk_level"        : get_risk_level(prob),
            "recommendation"    : get_recommendation(prob, is_fraud),
        })
    return {"count": len(results), "results": results}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns model metadata and top SHAP features."""
    shap_path    = MODELS_DIR / "shap_importance.csv"
    top_features = []
    if shap_path.exists():
        top_features = pd.read_csv(shap_path).head(10).to_dict(orient="records")

    return {
        "model"        : metadata["best_model"],
        "pr_auc"       : metadata["pr_auc"],
        "roc_auc"      : metadata["roc_auc"],
        "f1"           : metadata["f1"],
        "threshold"    : THRESHOLD,
        "n_features"   : len(feature_cols),
        "top_features" : top_features,
    }
 
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host      = "0.0.0.0",
        port      = 8000,
        reload    = True,
        log_level = "info",
    )