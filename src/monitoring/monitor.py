import json
import joblib
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from scipy import stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
 
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
PROC_DIR    = BASE_DIR / "data" / "processed"
MODELS_DIR  = BASE_DIR / "models"
MONITOR_DIR = BASE_DIR / "monitoring"
MONITOR_DIR.mkdir(exist_ok=True)
(MONITOR_DIR / "reports").mkdir(exist_ok=True)
(MONITOR_DIR / "logs").mkdir(exist_ok=True)

def load_artifacts():
    model     = joblib.load(MODELS_DIR / "best_model.pkl")
    artifacts = joblib.load(MODELS_DIR / "feature_artifacts.pkl")
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    return model, artifacts, metadata

def tune_threshold(model, X_val, y_val, target_recall=0.70):
    from sklearn.metrics import precision_recall_curve, f1_score

    log.info("Tuning decision threshold...")
    y_proba = model.predict_proba(X_val)[:, 1]

    precision_arr, recall_arr, thresholds = precision_recall_curve(y_val, y_proba)

    results = []
    for i, thresh in enumerate(thresholds):
        y_pred = (y_proba >= thresh).astype(int)
        results.append({
            "threshold" : round(float(thresh), 4),
            "precision" : round(float(precision_arr[i]), 4),
            "recall"    : round(float(recall_arr[i]), 4),
            "f1"        : round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        })

    df = pd.DataFrame(results)

    candidates = df[df["recall"] >= target_recall]
    best = candidates.loc[candidates["f1"].idxmax()] if len(candidates) > 0 \
           else df.loc[df["f1"].idxmax()]

    default_row = df.iloc[(df["threshold"] - 0.5).abs().argsort()[:1]]
    log.info(f"\n   Default threshold (0.5):")
    log.info(f"   Precision: {default_row['precision'].values[0]} | "
             f"Recall: {default_row['recall'].values[0]} | "
             f"F1: {default_row['f1'].values[0]}")
    log.info(f"\n   Optimal threshold → {best['threshold']}")
    log.info(f"   Precision: {best['precision']} | Recall: {best['recall']} | F1: {best['f1']}")
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(df["threshold"], df["precision"], label="Precision", color="steelblue", lw=2)
    axes[0].plot(df["threshold"], df["recall"],    label="Recall",    color="crimson",   lw=2)
    axes[0].plot(df["threshold"], df["f1"],        label="F1",        color="green",     lw=2)
    axes[0].axvline(x=best["threshold"], color="orange", linestyle="--",
                    label=f"Optimal ({best['threshold']})", lw=2)
    axes[0].axvline(x=0.5, color="gray", linestyle=":", label="Default (0.5)")
    axes[0].set_xlabel("Threshold")
    axes[0].set_title("Metrics vs Threshold")
    axes[0].legend()
    axes[0].set_xlim([0, 1])
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["recall"], df["precision"], color="purple", lw=2)
    axes[1].scatter([best["recall"]], [best["precision"]],
                    color="orange", s=150, zorder=5, label="Optimal point")
    axes[1].scatter([default_row["recall"].values[0]],
                    [default_row["precision"].values[0]],
                    color="gray", s=100, zorder=5, label="Default (0.5)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Threshold Tuning Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(MONITOR_DIR / "threshold_tuning.png", dpi=150)
    plt.close()

    # Update metadata
    with open(MODELS_DIR / "model_metadata.json") as f:
        metadata = json.load(f)
    metadata["threshold"]           = float(best["threshold"])
    metadata["threshold_recall"]    = float(best["recall"])
    metadata["threshold_precision"] = float(best["precision"])
    metadata["threshold_f1"]        = float(best["f1"])
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    log.info("   Saved: monitoring/threshold_tuning.png")
    log.info("   Updated: models/model_metadata.json")
    return float(best["threshold"]), df
 
def run_drift_detection(reference_data, current_data, top_n=20):
    log.info("\nRunning drift detection (KS Test)...")

    num_cols = reference_data.select_dtypes(include=[np.number]).columns[:top_n]
    drift_results = []

    for col in num_cols:
        ref = reference_data[col].dropna()
        cur = current_data[col].dropna()
        if len(ref) == 0 or len(cur) == 0:
            continue
        ks_stat, p_value = stats.ks_2samp(ref, cur)
        drift_results.append({
            "feature" : col,
            "ks_stat" : round(float(ks_stat), 4),
            "p_value" : round(float(p_value), 4),
            "drifted" : bool(p_value < 0.05),
        })

    drift_df        = pd.DataFrame(drift_results)
    drifted_cols    = drift_df[drift_df["drifted"]]
    drift_share_pct = round(len(drifted_cols) / len(drift_df) * 100, 2)
    drift_detected  = drift_share_pct > 20

    log.info(f"   Total features checked : {len(drift_df)}")
    log.info(f"   Drifted features       : {len(drifted_cols)} ({drift_share_pct}%)")
    log.info(f"   Drift alert            : {drift_detected}")

    if len(drifted_cols) > 0:
        log.info("\n   Top drifted features:")
        for _, row in drifted_cols.sort_values("ks_stat", ascending=False).head(5).iterrows():
            log.info(f"   → {row['feature']}: KS={row['ks_stat']}, p={row['p_value']}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["crimson" if d else "steelblue" for d in drift_df["drifted"]]
    axes[0].barh(drift_df["feature"].str[:25], drift_df["ks_stat"],
                 color=colors, edgecolor="white")
    axes[0].axvline(x=0.1, color="orange", linestyle="--", label="KS=0.1 warning")
    axes[0].set_title("KS Statistic per Feature\n(red = drifted)")
    axes[0].set_xlabel("KS Statistic")
    axes[0].legend()

    drift_counts = drift_df["drifted"].value_counts()
    values  = [drift_counts.get(False, 0), drift_counts.get(True, 0)]
    axes[1].pie(values, labels=["No Drift", "Drifted"],
                autopct="%1.1f%%", colors=["steelblue", "crimson"], startangle=90)
    axes[1].set_title(f"Drift Summary ({drift_share_pct}% drifted)")

    plt.suptitle("Data Drift Detection Report", fontsize=14, fontweight="bold")
    plt.tight_layout()
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = MONITOR_DIR / "reports" / f"drift_report_{ts}.png"
    plt.savefig(report_path, dpi=150)
    plt.close()

    drift_df.to_csv(MONITOR_DIR / "reports" / f"drift_details_{ts}.csv", index=False)

    summary = {
        "timestamp"       : ts,
        "drift_detected"  : drift_detected,
        "drift_share_pct" : drift_share_pct,
        "drifted_count"   : len(drifted_cols),
        "total_checked"   : len(drift_df),
    }
    with open(MONITOR_DIR / "logs" / "drift_log.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")

    log.info(f"   Saved: {report_path}")
    return summary, drift_detected, drift_df
 
class PredictionLogger:
    def __init__(self):
        self.log_path = MONITOR_DIR / "logs" / "predictions.jsonl"

    def log(self, features: dict, prob: float, is_fraud: bool, threshold: float):
        entry = {
            "timestamp"         : datetime.now().isoformat(),
            "fraud_probability" : round(prob, 4),
            "is_fraud"          : bool(is_fraud),
            "threshold"         : threshold,
            "TransactionAmt"    : float(features.get("TransactionAmt", 0)),
            "tx_hour"           : float(features.get("tx_hour", 0)),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_recent(self, n=1000) -> pd.DataFrame:
        if not self.log_path.exists():
            return pd.DataFrame()
        rows = []
        with open(self.log_path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line.strip()))
                except:
                    continue
        return pd.DataFrame(rows).tail(n)
 
def generate_monitoring_dashboard(pred_log: pd.DataFrame):
    if len(pred_log) < 5:
        log.info("Not enough predictions yet for dashboard")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(pred_log["fraud_probability"], bins=20,
                    color="steelblue", edgecolor="white")
    axes[0, 0].axvline(x=0.5, color="red", linestyle="--", label="Threshold 0.5")
    axes[0, 0].set_title("Fraud Probability Distribution")
    axes[0, 0].set_xlabel("Fraud Probability")
    axes[0, 0].legend()

    pred_log["timestamp"] = pd.to_datetime(pred_log["timestamp"])
    pred_log = pred_log.sort_values("timestamp")
    pred_log["rolling_fraud"] = pred_log["is_fraud"].rolling(
        min(50, len(pred_log)), min_periods=1).mean() * 100
    axes[0, 1].plot(range(len(pred_log)), pred_log["rolling_fraud"],
                    color="crimson", lw=2)
    axes[0, 1].set_title("Rolling Fraud Rate")
    axes[0, 1].set_ylabel("Fraud Rate %")
    axes[0, 1].set_xlabel("Prediction #")

    legit = pred_log[~pred_log["is_fraud"]]["TransactionAmt"].clip(upper=2000)
    fraud = pred_log[pred_log["is_fraud"]]["TransactionAmt"].clip(upper=2000)
    axes[1, 0].hist(legit, bins=20, alpha=0.6, color="steelblue", label="Legit")
    if len(fraud) > 0:
        axes[1, 0].hist(fraud, bins=20, alpha=0.6, color="crimson", label="Fraud")
    axes[1, 0].set_title("Amount by Prediction")
    axes[1, 0].legend()

    def risk(p):
        if p < 0.2:  return "LOW"
        if p < 0.5:  return "MEDIUM"
        if p < 0.75: return "HIGH"
        return "CRITICAL"

    pred_log["risk"] = pred_log["fraud_probability"].apply(risk)
    risk_counts = pred_log["risk"].value_counts()
    cmap = {"LOW": "green", "MEDIUM": "gold", "HIGH": "orange", "CRITICAL": "crimson"}
    axes[1, 1].pie(risk_counts.values,
                   labels=risk_counts.index,
                   autopct="%1.1f%%",
                   colors=[cmap.get(r, "gray") for r in risk_counts.index],
                   startangle=90)
    axes[1, 1].set_title("Risk Level Distribution")

    plt.suptitle("Live Monitoring Dashboard", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = MONITOR_DIR / "live_dashboard.png"
    plt.savefig(path, dpi=150)
    plt.close()
    log.info(f"Dashboard saved: {path}")
 
def main():
    print("=" * 60)
    print("   MONITORING + DRIFT DETECTION")
    print("=" * 60)

    model, artifacts, metadata = load_artifacts()

    X_val = pd.read_csv(PROC_DIR / "X_val.csv")
    y_val = pd.read_csv(PROC_DIR / "y_val.csv").squeeze()

    # Step 1: Tune threshold
    optimal_threshold, _ = tune_threshold(model, X_val, y_val, target_recall=0.70)
    print(f"\nOptimal threshold: {optimal_threshold}")

    # Step 2: Drift detection
    split    = int(len(X_val) * 0.8)
    summary, drift_detected, drift_df = run_drift_detection(
        X_val.iloc[:split], X_val.iloc[split:]
    )
    if drift_detected:
        print(f"\nDRIFT DETECTED — {summary['drift_share_pct']}% features drifted!")
    else:
        print(f"\nNo significant drift ({summary['drift_share_pct']}% features affected)")

    # Step 3: Prediction log dashboard
    logger   = PredictionLogger()
    pred_log = logger.load_recent(1000)
    if len(pred_log) > 0:
        generate_monitoring_dashboard(pred_log)
        print(f"\n{len(pred_log)} predictions logged | "
              f"Fraud rate: {pred_log['is_fraud'].mean()*100:.2f}%")
    else:
        print("\nNo predictions logged yet — use the API first, then re-run!")


if __name__ == "__main__":
    main()