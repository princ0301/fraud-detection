import pandas as pd
import numpy as np
import json
import joblib
import warnings
import matplotlib.pylab as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import optuna
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MLFLOW_URI        = f"sqlite:///{BASE_DIR}/mlflow_runs/mlflow.db"
EXPERIMENT_NAME   = "fraud-detection"

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def load_processed_data():
    print("Loading processed data...")
    X_train = pd.read_csv(PROC_DIR / 'X_train.csv')
    y_train = pd.read_csv(PROC_DIR / 'y_train.csv').squeeze()
    X_val   = pd.read_csv(PROC_DIR / 'X_val.csv')
    y_val   = pd.read_csv(PROC_DIR / 'y_val.csv').squeeze()

    print(f"   X_train : {X_train.shape} | Fraud rate: {y_train.mean()*100:.2f}%")
    print(f"   X_val   : {X_val.shape}   | Fraud rate: {y_val.mean()*100:.2f}%")
    return X_train, y_train, X_val, y_val

def compute_metrics(y_true, y_pred_proba, threshold=0.5):
    y_pred = (y_pred_proba >= threshold).astype(int)
    return {
        'pr_auc'    : round(average_precision_score(y_true, y_pred_proba), 4),
        'roc_auc'   : round(roc_auc_score(y_true, y_pred_proba), 4),
        'f1'        : round(f1_score(y_true, y_pred, zero_division=0), 4),
        'precision' : round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall'    : round(recall_score(y_true, y_pred, zero_division=0), 4),
    }

def plot_pr_curve(y_true, y_proba, model_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, color='crimson', lw=2,
             label=f'PR AUC = {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_confusion_matrix(y_true, y_proba, model_name, save_path, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix — {model_name}')
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black',
                     fontsize=14, fontweight='bold')
    plt.xticks([0, 1], ['Predicted Legit', 'Predicted Fraud'])
    plt.yticks([0, 1], ['Actual Legit', 'Actual Fraud'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def train_baseline(X_train, y_train, X_val, y_val):
    print("\n" + "─"*50)
    print("Training Baseline: Logistic Regression")
    print("─"*50)

    with mlflow.start_run(run_name="logistic_regression"):
        params = {'C': 0.1, 'max_iter': 1000, 'class_weight': 'balanced',
                  'random_state': 42, 'n_jobs': -1}
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba)
        mlflow.log_metrics(metrics)

        plot_pr_curve(y_val, y_proba, 'Logistic Regression',
                      MODELS_DIR / 'lr_pr_curve.png')
        mlflow.log_artifact(str(MODELS_DIR / 'lr_pr_curve.png'))

        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

    print(f"   PR-AUC : {metrics['pr_auc']}")
    print(f"   ROC-AUC: {metrics['roc_auc']}")
    print(f"   F1     : {metrics['f1']}")
    return model, metrics, run_id

def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=30):
    print("\n" + "─"*50)
    print("Tuning XGBoost with Optuna...")
    print("─"*50)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 200, 800),
            'max_depth'        : trial.suggest_int('max_depth', 4, 10),
            'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha'        : trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda'       : trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight' : scale_pos_weight,
            'random_state'     : 42,
            'eval_metric'      : 'aucpr',
            'use_label_encoder': False,
            'n_jobs'           : -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        y_proba = model.predict_proba(X_val)[:, 1]
        return average_precision_score(y_val, y_proba)
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({
        'scale_pos_weight' : scale_pos_weight,
        'random_state'     : 42,
        'eval_metric'      : 'aucpr',
        'use_label_encoder': False,
        'n_jobs'           : -1,
    })
    print(f"\n   Best PR-AUC (Optuna): {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")

    with mlflow.start_run(run_name="xgboost_tuned"):
        mlflow.log_params(best_params)

        model = XGBClassifier(**best_params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba)
        mlflow.log_metrics(metrics)

        plot_pr_curve(y_val, y_proba, 'XGBoost Tuned',
                      MODELS_DIR / 'xgb_pr_curve.png')
        plot_confusion_matrix(y_val, y_proba, 'XGBoost Tuned',
                              MODELS_DIR / 'xgb_confusion_matrix.png')
        mlflow.log_artifact(str(MODELS_DIR / 'xgb_pr_curve.png'))
        mlflow.log_artifact(str(MODELS_DIR / 'xgb_confusion_matrix.png'))

        mlflow.xgboost.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

    print(f"\n   PR-AUC : {metrics['pr_auc']}")
    print(f"   ROC-AUC: {metrics['roc_auc']}")
    print(f"   F1     : {metrics['f1']}")
    return model, metrics, run_id, best_params

def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\n" + "─"*50)
    print("Training LightGBM...")
    print("─"*50)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        'n_estimators'     : 500,
        'max_depth'        : 7,
        'learning_rate'    : 0.05,
        'num_leaves'       : 63,
        'subsample'        : 0.8,
        'colsample_bytree' : 0.8,
        'scale_pos_weight' : scale_pos_weight,
        'random_state'     : 42,
        'n_jobs'           : -1,
        'verbose'          : -1,
    }

    with mlflow.start_run(run_name="lightgbm"):
        mlflow.log_params(params)

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[])

        y_proba = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_proba)
        mlflow.log_metrics(metrics)

        plot_pr_curve(y_val, y_proba, 'LightGBM',
                      MODELS_DIR / 'lgbm_pr_curve.png')
        mlflow.log_artifact(str(MODELS_DIR / 'lgbm_pr_curve.png'))
        mlflow.sklearn.log_model(model, "model")
        run_id = mlflow.active_run().info.run_id

    print(f"   PR-AUC : {metrics['pr_auc']}")
    print(f"   ROC-AUC: {metrics['roc_auc']}")
    print(f"   F1     : {metrics['f1']}")
    return model, metrics, run_id
 
def generate_shap(model, X_val, model_name='XGBoost'):
    print(f"\nGenerating SHAP values for {model_name}...")
    sample = X_val.sample(min(500, len(X_val)), random_state=42)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
 
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample, plot_type='bar',
                      max_display=20, show=False)
    plt.title(f'SHAP Feature Importance — {model_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f" SHAP plot saved to models/shap_summary.png")
 
    shap_df = pd.DataFrame({
        'feature'     : X_val.columns,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    print("\n   Top 10 most important features (SHAP):")
    print(shap_df.head(10).to_string(index=False))

    shap_df.to_csv(MODELS_DIR / 'shap_importance.csv', index=False)
    return shap_df
 
def compare_and_save(results, X_val, y_val):
    print("\n" + "="*60)
    print("   MODEL COMPARISON")
    print("="*60)

    comparison = pd.DataFrame([
        {'Model': name, **metrics}
        for name, (model, metrics, *_) in results.items()
    ]).sort_values('pr_auc', ascending=False)

    print(comparison.to_string(index=False))
 
    best_name  = comparison.iloc[0]['Model']
    best_model = results[best_name][0]

    print(f"\nBest Model: {best_name}")
    print(f"   PR-AUC: {comparison.iloc[0]['pr_auc']}")
 
    y_proba = best_model.predict_proba(X_val)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)
    print(f"\n   Classification Report ({best_name}):")
    print(classification_report(y_val, y_pred, target_names=['Legit', 'Fraud']))
 
    joblib.dump(best_model, MODELS_DIR / 'best_model.pkl')
    print(f"\nBest model saved to: models/best_model.pkl")
 
    comparison.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
 
    metadata = {
        'best_model'     : best_name,
        'pr_auc'         : float(comparison.iloc[0]['pr_auc']),
        'roc_auc'        : float(comparison.iloc[0]['roc_auc']),
        'f1'             : float(comparison.iloc[0]['f1']),
        'threshold'      : 0.5,
        'mlflow_run_id'  : results[best_name][2],
    }
    with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    return best_model, best_name, comparison
 
def main():
    print("=" * 60)
    print("   MODEL TRAINING + MLFLOW")
    print("=" * 60)
 
    X_train, y_train, X_val, y_val = load_processed_data()

    results = {}
 
    lr_model, lr_metrics, lr_run = train_baseline(X_train, y_train, X_val, y_val)
    results['Logistic Regression'] = (lr_model, lr_metrics, lr_run)
 
    lgbm_model, lgbm_metrics, lgbm_run = train_lightgbm(X_train, y_train, X_val, y_val)
    results['LightGBM'] = (lgbm_model, lgbm_metrics, lgbm_run)
 
    xgb_model, xgb_metrics, xgb_run, xgb_params = tune_xgboost(
        X_train, y_train, X_val, y_val, n_trials=30
    )
    results['XGBoost Tuned'] = (xgb_model, xgb_metrics, xgb_run)
 
    best_model, best_name, comparison = compare_and_save(results, X_val, y_val)
 
    generate_shap(best_model, X_val, model_name=best_name)

if __name__ == '__main__':
    main()