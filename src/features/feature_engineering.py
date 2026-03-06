import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"
PROC_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    print("Loading raw data...")
    train_txn = pd.read_csv(RAW_DIR / 'train_transaction.csv')
    train_id  = pd.read_csv(RAW_DIR / 'train_identity.csv')
    df = train_txn.merge(train_id, on='TransactionID', how='left')
    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} cols")
    return df

def drop_high_missing(df, threshold=0.9):
    """Drop columns with more than `threshold` missing values."""
    missing_pct = df.isnull().mean()
    drop_cols = missing_pct[missing_pct > threshold].index.tolist()
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} columns with >{threshold*100:.0f}% missing")
    return df, drop_cols

def add_time_features(df):
    """Extract time-based signals from TransactionDT."""
    print("Adding time features...")

    df['tx_hour']     = (df['TransactionDT'] // 3600) % 24
    df['tx_day']      = (df['TransactionDT'] // (3600 * 24)) % 7    
    df['tx_day_abs']  =  df['TransactionDT'] // (3600 * 24)

    # Is transaction at night? (midnight to 6am = higher fraud risk)
    df['is_night']    = df['tx_hour'].between(0, 6).astype(int)

    # Is weekend proxy
    df['is_weekend']  = df['tx_day'].isin([5, 6]).astype(int)

    return df

def add_amount_features(df):
    """Log transform + amount-based signals."""
    print("Adding amount features...")

    df['log_amount'] = np.log1p(df['TransactionAmt'])
    df['amount_cents'] = (df['TransactionAmt'] % 1).round(2)
    df['is_round_amount'] = (df['amount_cents'] == 0).astype(int)

    card_amt_mean = df.groupby('card1')['TransactionAmt'].transform('mean')
    card_amt_std = df.groupby('card1')['TransactionAmt'].transform('std').fillna(1)
    df['amount_card_zscore'] = (df['TransactionAmt'] - card_amt_mean) / card_amt_std

    return df

def add_velocity_features(df):
    """Count-based features — how often is a card/email being used?"""
    print("Adding velocity features...")

    # Transaction count per card
    df['card1_count']  = df.groupby('card1')['TransactionID'].transform('count')
    df['card2_count']  = df.groupby('card2')['TransactionID'].transform('count')

    # Transaction count per email domain
    if 'P_emaildomain' in df.columns:
        df['pemail_count'] = df.groupby('P_emaildomain')['TransactionID'].transform('count')

    # Transaction count per addr
    if 'addr1' in df.columns:
        df['addr1_count']  = df.groupby('addr1')['TransactionID'].transform('count')

    return df

def target_encode(df, cat_cols, target='isFraud', smoothing=10):
    """
    Target encoding: replace category with its fraud rate.
    Smoothing prevents overfitting on rare categories.
    """
    print(f"Target encoding: {cat_cols}")
    global_mean = df[target].mean()
    encoders    = {}

    for col in cat_cols:
        if col not in df.columns:
            continue

        stats = df.groupby(col)[target].agg(['mean', 'count'])
        # Smoothed target encode formula
        smooth = (stats['count'] * stats['mean'] + smoothing * global_mean) / \
                 (stats['count'] + smoothing)
        encoders[col] = smooth.to_dict()
        df[f'{col}_encoded'] = df[col].map(smooth).fillna(global_mean)

    return df, encoders

def label_encode_cats(df, exclude_cols):
    """Label encode low-cardinality string columns."""
    print("Label encoding remaining categoricals...")
    label_encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_cols]

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna('missing')
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def fill_missing(df):
    """Fill remaining NaNs — median for numeric, -999 for categoricals."""
    print("Filling missing values...")

    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    return df

def pca_v_features(df, n_components=30):
    """Reduce 300+ V features to n_components principal components."""
    print(f"PCA on V features → {n_components} components...")
    v_cols = [c for c in df.columns if c.startswith('V')]

    if len(v_cols) == 0:
        print("   No V columns found, skipping PCA")
        return df, None

    v_data = df[v_cols].fillna(0)
    pca    = PCA(n_components=n_components, random_state=42)
    v_pca  = pca.fit_transform(v_data)

    # Add PCA components as new columns
    pca_df = pd.DataFrame(
        v_pca,
        columns=[f'V_pca_{i}' for i in range(n_components)],
        index=df.index
    )
    df = pd.concat([df.drop(columns=v_cols), pca_df], axis=1)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"   Variance explained by {n_components} components: {explained:.1f}%")

    return df, pca

def time_based_split(df, split_percentile=0.8):
    """Split based on time, NOT randomly — prevents data leakage."""
    print("Time-based train/val split...")

    split_day  = df['tx_day_abs'].quantile(split_percentile)
    train_mask = df['tx_day_abs'] <= split_day
    val_mask   = df['tx_day_abs'] >  split_day

    train_df = df[train_mask].copy()
    val_df   = df[val_mask].copy()

    print(f"   Train: {len(train_df):,} rows | Val: {len(val_df):,} rows")
    print(f"   Train fraud rate: {train_df['isFraud'].mean()*100:.2f}%")
    print(f"   Val   fraud rate: {val_df['isFraud'].mean()*100:.2f}%")

    return train_df, val_df

def run_pipeline():
    print("=" * 60)
    print("   FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
 
    df = load_data()
 
    df, dropped_cols = drop_high_missing(df, threshold=0.9)
 
    df = add_time_features(df)
    df = add_amount_features(df)
    df = add_velocity_features(df)
 
    target_encode_cols = ['card1', 'card2', 'card3', 'card5',
                          'P_emaildomain', 'R_emaildomain', 'addr1', 'addr2']
    df, te_encoders = target_encode(df, target_encode_cols)
 
    exclude = ['TransactionID', 'isFraud'] + target_encode_cols
    df, le_encoders = label_encode_cats(df, exclude_cols=exclude)
 
    df = fill_missing(df)
 
    df, pca_model = pca_v_features(df, n_components=30)
 
    drop_raw = ['TransactionID', 'TransactionDT']
    df = df.drop(columns=[c for c in drop_raw if c in df.columns])

    print(f"\nFinal feature set: {df.shape[1]} columns")
 
    train_df, val_df = time_based_split(df)
 
    feature_cols = [c for c in train_df.columns if c != 'isFraud']
    X_train = train_df[feature_cols]
    y_train = train_df['isFraud']
    X_val   = val_df[feature_cols]
    y_val   = val_df['isFraud']
 
    print("\nSaving processed data...")
    X_train.to_csv(PROC_DIR / 'X_train.csv', index=False)
    y_train.to_csv(PROC_DIR / 'y_train.csv', index=False)
    X_val.to_csv(PROC_DIR / 'X_val.csv', index=False)
    y_val.to_csv(PROC_DIR / 'y_val.csv', index=False)
 
    artifacts = {
        'target_encoders': te_encoders,
        'label_encoders':  le_encoders,
        'dropped_cols':    dropped_cols,
        'feature_cols':    feature_cols,
    }
    joblib.dump(artifacts, MODELS_DIR / 'feature_artifacts.pkl')
    if pca_model:
        joblib.dump(pca_model, MODELS_DIR / 'pca_model.pkl')
 
    summary = {
        'total_features':    len(feature_cols),
        'train_rows':        len(X_train),
        'val_rows':          len(X_val),
        'train_fraud_rate':  round(y_train.mean() * 100, 2),
        'val_fraud_rate':    round(y_val.mean() * 100, 2),
        'feature_cols':      feature_cols,
    }
    with open(PROC_DIR / 'feature_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("=" * 60)
    print(f"   X_train : {X_train.shape}")
    print(f"   X_val   : {X_val.shape}")
    print(f"   Artifacts saved to: models/")
    print(f"   Data saved to     : data/processed/")
    print("=" * 60)

    return X_train, y_train, X_val, y_val, feature_cols
 
if __name__ == '__main__':
    X_train, y_train, X_val, y_val, feature_cols = run_pipeline()
