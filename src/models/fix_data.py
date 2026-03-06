import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"

def fix_string_columns(df, df_name=""):
    """Find and encode any remaining string columns."""
    str_cols = df.select_dtypes(include=['object']).columns.tolist()

    if len(str_cols) == 0:
        print(f"{df_name}: No string columns found")
        return df

    print(f"{df_name}: Found {len(str_cols)} string columns → {str_cols}")

    for col in str_cols:
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna('missing')
        df[col] = le.fit_transform(df[col])
        print(f"  Encoded: {col}")

    return df

def fix_inf_nan(df, df_name=""):
    """Replace inf and remaining NaN with 0."""
    # Replace inf
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"  {df_name}: Found {inf_count} inf values → replacing with 0")
        df = df.replace([np.inf, -np.inf], 0)

    # Replace NaN
    nan_count = df.isnull().sum().sum()
    if nan_count > 0:
        print(f"  {df_name}: Found {nan_count} NaN values → replacing with 0")
        df = df.fillna(0)

    return df

def run_fix():
    print(" Running data fix...\n")

    for split in ['train', 'val']:
        X = pd.read_csv(PROC_DIR / f'X_{split}.csv')
        print(f"--- X_{split} ({X.shape}) ---")
        X = fix_string_columns(X, f"X_{split}")
        X = fix_inf_nan(X, f"X_{split}")
        X.to_csv(PROC_DIR / f'X_{split}.csv', index=False)
        print(f" Saved fixed X_{split}.csv\n")

if __name__ == '__main__':
    run_fix()