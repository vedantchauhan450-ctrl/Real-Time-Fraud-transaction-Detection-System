"""Loading, feature engineering and splitting for card-fraud data."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "Class"


def load_data(path):
    return pd.read_csv(path).drop_duplicates().reset_index(drop=True)


def engineer_features(df):
    """Log-scale the skewed Amount and encode hour-of-day cyclically."""
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    hour = (df["Time"] // 3600) % 24
    df["HourSin"] = np.sin(2 * np.pi * hour / 24)
    df["HourCos"] = np.cos(2 * np.pi * hour / 24)
    return df.drop(columns=["Time", "Amount"])


def split_data(df, seed=42):
    """Stratified 60/20/20 train/val/test (fraud is ~0.2% so stratify!)."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return X_tr, X_va, X_te, y_tr, y_va, y_te


def normal_only(X, y, max_rows=None, seed=42):
    """Anomaly detectors are fitted on legitimate transactions only."""
    Xn = X[y == 0]
    return Xn.sample(max_rows, random_state=seed) if max_rows and len(Xn) > max_rows else Xn
