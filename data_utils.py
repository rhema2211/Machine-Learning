"""
data_utils.py
-------------
Dataset loading and preprocessing, matching Section III-B of the paper.

Steps:
    1. Load marketing_campaign.csv (falls back to a synthetic dataset with
       the same schema if the file isn't present).
    2. Drop non-informative identifier columns: ID, Z_CostContact, Z_Revenue
       (and the free-text/date columns Dt_Customer, Education,
       Marital_Status, which are categorical/non-numeric and are not part
       of the 23 numeric clustering features).
    3. Drop rows containing missing values.
    4. Standardise the remaining numeric columns (zero mean, unit variance).

The paper reports this leaves 2,216 usable rows across 23 features.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

DROP_COLS = ["ID", "Z_CostContact", "Z_Revenue", "Dt_Customer", "Education", "Marital_Status"]

NUMERIC_FEATURES = [
    "Year_Birth", "Income", "Kidhome", "Teenhome", "Recency",
    "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts",
    "MntSweetProducts", "MntGoldProds", "NumDealsPurchases",
    "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases",
    "NumWebVisitsMonth", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5",
    "AcceptedCmp1", "AcceptedCmp2", "Complain", "Response",
]  # 23 features


def load_and_preprocess(path: str = "marketing_campaign.csv", n_rows: int = 2216):
    """Returns (X_standardised: np.ndarray, feature_names: list[str])."""
    if path and os.path.exists(path):
        df = pd.read_csv(path, sep="\t")
        print(f"Loaded real dataset from '{path}' with shape {df.shape}")
    else:
        print("No dataset file found — generating synthetic demo data "
              f"with the marketing_campaign schema ({n_rows} rows, 23 numeric features).")
        df = _generate_synthetic_dataset(n_rows)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    feature_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    df = df[feature_cols].dropna().reset_index(drop=True)

    print(f"Usable rows after dropping missing values: {len(df)} across {len(feature_cols)} features")

    X = df.to_numpy(dtype=float)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    return X_std, feature_cols


def _generate_synthetic_dataset(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {
        "ID": np.arange(n),
        "Year_Birth": rng.integers(1940, 2000, size=n),
        "Income": rng.lognormal(mean=10.85, sigma=0.35, size=n).round(2),
        "Kidhome": rng.integers(0, 3, size=n),
        "Teenhome": rng.integers(0, 3, size=n),
        "Recency": rng.integers(0, 100, size=n),
        "MntWines": rng.integers(0, 1500, size=n),
        "MntFruits": rng.integers(0, 200, size=n),
        "MntMeatProducts": rng.integers(0, 1800, size=n),
        "MntFishProducts": rng.integers(0, 260, size=n),
        "MntSweetProducts": rng.integers(0, 260, size=n),
        "MntGoldProds": rng.integers(0, 320, size=n),
        "NumDealsPurchases": rng.integers(0, 15, size=n),
        "NumWebPurchases": rng.integers(0, 27, size=n),
        "NumCatalogPurchases": rng.integers(0, 28, size=n),
        "NumStorePurchases": rng.integers(0, 13, size=n),
        "NumWebVisitsMonth": rng.integers(0, 20, size=n),
        "AcceptedCmp3": rng.integers(0, 2, size=n),
        "AcceptedCmp4": rng.integers(0, 2, size=n),
        "AcceptedCmp5": rng.integers(0, 2, size=n),
        "AcceptedCmp1": rng.integers(0, 2, size=n),
        "AcceptedCmp2": rng.integers(0, 2, size=n),
        "Complain": rng.integers(0, 2, size=n),
        "Response": rng.integers(0, 2, size=n),
        "Z_CostContact": np.full(n, 3),
        "Z_Revenue": np.full(n, 11),
    }
    return pd.DataFrame(data)
