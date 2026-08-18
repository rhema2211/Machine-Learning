"""
Marketing Campaign Dataset: Preprocessing, Statistical Analysis & Clustering
=============================================================================

Implements the methodology described in the paper's Section III (Methodology)
and Section IV (Results and Discussion):

    1. Dataset loading
    2. Data preprocessing (Label Encoding + One-Hot Encoding)
    3. Custom Minkowski distance (p = 1..10), verified against SciPy
    4. Custom dot product & Euclidean norm, verified against NumPy
    5. Custom mean / variance / standard deviation, verified against NumPy
    6. Histogram analysis (raw + density)
    7. K-Means clustering implemented from scratch, verified against
       scikit-learn's KMeans

Dataset
-------
Expects the well-known "marketing_campaign.csv" (Kaggle: Customer Personality
Analysis), which uses '\t' as a separator and includes columns such as:
Education, Marital_Status, Income, Kidhome, Teenhome, Recency, ...

If the file is not found, a synthetic dataset with the same schema is
generated so the pipeline can still be run and demoed end-to-end.

Usage
-----
    python marketing_campaign_analysis.py --data marketing_campaign.csv
    python marketing_campaign_analysis.py            # uses synthetic data
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski as scipy_minkowski
from sklearn.cluster import KMeans as SKKMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

NUMERICAL_COLS = ["Income", "Kidhome", "Teenhome", "Recency"]
CATEGORICAL_COLS = ["Education", "Marital_Status"]


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_dataset(path: str | None) -> pd.DataFrame:
    """Load marketing_campaign.csv if available, else synthesize a stand-in."""
    if path and os.path.exists(path):
        df = pd.read_csv(path, sep="\t")
        print(f"Loaded real dataset from '{path}' with shape {df.shape}")
    else:
        print("No dataset file found — generating synthetic demo data "
              "with the same schema (Table I in the paper).")
        df = _generate_synthetic_dataset(n=2000)

    # Keep only the columns we need for this pipeline, drop missing rows
    keep_cols = NUMERICAL_COLS + CATEGORICAL_COLS
    df = df[[c for c in keep_cols if c in df.columns]].dropna().reset_index(drop=True)
    return df


def _generate_synthetic_dataset(n: int) -> pd.DataFrame:
    educations = ["Basic", "2n Cycle", "Graduation", "Master", "PhD"]
    marital_statuses = ["Single", "Married", "Together", "Divorced", "Widow"]

    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.DataFrame({
        "Education": rng.choice(educations, size=n, p=[0.05, 0.10, 0.45, 0.25, 0.15]),
        "Marital_Status": rng.choice(marital_statuses, size=n),
        "Income": rng.lognormal(mean=10.85, sigma=0.35, size=n).round(2),
        "Kidhome": rng.integers(0, 3, size=n),
        "Teenhome": rng.integers(0, 3, size=n),
        "Recency": rng.integers(0, 100, size=n),
    })
    return df


# --------------------------------------------------------------------------- #
# 2. Preprocessing: Label Encoding + One-Hot Encoding  (Section III-B)
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_before = df.shape[1]

    # --- Label Encoding (kept as a separate reference column set) ---
    label_encoded = df.copy()
    label_encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        label_encoded[col + "_label"] = le.fit_transform(df[col])
        label_encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    # --- One-Hot Encoding (used downstream for distance/clustering) ---
    ohe_df = pd.get_dummies(df, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)

    n_after = ohe_df.shape[1]
    info = {
        "n_before": n_before,
        "n_after": n_after,
        "label_encoders": label_encoders,
    }

    print("\n=== Table III: Feature Count Before/After Encoding ===")
    print(f"Before Encoding : {n_before}")
    print(f"After One-Hot   : {n_after}")

    return ohe_df, info


# --------------------------------------------------------------------------- #
# 3. Minkowski distance (custom vs SciPy)   (Section III-C, III-D)
# --------------------------------------------------------------------------- #
def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """Custom Minkowski distance implementation from scratch."""
    return float(np.sum(np.abs(x - y) ** p) ** (1.0 / p))


def run_minkowski_analysis(vec_a: np.ndarray, vec_b: np.ndarray) -> pd.DataFrame:
    rows = []
    for p in range(1, 11):
        custom_d = minkowski_distance(vec_a, vec_b, p)
        scipy_d = scipy_minkowski(vec_a, vec_b, p)
        rows.append({"p": p, "Custom": round(custom_d, 4), "SciPy": round(scipy_d, 4)})

    table = pd.DataFrame(rows)
    print("\n=== Table IV/V: Minkowski Distance (Custom vs SciPy) ===")
    print(table.to_string(index=False))
    return table


def plot_minkowski(table: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(table["p"], table["Custom"], marker="o", color="tab:blue")
    plt.title("Minkowski Distance vs Order Parameter (p)")
    plt.xlabel("Order Parameter (p)")
    plt.ylabel("Minkowski Distance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Fig. 1/4 -> {out_path}")


# --------------------------------------------------------------------------- #
# 4. Dot product & Euclidean norm (custom vs NumPy)   (Section III-D, IV-E)
# --------------------------------------------------------------------------- #
def custom_dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(sum(ai * bi for ai, bi in zip(a, b)))


def custom_euclidean_norm(a: np.ndarray) -> float:
    return float(sum(ai ** 2 for ai in a) ** 0.5)


def run_vector_ops(vec_a: np.ndarray, vec_b: np.ndarray) -> pd.DataFrame:
    dot_custom, dot_numpy = custom_dot_product(vec_a, vec_b), np.dot(vec_a, vec_b)
    norm_custom, norm_numpy = custom_euclidean_norm(vec_a), np.linalg.norm(vec_a)

    table = pd.DataFrame({
        "Operation": ["Dot Product", "Vector Norm"],
        "Custom": [round(dot_custom, 4), round(norm_custom, 4)],
        "NumPy": [round(float(dot_numpy), 4), round(float(norm_numpy), 4)],
    })
    print("\n=== Table VI: Vector Operations (Custom vs NumPy) ===")
    print(table.to_string(index=False))
    return table


# --------------------------------------------------------------------------- #
# 5. Statistical measures (custom vs NumPy)   (Section III-E, IV-F)
# --------------------------------------------------------------------------- #
def custom_mean(x: np.ndarray) -> float:
    return float(sum(x) / len(x))


def custom_variance(x: np.ndarray) -> float:
    m = custom_mean(x)
    return float(sum((xi - m) ** 2 for xi in x) / len(x))


def custom_std(x: np.ndarray) -> float:
    return float(custom_variance(x) ** 0.5)


def run_statistics(x: np.ndarray) -> pd.DataFrame:
    table = pd.DataFrame({
        "Statistic": ["Mean", "Variance", "Standard Deviation"],
        "Custom": [round(custom_mean(x), 4), round(custom_variance(x), 4), round(custom_std(x), 4)],
        "NumPy": [round(float(np.mean(x)), 4), round(float(np.var(x)), 4), round(float(np.std(x)), 4)],
    })
    print("\n=== Table VII: Statistical Measures (Custom vs NumPy) ===")
    print(table.to_string(index=False))
    return table


# --------------------------------------------------------------------------- #
# 6. Histogram analysis   (Section III-F, IV-G)
# --------------------------------------------------------------------------- #
def plot_histograms(income: np.ndarray, out_dir: str) -> None:
    mean_income = np.mean(income)
    std_income = np.std(income)

    # Raw frequency histogram
    plt.figure(figsize=(7, 4))
    plt.hist(income, bins=30, color="tab:blue", edgecolor="white")
    plt.axvline(mean_income, color="red", linestyle="--", label=f"Mean: {mean_income:.2f}")
    plt.axvline(mean_income + std_income, color="gray", linestyle=":", label="+1 Std")
    plt.axvline(mean_income - std_income, color="gray", linestyle=":", label="-1 Std")
    plt.title("Histogram of Income")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    path1 = os.path.join(out_dir, "fig2_income_histogram.png")
    plt.savefig(path1, dpi=150)
    plt.close()
    print(f"Saved Fig. 2 -> {path1}")

    # Density histogram
    plt.figure(figsize=(7, 4))
    plt.hist(income, bins=30, density=True, color="tab:blue", edgecolor="white")
    plt.title("Density Histogram of Income")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.tight_layout()
    path2 = os.path.join(out_dir, "fig3_income_density.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Saved Fig. 3 -> {path2}")


# --------------------------------------------------------------------------- #
# 7. K-Means clustering from scratch (vs scikit-learn)   (Section III-G, IV)
# --------------------------------------------------------------------------- #
@dataclass
class KMeansResult:
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    n_iter: int


def kmeans_from_scratch(
    X: np.ndarray, k: int, max_iter: int = 300, tol: float = 1e-4, seed: int = RANDOM_SEED
) -> KMeansResult:
    rng = np.random.default_rng(seed)
    # Step 1: initialize centroids by picking k random points
    init_idx = rng.choice(len(X), size=k, replace=False)
    centroids = X[init_idx].copy()

    for it in range(1, max_iter + 1):
        # Step 2: distance of each sample from every centroid (Euclidean, p=2)
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        # Step 3: assign each sample to nearest centroid
        labels = np.argmin(dists, axis=1)

        # Step 4: recompute centroids
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
            for j in range(k)
        ])

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = np.argmin(dists, axis=1)
    inertia = float(np.sum((X - centroids[labels]) ** 2))

    return KMeansResult(centroids=centroids, labels=labels, inertia=inertia, n_iter=it)


def elbow_method(X: np.ndarray, k_range: range, out_path: str) -> None:
    inertias = []
    for k in k_range:
        result = kmeans_from_scratch(X, k)
        inertias.append(result.inertia)

    plt.figure(figsize=(7, 4))
    plt.plot(list(k_range), inertias, marker="o", color="tab:blue")
    plt.title("Elbow Method for Optimal K")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Fig. 5 (elbow) -> {out_path}")


def run_kmeans_comparison(X: np.ndarray, k: int) -> pd.DataFrame:
    scratch = kmeans_from_scratch(X, k)
    sk = SKKMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X)

    counts_scratch = pd.Series(scratch.labels).value_counts().sort_index()
    counts_sklearn = pd.Series(sk.labels_).value_counts().sort_index()

    table = pd.DataFrame({
        "Cluster": [f"Cluster {i+1}" for i in range(k)],
        "Custom KMeans (n)": counts_scratch.values,
        "sklearn KMeans (n)": counts_sklearn.values,
    })

    print("\n=== Table VIII: Distribution of Samples Across Clusters ===")
    print(table.to_string(index=False))
    print(f"\nCustom KMeans inertia : {scratch.inertia:.2f}")
    print(f"sklearn KMeans inertia: {sk.inertia_:.2f}")
    return table


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Marketing campaign preprocessing & clustering pipeline")
    parser.add_argument("--data", type=str, default="marketing_campaign.csv", help="Path to marketing_campaign.csv")
    parser.add_argument("--k", type=int, default=3, help="Number of clusters for K-Means")
    parser.add_argument("--out", type=str, default="outputs", help="Directory to save figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Load
    df = load_dataset(args.data)

    # 2. Preprocess
    encoded_df, info = preprocess(df)

    # 3. Minkowski distance demo: compare first two customer rows (numeric cols only)
    numeric_matrix = encoded_df[NUMERICAL_COLS].to_numpy(dtype=float)
    vec_a, vec_b = numeric_matrix[0], numeric_matrix[1]
    mink_table = run_minkowski_analysis(vec_a, vec_b)
    plot_minkowski(mink_table, os.path.join(args.out, "fig1_minkowski_distance.png"))

    # 4. Vector operations
    run_vector_ops(vec_a, vec_b)

    # 5. Statistics on Income
    income = df["Income"].to_numpy(dtype=float)
    run_statistics(income)

    # 6. Histograms
    plot_histograms(income, args.out)

    # 7. K-Means clustering (on standardized numeric + one-hot features)
    X = encoded_df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    elbow_method(X_std, range(2, 8), os.path.join(args.out, "fig5_elbow_method.png"))
    run_kmeans_comparison(X_std, k=args.k)

    print(f"\nAll figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
