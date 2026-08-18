"""
run_performance_comparison.py
------------------------------
Reproduces the performance and quality comparison described in
Sections III-C and IV of the paper:

  - Table II: wall-clock time (mean +/- std over 5 repetitions) for the
    Original vs. AI-generated K-Means, for K = 2, 3, 4, plus speedup.
  - Table III: inertia (Original vs. AI), their difference, and cluster
    agreement between the two implementations (labels aligned with the
    Hungarian algorithm before comparing, since cluster IDs are
    arbitrary).
  - Fig. 1: execution-time bar chart.
  - Fig. 2: inertia bar chart.

Usage:
    python run_performance_comparison.py --data marketing_campaign.csv
    python run_performance_comparison.py            # synthetic data
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

from data_utils import load_and_preprocess
from kmeans_original import KMeansOriginal
from kmeans_ai import KMeansAI

REPETITIONS = 5
K_VALUES = [2, 3, 4]
MAX_ITERATIONS = 100
TOL = 1e-4


def time_runs(model_cls, X: np.ndarray, k: int, n_reps: int = REPETITIONS):
    """Run a model n_reps times, returning (times, last_fitted_model)."""
    times = []
    model = None
    for rep in range(n_reps):
        model = model_cls(n_clusters=k, max_iterations=MAX_ITERATIONS, tol=TOL, random_state=rep)
        start = time.perf_counter()
        model.fit(X)
        times.append(time.perf_counter() - start)
    return np.array(times), model


def cluster_agreement(labels_a: np.ndarray, labels_b: np.ndarray, k: int) -> float:
    """
    Cluster labels are arbitrary permutations of {0..k-1} between two
    independent runs. Find the best label alignment (Hungarian algorithm
    on the confusion matrix) before computing point-wise agreement.
    """
    confusion = np.zeros((k, k), dtype=int)
    for a, b in zip(labels_a, labels_b):
        confusion[a, b] += 1

    row_idx, col_idx = linear_sum_assignment(-confusion)
    matched = confusion[row_idx, col_idx].sum()
    return matched / len(labels_a)


def run_performance_table(X: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_VALUES:
        orig_times, _ = time_runs(KMeansOriginal, X, k)
        ai_times, _ = time_runs(KMeansAI, X, k)

        speedup = orig_times.mean() / ai_times.mean()
        rows.append({
            "K": k,
            "Original mean (s)": orig_times.mean(),
            "Original std (s)": orig_times.std(),
            "AI mean (s)": ai_times.mean(),
            "AI std (s)": ai_times.std(),
            "Speedup": speedup,
        })

    table = pd.DataFrame(rows)
    print("\n=== Table II: Performance Comparison ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return table


def run_quality_table(X: np.ndarray) -> tuple[pd.DataFrame, dict]:
    rows = []
    fitted = {}
    for k in K_VALUES:
        orig = KMeansOriginal(n_clusters=k, max_iterations=MAX_ITERATIONS, tol=TOL, random_state=0).fit(X)
        ai = KMeansAI(n_clusters=k, max_iterations=MAX_ITERATIONS, tol=TOL, random_state=0).fit(X)

        agreement = cluster_agreement(orig.labels_, ai.labels_, k)
        n = len(X)
        rows.append({
            "K": k,
            "Inertia (Original)": orig.inertia_,
            "Inertia (AI)": ai.inertia_,
            "Difference": abs(orig.inertia_ - ai.inertia_),
            "Cluster Agreement": f"{int(round(agreement * n))} / {n} ({agreement * 100:.1f}%)",
        })
        fitted[k] = (orig, ai)

    table = pd.DataFrame(rows)
    print("\n=== Table III: Clustering Quality Comparison ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return table, fitted


def plot_execution_time(perf_table: pd.DataFrame, out_path: str) -> None:
    x = np.arange(len(perf_table))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, perf_table["Original mean (s)"], width, label="Original", color="tab:blue")
    plt.bar(x + width / 2, perf_table["AI mean (s)"], width, label="AI-Generated", color="tab:orange")
    plt.xticks(x, perf_table["K"])
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Fig. 1 -> {out_path}")


def plot_inertia(quality_table: pd.DataFrame, out_path: str) -> None:
    x = np.arange(len(quality_table))
    width = 0.35

    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, quality_table["Inertia (Original)"], width, label="Original", color="tab:blue")
    plt.bar(x + width / 2, quality_table["Inertia (AI)"], width, label="AI-Generated", color="tab:orange")
    plt.xticks(x, quality_table["K"])
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Inertia Comparison (Lower is Better)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved Fig. 2 -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Original vs. AI-generated K-Means comparison")
    parser.add_argument("--data", type=str, default="marketing_campaign.csv")
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    X, feature_names = load_and_preprocess(args.data)

    perf_table = run_performance_table(X)
    quality_table, _ = run_quality_table(X)

    plot_execution_time(perf_table, os.path.join(args.out, "fig1_execution_time.png"))
    plot_inertia(quality_table, os.path.join(args.out, "fig2_inertia_comparison.png"))

    print(f"\nAll figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
