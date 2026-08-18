"""
kmeans_ai.py
------------
AI-generated implementation, produced through a series of prompts to
ChatGPT (Section III-A of the paper).

KMeansAI uses the same K-means++ initialisation idea as the original,
hand-written version, but the cluster-assignment step is vectorised with
scipy.spatial.distance.cdist instead of explicit Python loops.

The supporting functions below (Minkowski distance, dot product,
Euclidean norm, mean/variance/standard deviation, label encoding and
one-hot encoding) were generated the same way and are used purely as a
reference implementation for the unit tests in test_kmeans.py. Each
block is commented with "AI-generated" to mark its origin, per the
paper's methodology.

Note on the encoding bug described in the paper (Section IV-A):
The first version of ai_label_encode() returned NumPy int64 values,
which failed a unit test that checked for the built-in Python int type.
The fix (kept here) is to cast explicitly to native Python int.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist


# ===================================================================== #
# AI-generated: K-Means (vectorised assignment)
# ===================================================================== #
class KMeansAI:
    def __init__(self, n_clusters: int, max_iterations: int = 100, tol: float = 1e-4, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        centroids = [X[rng.integers(0, n_samples)]]
        for _ in range(1, self.n_clusters):
            dist_sq = cdist(X, np.array(centroids)).min(axis=1) ** 2
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        # Vectorised: one cdist call computes all pairwise distances at once,
        # replacing the original implementation's nested Python loops.
        distances = cdist(X, centroids)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray, prev_centroids: np.ndarray) -> np.ndarray:
        new_centroids = np.empty_like(prev_centroids)
        for j in range(self.n_clusters):
            mask = labels == j
            new_centroids[j] = X[mask].mean(axis=0) if np.any(mask) else prev_centroids[j]
        return new_centroids

    def fit(self, X: np.ndarray) -> "KMeansAI":
        X = np.asarray(X, dtype=float)
        centroids = self._kmeans_plus_plus_init(X)

        for it in range(1, self.max_iterations + 1):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels, centroids)

            shift = np.linalg.norm(new_centroids - centroids)
            centroids = new_centroids
            if shift < self.tol:
                break

        self.n_iter_ = it
        self.centroids_ = centroids
        self.labels_ = self._assign_clusters(X, centroids)
        self.inertia_ = float(np.sum((X - centroids[self.labels_]) ** 2))
        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels_


# ===================================================================== #
# AI-generated: distance functions
# ===================================================================== #
def minkowski_distance(x: np.ndarray, y: np.ndarray, p: float) -> float:
    """AI-generated. Custom Minkowski distance, p >= 1."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return float(np.sum(np.abs(x - y) ** p) ** (1.0 / p))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """AI-generated. Custom dot product."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.sum(a * b))


def euclidean_norm(a: np.ndarray) -> float:
    """AI-generated. Custom Euclidean (L2) norm."""
    a = np.asarray(a, dtype=float)
    return float(np.sqrt(np.sum(a ** 2)))


# ===================================================================== #
# AI-generated: statistical functions
# ===================================================================== #
def compute_mean(x: np.ndarray) -> float:
    """AI-generated. Custom mean."""
    x = np.asarray(x, dtype=float)
    return float(np.sum(x) / len(x))


def compute_variance(x: np.ndarray) -> float:
    """AI-generated. Custom population variance."""
    x = np.asarray(x, dtype=float)
    m = compute_mean(x)
    return float(np.sum((x - m) ** 2) / len(x))


def compute_std(x: np.ndarray) -> float:
    """AI-generated. Custom standard deviation."""
    return float(np.sqrt(compute_variance(x)))


# ===================================================================== #
# AI-generated: encoding functions
# ===================================================================== #
def ai_label_encode(column) -> list[int]:
    """
    AI-generated. Label-encodes a categorical column.

    Bug fix applied (see paper, Section IV-A): the first version returned
    NumPy int64 values, which failed a unit test asserting the built-in
    Python `int` type. Fixed here by casting explicitly to native `int`.
    """
    categories = sorted(set(column))
    mapping = {cat: int(i) for i, cat in enumerate(categories)}
    return [int(mapping[val]) for val in column]


def ai_one_hot_encode(column) -> tuple[list[str], np.ndarray]:
    """AI-generated. One-hot encodes a categorical column."""
    categories = sorted(set(column))
    encoded = np.zeros((len(column), len(categories)), dtype=int)
    for row_idx, val in enumerate(column):
        col_idx = categories.index(val)
        encoded[row_idx, col_idx] = 1
    return categories, encoded
