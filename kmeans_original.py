"""
kmeans_original.py
-------------------
Original, hand-written K-Means implementation (Lab 03 style).

Characteristics (as described in the paper, Section III-A):
  - Plain Python / NumPy only.
  - Centroids initialised with a manually written K-means++ procedure.
  - Cluster assignment done through explicit Python loops over every
    data point and every centroid (not vectorised).

This is the "Original" implementation referenced throughout the paper's
Tables II and III and Figures 1-2.
"""

from __future__ import annotations
import numpy as np


class KMeansOriginal:
    def __init__(self, n_clusters: int, max_iterations: int = 100, tol: float = 1e-4, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tol = tol
        self.random_state = random_state

        self.centroids_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int = 0

    # ------------------------------------------------------------------ #
    # K-means++ initialisation, written by hand
    # ------------------------------------------------------------------ #
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        centroids = []
        first_idx = rng.integers(0, n_samples)
        centroids.append(X[first_idx])

        for _ in range(1, self.n_clusters):
            # distance of every point to the nearest already-chosen centroid
            dist_sq = np.array([
                min(np.sum((x - c) ** 2) for c in centroids)
                for x in X
            ])
            probs = dist_sq / dist_sq.sum()
            next_idx = rng.choice(n_samples, p=probs)
            centroids.append(X[next_idx])

        return np.array(centroids)

    # ------------------------------------------------------------------ #
    # Explicit loop-based assignment step (the slow part)
    # ------------------------------------------------------------------ #
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=int)

        for i in range(n_samples):
            best_dist = np.inf
            best_j = 0
            for j in range(self.n_clusters):
                d = 0.0
                for f in range(X.shape[1]):
                    diff = X[i, f] - centroids[j, f]
                    d += diff * diff
                if d < best_dist:
                    best_dist = d
                    best_j = j
            labels[i] = best_j

        return labels

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray, prev_centroids: np.ndarray) -> np.ndarray:
        new_centroids = np.empty_like(prev_centroids)
        for j in range(self.n_clusters):
            mask = labels == j
            if np.any(mask):
                new_centroids[j] = X[mask].mean(axis=0)
            else:
                new_centroids[j] = prev_centroids[j]
        return new_centroids

    def fit(self, X: np.ndarray) -> "KMeansOriginal":
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
