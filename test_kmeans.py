"""
test_kmeans.py
--------------
Unit test suite referenced in Table I of the paper.

    Test Class                Cases   Covers
    TestKMeans                 4      Initialisation, fit_predict, inertia, convergence
    TestDistanceFunctions       4      Minkowski (p=1, p=2), dot product, Euclidean norm
    TestStatisticalFunctions    3      Mean, variance, standard deviation
    TestEncodingFunctions       2      Label encoding, one-hot encoding
    ------------------------------------------------------------------
    Total                      13

Run with:
    pytest test_kmeans.py -v
"""

import numpy as np
import pytest
from scipy.spatial.distance import minkowski as scipy_minkowski

from kmeans_ai import (
    KMeansAI,
    minkowski_distance,
    dot_product,
    euclidean_norm,
    compute_mean,
    compute_variance,
    compute_std,
    ai_label_encode,
    ai_one_hot_encode,
)


# ===================================================================== #
# TestKMeans (4 cases)
# ===================================================================== #
class TestKMeans:
    def setup_method(self):
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(loc=[0, 0], scale=0.3, size=(30, 2))
        cluster_b = rng.normal(loc=[5, 5], scale=0.3, size=(30, 2))
        self.X = np.vstack([cluster_a, cluster_b])

    def test_initialisation(self):
        model = KMeansAI(n_clusters=2, random_state=0)
        assert model.n_clusters == 2
        assert model.centroids_ is None
        assert model.labels_ is None

    def test_fit_predict_shape_and_range(self):
        model = KMeansAI(n_clusters=2, random_state=0)
        labels = model.fit_predict(self.X)
        assert labels.shape == (self.X.shape[0],)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_inertia_is_nonnegative(self):
        model = KMeansAI(n_clusters=2, random_state=0)
        model.fit(self.X)
        assert model.inertia_ >= 0

    def test_convergence_within_iteration_limit(self):
        model = KMeansAI(n_clusters=2, max_iterations=100, tol=1e-4, random_state=0)
        model.fit(self.X)
        assert model.n_iter_ <= 100
        assert not np.any(np.isnan(model.centroids_))


# ===================================================================== #
# TestDistanceFunctions (4 cases)
# ===================================================================== #
class TestDistanceFunctions:
    def setup_method(self):
        self.a = np.array([1.0, 2.0, 3.0])
        self.b = np.array([4.0, 6.0, 3.0])

    def test_minkowski_p1_matches_scipy(self):
        custom = minkowski_distance(self.a, self.b, p=1)
        reference = scipy_minkowski(self.a, self.b, p=1)
        assert custom == pytest.approx(reference)

    def test_minkowski_p2_matches_scipy(self):
        custom = minkowski_distance(self.a, self.b, p=2)
        reference = scipy_minkowski(self.a, self.b, p=2)
        assert custom == pytest.approx(reference)

    def test_dot_product_matches_numpy(self):
        assert dot_product(self.a, self.b) == pytest.approx(np.dot(self.a, self.b))

    def test_euclidean_norm_matches_numpy(self):
        assert euclidean_norm(self.a) == pytest.approx(np.linalg.norm(self.a))


# ===================================================================== #
# TestStatisticalFunctions (3 cases)
# ===================================================================== #
class TestStatisticalFunctions:
    def setup_method(self):
        self.x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

    def test_mean_matches_numpy(self):
        assert compute_mean(self.x) == pytest.approx(np.mean(self.x))

    def test_variance_matches_numpy(self):
        assert compute_variance(self.x) == pytest.approx(np.var(self.x))

    def test_std_matches_numpy(self):
        assert compute_std(self.x) == pytest.approx(np.std(self.x))


# ===================================================================== #
# TestEncodingFunctions (2 cases)
# ===================================================================== #
class TestEncodingFunctions:
    def test_label_encode_returns_native_int(self):
        column = ["Graduation", "PhD", "Master", "Graduation", "Basic"]
        encoded = ai_label_encode(column)
        assert len(encoded) == len(column)
        # Regression test for the NumPy int64 vs. Python int bug (Section IV-A)
        assert all(isinstance(v, (int, np.integer)) for v in encoded)
        assert len(set(encoded)) == len(set(column))

    def test_one_hot_encode_rows_sum_to_one(self):
        column = ["Single", "Married", "Single", "Divorced"]
        categories, encoded = ai_one_hot_encode(column)
        assert encoded.shape == (len(column), len(categories))
        assert np.all(encoded.sum(axis=1) == 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
