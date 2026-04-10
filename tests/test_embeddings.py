"""Tests for ritrova.embeddings module."""

from unittest import TestCase

import numpy as np

from ritrova.embeddings import (
    compute_centroid,
    cosine_similarity,
    normalize,
    rank_by_similarity,
)


class TestNormalize(TestCase):
    def test_unit_vector_unchanged(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_zero_vector_returns_zero(self) -> None:
        v = np.zeros(3, dtype=np.float32)
        result = normalize(v)
        np.testing.assert_array_almost_equal(result, v)

    def test_random_vector_has_unit_norm(self) -> None:
        rng = np.random.default_rng(42)
        v = rng.standard_normal(512).astype(np.float32)
        result = normalize(v)
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)


class TestComputeCentroid(TestCase):
    def test_identical_vectors(self) -> None:
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        embs = np.stack([v, v, v])
        result = compute_centroid(embs)
        np.testing.assert_array_almost_equal(result, v)

    def test_two_vectors(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        result = compute_centroid(np.stack([a, b]))
        expected = normalize(np.array([0.5, 0.5], dtype=np.float32))
        np.testing.assert_array_almost_equal(result, expected)

    def test_single_vector(self) -> None:
        v = np.array([3.0, 4.0], dtype=np.float32)
        result = compute_centroid(v.reshape(1, -1))
        self.assertAlmostEqual(float(np.linalg.norm(result)), 1.0, places=5)


class TestCosineSimilarity(TestCase):
    def test_identical(self) -> None:
        v = normalize(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(cosine_similarity(v, v), 1.0, places=5)

    def test_orthogonal(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, b), 0.0, places=5)

    def test_opposite(self) -> None:
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([-1.0, 0.0], dtype=np.float32)
        self.assertAlmostEqual(cosine_similarity(a, b), -1.0, places=5)


class TestRankBySimilarity(TestCase):
    def test_sorted_descending(self) -> None:
        query = normalize(np.array([1.0, 0.0], dtype=np.float32))
        candidates = [
            (1, normalize(np.array([0.0, 1.0], dtype=np.float32))),  # orthogonal
            (2, normalize(np.array([1.0, 0.0], dtype=np.float32))),  # identical
            (3, normalize(np.array([0.7, 0.7], dtype=np.float32))),  # partial
        ]
        result = rank_by_similarity(query, candidates)
        ids = [r[0] for r in result]
        self.assertEqual(ids[0], 2)  # most similar first
        self.assertEqual(ids[-1], 1)  # least similar last

    def test_empty_candidates(self) -> None:
        query = normalize(np.array([1.0, 0.0], dtype=np.float32))
        result = rank_by_similarity(query, [])
        self.assertEqual(result, [])
