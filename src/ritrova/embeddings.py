"""Pure math utilities for face embedding operations."""

import numpy as np
from numpy.typing import NDArray


def normalize(v: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize a vector to unit length. Returns zero vector if input is zero."""
    norm = np.linalg.norm(v)
    if norm > 0:
        return v / norm
    return v


def compute_centroid(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """Compute the normalized centroid of a set of embeddings."""
    centroid: NDArray[np.float32] = embeddings.mean(axis=0)
    return normalize(centroid)


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Cosine similarity between two vectors (assumed normalized)."""
    return float(a @ b)


def rank_by_similarity(
    query: NDArray[np.float32],
    candidates: list[tuple[int, NDArray[np.float32]]],
) -> list[tuple[int, float]]:
    """Rank candidates by cosine similarity to query, descending.

    Args:
        query: Normalized query embedding.
        candidates: List of (id, normalized_embedding) pairs.

    Returns:
        List of (id, similarity) sorted by similarity descending.
    """
    if not candidates:
        return []
    ids = [c[0] for c in candidates]
    matrix = np.array([c[1] for c in candidates])
    sims = matrix @ query
    ranked = sorted(zip(ids, sims.tolist(), strict=True), key=lambda x: x[1], reverse=True)
    return [(cid, float(sim)) for cid, sim in ranked]
