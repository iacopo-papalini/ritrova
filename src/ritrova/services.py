"""Business logic services extracted from presentation layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .db import Face, FaceDB
from .embeddings import compute_centroid, cosine_similarity, normalize


def compute_cluster_hint(db: FaceDB, cluster_id: int) -> dict[str, Any] | None:
    """Return the best matching person for a cluster, or None.

    Returns dict with keys: person_id, name, sim (percentage).
    """
    faces = db.get_cluster_faces(cluster_id, limit=200)
    if not faces:
        return None

    cluster_species = faces[0].species
    pet_species = db.PET_SPECIES
    species = "pet" if cluster_species in pet_species else cluster_species

    centroid = compute_centroid(np.array([f.embedding for f in faces]))
    person_centroids = db.get_person_centroids(species=species)

    best_name: str | None = None
    best_sim = 0.0
    best_pid: int | None = None

    for pid, name, p_centroid in person_centroids:
        sim = cosine_similarity(centroid, p_centroid)
        if sim > best_sim:
            best_sim = sim
            best_name = name
            best_pid = pid

    if best_name is None:
        return None
    return {"person_id": best_pid, "name": best_name, "sim": round(best_sim * 100, 1)}


def compute_singleton_hints(
    db: FaceDB,
    faces: list[Face],
    species: str,
) -> dict[int, dict[str, Any]]:
    """For each singleton face, find the nearest person by centroid similarity.

    Returns {face_id: {"person_id": int, "name": str, "sim": float}}.
    """
    person_centroids = db.get_person_centroids(species=species)
    if not person_centroids:
        return {}

    centroid_matrix = np.array([pc[2] for pc in person_centroids])

    hints: dict[int, dict[str, Any]] = {}
    for face in faces:
        emb = normalize(face.embedding)
        sims = centroid_matrix @ emb
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        pid, pname, _ = person_centroids[best_idx]
        hints[face.id] = {
            "person_id": pid,
            "name": pname,
            "sim": round(best_sim * 100, 1),
        }

    return hints
