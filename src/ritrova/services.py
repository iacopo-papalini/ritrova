"""Business logic services extracted from presentation layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .db import Face, FaceDB
from .embeddings import compute_centroid, cosine_similarity, normalize


def compute_cluster_hint(db: FaceDB, cluster_id: int) -> dict[str, Any] | None:
    """Return the best matching subject for a cluster, or None.

    Returns dict with keys: person_id, name, sim (percentage).
    (person_id kept because the FK column on faces is still person_id.)
    """
    faces = db.get_cluster_faces(cluster_id, limit=200)
    if not faces:
        return None

    cluster_species = faces[0].species
    # Map face species to subject kind
    kind = "pet" if cluster_species in db.PET_SPECIES else "person"

    centroid = compute_centroid(np.array([f.embedding for f in faces]))
    subject_centroids = db.get_subject_centroids(kind=kind)

    best_name: str | None = None
    best_sim = 0.0
    best_sid: int | None = None

    for sid, name, s_centroid in subject_centroids:
        sim = cosine_similarity(centroid, s_centroid)
        if sim > best_sim:
            best_sim = sim
            best_name = name
            best_sid = sid

    if best_name is None:
        return None
    return {"person_id": best_sid, "name": best_name, "sim": round(best_sim * 100, 1)}


def compute_singleton_hints(
    db: FaceDB,
    faces: list[Face],
    kind: str,
) -> dict[int, dict[str, Any]]:
    """For each singleton face, find the nearest subject by centroid similarity.

    Returns {face_id: {"person_id": int, "name": str, "sim": float}}.
    (person_id kept because the FK column on faces is still person_id.)
    """
    subject_centroids = db.get_subject_centroids(kind=kind)
    if not subject_centroids:
        return {}

    centroid_matrix = np.array([sc[2] for sc in subject_centroids])

    hints: dict[int, dict[str, Any]] = {}
    for face in faces:
        emb = normalize(face.embedding)
        sims = centroid_matrix @ emb
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        sid, sname, _ = subject_centroids[best_idx]
        hints[face.id] = {
            "person_id": sid,
            "name": sname,
            "sim": round(best_sim * 100, 1),
        }

    return hints
