"""Business logic services extracted from presentation layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .db import Face, FaceDB, Person
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

    centroid = compute_centroid(np.array([f.embedding for f in faces]))

    best_name: str | None = None
    best_sim = 0.0
    best_pid: int | None = None

    for person in db.get_persons():
        pfaces = db.get_person_faces(person.id, limit=200)
        if not pfaces:
            continue
        face_sp = pfaces[0].species
        if cluster_species in pet_species and face_sp not in pet_species:
            continue
        if cluster_species not in pet_species and face_sp != cluster_species:
            continue
        p_centroid = compute_centroid(np.array([f.embedding for f in pfaces]))
        sim = cosine_similarity(centroid, p_centroid)
        if sim > best_sim:
            best_sim = sim
            best_name = person.name
            best_pid = person.id

    if best_name is None:
        return None
    return {"person_id": best_pid, "name": best_name, "sim": round(best_sim * 100, 1)}


def compute_singleton_hints(
    db: FaceDB,
    faces: list[Face],
    persons: list[Person],
    species: str,
) -> dict[int, dict[str, Any]]:
    """For each singleton face, find the nearest person by centroid similarity.

    Returns {face_id: {"person_id": int, "name": str, "sim": float}}.
    """
    pet_species = db.PET_SPECIES

    person_centroids: list[tuple[int, str, np.ndarray]] = []
    for p in persons:
        pfaces = db.get_person_faces(p.id, limit=200)
        if not pfaces:
            continue
        face_sp = pfaces[0].species
        if species == "pet" and face_sp not in pet_species:
            continue
        if species != "pet" and face_sp != species:
            continue
        centroid = compute_centroid(np.array([f.embedding for f in pfaces]))
        person_centroids.append((p.id, p.name, centroid))

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


def filter_persons_by_species(db: FaceDB, persons: list[Person], species: str) -> list[Person]:
    """Filter a list of persons to those who have faces of the given species."""
    return [p for p in persons if db.has_person_species(p.id, species)]
