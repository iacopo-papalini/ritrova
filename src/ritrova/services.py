"""Business logic services extracted from presentation layers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cluster import EMBEDDING_DIMS
from .db import FaceDB, Finding
from .embeddings import compute_centroid, cosine_similarity, normalize


def compute_cluster_hint(db: FaceDB, cluster_id: int) -> dict[str, Any] | None:
    """Return the best matching subject for a cluster, or None.

    Returns dict with keys: person_id, name, sim (percentage).
    (person_id kept because the FK column on findings is still person_id.)
    """
    findings = db.get_cluster_findings(cluster_id, limit=200)
    if not findings:
        return None

    cluster_species = findings[0].species
    kind = "pet" if cluster_species in db.PET_SPECIES else "person"
    dim = EMBEDDING_DIMS.get(kind)

    centroid = compute_centroid(np.array([f.embedding for f in findings]))
    subject_centroids = db.get_subject_centroids(kind=kind, embedding_dim=dim)

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
    findings: list[Finding],
    kind: str,
) -> dict[int, dict[str, Any]]:
    """For each singleton finding, find the nearest subject by centroid similarity.

    Returns {finding_id: {"person_id": int, "name": str, "sim": float}}.
    (person_id kept because the FK column on findings is still person_id.)
    """
    dim = EMBEDDING_DIMS.get(kind)
    subject_centroids = db.get_subject_centroids(kind=kind, embedding_dim=dim)
    if not subject_centroids:
        return {}

    centroid_matrix = np.array([sc[2] for sc in subject_centroids])

    hints: dict[int, dict[str, Any]] = {}
    for finding in findings:
        emb = normalize(finding.embedding)
        sims = centroid_matrix @ emb
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])
        sid, sname, _ = subject_centroids[best_idx]
        hints[finding.id] = {
            "person_id": sid,
            "name": sname,
            "sim": round(best_sim * 100, 1),
        }

    return hints
