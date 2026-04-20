"""Face clustering and similarity search."""

import logging
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .db import FaceDB, Finding
from .embeddings import compute_centroid, cosine_similarity, normalize

logger = logging.getLogger(__name__)

# Embedding dimensions by subject kind — determines which vector space to use
EMBEDDING_DIMS: dict[str, int] = {"person": 512, "pet": 768}


@dataclass
class MergeSuggestion:
    cluster_a: int
    cluster_b: int
    similarity_pct: float
    size_a: int
    size_b: int
    sample_finding_ids_a: list[int]
    sample_finding_ids_b: list[int]
    kind_a: str = "cluster"  # "subject" or "cluster"
    kind_b: str = "cluster"


def _findings_centroid(findings: list[Finding], embedding_dim: int | None = None) -> np.ndarray:
    """Compute normalized centroid from a list of Finding objects.

    If embedding_dim is specified, only use findings with that dimension.
    If not specified, use the majority dimension (most common among findings).
    """
    if not findings:
        return np.zeros(512, dtype=np.float32)
    if embedding_dim is None:
        # Pick the majority dimension
        dims = [len(f.embedding) for f in findings]
        embedding_dim = max(set(dims), key=dims.count)
    compatible = [f.embedding for f in findings if len(f.embedding) == embedding_dim]
    if not compatible:
        return np.zeros(embedding_dim, dtype=np.float32)
    return compute_centroid(np.array(compatible))


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms: np.ndarray = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    result: np.ndarray = embeddings / norms
    return result


def _cluster_exact(embeddings: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    """Exact hierarchical clustering with complete linkage. O(n^2)."""
    if len(embeddings) < 2:
        return np.full(len(embeddings), -1, dtype=np.int64)
    logger.info("Exact clustering: computing pairwise distances for %d faces...", len(embeddings))
    distances = pdist(embeddings, metric="cosine")
    Z = linkage(distances, method="complete")
    labels: np.ndarray = fcluster(Z, t=threshold, criterion="distance") - 1

    cluster_sizes = np.bincount(labels)
    small = set(np.where(cluster_sizes < min_size)[0])
    for i in range(len(labels)):
        if labels[i] in small:
            labels[i] = -1
    return labels


def _cluster_faiss(embeddings: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    """Two-phase clustering: FAISS pre-grouping + exact verification.

    Phase 1: FAISS brute-force range search to find neighbor pairs.
             Still O(n^2) in theory but much faster than scipy pdist
             due to SIMD-optimized native code.
    Phase 2: Connected components -> candidate groups. For each group,
             run exact complete-linkage clustering (O(k^2) per group, k << n).
             Avoids storing the full n*n distance matrix.
    """
    n, d = embeddings.shape
    logger.info("FAISS clustering: building index for %d faces (dim=%d)...", n, d)

    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    # Inner product >= (1 - threshold) means cosine distance <= threshold
    sim_threshold = 1.0 - threshold
    logger.info("FAISS range search (similarity >= %.2f)...", sim_threshold)
    lims, _dists, neighbors = index.range_search(embeddings, sim_threshold)

    # Build adjacency list from range search results
    adj: list[set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j_idx in range(lims[i], lims[i + 1]):
            j = int(neighbors[j_idx])
            if i != j:
                adj[i].add(j)
                adj[j].add(i)

    # Phase 2: connected components via BFS
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0
    for start in range(n):
        if labels[start] != -1:
            continue
        # BFS
        component = []
        queue = [start]
        labels[start] = cluster_id
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in adj[node]:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster_id
                    queue.append(neighbor)

        # Exact verification: run complete linkage within this component
        # to prevent chaining (connected components = single linkage)
        if len(component) >= min_size:
            comp_embs = embeddings[component]
            if len(component) <= 500:
                sub_labels = _cluster_exact(comp_embs, threshold, min_size)
            else:
                # Very large component: use a tighter threshold to break it up,
                # then recurse on sub-groups
                sub_labels = _cluster_exact_or_split(comp_embs, threshold, min_size)

            # Remap sub-labels to global labels
            unique_sub = set(sub_labels) - {-1}
            sub_to_global = {s: cluster_id + i for i, s in enumerate(sorted(unique_sub))}
            for idx, sub_label in zip(component, sub_labels, strict=True):
                if sub_label == -1:
                    labels[idx] = -1
                else:
                    labels[idx] = sub_to_global[sub_label]
            cluster_id += len(unique_sub)
        else:
            for idx in component:
                labels[idx] = -1
            cluster_id += 1

    return labels


def _cluster_exact_or_split(embeddings: np.ndarray, threshold: float, min_size: int) -> np.ndarray:
    """For large components: split into sub-groups via k-means, then exact cluster each."""
    n = len(embeddings)
    n_splits = max(2, n // 300)
    logger.info("Splitting large component (%d faces) into %d sub-groups...", n, n_splits)

    kmeans = faiss.Kmeans(embeddings.shape[1], n_splits, niter=20, verbose=False)
    kmeans.train(embeddings)
    assert kmeans.index is not None
    _, assignments = kmeans.index.search(embeddings, 1)
    assignments = assignments.flatten()

    labels = np.full(n, -1, dtype=np.int64)
    next_label = 0
    for group_id in range(n_splits):
        mask = assignments == group_id
        if mask.sum() < min_size:
            continue
        group_embs = embeddings[mask]
        sub_labels = _cluster_exact(group_embs, threshold, min_size)
        group_indices = np.where(mask)[0]
        unique_sub = set(sub_labels) - {-1}
        sub_to_global = {s: next_label + i for i, s in enumerate(sorted(unique_sub))}
        for idx, sub_label in zip(group_indices, sub_labels, strict=True):
            if sub_label != -1:
                labels[idx] = sub_to_global[sub_label]
        next_label += len(unique_sub)

    return labels


def cluster_faces(
    db: FaceDB,
    threshold: float = 0.45,
    min_size: int = 2,
    species: str = "human",
) -> dict[str, Any]:
    """Cluster face embeddings using FAISS-accelerated two-phase clustering.

    Phase 1: FAISS range search finds candidate neighbor pairs (brute-force
    but SIMD-optimized, much faster than scipy pdist).
    Phase 2: Connected components -> candidate groups, then exact
    complete-linkage verification within each group (O(k^2), k << n).
    Avoids storing the full n*n distance matrix in memory.

    Args:
        db: Database instance.
        threshold: Max cosine distance within a cluster (0.3-0.5 for ArcFace).
        min_size: Discard clusters smaller than this.
        species: Only cluster faces of this species.
    """
    db.clear_clusters(species=species)

    # Offset cluster IDs to avoid collision with other species' clusters
    max_existing = db.query("SELECT COALESCE(MAX(cluster_id), -1) FROM cluster_findings")
    cluster_offset = max_existing[0][0] + 1 if max_existing else 0

    dim = EMBEDDING_DIMS.get("pet" if species in db.PET_SPECIES or species == "pet" else "person")
    logger.info("Loading %s embeddings (dim=%s)...", species, dim)
    data = db.get_all_embeddings(species=species, embedding_dim=dim)
    if not data:
        return {"total_faces": 0, "clusters": 0, "noise": 0}

    face_ids = [d[0] for d in data]
    embeddings = _normalize_embeddings(np.array([d[1] for d in data]))

    logger.info("Clustering %d face embeddings...", len(embeddings))
    labels = _cluster_faiss(embeddings, threshold, min_size)

    face_cluster_map = {
        fid: int(label) + cluster_offset
        for fid, label in zip(face_ids, labels, strict=True)
        if label >= 0
    }

    logger.info("Updating %d face cluster assignments...", len(face_cluster_map))
    db.update_cluster_ids(face_cluster_map)

    n_clusters = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    positive_labels = labels[labels >= 0]

    return {
        "total_faces": len(face_ids),
        "clusters": n_clusters,
        "noise": n_noise,
        "largest_cluster": (int(np.bincount(positive_labels).max()) if n_clusters > 0 else 0),
    }


def find_similar_unclustered(
    db: FaceDB, subject_id: int, min_similarity: float = 0.55, limit: int = 100
) -> list[tuple[int, float]]:
    """Find unclustered faces similar to a confirmed subject.

    Returns list of (face_id, similarity_pct) sorted by similarity desc.
    """
    subject_findings = db.get_subject_findings(subject_id, limit=500)
    if not subject_findings:
        return []

    centroid = _findings_centroid(subject_findings)
    dim = len(centroid)
    species = "pet" if dim == EMBEDDING_DIMS["pet"] else "human"

    unclustered = db.get_unclustered_embeddings(species=species, embedding_dim=dim)
    if not unclustered:
        return []

    candidates = []
    for fid, emb in unclustered:
        emb = normalize(emb)
        sim = cosine_similarity(centroid, emb)
        if sim >= min_similarity:
            candidates.append((fid, round(sim * 100, 1)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:limit]


def auto_assign(
    db: FaceDB,
    min_similarity: float = 0.50,
    kind: str = "person",
) -> dict[str, Any]:
    """Bulk-assign unnamed clusters to existing named subjects by centroid similarity.

    For each unnamed cluster, find the best-matching subject. If similarity
    exceeds the threshold, assign all faces in that cluster to the subject.

    Returns stats dict.
    """
    species = FaceDB.KIND_TO_SPECIES[kind]
    dim = EMBEDDING_DIMS.get(kind)
    subjects = db.get_subjects_by_kind(kind)
    if not subjects:
        logger.info("No named subjects to match against.")
        return {"assigned_clusters": 0, "assigned_faces": 0, "unmatched": 0}

    logger.info("Computing centroids for %d subjects...", len(subjects))
    subject_centroids = db.get_subject_centroids(kind=kind, embedding_dim=dim)

    if not subject_centroids:
        logger.info("No subjects with matching kind.")
        return {"assigned_clusters": 0, "assigned_faces": 0, "unmatched": 0}

    centroid_matrix = np.array([sc[2] for sc in subject_centroids])

    # Process unnamed clusters
    unnamed = db.get_unnamed_clusters(species=species)
    logger.info("Matching %d unnamed clusters...", len(unnamed))

    assigned_clusters = 0
    assigned_faces = 0
    unmatched = 0

    for cluster in unnamed:
        findings = db.get_cluster_findings(cluster["cluster_id"], limit=500)
        if not findings:
            continue
        cluster_centroid = _findings_centroid(findings)

        sims = centroid_matrix @ cluster_centroid
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])

        if best_sim >= min_similarity:
            sid, _name, _ = subject_centroids[best_idx]
            db.assign_cluster_to_subject(cluster["cluster_id"], sid)
            assigned_clusters += 1
            assigned_faces += cluster["face_count"]
            if assigned_clusters % 100 == 0:
                logger.info(
                    "  assigned %d clusters, %d faces...",
                    assigned_clusters,
                    assigned_faces,
                )
        else:
            unmatched += 1

    logger.info(
        "Clusters: assigned %d (%d faces), unmatched %d",
        assigned_clusters,
        assigned_faces,
        unmatched,
    )

    # Also sweep unclustered singletons
    unclustered = db.get_unclustered_embeddings(species=species, embedding_dim=dim)

    assigned_singletons = 0
    if unclustered:
        logger.info("Sweeping %d unclustered singletons...", len(unclustered))
        for fid, emb in unclustered:
            emb = normalize(emb)
            sims = centroid_matrix @ emb
            best_idx = int(sims.argmax())
            best_sim = float(sims[best_idx])
            if best_sim >= min_similarity:
                sid = subject_centroids[best_idx][0]
                db.assign_finding_to_subject(fid, sid)
                assigned_singletons += 1

        logger.info(
            "Singletons: assigned %d, skipped %d",
            assigned_singletons,
            len(unclustered) - assigned_singletons,
        )

    return {
        "assigned_clusters": assigned_clusters,
        "assigned_faces": assigned_faces + assigned_singletons,
        "assigned_singletons": assigned_singletons,
        "unmatched": unmatched,
    }


def compare_subjects(
    db: FaceDB,
    subject_a_id: int,
    subject_b_id: int,
) -> dict[str, Any]:
    """Compare two subjects and find faces that might be misassigned.

    Returns faces from A that are closer to B's centroid, and vice versa.
    """
    findings_a = db.get_subject_findings(subject_a_id, limit=5000)
    findings_b = db.get_subject_findings(subject_b_id, limit=5000)
    if not findings_a or not findings_b:
        return {"swaps_a_to_b": [], "swaps_b_to_a": []}

    # Use consistent dimension (from majority of subject A)
    centroid_a = _findings_centroid(findings_a)
    dim = len(centroid_a)
    centroid_b = _findings_centroid(findings_b, embedding_dim=dim)

    def _find_swaps(
        findings: list[Finding], own_centroid: np.ndarray, other_centroid: np.ndarray
    ) -> list[tuple[int, float, float, str]]:
        swaps = []
        for finding in findings:
            if len(finding.embedding) != dim:
                continue
            emb = normalize(finding.embedding)
            sim_own = cosine_similarity(emb, own_centroid)
            sim_other = cosine_similarity(emb, other_centroid)
            if sim_other > sim_own:
                source = db.get_source(finding.source_id)
                path = source.file_path if source else ""
                swaps.append((finding.id, round(sim_own * 100, 1), round(sim_other * 100, 1), path))
        swaps.sort(key=lambda x: x[2] - x[1], reverse=True)
        return swaps

    return {
        "swaps_a_to_b": _find_swaps(findings_a, centroid_a, centroid_b),
        "swaps_b_to_a": _find_swaps(findings_b, centroid_b, centroid_a),
    }


def auto_merge_clusters(
    db: FaceDB,
    min_similarity: float = 0.70,
    species: str = "human",
) -> dict[str, Any]:
    """Auto-merge unnamed cluster pairs whose centroids exceed the similarity threshold."""
    suggestions = suggest_merges(db, min_similarity=min_similarity * 100, species=species)

    merged = 0
    faces_moved = 0
    for s in suggestions:
        if s.kind_a == "subject" or s.kind_b == "subject":
            continue
        if s.size_a >= s.size_b:
            target, source, source_size = s.cluster_a, s.cluster_b, s.size_b
        else:
            target, source, source_size = s.cluster_b, s.cluster_a, s.size_a
        db.merge_clusters(source, target)
        merged += 1
        faces_moved += source_size

    remaining = len(db.get_unnamed_clusters(species=species))
    return {"merged": merged, "faces_moved": faces_moved, "remaining_clusters": remaining}


def rank_subjects_for_cluster(db: FaceDB, cluster_id: int) -> list[tuple[int, str, int, float]]:
    """Rank existing subjects by similarity to a cluster.

    Returns list of (subject_id, name, face_count, similarity_pct) sorted desc.
    """
    cluster_findings = db.get_cluster_findings(cluster_id, limit=200)
    if not cluster_findings:
        return []

    centroid = _findings_centroid(cluster_findings)
    # Derive kind from actual embedding dimension, not species label
    # (species may have been corrected but embedding stays in its original space)
    dim = len(centroid)
    kind = "pet" if dim == EMBEDDING_DIMS["pet"] else "person"

    subject_centroids = db.get_subject_centroids(kind=kind, embedding_dim=dim)
    subjects_by_kind = {s.id: s.face_count for s in db.get_subjects_by_kind(kind)}
    results = []
    for sid, name, s_centroid in subject_centroids:
        sim = round(cosine_similarity(centroid, s_centroid) * 100, 1)
        results.append((sid, name, subjects_by_kind.get(sid, 0), sim))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def find_similar_cluster(
    db: FaceDB,
    subject_id: int,
    min_similarity: float = 0.35,
    species: str = "human",
) -> int | None:
    """Find the unnamed cluster most similar to a subject. Returns cluster_id or None."""
    subject_findings = db.get_subject_findings(subject_id, limit=500)
    if not subject_findings:
        return None

    centroid = _findings_centroid(subject_findings)
    dim = len(centroid)

    best_cluster = None
    best_sim = min_similarity

    for cluster in db.get_unnamed_clusters(species=species):
        findings = db.get_cluster_findings(cluster["cluster_id"], limit=100)
        if not findings:
            continue
        c = _findings_centroid(findings, embedding_dim=dim)
        if len(c) != dim:
            continue
        sim = cosine_similarity(centroid, c)
        if sim > best_sim:
            best_sim = sim
            best_cluster = cluster["cluster_id"]

    return best_cluster


def suggest_merges(
    db: FaceDB,
    min_similarity: float = 40.0,
    species: str = "human",
) -> list[MergeSuggestion]:
    """Compare cluster/subject centroids and suggest likely merges.

    Works across both unnamed clusters and named subjects — any pair whose
    centroid cosine similarity exceeds *min_similarity* % is returned,
    sorted by similarity descending.
    """
    # Internal: domain layer still thinks in singular subject `kind` for
    # subject lookups and embedding-dim selection.
    kind = db._kind_for_species(species)  # noqa: SLF001 — DB-internal mapper
    dim = EMBEDDING_DIMS.get(kind, 512)
    groups: list[tuple[str, int, list[int], np.ndarray]] = []

    for subject in db.get_subjects_by_kind(kind):
        findings = db.get_subject_findings(subject.id, limit=500)
        compatible = [f for f in findings if len(f.embedding) == dim]
        if not compatible:
            continue
        embs = np.array([f.embedding for f in compatible])
        groups.append(("subject", subject.id, [f.id for f in compatible], embs))

    for cluster in db.get_unnamed_clusters(species=species):
        findings = db.get_cluster_findings(cluster["cluster_id"], limit=500)
        compatible = [f for f in findings if len(f.embedding) == dim]
        if not compatible:
            continue
        embs = np.array([f.embedding for f in compatible])
        groups.append(
            (
                "cluster",
                cluster["cluster_id"],
                [f.id for f in compatible],
                embs,
            )
        )

    if len(groups) < 2:
        return []

    centroids = np.array([compute_centroid(embs) for _kind, _gid, _fids, embs in groups])
    sim_matrix = centroids @ centroids.T

    suggestions = []
    n = len(groups)
    for i in range(n):
        for j in range(i + 1, n):
            kind_a, gid_a, fids_a, _ = groups[i]
            kind_b, gid_b, fids_b, _ = groups[j]
            # Skip subject-to-subject pairs — those are intentionally different
            if kind_a == "subject" and kind_b == "subject":
                continue
            pct = float(sim_matrix[i, j]) * 100
            if pct < min_similarity:
                continue
            suggestions.append(
                MergeSuggestion(
                    cluster_a=gid_a,
                    cluster_b=gid_b,
                    similarity_pct=round(pct, 1),
                    size_a=len(fids_a),
                    size_b=len(fids_b),
                    sample_finding_ids_a=fids_a[:4],
                    sample_finding_ids_b=fids_b[:4],
                    kind_a=kind_a,
                    kind_b=kind_b,
                )
            )

    suggestions.sort(key=lambda s: s.similarity_pct, reverse=True)
    return suggestions
