"""Face clustering and similarity search."""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from .db import FaceDB


@dataclass
class MergeSuggestion:
    cluster_a: int
    cluster_b: int
    similarity_pct: float
    size_a: int
    size_b: int
    sample_face_ids_a: list[int]
    sample_face_ids_b: list[int]


def cluster_faces(
    db: FaceDB,
    threshold: float = 0.45,
    min_size: int = 2,
    species: str = "human",
) -> dict:
    """Cluster face embeddings using agglomerative clustering (complete linkage).

    Complete linkage requires ALL members of a cluster to be within the
    distance threshold of each other — no chaining.

    Args:
        db: Database instance.
        threshold: Max cosine distance within a cluster (0.3-0.5 for ArcFace).
        min_size: Discard clusters smaller than this.
        species: Only cluster faces of this species.
    """
    db.clear_clusters()

    print(f"Loading {species} embeddings...")
    data = db.get_all_embeddings(species=species)
    if not data:
        return {"total_faces": 0, "clusters": 0, "noise": 0}

    face_ids = [d[0] for d in data]
    embeddings = np.array([d[1] for d in data])

    print(f"Clustering {len(embeddings)} face embeddings...")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    print("Computing pairwise distances...")
    distances = pdist(embeddings, metric="cosine")

    print("Running hierarchical clustering (complete linkage)...")
    Z = linkage(distances, method="complete")

    labels = fcluster(Z, t=threshold, criterion="distance")
    # fcluster labels start at 1; shift to 0-based
    labels = labels - 1

    # Filter out small clusters — remap them to -1 (noise)
    cluster_sizes = np.bincount(labels)
    small_clusters = set(np.where(cluster_sizes < min_size)[0])
    for i in range(len(labels)):
        if labels[i] in small_clusters:
            labels[i] = -1

    face_cluster_map = {fid: int(label) for fid, label in zip(face_ids, labels) if label >= 0}

    print(f"Updating {len(face_cluster_map)} face cluster assignments...")
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
    db: FaceDB, person_id: int, min_similarity: float = 0.55, limit: int = 100
) -> list[tuple[int, float]]:
    """Find unclustered faces similar to a confirmed person.

    Returns list of (face_id, similarity_pct) sorted by similarity desc.
    """
    person_faces = db.get_person_faces(person_id, limit=500)
    if not person_faces:
        return []

    # Compute person centroid
    embs = np.array([f.embedding for f in person_faces])
    centroid = embs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    # Get unclustered, unassigned faces
    # Match species of the person's faces
    species = person_faces[0].species if person_faces else "human"
    rows = db.query(
        "SELECT id, embedding FROM faces "
        "WHERE person_id IS NULL AND cluster_id IS NULL AND species = ? "
        "AND id NOT IN (SELECT face_id FROM dismissed_faces)",
        (species,),
    )

    if not rows:
        return []

    candidates = []
    for row in rows:
        emb = np.frombuffer(row[1], dtype=np.float32)
        emb_norm = np.linalg.norm(emb)
        if emb_norm > 0:
            emb = emb / emb_norm
        sim = float(centroid @ emb)
        if sim >= min_similarity:
            candidates.append((row[0], round(sim * 100, 1)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:limit]


def auto_assign(
    db: FaceDB,
    min_similarity: float = 0.50,
    species: str = "human",
) -> dict:
    """Bulk-assign unnamed clusters to existing named persons by centroid similarity.

    For each unnamed cluster, find the best-matching person. If similarity
    exceeds the threshold, assign all faces in that cluster to the person.

    Returns stats dict.
    """
    persons = db.get_persons()
    if not persons:
        print("No named persons to match against.")
        return {"assigned_clusters": 0, "assigned_faces": 0, "unmatched": 0}

    # Build person centroids
    print(f"Computing centroids for {len(persons)} persons...")
    person_centroids = []
    for person in persons:
        faces = db.get_person_faces(person.id, limit=500)
        if not faces or faces[0].species != species:
            continue
        embs = np.array([f.embedding for f in faces])
        c = embs.mean(axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        person_centroids.append((person.id, person.name, c))

    if not person_centroids:
        print("No persons with matching species.")
        return {"assigned_clusters": 0, "assigned_faces": 0, "unmatched": 0}

    centroid_matrix = np.array([pc[2] for pc in person_centroids])

    # Process unnamed clusters
    unnamed = db.get_unnamed_clusters(species=species)
    print(f"Matching {len(unnamed)} unnamed clusters...")

    assigned_clusters = 0
    assigned_faces = 0
    unmatched = 0

    for cluster in unnamed:
        faces = db.get_cluster_faces(cluster["cluster_id"], limit=500)
        if not faces:
            continue
        embs = np.array([f.embedding for f in faces])
        cluster_centroid = embs.mean(axis=0)
        norm = np.linalg.norm(cluster_centroid)
        if norm > 0:
            cluster_centroid = cluster_centroid / norm

        sims = centroid_matrix @ cluster_centroid
        best_idx = int(sims.argmax())
        best_sim = float(sims[best_idx])

        if best_sim >= min_similarity:
            pid, pname, _ = person_centroids[best_idx]
            db.assign_cluster_to_person(cluster["cluster_id"], pid)
            assigned_clusters += 1
            assigned_faces += cluster["face_count"]
            if assigned_clusters % 100 == 0:
                print(
                    f"\r  assigned {assigned_clusters} clusters, {assigned_faces} faces...",
                    end="",
                    flush=True,
                )
        else:
            unmatched += 1

    print(
        f"\n  Clusters: assigned {assigned_clusters} ({assigned_faces} faces), unmatched {unmatched}"
    )

    # Also sweep unclustered singletons
    rows = db.query(
        "SELECT id, embedding FROM faces "
        "WHERE person_id IS NULL AND cluster_id IS NULL AND species = ? "
        "AND id NOT IN (SELECT face_id FROM dismissed_faces)",
        (species,) if species != "pet" else db.PET_SPECIES[:1],
    )
    # Handle pet species properly
    if species == "pet":
        clause, params = db._species_filter(species)
        rows = db.query(
            f"SELECT id, embedding FROM faces "
            f"WHERE person_id IS NULL AND cluster_id IS NULL AND {clause} "
            f"AND id NOT IN (SELECT face_id FROM dismissed_faces)",
            params,
        )

    assigned_singletons = 0
    if rows:
        print(f"  Sweeping {len(rows)} unclustered singletons...")
        for r in rows:
            emb = np.frombuffer(r[1], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            sims = centroid_matrix @ emb
            best_idx = int(sims.argmax())
            best_sim = float(sims[best_idx])
            if best_sim >= min_similarity:
                pid = person_centroids[best_idx][0]
                db.assign_face_to_person(r[0], pid)
                assigned_singletons += 1

        print(
            f"  Singletons: assigned {assigned_singletons}, skipped {len(rows) - assigned_singletons}"
        )

    return {
        "assigned_clusters": assigned_clusters,
        "assigned_faces": assigned_faces + assigned_singletons,
        "assigned_singletons": assigned_singletons,
        "unmatched": unmatched,
    }


def compare_persons(
    db: FaceDB,
    person_a_id: int,
    person_b_id: int,
) -> dict:
    """Compare two persons and find faces that might be misassigned.

    Returns faces from A that are closer to B's centroid, and vice versa.
    """
    faces_a = db.get_person_faces(person_a_id, limit=5000)
    faces_b = db.get_person_faces(person_b_id, limit=5000)
    if not faces_a or not faces_b:
        return {"swaps_a_to_b": [], "swaps_b_to_a": []}

    # Compute centroids
    embs_a = np.array([f.embedding for f in faces_a])
    centroid_a = embs_a.mean(axis=0)
    centroid_a = centroid_a / np.linalg.norm(centroid_a)

    embs_b = np.array([f.embedding for f in faces_b])
    centroid_b = embs_b.mean(axis=0)
    centroid_b = centroid_b / np.linalg.norm(centroid_b)

    # Find faces in A closer to B
    swaps_a_to_b = []
    for face, emb in zip(faces_a, embs_a):
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        sim_a = float(emb @ centroid_a)
        sim_b = float(emb @ centroid_b)
        if sim_b > sim_a:
            photo = db.get_photo(face.photo_id)
            path = photo.file_path if photo else ""
            swaps_a_to_b.append((face.id, round(sim_a * 100, 1), round(sim_b * 100, 1), path))

    # Find faces in B closer to A
    swaps_b_to_a = []
    for face, emb in zip(faces_b, embs_b):
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        sim_a = float(emb @ centroid_a)
        sim_b = float(emb @ centroid_b)
        if sim_a > sim_b:
            photo = db.get_photo(face.photo_id)
            path = photo.file_path if photo else ""
            swaps_b_to_a.append((face.id, round(sim_b * 100, 1), round(sim_a * 100, 1), path))

    # Sort by how "wrong" they are (bigger gap = more likely misassigned)
    swaps_a_to_b.sort(key=lambda x: x[2] - x[1], reverse=True)
    swaps_b_to_a.sort(key=lambda x: x[2] - x[1], reverse=True)

    return {"swaps_a_to_b": swaps_a_to_b, "swaps_b_to_a": swaps_b_to_a}


def auto_merge_clusters(
    db: FaceDB,
    min_similarity: float = 0.70,
    species: str = "human",
) -> dict:
    """Auto-merge unnamed cluster pairs whose centroids exceed the similarity threshold."""
    suggestions = suggest_merges(db, min_similarity=min_similarity * 100, species=species)

    # Only merge unnamed clusters (skip any involving named persons)
    named_ids = {p.id for p in db.get_persons()}

    merged = 0
    faces_moved = 0
    for s in suggestions:
        if s.cluster_a in named_ids or s.cluster_b in named_ids:
            continue
        if s.size_a >= s.size_b:
            target, source, source_size = s.cluster_a, s.cluster_b, s.size_b
        else:
            target, source, source_size = s.cluster_b, s.cluster_a, s.size_a
        db.run("UPDATE faces SET cluster_id = ? WHERE cluster_id = ?", (target, source))
        merged += 1
        faces_moved += source_size

    remaining = len(db.get_unnamed_clusters(species=species))
    return {"merged": merged, "faces_moved": faces_moved, "remaining_clusters": remaining}


def rank_persons_for_cluster(db: FaceDB, cluster_id: int) -> list[tuple[int, str, int, float]]:
    """Rank existing persons by similarity to a cluster.

    Returns list of (person_id, name, face_count, similarity_pct) sorted desc.
    """
    cluster_faces = db.get_cluster_faces(cluster_id, limit=200)
    if not cluster_faces:
        return []

    cluster_embs = np.array([f.embedding for f in cluster_faces])
    centroid = cluster_embs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    # Only show persons that share this species
    cluster_species = cluster_faces[0].species if cluster_faces else "human"

    results = []
    for person in db.get_persons():
        pfaces = db.get_person_faces(person.id, limit=200)
        if pfaces and pfaces[0].species != cluster_species:
            continue
        if not pfaces:
            continue
        p_centroid = np.array([f.embedding for f in pfaces]).mean(axis=0)
        p_norm = np.linalg.norm(p_centroid)
        if p_norm > 0:
            p_centroid = p_centroid / p_norm
        sim = round(float(centroid @ p_centroid) * 100, 1)
        results.append((person.id, person.name, person.face_count, sim))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def find_similar_cluster(db: FaceDB, person_id: int, min_similarity: float = 0.35) -> int | None:
    """Find the unnamed cluster most similar to a person. Returns cluster_id or None."""
    person_faces = db.get_person_faces(person_id, limit=500)
    if not person_faces:
        return None

    centroid = np.array([f.embedding for f in person_faces]).mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    best_cluster = None
    best_sim = min_similarity

    for cluster in db.get_unnamed_clusters():
        faces = db.get_cluster_faces(cluster["cluster_id"], limit=100)
        if not faces:
            continue
        c = np.array([f.embedding for f in faces]).mean(axis=0)
        c_norm = np.linalg.norm(c)
        if c_norm > 0:
            c = c / c_norm
        sim = float(centroid @ c)
        if sim > best_sim:
            best_sim = sim
            best_cluster = cluster["cluster_id"]

    return best_cluster


def suggest_merges(
    db: FaceDB,
    min_similarity: float = 40.0,
    species: str = "human",
) -> list[MergeSuggestion]:
    """Compare cluster/person centroids and suggest likely merges.

    Works across both unnamed clusters and named persons — any pair whose
    centroid cosine similarity exceeds *min_similarity* % is returned,
    sorted by similarity descending.
    """
    groups: list[tuple[str, int, list[int], np.ndarray]] = []
    pet_species = db.PET_SPECIES

    for person in db.get_persons():
        faces = db.get_person_faces(person.id, limit=500)
        if not faces:
            continue
        face_species = faces[0].species
        if species == "pet" and face_species not in pet_species:
            continue
        if species != "pet" and face_species != species:
            continue
        embs = np.array([f.embedding for f in faces])
        groups.append(("person", person.id, [f.id for f in faces], embs))

    for cluster in db.get_unnamed_clusters(species=species):
        faces = db.get_cluster_faces(cluster["cluster_id"], limit=500)
        if not faces:
            continue
        embs = np.array([f.embedding for f in faces])
        groups.append(
            (
                "cluster",
                cluster["cluster_id"],
                [f.id for f in faces],
                embs,
            )
        )

    if len(groups) < 2:
        return []

    centroids = []
    for _kind, _gid, _fids, embs in groups:
        c = embs.mean(axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        centroids.append(c)

    centroids_matrix = np.array(centroids)
    sim_matrix = centroids_matrix @ centroids_matrix.T

    suggestions = []
    n = len(groups)
    for i in range(n):
        for j in range(i + 1, n):
            pct = float(sim_matrix[i, j]) * 100
            if pct < min_similarity:
                continue
            kind_a, gid_a, fids_a, _ = groups[i]
            kind_b, gid_b, fids_b, _ = groups[j]
            suggestions.append(
                MergeSuggestion(
                    cluster_a=gid_a,
                    cluster_b=gid_b,
                    similarity_pct=round(pct, 1),
                    size_a=len(fids_a),
                    size_b=len(fids_b),
                    sample_face_ids_a=fids_a[:4],
                    sample_face_ids_b=fids_b[:4],
                )
            )

    suggestions.sort(key=lambda s: s.similarity_pct, reverse=True)
    return suggestions
