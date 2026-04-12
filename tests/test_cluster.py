"""Integration tests for ritrova.cluster module (real db, no mocking)."""

from unittest import TestCase

import numpy as np
import pytest

from ritrova.cluster import (
    _cluster_exact,
    _cluster_faiss,
    _normalize_embeddings,
    auto_assign,
    auto_merge_clusters,
    cluster_faces,
    compare_subjects,
    find_similar_cluster,
    find_similar_unclustered,
    rank_subjects_for_cluster,
    suggest_merges,
)
from ritrova.db import FaceDB


def _emb(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_face(db: FaceDB, path: str, seed: int, species: str = "human") -> int:
    pid = db.add_photo(path, 100, 100)
    db.add_faces_batch([(pid, (10, 10, 50, 50), _emb(seed), 0.95)], species=species)
    return db.get_photo_faces(pid)[0].id


class TestAutoAssign(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_matches_cluster_to_subject(self) -> None:
        # Subject "Alice" with seed=1 embedding
        sid = self.db.create_subject("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        # Unnamed cluster with same embedding (seed=1)
        fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        fid3 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = auto_assign(self.db, min_similarity=0.50)
        assert result["assigned_clusters"] == 1

        # Verify faces assigned to Alice
        face = self.db.get_face(fid2)
        assert face is not None
        assert face.person_id == sid

    def test_below_threshold_skips(self) -> None:
        sid = self.db.create_subject("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        # Cluster with very different embedding
        fid2 = _add_face(self.db, "/b1.jpg", seed=999)
        fid3 = _add_face(self.db, "/b2.jpg", seed=999)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = auto_assign(self.db, min_similarity=0.95)
        assert result["assigned_clusters"] == 0
        assert result["unmatched"] == 1

    def test_sweeps_singletons(self) -> None:
        sid = self.db.create_subject("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        # Unclustered face with same embedding
        _add_face(self.db, "/b1.jpg", seed=1)

        result = auto_assign(self.db, min_similarity=0.50)
        assert result["assigned_singletons"] >= 1

    def test_no_subjects(self) -> None:
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        fid2 = _add_face(self.db, "/a2.jpg", seed=1)
        self.db.update_cluster_ids({fid: 10, fid2: 10})

        result = auto_assign(self.db)
        assert result["assigned_clusters"] == 0


class TestAutoMergeClusters(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_similar_clusters_merge(self) -> None:
        # Two clusters with identical embeddings
        fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        fid2 = _add_face(self.db, "/a2.jpg", seed=1)
        fid3 = _add_face(self.db, "/b1.jpg", seed=1)
        fid4 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        result = auto_merge_clusters(self.db, min_similarity=0.50)
        assert result["merged"] >= 1

    def test_keeps_larger_cluster(self) -> None:
        fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        fid2 = _add_face(self.db, "/a2.jpg", seed=1)
        fid3 = _add_face(self.db, "/a3.jpg", seed=1)
        fid4 = _add_face(self.db, "/b1.jpg", seed=1)
        fid5 = _add_face(self.db, "/b2.jpg", seed=1)
        # Cluster 10 has 3 faces, cluster 20 has 2
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 10, fid4: 20, fid5: 20})

        auto_merge_clusters(self.db, min_similarity=0.50)
        # All should now be in cluster 10 (larger)
        face = self.db.get_face(fid4)
        assert face is not None
        assert face.cluster_id == 10


class TestSuggestMerges(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_finds_pair(self) -> None:
        fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        fid2 = _add_face(self.db, "/a2.jpg", seed=1)
        fid3 = _add_face(self.db, "/b1.jpg", seed=1)
        fid4 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        suggestions = suggest_merges(self.db, min_similarity=50.0)
        assert len(suggestions) >= 1
        assert suggestions[0].similarity_pct > 90

    def test_respects_threshold(self) -> None:
        fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        fid2 = _add_face(self.db, "/a2.jpg", seed=1)
        fid3 = _add_face(self.db, "/b1.jpg", seed=999)
        fid4 = _add_face(self.db, "/b2.jpg", seed=999)
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        suggestions = suggest_merges(self.db, min_similarity=99.0)
        assert len(suggestions) == 0

    def test_skips_subject_to_subject_pairs(self) -> None:
        """Two named subjects with similar embeddings should NOT be suggested."""
        sid_a = self.db.create_subject("Alice")
        fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid1, sid_a)

        sid_b = self.db.create_subject("Alice2")
        fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        self.db.assign_face_to_subject(fid2, sid_b)

        suggestions = suggest_merges(self.db, min_similarity=50.0)
        assert len(suggestions) == 0

    def test_kind_filter(self) -> None:
        fid1 = _add_face(self.db, "/a1.jpg", seed=1, species="human")
        fid2 = _add_face(self.db, "/a2.jpg", seed=1, species="human")
        fid3 = _add_face(self.db, "/b1.jpg", seed=1, species="dog")
        fid4 = _add_face(self.db, "/b2.jpg", seed=1, species="dog")
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        person_suggestions = suggest_merges(self.db, min_similarity=50.0, kind="person")
        # Only cluster 10 is human, so no pairs
        assert len(person_suggestions) == 0


class TestFindSimilarCluster(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_best_match(self) -> None:
        sid = self.db.create_subject("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        fid3 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = find_similar_cluster(self.db, sid)
        assert result == 10

    def test_none_below_threshold(self) -> None:
        sid = self.db.create_subject("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        fid2 = _add_face(self.db, "/b1.jpg", seed=999)
        fid3 = _add_face(self.db, "/b2.jpg", seed=999)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = find_similar_cluster(self.db, sid, min_similarity=0.99)
        assert result is None


class TestClusterFaces(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_empty_db(self) -> None:
        result = cluster_faces(self.db)
        assert result["total_faces"] == 0
        assert result["clusters"] == 0

    def test_identical_embeddings_cluster_together(self) -> None:
        _add_face(self.db, "/a.jpg", seed=1)
        _add_face(self.db, "/b.jpg", seed=1)
        _add_face(self.db, "/c.jpg", seed=1)

        result = cluster_faces(self.db)
        assert result["total_faces"] == 3
        assert result["clusters"] >= 1

    def test_distant_embeddings_stay_separate(self) -> None:
        _add_face(self.db, "/a.jpg", seed=1)
        _add_face(self.db, "/b.jpg", seed=1)
        _add_face(self.db, "/c.jpg", seed=999)
        _add_face(self.db, "/d.jpg", seed=999)

        result = cluster_faces(self.db, threshold=0.3)
        assert result["clusters"] >= 2

    def test_species_isolation(self) -> None:
        """Clustering humans doesn't wipe pet clusters."""
        _add_face(self.db, "/dog1.jpg", seed=1, species="dog")
        _add_face(self.db, "/dog2.jpg", seed=1, species="dog")
        self.db.update_cluster_ids(
            {
                self.db.get_all_embeddings(species="dog")[0][0]: 100,
                self.db.get_all_embeddings(species="dog")[1][0]: 100,
            }
        )

        _add_face(self.db, "/h1.jpg", seed=2)
        _add_face(self.db, "/h2.jpg", seed=2)

        cluster_faces(self.db, species="human")

        # Pet cluster should still exist
        dog_faces = self.db.get_cluster_faces(100)
        assert len(dog_faces) == 2


class TestFindSimilarUnclustered(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_finds_similar(self) -> None:
        sid = self.db.create_subject("Alice")
        fid = _add_face(self.db, "/a.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        # Unclustered face with same embedding
        _add_face(self.db, "/b.jpg", seed=1)

        results = find_similar_unclustered(self.db, sid, min_similarity=0.5)
        assert len(results) >= 1
        assert results[0][1] > 90  # similarity percentage

    def test_no_match_below_threshold(self) -> None:
        sid = self.db.create_subject("Alice")
        fid = _add_face(self.db, "/a.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid)

        _add_face(self.db, "/b.jpg", seed=999)

        results = find_similar_unclustered(self.db, sid, min_similarity=0.99)
        assert len(results) == 0

    def test_empty_subject(self) -> None:
        sid = self.db.create_subject("Alice")
        results = find_similar_unclustered(self.db, sid)
        assert results == []


class TestCompareSubjects(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_finds_swaps(self) -> None:
        """When a face is closer to the other subject's centroid, it's flagged."""
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")

        # Give Alice a face with seed=1 centroid
        fid_a = _add_face(self.db, "/a.jpg", seed=1)
        self.db.assign_face_to_subject(fid_a, sid_a)

        # Give Bob a face with seed=999 centroid
        fid_b = _add_face(self.db, "/b.jpg", seed=999)
        self.db.assign_face_to_subject(fid_b, sid_b)

        # Assign a seed=999 face to Alice — should be flagged as swap to Bob
        fid_wrong = _add_face(self.db, "/wrong.jpg", seed=999)
        self.db.assign_face_to_subject(fid_wrong, sid_a)

        result = compare_subjects(self.db, sid_a, sid_b)
        assert len(result["swaps_a_to_b"]) >= 1

    def test_empty_subject(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        result = compare_subjects(self.db, sid_a, sid_b)
        assert result["swaps_a_to_b"] == []
        assert result["swaps_b_to_a"] == []


class TestRankSubjectsForCluster(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_ranks_by_similarity(self) -> None:
        sid_close = self.db.create_subject("Close")
        fid = _add_face(self.db, "/close.jpg", seed=1)
        self.db.assign_face_to_subject(fid, sid_close)

        sid_far = self.db.create_subject("Far")
        fid2 = _add_face(self.db, "/far.jpg", seed=999)
        self.db.assign_face_to_subject(fid2, sid_far)

        # Cluster with seed=1 embedding
        fid3 = _add_face(self.db, "/c1.jpg", seed=1)
        fid4 = _add_face(self.db, "/c2.jpg", seed=1)
        self.db.update_cluster_ids({fid3: 10, fid4: 10})

        ranked = rank_subjects_for_cluster(self.db, 10)
        assert len(ranked) == 2
        assert ranked[0][1] == "Close"  # most similar first

    def test_empty_cluster(self) -> None:
        assert rank_subjects_for_cluster(self.db, 999) == []


def _make_group(n: int, seed: int, dim: int = 512) -> np.ndarray:
    """Create n similar embeddings (same base + small noise), normalized."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(dim).astype(np.float32)
    noise = rng.standard_normal((n, dim)).astype(np.float32) * 0.05
    embs = base + noise
    return _normalize_embeddings(embs)


class TestClusterExact(TestCase):
    def test_identical_embeddings(self) -> None:
        embs = _make_group(5, seed=1)
        labels = _cluster_exact(embs, threshold=0.45, min_size=2)
        assert len(set(labels) - {-1}) == 1

    def test_two_distinct_groups(self) -> None:
        group_a = _make_group(5, seed=1)
        group_b = _make_group(5, seed=999)
        embs = np.vstack([group_a, group_b])
        labels = _cluster_exact(embs, threshold=0.3, min_size=2)
        assert len(set(labels) - {-1}) >= 2
        # Faces 0-4 should be in one cluster, 5-9 in another
        assert labels[0] == labels[4]
        assert labels[5] == labels[9]
        assert labels[0] != labels[5]

    def test_min_size_filters_small(self) -> None:
        embs = _make_group(1, seed=1)
        labels = _cluster_exact(embs, threshold=0.45, min_size=2)
        assert all(label == -1 for label in labels)


class TestClusterFaiss(TestCase):
    def test_identical_embeddings(self) -> None:
        embs = _make_group(10, seed=1)
        labels = _cluster_faiss(embs, threshold=0.45, min_size=2)
        assert len(set(labels) - {-1}) == 1

    def test_two_distinct_groups(self) -> None:
        group_a = _make_group(10, seed=1)
        group_b = _make_group(10, seed=999)
        embs = np.vstack([group_a, group_b])
        labels = _cluster_faiss(embs, threshold=0.3, min_size=2)
        assert len(set(labels) - {-1}) >= 2
        # First 10 in one cluster, last 10 in another
        assert labels[0] == labels[9]
        assert labels[10] == labels[19]
        assert labels[0] != labels[10]

    def test_no_chaining(self) -> None:
        """Verify complete-linkage verification prevents chaining.

        Create a chain A-B-C where A~B and B~C but A is far from C.
        Single linkage would merge all three; complete linkage should not.
        """
        rng = np.random.default_rng(42)
        dim = 512
        a = rng.standard_normal(dim).astype(np.float32)
        a /= np.linalg.norm(a)

        # B is close to A (similarity ~0.85)
        b = a + rng.standard_normal(dim).astype(np.float32) * 0.2
        b /= np.linalg.norm(b)

        # C is close to B but far from A
        c = b + rng.standard_normal(dim).astype(np.float32) * 0.2
        c /= np.linalg.norm(c)

        # Make groups of 3 around each anchor
        embs_a = _normalize_embeddings(a + rng.standard_normal((3, dim)).astype(np.float32) * 0.03)
        embs_b = _normalize_embeddings(b + rng.standard_normal((3, dim)).astype(np.float32) * 0.03)
        embs_c = _normalize_embeddings(c + rng.standard_normal((3, dim)).astype(np.float32) * 0.03)

        embs = np.vstack([embs_a, embs_b, embs_c])

        # Use a threshold that allows A-B and B-C but not A-C
        # Cosine distance between A and C should be > threshold
        sim_ac = float(a @ c)
        threshold = 1.0 - sim_ac + 0.05  # just above A-C distance

        labels = _cluster_faiss(embs, threshold=threshold, min_size=2)
        # A and C should NOT be in the same cluster
        cluster_a = labels[0]
        cluster_c = labels[6]
        if cluster_a != -1 and cluster_c != -1:
            assert cluster_a != cluster_c, (
                "Chaining detected: A and C should not be in same cluster"
            )

    def test_min_size_filters(self) -> None:
        """Single isolated face should be noise."""
        group = _make_group(5, seed=1)
        loner = _make_group(1, seed=999)
        embs = np.vstack([group, loner])
        labels = _cluster_faiss(embs, threshold=0.45, min_size=2)
        assert labels[-1] == -1

    def test_matches_exact_on_small_input(self) -> None:
        """FAISS and exact should produce same clusters for small input."""
        group_a = _make_group(5, seed=1)
        group_b = _make_group(5, seed=999)
        embs = np.vstack([group_a, group_b])

        labels_exact = _cluster_exact(embs, threshold=0.3, min_size=2)
        labels_faiss = _cluster_faiss(embs, threshold=0.3, min_size=2)

        # Same number of clusters
        n_exact = len(set(labels_exact) - {-1})
        n_faiss = len(set(labels_faiss) - {-1})
        assert n_exact == n_faiss

        # Same groupings (labels may differ but membership should match)
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                same_exact = labels_exact[i] == labels_exact[j] and labels_exact[i] != -1
                same_faiss = labels_faiss[i] == labels_faiss[j] and labels_faiss[i] != -1
                assert same_exact == same_faiss, (
                    f"Faces {i},{j}: exact says {'same' if same_exact else 'diff'}, "
                    f"faiss says {'same' if same_faiss else 'diff'}"
                )
