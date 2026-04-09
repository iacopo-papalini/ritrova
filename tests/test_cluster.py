"""Integration tests for face_recog.cluster module (real db, no mocking)."""

from unittest import TestCase

import numpy as np
import pytest

from face_recog.cluster import (
    auto_assign,
    auto_merge_clusters,
    cluster_faces,
    compare_persons,
    find_similar_cluster,
    find_similar_unclustered,
    rank_persons_for_cluster,
    suggest_merges,
)
from face_recog.db import FaceDB


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

    def test_matches_cluster_to_person(self) -> None:
        # Person "Alice" with seed=1 embedding
        pid = self.db.create_person("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid)

        # Unnamed cluster with same embedding (seed=1)
        fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        fid3 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = auto_assign(self.db, min_similarity=0.50)
        assert result["assigned_clusters"] == 1

        # Verify faces assigned to Alice
        face = self.db.get_face(fid2)
        assert face is not None
        assert face.person_id == pid

    def test_below_threshold_skips(self) -> None:
        pid = self.db.create_person("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid)

        # Cluster with very different embedding
        fid2 = _add_face(self.db, "/b1.jpg", seed=999)
        fid3 = _add_face(self.db, "/b2.jpg", seed=999)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = auto_assign(self.db, min_similarity=0.95)
        assert result["assigned_clusters"] == 0
        assert result["unmatched"] == 1

    def test_sweeps_singletons(self) -> None:
        pid = self.db.create_person("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid)

        # Unclustered face with same embedding
        _add_face(self.db, "/b1.jpg", seed=1)

        result = auto_assign(self.db, min_similarity=0.50)
        assert result["assigned_singletons"] >= 1

    def test_no_persons(self) -> None:
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

    def test_skips_person_to_person_pairs(self) -> None:
        """Two named persons with similar embeddings should NOT be suggested."""
        pid_a = self.db.create_person("Alice")
        fid1 = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_person(fid1, pid_a)

        pid_b = self.db.create_person("Alice2")
        fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        self.db.assign_face_to_person(fid2, pid_b)

        suggestions = suggest_merges(self.db, min_similarity=50.0)
        assert len(suggestions) == 0

    def test_species_filter(self) -> None:
        fid1 = _add_face(self.db, "/a1.jpg", seed=1, species="human")
        fid2 = _add_face(self.db, "/a2.jpg", seed=1, species="human")
        fid3 = _add_face(self.db, "/b1.jpg", seed=1, species="dog")
        fid4 = _add_face(self.db, "/b2.jpg", seed=1, species="dog")
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        human_suggestions = suggest_merges(self.db, min_similarity=50.0, species="human")
        # Only cluster 10 is human, so no pairs
        assert len(human_suggestions) == 0


class TestFindSimilarCluster(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_best_match(self) -> None:
        pid = self.db.create_person("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid)

        fid2 = _add_face(self.db, "/b1.jpg", seed=1)
        fid3 = _add_face(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = find_similar_cluster(self.db, pid)
        assert result == 10

    def test_none_below_threshold(self) -> None:
        pid = self.db.create_person("Alice")
        fid = _add_face(self.db, "/a1.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid)

        fid2 = _add_face(self.db, "/b1.jpg", seed=999)
        fid3 = _add_face(self.db, "/b2.jpg", seed=999)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = find_similar_cluster(self.db, pid, min_similarity=0.99)
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
        pid = self.db.create_person("Alice")
        fid = _add_face(self.db, "/a.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid)

        # Unclustered face with same embedding
        _add_face(self.db, "/b.jpg", seed=1)

        results = find_similar_unclustered(self.db, pid, min_similarity=0.5)
        assert len(results) >= 1
        assert results[0][1] > 90  # similarity percentage

    def test_no_match_below_threshold(self) -> None:
        pid = self.db.create_person("Alice")
        fid = _add_face(self.db, "/a.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid)

        _add_face(self.db, "/b.jpg", seed=999)

        results = find_similar_unclustered(self.db, pid, min_similarity=0.99)
        assert len(results) == 0

    def test_empty_person(self) -> None:
        pid = self.db.create_person("Alice")
        results = find_similar_unclustered(self.db, pid)
        assert results == []


class TestComparePersons(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_finds_swaps(self) -> None:
        """When a face is closer to the other person's centroid, it's flagged."""
        pid_a = self.db.create_person("Alice")
        pid_b = self.db.create_person("Bob")

        # Give Alice a face with seed=1 centroid
        fid_a = _add_face(self.db, "/a.jpg", seed=1)
        self.db.assign_face_to_person(fid_a, pid_a)

        # Give Bob a face with seed=999 centroid
        fid_b = _add_face(self.db, "/b.jpg", seed=999)
        self.db.assign_face_to_person(fid_b, pid_b)

        # Assign a seed=999 face to Alice — should be flagged as swap to Bob
        fid_wrong = _add_face(self.db, "/wrong.jpg", seed=999)
        self.db.assign_face_to_person(fid_wrong, pid_a)

        result = compare_persons(self.db, pid_a, pid_b)
        assert len(result["swaps_a_to_b"]) >= 1

    def test_empty_person(self) -> None:
        pid_a = self.db.create_person("Alice")
        pid_b = self.db.create_person("Bob")
        result = compare_persons(self.db, pid_a, pid_b)
        assert result["swaps_a_to_b"] == []
        assert result["swaps_b_to_a"] == []


class TestRankPersonsForCluster(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_ranks_by_similarity(self) -> None:
        pid_close = self.db.create_person("Close")
        fid = _add_face(self.db, "/close.jpg", seed=1)
        self.db.assign_face_to_person(fid, pid_close)

        pid_far = self.db.create_person("Far")
        fid2 = _add_face(self.db, "/far.jpg", seed=999)
        self.db.assign_face_to_person(fid2, pid_far)

        # Cluster with seed=1 embedding
        fid3 = _add_face(self.db, "/c1.jpg", seed=1)
        fid4 = _add_face(self.db, "/c2.jpg", seed=1)
        self.db.update_cluster_ids({fid3: 10, fid4: 10})

        ranked = rank_persons_for_cluster(self.db, 10)
        assert len(ranked) == 2
        assert ranked[0][1] == "Close"  # most similar first

    def test_empty_cluster(self) -> None:
        assert rank_persons_for_cluster(self.db, 999) == []
