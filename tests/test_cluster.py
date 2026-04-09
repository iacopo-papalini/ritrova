"""Integration tests for face_recog.cluster module (real db, no mocking)."""

from unittest import TestCase

import numpy as np
import pytest

from face_recog.cluster import (
    auto_assign,
    auto_merge_clusters,
    find_similar_cluster,
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
