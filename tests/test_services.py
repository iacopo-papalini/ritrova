"""Tests for ritrova.services module."""

from unittest import TestCase

import numpy as np
import pytest

from ritrova.db import FaceDB
from ritrova.services import (
    compute_cluster_hint,
    compute_singleton_hints,
)


def _make_embedding(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_source_with_finding(
    db: FaceDB,
    path: str = "/test/photo.jpg",
    species: str = "human",
    embedding: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[int, int]:
    pid = db.add_source(path, width=100, height=100)
    emb = embedding if embedding is not None else _make_embedding(seed)
    db.add_findings_batch([(pid, (10, 10, 50, 50), emb, 0.95)], species=species)
    findings = db.get_source_findings(pid)
    return pid, findings[0].id


class TestComputeClusterHint(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_best_match(self) -> None:
        emb = _make_embedding(seed=1)
        # Create a subject with similar embedding
        sid = self.db.create_subject("Alice")
        _, fid = _add_source_with_finding(self.db, path="/a.jpg", embedding=emb, seed=1)
        self.db.assign_finding_to_subject(fid, sid)

        # Create a cluster with the same embedding
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg", embedding=emb, seed=1)
        _, fid3 = _add_source_with_finding(self.db, path="/c.jpg", embedding=emb, seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = compute_cluster_hint(self.db, 10)
        assert result is not None
        assert result["name"] == "Alice"
        assert result["person_id"] == sid
        assert result["sim"] > 90

    def test_no_subjects(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 10, fid2: 10})
        result = compute_cluster_hint(self.db, 10)
        assert result is None

    def test_empty_cluster(self) -> None:
        result = compute_cluster_hint(self.db, 999)
        assert result is None

    def test_species_filtering(self) -> None:
        # Subject with human findings
        sid = self.db.create_subject("Alice", kind="person")
        _, fid = _add_source_with_finding(self.db, path="/a.jpg", species="human", seed=1)
        self.db.assign_finding_to_subject(fid, sid)

        # Cluster with dog findings — should NOT match human subject
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg", species="dog", seed=1)
        _, fid3 = _add_source_with_finding(self.db, path="/c.jpg", species="dog", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = compute_cluster_hint(self.db, 10)
        assert result is None


class TestComputeSingletonHints(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_compute_hints(self) -> None:
        emb = _make_embedding(seed=1)
        sid = self.db.create_subject("Alice")
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg", embedding=emb)
        self.db.assign_finding_to_subject(fid1, sid)

        # Singleton finding with same embedding
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg", embedding=emb)
        finding = self.db.get_finding(fid2)
        assert finding is not None
        subject = self.db.get_subject(sid)
        assert subject is not None

        hints = compute_singleton_hints(self.db, [finding], "person")
        assert fid2 in hints
        assert hints[fid2]["name"] == "Alice"
        assert hints[fid2]["sim"] > 90

    def test_no_subjects(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        finding = self.db.get_finding(fid)
        assert finding is not None
        hints = compute_singleton_hints(self.db, [finding], "person")
        assert hints == {}


class TestGetSubjectsByKind(TestCase):
    """Tests for db.get_subjects_by_kind."""

    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_person_filter(self) -> None:
        sid_human = self.db.create_subject("Alice", kind="person")
        _, fid = _add_source_with_finding(self.db, path="/a.jpg", species="human")
        self.db.assign_finding_to_subject(fid, sid_human)

        sid_dog = self.db.create_subject("Figaro", kind="pet")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        self.db.assign_finding_to_subject(fid2, sid_dog)

        person_subjects = self.db.get_subjects_by_kind("person")
        assert len(person_subjects) == 1
        assert person_subjects[0].name == "Alice"

    def test_pet_filter(self) -> None:
        sid_human = self.db.create_subject("Alice", kind="person")
        _, fid = _add_source_with_finding(self.db, path="/a.jpg", species="human")
        self.db.assign_finding_to_subject(fid, sid_human)

        sid_dog = self.db.create_subject("Figaro", kind="pet")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        self.db.assign_finding_to_subject(fid2, sid_dog)

        pet_subjects = self.db.get_subjects_by_kind("pet")
        assert len(pet_subjects) == 1
        assert pet_subjects[0].name == "Figaro"
