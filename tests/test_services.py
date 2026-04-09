"""Tests for face_recog.services module."""

from unittest import TestCase

import numpy as np
import pytest

from face_recog.db import FaceDB
from face_recog.services import (
    compute_cluster_hint,
    compute_singleton_hints,
    filter_persons_by_species,
)


def _make_embedding(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_photo_with_face(
    db: FaceDB,
    path: str = "/test/photo.jpg",
    species: str = "human",
    embedding: np.ndarray | None = None,
    seed: int = 42,
) -> tuple[int, int]:
    pid = db.add_photo(path, 100, 100)
    emb = embedding if embedding is not None else _make_embedding(seed)
    db.add_faces_batch([(pid, (10, 10, 50, 50), emb, 0.95)], species=species)
    faces = db.get_photo_faces(pid)
    return pid, faces[0].id


class TestComputeClusterHint(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_best_match(self) -> None:
        emb = _make_embedding(seed=1)
        # Create a person with similar embedding
        pid = self.db.create_person("Alice")
        _, fid = _add_photo_with_face(self.db, path="/a.jpg", embedding=emb, seed=1)
        self.db.assign_face_to_person(fid, pid)

        # Create a cluster with the same embedding
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg", embedding=emb, seed=1)
        _, fid3 = _add_photo_with_face(self.db, path="/c.jpg", embedding=emb, seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = compute_cluster_hint(self.db, 10)
        assert result is not None
        assert result["name"] == "Alice"
        assert result["person_id"] == pid
        assert result["sim"] > 90

    def test_no_persons(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 10, fid2: 10})
        result = compute_cluster_hint(self.db, 10)
        assert result is None

    def test_empty_cluster(self) -> None:
        result = compute_cluster_hint(self.db, 999)
        assert result is None

    def test_species_filtering(self) -> None:
        # Person with human faces
        pid = self.db.create_person("Alice")
        _, fid = _add_photo_with_face(self.db, path="/a.jpg", species="human", seed=1)
        self.db.assign_face_to_person(fid, pid)

        # Cluster with dog faces — should NOT match human person
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg", species="dog", seed=1)
        _, fid3 = _add_photo_with_face(self.db, path="/c.jpg", species="dog", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        result = compute_cluster_hint(self.db, 10)
        assert result is None


class TestComputeSingletonHints(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_compute_hints(self) -> None:
        emb = _make_embedding(seed=1)
        pid = self.db.create_person("Alice")
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg", embedding=emb)
        self.db.assign_face_to_person(fid1, pid)

        # Singleton face with same embedding
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg", embedding=emb)
        face = self.db.get_face(fid2)
        assert face is not None
        person = self.db.get_person(pid)
        assert person is not None

        hints = compute_singleton_hints(self.db, [face], [person], "human")
        assert fid2 in hints
        assert hints[fid2]["name"] == "Alice"
        assert hints[fid2]["sim"] > 90

    def test_no_persons(self) -> None:
        _, fid = _add_photo_with_face(self.db)
        face = self.db.get_face(fid)
        assert face is not None
        hints = compute_singleton_hints(self.db, [face], [], "human")
        assert hints == {}


class TestFilterPersonsBySpecies(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_human_filter(self) -> None:
        pid_human = self.db.create_person("Alice")
        _, fid = _add_photo_with_face(self.db, path="/a.jpg", species="human")
        self.db.assign_face_to_person(fid, pid_human)

        pid_dog = self.db.create_person("Figaro")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg", species="dog")
        self.db.assign_face_to_person(fid2, pid_dog)

        all_persons = self.db.get_persons()
        human_persons = filter_persons_by_species(self.db, all_persons, "human")
        assert len(human_persons) == 1
        assert human_persons[0].name == "Alice"

    def test_pet_filter(self) -> None:
        pid_human = self.db.create_person("Alice")
        _, fid = _add_photo_with_face(self.db, path="/a.jpg", species="human")
        self.db.assign_face_to_person(fid, pid_human)

        pid_dog = self.db.create_person("Figaro")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg", species="dog")
        self.db.assign_face_to_person(fid2, pid_dog)

        all_persons = self.db.get_persons()
        pet_persons = filter_persons_by_species(self.db, all_persons, "pet")
        assert len(pet_persons) == 1
        assert pet_persons[0].name == "Figaro"
