"""Comprehensive tests for ritrova.db module."""

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from ritrova.db import FaceDB


class TestPhotoOperations(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_add_and_get_photo(self) -> None:
        pid = self.db.add_photo("/test/photo.jpg", 1920, 1080, taken_at="2024-01-01")
        photo = self.db.get_photo(pid)
        assert photo is not None
        assert photo.file_path == "/test/photo.jpg"
        assert photo.width == 1920
        assert photo.height == 1080
        assert photo.taken_at == "2024-01-01"

    def test_get_nonexistent_photo(self) -> None:
        assert self.db.get_photo(999) is None

    def test_is_photo_scanned(self) -> None:
        assert not self.db.is_photo_scanned("/test/photo.jpg")
        self.db.add_photo("/test/photo.jpg", 100, 100)
        assert self.db.is_photo_scanned("/test/photo.jpg")

    def test_is_pet_scanned(self) -> None:
        assert not self.db.is_pet_scanned("/test/photo.jpg")
        self.db.add_photo("/test/photo.jpg__pets", 100, 100)
        assert self.db.is_pet_scanned("/test/photo.jpg")

    def test_is_video_scanned(self) -> None:
        assert not self.db.is_video_scanned("/test/video.mp4")
        self.db.add_photo("/test/frame_001.jpg", 100, 100, video_path="/test/video.mp4")
        assert self.db.is_video_scanned("/test/video.mp4")

    def test_photo_count(self) -> None:
        assert self.db.get_photo_count() == 0
        self.db.add_photo("/test/a.jpg", 100, 100)
        self.db.add_photo("/test/b.jpg", 100, 100)
        assert self.db.get_photo_count() == 2

    def test_duplicate_photo_path_raises(self) -> None:
        self.db.add_photo("/test/photo.jpg", 100, 100)
        with pytest.raises(Exception):  # noqa: B017
            self.db.add_photo("/test/photo.jpg", 200, 200)


def _make_embedding(dim: int = 512) -> np.ndarray:
    """Create a random normalized embedding."""
    v = np.random.default_rng(42).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_photo_with_face(
    db: FaceDB,
    path: str = "/test/photo.jpg",
    species: str = "human",
    embedding: np.ndarray | None = None,
) -> tuple[int, int]:
    """Helper: add a photo + one face, return (photo_id, face_id)."""
    pid = db.add_photo(path, 100, 100)
    emb = embedding if embedding is not None else _make_embedding()
    db.add_faces_batch([(pid, (10, 10, 50, 50), emb, 0.95)], species=species)
    faces = db.get_photo_faces(pid)
    return pid, faces[0].id


class TestFaceOperations(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_add_and_get_face(self) -> None:
        pid = self.db.add_photo("/test/photo.jpg", 100, 100)
        emb = _make_embedding()
        self.db.add_faces_batch([(pid, (10, 20, 30, 40), emb, 0.95)])
        faces = self.db.get_photo_faces(pid)
        assert len(faces) == 1
        f = faces[0]
        assert f.bbox_x == 10
        assert f.bbox_y == 20
        assert f.bbox_w == 30
        assert f.bbox_h == 40
        assert f.confidence == pytest.approx(0.95)
        assert f.species == "human"

    def test_get_face_by_id(self) -> None:
        _, fid = _add_photo_with_face(self.db)
        face = self.db.get_face(fid)
        assert face is not None
        assert face.id == fid

    def test_get_nonexistent_face(self) -> None:
        assert self.db.get_face(999) is None

    def test_embedding_round_trip(self) -> None:
        emb = _make_embedding()
        _, fid = _add_photo_with_face(self.db, embedding=emb)
        face = self.db.get_face(fid)
        assert face is not None
        np.testing.assert_array_almost_equal(face.embedding, emb)

    def test_face_count(self) -> None:
        assert self.db.get_face_count() == 0
        _add_photo_with_face(self.db, path="/a.jpg")
        _add_photo_with_face(self.db, path="/b.jpg")
        assert self.db.get_face_count() == 2

    def test_get_all_embeddings_excludes_dismissed(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        assert len(self.db.get_all_embeddings()) == 2
        self.db.dismiss_faces([fid1])
        assert len(self.db.get_all_embeddings()) == 1

    def test_get_all_embeddings_species_filter(self) -> None:
        _add_photo_with_face(self.db, path="/a.jpg", species="human")
        _add_photo_with_face(self.db, path="/b.jpg", species="dog")
        assert len(self.db.get_all_embeddings(species="human")) == 1
        assert len(self.db.get_all_embeddings(species="dog")) == 1

    def test_cluster_operations(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        _, fid3 = _add_photo_with_face(self.db, path="/c.jpg")

        self.db.update_cluster_ids({fid1: 1, fid2: 1, fid3: 2})
        assert sorted(self.db.get_cluster_ids()) == [1, 2]
        assert self.db.get_cluster_face_count(1) == 2
        assert self.db.get_cluster_face_count(2) == 1

        faces = self.db.get_cluster_faces(1)
        assert len(faces) == 2

        self.db.clear_clusters()
        assert self.db.get_cluster_ids() == []

    def test_clear_clusters_species_scoped(self) -> None:
        _, fid_human = _add_photo_with_face(self.db, path="/h.jpg", species="human")
        _, fid_dog = _add_photo_with_face(self.db, path="/d.jpg", species="dog")
        self.db.update_cluster_ids({fid_human: 1, fid_dog: 2})

        # Clear only human clusters
        self.db.clear_clusters(species="human")
        human_face = self.db.get_face(fid_human)
        dog_face = self.db.get_face(fid_dog)
        assert human_face is not None
        assert human_face.cluster_id is None
        assert dog_face is not None
        assert dog_face.cluster_id == 2

    def test_get_cluster_face_ids(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 5, fid2: 5})
        ids = self.db.get_cluster_face_ids(5)
        assert sorted(ids) == sorted([fid1, fid2])

    def test_singleton_faces(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        # Both unclustered — both are singletons
        singletons = self.db.get_singleton_faces()
        assert len(singletons) == 2

        # Put one in a cluster of size 2 — not a singleton anymore
        _, fid3 = _add_photo_with_face(self.db, path="/c.jpg")
        self.db.update_cluster_ids({fid1: 1, fid3: 1})
        singletons = self.db.get_singleton_faces()
        # fid2 is unclustered, fid1/fid3 in cluster of 2 — only fid2 is singleton
        assert len(singletons) == 1
        assert singletons[0].id == fid2

    def test_singleton_count(self) -> None:
        _add_photo_with_face(self.db, path="/a.jpg")
        _add_photo_with_face(self.db, path="/b.jpg")
        assert self.db.get_singleton_count() == 2

    def test_dismiss_faces(self) -> None:
        _, fid = _add_photo_with_face(self.db)
        person_id = self.db.create_person("Test")
        self.db.assign_face_to_person(fid, person_id)
        self.db.update_cluster_ids({fid: 1})

        self.db.dismiss_faces([fid])
        face = self.db.get_face(fid)
        assert face is not None
        assert face.person_id is None
        assert face.cluster_id is None

    def test_unassign_face(self) -> None:
        _, fid = _add_photo_with_face(self.db)
        person_id = self.db.create_person("Test")
        self.db.assign_face_to_person(fid, person_id)
        face = self.db.get_face(fid)
        assert face is not None
        assert face.person_id == person_id

        self.db.unassign_face(fid)
        face = self.db.get_face(fid)
        assert face is not None
        assert face.person_id is None

    def test_exclude_faces(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 1, fid2: 1})

        self.db.exclude_faces([fid1], cluster_id=1)
        face = self.db.get_face(fid1)
        assert face is not None
        assert face.cluster_id is None
        # fid2 still in cluster
        face2 = self.db.get_face(fid2)
        assert face2 is not None
        assert face2.cluster_id == 1

    def test_merge_clusters(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 1, fid2: 2})

        self.db.merge_clusters(source_id=2, target_id=1)
        face = self.db.get_face(fid2)
        assert face is not None
        assert face.cluster_id == 1


class TestPersonOperations(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_create_person(self) -> None:
        pid = self.db.create_person("Alice")
        person = self.db.get_person(pid)
        assert person is not None
        assert person.name == "Alice"
        assert person.face_count == 0

    def test_create_person_idempotent(self) -> None:
        pid1 = self.db.create_person("Alice")
        pid2 = self.db.create_person("Alice")
        assert pid1 == pid2

    def test_get_persons(self) -> None:
        self.db.create_person("Bob")
        self.db.create_person("Alice")
        persons = self.db.get_persons()
        assert len(persons) == 2
        # Ordered by name
        assert persons[0].name == "Alice"
        assert persons[1].name == "Bob"

    def test_rename_person(self) -> None:
        pid = self.db.create_person("Alice")
        self.db.rename_person(pid, "Alicia")
        person = self.db.get_person(pid)
        assert person is not None
        assert person.name == "Alicia"

    def test_assign_face_to_person(self) -> None:
        _, fid = _add_photo_with_face(self.db)
        pid = self.db.create_person("Alice")
        self.db.assign_face_to_person(fid, pid)
        person = self.db.get_person(pid)
        assert person is not None
        assert person.face_count == 1

    def test_assign_cluster_to_person(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 1, fid2: 1})
        pid = self.db.create_person("Alice")
        self.db.assign_cluster_to_person(1, pid)
        person = self.db.get_person(pid)
        assert person is not None
        assert person.face_count == 2

    def test_merge_persons(self) -> None:
        pid_a = self.db.create_person("Alice")
        pid_b = self.db.create_person("Alice2")
        _, fid = _add_photo_with_face(self.db)
        self.db.assign_face_to_person(fid, pid_a)

        self.db.merge_persons(pid_a, pid_b)
        # Source deleted
        assert self.db.get_person(pid_a) is None
        # Target has the face
        person = self.db.get_person(pid_b)
        assert person is not None
        assert person.face_count == 1

    def test_delete_person(self) -> None:
        pid = self.db.create_person("Alice")
        _, fid = _add_photo_with_face(self.db)
        self.db.assign_face_to_person(fid, pid)

        self.db.delete_person(pid)
        assert self.db.get_person(pid) is None
        # Face should be unassigned, not deleted
        face = self.db.get_face(fid)
        assert face is not None
        assert face.person_id is None

    def test_search_persons(self) -> None:
        self.db.create_person("Alice")
        self.db.create_person("Bob")
        self.db.create_person("Alicia")
        results = self.db.search_persons("Ali")
        assert len(results) == 2

    def test_get_person_faces(self) -> None:
        pid = self.db.create_person("Alice")
        _, fid = _add_photo_with_face(self.db)
        self.db.assign_face_to_person(fid, pid)
        faces = self.db.get_person_faces(pid)
        assert len(faces) == 1
        assert faces[0].id == fid

    def test_get_person_photos(self) -> None:
        pid = self.db.create_person("Alice")
        photo_id, fid = _add_photo_with_face(self.db)
        self.db.assign_face_to_person(fid, pid)
        photos = self.db.get_person_photos(pid)
        assert len(photos) == 1
        assert photos[0].id == photo_id

    def test_has_person_species_human(self) -> None:
        pid = self.db.create_person("Alice")
        _add_photo_with_face(self.db, path="/a.jpg", species="human")
        faces = self.db.get_photo_faces(
            self.db.query("SELECT id FROM photos WHERE file_path = '/a.jpg'")[0][0]
        )
        self.db.assign_face_to_person(faces[0].id, pid)
        assert self.db.has_person_species(pid, "human")
        assert not self.db.has_person_species(pid, "pet")

    def test_has_person_species_pet(self) -> None:
        pid = self.db.create_person("Figaro")
        _add_photo_with_face(self.db, path="/a.jpg", species="dog")
        faces = self.db.get_photo_faces(
            self.db.query("SELECT id FROM photos WHERE file_path = '/a.jpg'")[0][0]
        )
        self.db.assign_face_to_person(faces[0].id, pid)
        assert self.db.has_person_species(pid, "pet")
        assert not self.db.has_person_species(pid, "human")


class TestSpeciesFilter(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_human_filter(self) -> None:
        clause, params = self.db.species_filter("human")
        assert "species = ?" in clause
        assert params == ("human",)

    def test_pet_filter(self) -> None:
        clause, params = self.db.species_filter("pet")
        assert "IN" in clause
        assert len(params) == len(self.db.PET_SPECIES)

    def test_unnamed_clusters_species(self) -> None:
        _add_photo_with_face(self.db, path="/a.jpg", species="human")
        _add_photo_with_face(self.db, path="/b.jpg", species="human")
        _add_photo_with_face(self.db, path="/c.jpg", species="dog")
        _add_photo_with_face(self.db, path="/d.jpg", species="dog")

        # Get face ids and cluster them
        human_embs = self.db.get_all_embeddings(species="human")
        dog_embs = self.db.get_all_embeddings(species="dog")
        self.db.update_cluster_ids(
            {human_embs[0][0]: 1, human_embs[1][0]: 1, dog_embs[0][0]: 2, dog_embs[1][0]: 2}
        )

        human_clusters = self.db.get_unnamed_clusters(species="human")
        assert len(human_clusters) == 1
        assert human_clusters[0]["cluster_id"] == 1

        pet_clusters = self.db.get_unnamed_clusters(species="pet")
        assert len(pet_clusters) == 1
        assert pet_clusters[0]["cluster_id"] == 2


class TestStats(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_empty_db_stats(self) -> None:
        stats = self.db.get_stats()
        assert stats["total_photos"] == 0
        assert stats["total_faces"] == 0
        assert stats["total_persons"] == 0
        assert stats["named_faces"] == 0
        assert stats["unnamed_clusters"] == 0
        assert stats["unclustered_faces"] == 0
        assert stats["dismissed_faces"] == 0

    def test_populated_stats(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        _, fid3 = _add_photo_with_face(self.db, path="/c.jpg")

        pid = self.db.create_person("Alice")
        self.db.assign_face_to_person(fid1, pid)
        self.db.update_cluster_ids({fid2: 1, fid3: 1})

        stats = self.db.get_stats()
        assert stats["total_photos"] == 3
        assert stats["total_faces"] == 3
        assert stats["total_persons"] == 1
        assert stats["named_faces"] == 1
        assert stats["unnamed_clusters"] == 1
        assert stats["unclustered_faces"] == 1  # fid1 assigned but unclustered


class TestGetUnclusteredEmbeddings(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_returns_unclustered_unassigned(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        _, fid3 = _add_photo_with_face(self.db, path="/c.jpg")

        # Cluster fid1, assign fid2 to a person
        self.db.update_cluster_ids({fid1: 1})
        pid = self.db.create_person("Alice")
        self.db.assign_face_to_person(fid2, pid)

        result = self.db.get_unclustered_embeddings(species="human")
        assert len(result) == 1
        assert result[0][0] == fid3

    def test_excludes_dismissed(self) -> None:
        _, fid1 = _add_photo_with_face(self.db, path="/a.jpg")
        _, fid2 = _add_photo_with_face(self.db, path="/b.jpg")
        self.db.dismiss_faces([fid1])
        result = self.db.get_unclustered_embeddings(species="human")
        assert len(result) == 1
        assert result[0][0] == fid2

    def test_species_filter(self) -> None:
        _add_photo_with_face(self.db, path="/a.jpg", species="human")
        _add_photo_with_face(self.db, path="/b.jpg", species="dog")
        assert len(self.db.get_unclustered_embeddings(species="human")) == 1
        assert len(self.db.get_unclustered_embeddings(species="dog")) == 1
        assert len(self.db.get_unclustered_embeddings(species="pet")) == 1


class TestExport(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_export_json_structure(self) -> None:
        import json

        pid = self.db.create_person("Alice")
        _, fid = _add_photo_with_face(self.db)
        self.db.assign_face_to_person(fid, pid)

        data = json.loads(self.db.export_json())
        assert "persons" in data
        assert "unnamed_faces" in data
        assert len(data["persons"]) == 1
        assert data["persons"][0]["name"] == "Alice"
        assert len(data["persons"][0]["photos"]) == 1

    def test_export_empty_db(self) -> None:
        import json

        data = json.loads(self.db.export_json())
        assert data["persons"] == []
        assert data["unnamed_faces"] == []


class TestPathResolution(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.base = tmp_path / "photos"
        self.base.mkdir()
        self.db = FaceDB(tmp_path / "test.db", base_dir=self.base)

    def test_resolve_relative_path(self) -> None:
        resolved = self.db.resolve_path("subfolder/photo.jpg")
        assert resolved == self.base / "subfolder" / "photo.jpg"

    def test_resolve_strips_pets_suffix(self) -> None:
        resolved = self.db.resolve_path("photo.jpg__pets")
        assert resolved == self.base / "photo.jpg"

    def test_resolve_rejects_dotdot(self) -> None:
        with pytest.raises(ValueError, match="\\.\\."):
            self.db.resolve_path("../etc/passwd")

    def test_resolve_absolute_path_passthrough(self) -> None:
        """Legacy absolute paths are returned as-is."""
        resolved = self.db.resolve_path("/absolute/path/photo.jpg")
        assert resolved == Path("/absolute/path/photo.jpg")

    def test_to_relative(self) -> None:
        abs_path = str(self.base / "subfolder" / "photo.jpg")
        rel = self.db.to_relative(abs_path)
        assert rel == "subfolder/photo.jpg"

    def test_to_relative_outside_base(self) -> None:
        """Paths outside base_dir are returned as-is."""
        rel = self.db.to_relative("/somewhere/else/photo.jpg")
        assert rel == "/somewhere/else/photo.jpg"

    def test_no_base_dir_passthrough(self) -> None:
        db_no_base = FaceDB(self.db.db_path.parent / "test2.db")
        assert db_no_base.resolve_path("photo.jpg") == Path("photo.jpg")
        assert db_no_base.to_relative("/abs/photo.jpg") == "/abs/photo.jpg"
        db_no_base.close()
