"""Comprehensive tests for ritrova.db module."""

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from ritrova.db import FaceDB


class TestSourceOperations(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_add_and_get_source(self) -> None:
        pid = self.db.add_source("/test/photo.jpg", width=1920, height=1080, taken_at="2024-01-01")
        source = self.db.get_source(pid)
        assert source is not None
        assert source.file_path == "/test/photo.jpg"
        assert source.width == 1920
        assert source.height == 1080
        assert source.taken_at == "2024-01-01"

    def test_get_nonexistent_source(self) -> None:
        assert self.db.get_source(999) is None

    def test_is_scanned_human(self) -> None:
        assert not self.db.is_scanned("/test/photo.jpg", "human")
        sid = self.db.add_source("/test/photo.jpg", width=100, height=100)
        self.db.record_scan(sid, "human")
        assert self.db.is_scanned("/test/photo.jpg", "human")

    def test_is_scanned_pet(self) -> None:
        assert not self.db.is_scanned("/test/photo.jpg", "pet")
        sid = self.db.add_source("/test/photo.jpg", width=100, height=100)
        self.db.record_scan(sid, "pet")
        assert self.db.is_scanned("/test/photo.jpg", "pet")

    def test_is_scanned_video(self) -> None:
        assert not self.db.is_scanned("/test/video.mp4", "human")
        sid = self.db.add_source("/test/video.mp4", source_type="video", width=100, height=100)
        self.db.record_scan(sid, "human")
        assert self.db.is_scanned("/test/video.mp4", "human")

    def test_source_count(self) -> None:
        assert self.db.get_source_count() == 0
        self.db.add_source("/test/a.jpg", width=100, height=100)
        self.db.add_source("/test/b.jpg", width=100, height=100)
        assert self.db.get_source_count() == 2

    def test_duplicate_source_path_raises(self) -> None:
        self.db.add_source("/test/photo.jpg", width=100, height=100)
        with pytest.raises(Exception):  # noqa: B017
            self.db.add_source("/test/photo.jpg", width=200, height=200)


def _make_embedding(dim: int = 512) -> np.ndarray:
    """Create a random normalized embedding."""
    v = np.random.default_rng(42).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_source_with_finding(
    db: FaceDB,
    path: str = "/test/photo.jpg",
    species: str = "human",
    embedding: np.ndarray | None = None,
) -> tuple[int, int]:
    """Helper: add a source + one finding, return (source_id, finding_id)."""
    pid = db.add_source(path, width=100, height=100)
    emb = embedding if embedding is not None else _make_embedding()
    db.add_findings_batch([(pid, (10, 10, 50, 50), emb, 0.95)], species=species)
    findings = db.get_source_findings(pid)
    return pid, findings[0].id


class TestFindingOperations(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_add_and_get_finding(self) -> None:
        pid = self.db.add_source("/test/photo.jpg", width=100, height=100)
        emb = _make_embedding()
        self.db.add_findings_batch([(pid, (10, 20, 30, 40), emb, 0.95)])
        findings = self.db.get_source_findings(pid)
        assert len(findings) == 1
        f = findings[0]
        assert f.bbox_x == 10
        assert f.bbox_y == 20
        assert f.bbox_w == 30
        assert f.bbox_h == 40
        assert f.confidence == pytest.approx(0.95)
        assert f.species == "human"

    def test_get_finding_by_id(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        finding = self.db.get_finding(fid)
        assert finding is not None
        assert finding.id == fid

    def test_get_nonexistent_finding(self) -> None:
        assert self.db.get_finding(999) is None

    def test_embedding_round_trip(self) -> None:
        emb = _make_embedding()
        _, fid = _add_source_with_finding(self.db, embedding=emb)
        finding = self.db.get_finding(fid)
        assert finding is not None
        np.testing.assert_array_almost_equal(finding.embedding, emb)

    def test_finding_count(self) -> None:
        assert self.db.get_finding_count() == 0
        _add_source_with_finding(self.db, path="/a.jpg")
        _add_source_with_finding(self.db, path="/b.jpg")
        assert self.db.get_finding_count() == 2

    def test_get_all_embeddings_excludes_dismissed(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        assert len(self.db.get_all_embeddings()) == 2
        self.db.dismiss_findings([fid1])
        assert len(self.db.get_all_embeddings()) == 1

    def test_get_all_embeddings_species_filter(self) -> None:
        _add_source_with_finding(self.db, path="/a.jpg", species="human")
        _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        assert len(self.db.get_all_embeddings(species="human")) == 1
        assert len(self.db.get_all_embeddings(species="dog")) == 1

    def test_cluster_operations(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        _, fid3 = _add_source_with_finding(self.db, path="/c.jpg")

        self.db.update_cluster_ids({fid1: 1, fid2: 1, fid3: 2})
        assert sorted(self.db.get_cluster_ids()) == [1, 2]
        assert self.db.get_cluster_finding_count(1) == 2
        assert self.db.get_cluster_finding_count(2) == 1

        findings = self.db.get_cluster_findings(1)
        assert len(findings) == 2

        self.db.clear_clusters()
        assert self.db.get_cluster_ids() == []

    def test_clear_clusters_species_scoped(self) -> None:
        _, fid_human = _add_source_with_finding(self.db, path="/h.jpg", species="human")
        _, fid_dog = _add_source_with_finding(self.db, path="/d.jpg", species="dog")
        self.db.update_cluster_ids({fid_human: 1, fid_dog: 2})

        # Clear only human clusters
        self.db.clear_clusters(species="human")
        human_finding = self.db.get_finding(fid_human)
        dog_finding = self.db.get_finding(fid_dog)
        assert human_finding is not None
        assert human_finding.cluster_id is None
        assert dog_finding is not None
        assert dog_finding.cluster_id == 2

    def test_get_cluster_finding_ids(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 5, fid2: 5})
        ids = self.db.get_cluster_finding_ids(5)
        assert sorted(ids) == sorted([fid1, fid2])

    def test_singleton_findings(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        # Both unclustered — both are singletons
        singletons = self.db.get_singleton_findings()
        assert len(singletons) == 2

        # Put one in a cluster of size 2 — not a singleton anymore
        _, fid3 = _add_source_with_finding(self.db, path="/c.jpg")
        self.db.update_cluster_ids({fid1: 1, fid3: 1})
        singletons = self.db.get_singleton_findings()
        # fid2 is unclustered, fid1/fid3 in cluster of 2 — only fid2 is singleton
        assert len(singletons) == 1
        assert singletons[0].id == fid2

    def test_singleton_count(self) -> None:
        _add_source_with_finding(self.db, path="/a.jpg")
        _add_source_with_finding(self.db, path="/b.jpg")
        assert self.db.get_singleton_count() == 2

    def test_dismiss_findings(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        subject_id = self.db.create_subject("Test")
        self.db.assign_finding_to_subject(fid, subject_id)
        self.db.update_cluster_ids({fid: 1})

        self.db.dismiss_findings([fid])
        finding = self.db.get_finding(fid)
        assert finding is not None
        assert finding.person_id is None
        assert finding.cluster_id is None

    def test_unassign_finding(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        subject_id = self.db.create_subject("Test")
        self.db.assign_finding_to_subject(fid, subject_id)
        finding = self.db.get_finding(fid)
        assert finding is not None
        assert finding.person_id == subject_id

        self.db.unassign_finding(fid)
        finding = self.db.get_finding(fid)
        assert finding is not None
        assert finding.person_id is None

    def test_exclude_findings(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 1, fid2: 1})

        self.db.exclude_findings([fid1], cluster_id=1)
        finding = self.db.get_finding(fid1)
        assert finding is not None
        assert finding.cluster_id is None
        # fid2 still in cluster
        finding2 = self.db.get_finding(fid2)
        assert finding2 is not None
        assert finding2.cluster_id == 1

    def test_merge_clusters(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 1, fid2: 2})

        self.db.merge_clusters(source_id=2, target_id=1)
        finding = self.db.get_finding(fid2)
        assert finding is not None
        assert finding.cluster_id == 1


class TestSubjectOperations(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_create_subject(self) -> None:
        sid = self.db.create_subject("Alice")
        subject = self.db.get_subject(sid)
        assert subject is not None
        assert subject.name == "Alice"
        assert subject.face_count == 0

    def test_create_subject_idempotent(self) -> None:
        sid1 = self.db.create_subject("Alice")
        sid2 = self.db.create_subject("Alice")
        assert sid1 == sid2

    def test_get_subjects(self) -> None:
        self.db.create_subject("Bob")
        self.db.create_subject("Alice")
        subjects = self.db.get_subjects()
        assert len(subjects) == 2
        # Ordered by name
        assert subjects[0].name == "Alice"
        assert subjects[1].name == "Bob"

    def test_rename_subject(self) -> None:
        sid = self.db.create_subject("Alice")
        self.db.rename_subject(sid, "Alicia")
        subject = self.db.get_subject(sid)
        assert subject is not None
        assert subject.name == "Alicia"

    def test_assign_finding_to_subject(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        sid = self.db.create_subject("Alice")
        self.db.assign_finding_to_subject(fid, sid)
        subject = self.db.get_subject(sid)
        assert subject is not None
        assert subject.face_count == 1

    def test_assign_cluster_to_subject(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.update_cluster_ids({fid1: 1, fid2: 1})
        sid = self.db.create_subject("Alice")
        self.db.assign_cluster_to_subject(1, sid)
        subject = self.db.get_subject(sid)
        assert subject is not None
        assert subject.face_count == 2

    def test_merge_subjects(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Alice2")
        _, fid = _add_source_with_finding(self.db)
        self.db.assign_finding_to_subject(fid, sid_a)

        self.db.merge_subjects(sid_a, sid_b)
        # Source deleted
        assert self.db.get_subject(sid_a) is None
        # Target has the finding
        subject = self.db.get_subject(sid_b)
        assert subject is not None
        assert subject.face_count == 1

    def test_delete_subject(self) -> None:
        sid = self.db.create_subject("Alice")
        _, fid = _add_source_with_finding(self.db)
        self.db.assign_finding_to_subject(fid, sid)

        self.db.delete_subject(sid)
        assert self.db.get_subject(sid) is None
        # Finding should be unassigned, not deleted
        finding = self.db.get_finding(fid)
        assert finding is not None
        assert finding.person_id is None

    def test_search_subjects(self) -> None:
        self.db.create_subject("Alice")
        self.db.create_subject("Bob")
        self.db.create_subject("Alicia")
        results = self.db.search_subjects("Ali")
        assert len(results) == 2

    def test_get_subject_findings(self) -> None:
        sid = self.db.create_subject("Alice")
        _, fid = _add_source_with_finding(self.db)
        self.db.assign_finding_to_subject(fid, sid)
        findings = self.db.get_subject_findings(sid)
        assert len(findings) == 1
        assert findings[0].id == fid

    def test_get_subject_sources(self) -> None:
        sid = self.db.create_subject("Alice")
        source_id, fid = _add_source_with_finding(self.db)
        self.db.assign_finding_to_subject(fid, sid)
        sources = self.db.get_subject_sources(sid)
        assert len(sources) == 1
        assert sources[0].id == source_id

    def test_subject_kind_person(self) -> None:
        sid = self.db.create_subject("Alice", kind="person")
        subject = self.db.get_subject(sid)
        assert subject is not None
        assert subject.kind == "person"

    def test_subject_kind_pet(self) -> None:
        sid = self.db.create_subject("Figaro", kind="pet")
        subject = self.db.get_subject(sid)
        assert subject is not None
        assert subject.kind == "pet"


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
        _add_source_with_finding(self.db, path="/a.jpg", species="human")
        _add_source_with_finding(self.db, path="/b.jpg", species="human")
        _add_source_with_finding(self.db, path="/c.jpg", species="dog")
        _add_source_with_finding(self.db, path="/d.jpg", species="dog")

        # Get finding ids and cluster them
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
        assert stats["total_sources"] == 0
        assert stats["total_findings"] == 0
        assert stats["total_subjects"] == 0
        assert stats["named_findings"] == 0
        assert stats["unnamed_clusters"] == 0
        assert stats["unclustered_findings"] == 0
        assert stats["dismissed_findings"] == 0

    def test_populated_stats(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        _, fid3 = _add_source_with_finding(self.db, path="/c.jpg")

        sid = self.db.create_subject("Alice")
        self.db.assign_finding_to_subject(fid1, sid)
        self.db.update_cluster_ids({fid2: 1, fid3: 1})

        stats = self.db.get_stats()
        assert stats["total_sources"] == 3
        assert stats["total_findings"] == 3
        assert stats["total_subjects"] == 1
        assert stats["named_findings"] == 1
        assert stats["unnamed_clusters"] == 1
        assert stats["unclustered_findings"] == 1  # fid1 assigned but unclustered


class TestGetUnclusteredEmbeddings(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_returns_unclustered_unassigned(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        _, fid3 = _add_source_with_finding(self.db, path="/c.jpg")

        # Cluster fid1, assign fid2 to a subject
        self.db.update_cluster_ids({fid1: 1})
        sid = self.db.create_subject("Alice")
        self.db.assign_finding_to_subject(fid2, sid)

        result = self.db.get_unclustered_embeddings(species="human")
        assert len(result) == 1
        assert result[0][0] == fid3

    def test_excludes_dismissed(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.dismiss_findings([fid1])
        result = self.db.get_unclustered_embeddings(species="human")
        assert len(result) == 1
        assert result[0][0] == fid2

    def test_species_filter(self) -> None:
        _add_source_with_finding(self.db, path="/a.jpg", species="human")
        _add_source_with_finding(self.db, path="/b.jpg", species="dog")
        assert len(self.db.get_unclustered_embeddings(species="human")) == 1
        assert len(self.db.get_unclustered_embeddings(species="dog")) == 1
        assert len(self.db.get_unclustered_embeddings(species="pet")) == 1


class TestExport(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_export_json_structure(self) -> None:
        import json

        sid = self.db.create_subject("Alice")
        _, fid = _add_source_with_finding(self.db)
        self.db.assign_finding_to_subject(fid, sid)

        data = json.loads(self.db.export_json())
        assert "subjects" in data
        assert "unnamed_findings" in data
        assert len(data["subjects"]) == 1
        assert data["subjects"][0]["name"] == "Alice"
        assert len(data["subjects"][0]["sources"]) == 1

    def test_export_empty_db(self) -> None:
        import json

        data = json.loads(self.db.export_json())
        assert data["subjects"] == []
        assert data["unnamed_findings"] == []


class TestPathResolution(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.base = tmp_path / "photos"
        self.base.mkdir()
        self.db = FaceDB(tmp_path / "test.db", base_dir=self.base)

    def test_resolve_relative_path(self) -> None:
        resolved = self.db.resolve_path("subfolder/photo.jpg")
        assert resolved == self.base / "subfolder" / "photo.jpg"

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
