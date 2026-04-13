"""Comprehensive tests for ritrova.db module."""

import sqlite3
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from ritrova.db import FaceDB


def _seed_legacy_db(
    path: Path,
    *,
    sources: list[tuple[int, str, str]],
    scans: list[tuple[int, int, str]],
    findings: list[tuple[int, int, str | None]],
) -> None:
    """Write a pre-migration DB by hand: no `findings.scan_id` column.

    Mimics the on-disk shape of an upgrade scenario so we can verify that
    `FaceDB.__init__` successfully migrates + backfills + enforces NOT NULL.

    `sources` rows: (id, file_path, type)
    `scans`   rows: (id, source_id, scan_type)
    `findings` rows: (source_id, embedding_dim, frame_path)
    """
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sources (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL DEFAULT 'photo',
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            taken_at TEXT, latitude REAL, longitude REAL
        );
        CREATE TABLE scans (
            id INTEGER PRIMARY KEY,
            source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            scan_type TEXT NOT NULL,
            scanned_at TEXT NOT NULL DEFAULT '2024-01-01',
            detection_strategy TEXT NOT NULL DEFAULT 'unknown',
            UNIQUE (source_id, scan_type)
        );
        CREATE TABLE subjects (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'person',
            created_at TEXT NOT NULL DEFAULT '2024-01-01'
        );
        CREATE TABLE findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            bbox_x INTEGER NOT NULL DEFAULT 0,
            bbox_y INTEGER NOT NULL DEFAULT 0,
            bbox_w INTEGER NOT NULL DEFAULT 0,
            bbox_h INTEGER NOT NULL DEFAULT 0,
            embedding BLOB NOT NULL,
            person_id INTEGER REFERENCES subjects(id) ON DELETE SET NULL,
            cluster_id INTEGER,
            confidence REAL NOT NULL DEFAULT 0.0,
            species TEXT NOT NULL DEFAULT 'human',
            detected_at TEXT NOT NULL DEFAULT '2024-01-01',
            frame_path TEXT,
            embedding_dim INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE dismissed_findings (
            finding_id INTEGER PRIMARY KEY REFERENCES findings(id) ON DELETE CASCADE
        );
        CREATE UNIQUE INDEX idx_subjects_name_kind ON subjects(name, kind);
        """
    )
    for sid, fp, t in sources:
        conn.execute("INSERT INTO sources (id, file_path, type) VALUES (?, ?, ?)", (sid, fp, t))
    for scan_id, source_id, scan_type in scans:
        conn.execute(
            "INSERT INTO scans (id, source_id, scan_type) VALUES (?, ?, ?)",
            (scan_id, source_id, scan_type),
        )
    for source_id, dim, frame_path in findings:
        emb = np.zeros(dim, dtype=np.float32)
        emb[0] = 1.0  # non-empty blob for length() check
        conn.execute(
            "INSERT INTO findings (source_id, embedding, embedding_dim, frame_path) "
            "VALUES (?, ?, ?, ?)",
            (source_id, emb.tobytes(), dim, frame_path),
        )
    conn.commit()
    conn.close()


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
    from ._helpers import add_findings

    pid = db.add_source(path, width=100, height=100)
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    emb = embedding if embedding is not None else _make_embedding(dim=dim)
    add_findings(db, [(pid, (10, 10, 50, 50), emb, 0.95)], species=species)
    findings = db.get_source_findings(pid)
    return pid, findings[0].id


class TestFindingOperations(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_add_and_get_finding(self) -> None:
        from ._helpers import add_findings

        pid = self.db.add_source("/test/photo.jpg", width=100, height=100)
        emb = _make_embedding()
        add_findings(self.db, [(pid, (10, 20, 30, 40), emb, 0.95)])
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
    def _setup_db(self, db: FaceDB, tmp_path: Path) -> None:
        self.db = db
        self.tmp_path = tmp_path

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

    def test_record_scan_returns_id(self) -> None:
        sid = self.db.add_source("/test/p.jpg", width=10, height=10)
        scan_id = self.db.record_scan(sid, "human")
        assert isinstance(scan_id, int)
        assert scan_id > 0

    def test_add_findings_batch_requires_scan_id_keyword(self) -> None:
        sid = self.db.add_source("/test/p.jpg", width=10, height=10)
        emb = _make_embedding()
        # Positional scan_id is rejected (kwarg-only).
        with pytest.raises(TypeError):
            self.db.add_findings_batch([(sid, (0, 0, 10, 10), emb, 0.9)], 1)
        # Missing entirely is rejected too.
        with pytest.raises(TypeError):
            self.db.add_findings_batch([(sid, (0, 0, 10, 10), emb, 0.9)])

    def test_delete_scan_cascades_to_findings(self) -> None:
        sid = self.db.add_source("/test/p.jpg", width=10, height=10)
        scan_id = self.db.record_scan(sid, "human")
        self.db.add_findings_batch([(sid, (0, 0, 10, 10), _make_embedding(), 0.9)], scan_id=scan_id)
        assert len(self.db.get_source_findings(sid)) == 1

        result = self.db.delete_scan(scan_id)
        assert result["deleted_findings"] == 1
        assert result["deleted_with_assignments"] == 0
        assert self.db.get_source_findings(sid) == []
        # Scan row is gone.
        assert (
            self.db.conn.execute("SELECT 1 FROM scans WHERE id = ?", (scan_id,)).fetchone() is None
        )

    def test_delete_scan_counts_manual_assignments(self) -> None:
        subject = self.db.create_subject("Alice")
        sid = self.db.add_source("/test/p.jpg", width=10, height=10)
        scan_id = self.db.record_scan(sid, "human")
        self.db.add_findings_batch(
            [
                (sid, (0, 0, 10, 10), _make_embedding(), 0.9),
                (sid, (5, 5, 10, 10), _make_embedding(), 0.9),
            ],
            scan_id=scan_id,
        )
        findings = self.db.get_source_findings(sid)
        self.db.assign_finding_to_subject(findings[0].id, subject)

        result = self.db.delete_scan(scan_id)
        assert result["deleted_findings"] == 2
        assert result["deleted_with_assignments"] == 1

    def test_delete_scan_unknown_id_raises(self) -> None:
        with pytest.raises(ValueError, match="No such scan"):
            self.db.delete_scan(99999)

    def test_find_scans_by_id_and_pattern(self) -> None:
        s1 = self.db.add_source("/2024/jan/a.jpg", width=10, height=10)
        s2 = self.db.add_source("/2024/feb/b.jpg", width=10, height=10)
        s3 = self.db.add_source("/2025/jan/c.jpg", width=10, height=10)
        sc1 = self.db.record_scan(s1, "human")
        sc2 = self.db.record_scan(s2, "human")
        self.db.record_scan(s3, "human")

        # Pattern only — both 2024 sources match.
        rows = self.db.find_scans(source_pattern="*2024*")
        assert {r["id"] for r in rows} == {sc1, sc2}

        # ID only — the single scan.
        rows = self.db.find_scans(scan_id=sc1)
        assert len(rows) == 1 and rows[0]["id"] == sc1
        assert rows[0]["finding_count"] == 0

        # Both filters — intersection.
        rows = self.db.find_scans(scan_id=sc1, source_pattern="*feb*")
        assert rows == []  # sc1 is on /jan/, doesn't match /feb/

        # No filters — everything.
        rows = self.db.find_scans()
        assert len(rows) == 3

    def test_findings_table_has_not_null_scan_id_after_migration(self) -> None:
        info = self.db.conn.execute("PRAGMA table_info(findings)").fetchall()
        scan_col = next(c for c in info if c["name"] == "scan_id")
        assert scan_col["notnull"] == 1, "scan_id must be NOT NULL post-migration"

    def test_backfill_links_legacy_findings_by_embedding_dim(self) -> None:
        """Build a pre-migration DB by hand (no `scan_id` column), then open it via
        FaceDB and assert the migration links each legacy finding to the right scan
        based on embedding dimension."""
        path = self.tmp_path / "legacy.db"
        _seed_legacy_db(
            path,
            sources=[(1, "/legacy.jpg", "photo")],
            scans=[(1, 1, "human"), (2, 1, "pet")],  # both scan types on same source
            findings=[
                # (source_id, dim, frame_path) — scan_id deliberately omitted
                (1, 512, None),  # → human scan
                (1, 768, None),  # → pet scan
            ],
        )

        db = FaceDB(path)  # triggers ALTER + backfill + NOT NULL rebuild
        try:
            findings = db.get_source_findings(1)
            assert {f.embedding_dim or len(f.embedding): f.scan_id for f in findings} == {
                512: 1,
                768: 2,
            }
        finally:
            db.close()

    def test_backfill_creates_synthetic_scan_for_orphan(self) -> None:
        """A pre-migration finding whose scan row no longer exists → synthetic
        `legacy_backfill` scan created and linked to it."""
        path = self.tmp_path / "orphan.db"
        _seed_legacy_db(
            path,
            sources=[(1, "/orphan.jpg", "photo")],
            scans=[],  # no scan rows — finding is fully orphaned
            findings=[(1, 512, None)],
        )

        db = FaceDB(path)
        try:
            scans = db.find_scans()
            assert len(scans) == 1
            assert scans[0]["scan_type"] == "human"
            assert scans[0]["detection_strategy"] == "legacy_backfill"
            assert scans[0]["finding_count"] == 1
            # And the finding got linked to that synthetic scan.
            findings = db.get_source_findings(1)
            assert findings[0].scan_id == scans[0]["id"]
        finally:
            db.close()

    def test_get_subject_sources_with_findings_videos(self) -> None:
        """Filter to videos only; return each video paired with the subject's findings on it.

        Regression guard: previously `SELECT s.*, f.*` collided on `id` and `dict(sqlite3.Row)`
        kept only the first one (source's), so every Finding inherited the source's id. The
        guard below asserts the **exact set** of finding ids per source matches what
        `get_source_findings` returns (which is unambiguous), and uses two findings per video
        so an off-by-one or id-substitution bug is detectable even when ids happen to align.
        """
        sid = self.db.create_subject("Alice")

        # Two video sources, each with TWO findings assigned to Alice. Multiple findings per
        # source means the buggy "all findings inherit source.id" output would collapse to a
        # single finding (set len == 1) instead of the expected 2.
        v1_sid = self.db.add_source(
            "/test/movies/wedding.mp4", source_type="video", width=100, height=100
        )
        v2_sid = self.db.add_source(
            "/test/movies/party.mp4", source_type="video", width=100, height=100
        )
        from ._helpers import add_findings

        add_findings(
            self.db,
            [
                (v1_sid, (0, 0, 10, 10), _make_embedding(), 0.9),
                (v1_sid, (5, 5, 10, 10), _make_embedding(), 0.9),
            ],
            species="human",
            frame_path="tmp/frames/v1.jpg",
        )
        add_findings(
            self.db,
            [
                (v2_sid, (0, 0, 10, 10), _make_embedding(), 0.9),
                (v2_sid, (5, 5, 10, 10), _make_embedding(), 0.9),
            ],
            species="human",
            frame_path="tmp/frames/v2.jpg",
        )
        for f in self.db.get_source_findings(v1_sid) + self.db.get_source_findings(v2_sid):
            self.db.assign_finding_to_subject(f.id, sid)

        # A photo source with the same subject — should be excluded by the type filter.
        photo_sid, photo_fid = _add_source_with_finding(self.db, path="/test/photos/p.jpg")
        self.db.assign_finding_to_subject(photo_fid, sid)

        videos = self.db.get_subject_sources_with_findings(sid, source_type="video")
        assert {src.id for src, _ in videos} == {v1_sid, v2_sid}

        canonical = {
            v1_sid: {f.id for f in self.db.get_source_findings(v1_sid)},
            v2_sid: {f.id for f in self.db.get_source_findings(v2_sid)},
        }
        for src, findings in videos:
            assert src.type == "video"
            # Set equality catches both id collapse (would yield fewer findings) and id mixup
            # (would yield ids not present in the canonical per-source lookup).
            assert {f.id for f in findings} == canonical[src.id]
            for f in findings:
                assert f.source_id == src.id
                assert f.frame_path is not None

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
