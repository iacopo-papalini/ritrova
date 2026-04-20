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
        assert finding.subject_id is None
        assert finding.cluster_id is None

    def test_unassign_finding(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        subject_id = self.db.create_subject("Test")
        self.db.assign_finding_to_subject(fid, subject_id)
        finding = self.db.get_finding(fid)
        assert finding is not None
        assert finding.subject_id == subject_id

        self.db.unassign_finding(fid)
        finding = self.db.get_finding(fid)
        assert finding is not None
        assert finding.subject_id is None

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
        assert finding.subject_id is None

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


class TestCircles(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_db(self, db: FaceDB) -> None:
        self.db = db

    def test_create_circle_idempotent(self) -> None:
        cid1 = self.db.create_circle("family")
        cid2 = self.db.create_circle("family")
        assert cid1 == cid2

    def test_create_circle_rejects_blank(self) -> None:
        with pytest.raises(ValueError):
            self.db.create_circle("   ")

    def test_list_circles_returns_user_created_only(self) -> None:
        self.db.create_circle("family")
        names = [c.name for c in self.db.list_circles()]
        # No auto-seeded circles — the user manages the set.
        assert names == ["family"]

    def test_membership_add_remove_roundtrip(self) -> None:
        sid = self.db.create_subject("Alice")
        cid = self.db.create_circle("family", description="nuclear only")
        assert self.db.add_subject_to_circle(sid, cid) is True
        # second add is a no-op
        assert self.db.add_subject_to_circle(sid, cid) is False
        circles = self.db.get_subject_circles(sid)
        assert [c.name for c in circles] == ["family"]
        assert self.db.remove_subject_from_circle(sid, cid) is True
        assert self.db.get_subject_circles(sid) == []

    def test_member_count_reflects_state(self) -> None:
        cid = self.db.create_circle("friends")
        self.db.add_subject_to_circle(self.db.create_subject("A"), cid)
        self.db.add_subject_to_circle(self.db.create_subject("B"), cid)
        assert self.db.get_circle(cid).member_count == 2

    def test_delete_subject_cascades_to_memberships(self) -> None:
        sid = self.db.create_subject("Alice")
        cid = self.db.create_circle("family")
        self.db.add_subject_to_circle(sid, cid)
        self.db.delete_subject(sid)
        assert self.db.get_circle(cid).member_count == 0

    def test_delete_circle_cascades_to_memberships(self) -> None:
        sid = self.db.create_subject("Alice")
        cid = self.db.create_circle("acquaintances")
        self.db.add_subject_to_circle(sid, cid)
        self.db.delete_circle(cid)
        assert self.db.get_subject_circles(sid) == []

    def test_subjects_in_any_circle(self) -> None:
        fam = self.db.create_circle("family")
        acq = self.db.create_circle("acquaintances")
        a, b, c = (
            self.db.create_subject("A"),
            self.db.create_subject("B"),
            self.db.create_subject("C"),
        )
        self.db.add_subject_to_circle(a, fam)
        self.db.add_subject_to_circle(b, acq)
        assert self.db.subjects_in_any_circle([fam, acq]) == {a, b}
        assert self.db.subjects_in_any_circle([fam]) == {a}
        assert self.db.subjects_in_any_circle([]) == set()
        assert c not in self.db.subjects_in_any_circle([fam, acq])

    def test_rename_circle(self) -> None:
        cid = self.db.create_circle("frineds")  # typo
        self.db.rename_circle(cid, "friends")
        assert self.db.get_circle(cid).name == "friends"


def _build_pre_refactor_db(
    path: Path,
    *,
    subject_ids: dict[int, int] | None = None,
    cluster_ids: dict[int, int] | None = None,
    dismissed_ids: list[int] | None = None,
    strangers_circle_member_ids: list[int] | None = None,
) -> None:
    """Build a DB with the pre-Apr-2026 schema (findings has person_id /
    cluster_id columns; separate dismissed_findings table) plus the given
    state. Opening it with FaceDB triggers the migration and drops the
    old columns."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL DEFAULT 'photo',
            width INTEGER NOT NULL DEFAULT 0,
            height INTEGER NOT NULL DEFAULT 0,
            taken_at TEXT, latitude REAL, longitude REAL
        );
        CREATE TABLE scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            scan_type TEXT NOT NULL,
            scanned_at TEXT NOT NULL DEFAULT '2024-01-01',
            detection_strategy TEXT NOT NULL DEFAULT 'unknown',
            UNIQUE (source_id, scan_type)
        );
        CREATE TABLE subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
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
            embedding_dim INTEGER NOT NULL DEFAULT 0,
            scan_id INTEGER REFERENCES scans(id) ON DELETE CASCADE,
            frame_number INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE dismissed_findings (
            finding_id INTEGER PRIMARY KEY REFERENCES findings(id) ON DELETE CASCADE
        );
        CREATE TABLE circles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            created_at TEXT NOT NULL
        );
        CREATE TABLE subject_circles (
            subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
            circle_id INTEGER NOT NULL REFERENCES circles(id) ON DELETE CASCADE,
            added_at TEXT NOT NULL,
            PRIMARY KEY (subject_id, circle_id)
        );
        CREATE UNIQUE INDEX idx_subjects_name_kind ON subjects(name, kind);
        """
    )
    # Seed a minimal source + scan + findings so subject_ids/cluster_ids have targets.
    conn.execute("INSERT INTO sources (id, file_path) VALUES (1, '/migration-test.jpg')")
    conn.execute("INSERT INTO scans (id, source_id, scan_type) VALUES (1, 1, 'subjects')")
    emb = np.zeros(512, dtype=np.float32)
    emb[0] = 1.0
    # Collect all finding ids referenced by the seeds.
    all_fids: set[int] = set()
    if subject_ids:
        all_fids.update(subject_ids.keys())
    if cluster_ids:
        all_fids.update(cluster_ids.keys())
    if dismissed_ids:
        all_fids.update(dismissed_ids)
    for fid in sorted(all_fids):
        conn.execute(
            "INSERT INTO findings (id, source_id, embedding, scan_id) VALUES (?, 1, ?, 1)",
            (fid, emb.tobytes()),
        )
    if subject_ids:
        # Create subjects referenced by the id map.
        for subject_id in set(subject_ids.values()):
            conn.execute(
                "INSERT INTO subjects (id, name, created_at) VALUES (?, ?, '2024-01-01')",
                (subject_id, f"Subject {subject_id}"),
            )
        for fid, sid in subject_ids.items():
            # legacy pre-refactor column — intentional; tests the migration path.
            conn.execute("UPDATE findings SET person_id = ? WHERE id = ?", (sid, fid))
    if cluster_ids:
        for fid, cid in cluster_ids.items():
            conn.execute("UPDATE findings SET cluster_id = ? WHERE id = ?", (cid, fid))
    if dismissed_ids:
        for fid in dismissed_ids:
            conn.execute("INSERT INTO dismissed_findings (finding_id) VALUES (?)", (fid,))
    if strangers_circle_member_ids:
        conn.execute(
            "INSERT INTO circles (id, name, created_at) VALUES (1, 'Strangers', '2024-01-01')"
        )
        for sid in strangers_circle_member_ids:
            conn.execute(
                "INSERT INTO subject_circles (subject_id, circle_id, added_at) "
                "VALUES (?, 1, '2024-01-01')",
                (sid,),
            )
    conn.commit()
    conn.close()


class TestFindingAssignmentMigration(TestCase):
    """Apr 2026 refactor: open a DB with the pre-refactor schema, verify
    FaceDB's migration copies state into the new tables and drops the
    obsolete columns + dismissed_findings table."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    def test_clusters_migrated(self) -> None:
        path = self.tmp_path / "old.db"
        _build_pre_refactor_db(path, cluster_ids={10: 7, 11: 7})
        db = FaceDB(path)
        rows = db.conn.execute(
            "SELECT finding_id, cluster_id FROM cluster_findings ORDER BY finding_id"
        ).fetchall()
        assert [(r[0], r[1]) for r in rows] == [(10, 7), (11, 7)]
        db.close()

    def test_named_subjects_migrated(self) -> None:
        path = self.tmp_path / "old.db"
        _build_pre_refactor_db(path, subject_ids={10: 100})
        db = FaceDB(path)
        row = db.conn.execute(
            "SELECT subject_id, exclusion_reason FROM finding_assignment WHERE finding_id = ?",
            (10,),
        ).fetchone()
        assert (row[0], row[1]) == (100, None)
        db.close()

    def test_dismissed_findings_migrated(self) -> None:
        path = self.tmp_path / "old.db"
        _build_pre_refactor_db(path, dismissed_ids=[10])
        db = FaceDB(path)
        row = db.conn.execute(
            "SELECT subject_id, exclusion_reason FROM finding_assignment WHERE finding_id = ?",
            (10,),
        ).fetchone()
        assert (row[0], row[1]) == (None, "not_a_face")
        db.close()

    def test_strangers_dissolved_to_exclusion_reason(self) -> None:
        path = self.tmp_path / "old.db"
        _build_pre_refactor_db(
            path,
            subject_ids={10: 100},
            strangers_circle_member_ids=[100],
        )
        db = FaceDB(path)
        assert db.get_subject(100) is None
        assert db.get_circle_by_name("Strangers") is None
        row = db.conn.execute(
            "SELECT subject_id, exclusion_reason FROM finding_assignment WHERE finding_id = ?",
            (10,),
        ).fetchone()
        assert (row[0], row[1]) == (None, "stranger")
        db.close()

    def test_migration_is_idempotent(self) -> None:
        """Reopening a migrated DB doesn't duplicate rows."""
        path = self.tmp_path / "old.db"
        _build_pre_refactor_db(path, subject_ids={10: 100}, cluster_ids={10: 7})
        db1 = FaceDB(path)
        db1.close()
        db2 = FaceDB(path)
        assert db2.conn.execute("SELECT COUNT(*) FROM finding_assignment").fetchone()[0] == 1
        assert db2.conn.execute("SELECT COUNT(*) FROM cluster_findings").fetchone()[0] == 1
        db2.close()

    def test_old_columns_dropped_after_migration(self) -> None:
        """After opening, findings.person_id / cluster_id and
        dismissed_findings are gone."""
        path = self.tmp_path / "old.db"
        _build_pre_refactor_db(path, subject_ids={10: 100})
        db = FaceDB(path)
        cols = {r[1] for r in db.conn.execute("PRAGMA table_info(findings)").fetchall()}
        assert "person_id" not in cols
        assert "cluster_id" not in cols
        row = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='dismissed_findings'"
        ).fetchone()
        assert row is None
        db.close()


class TestFindingAssignmentConstraints(TestCase):
    """XOR CHECK + enum CHECK on finding_assignment."""

    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.db = db

    def test_check_constraint_blocks_both_null(self) -> None:
        _, fid = _add_source_with_finding(self.db, path="/a.jpg")
        with pytest.raises(sqlite3.IntegrityError):
            self.db.conn.execute(
                "INSERT INTO finding_assignment(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, NULL, NULL, '2026-04-19')",
                (fid,),
            )

    def test_check_constraint_blocks_both_set(self) -> None:
        _, fid = _add_source_with_finding(self.db, path="/a.jpg")
        sid = self.db.create_subject("Alice")
        with pytest.raises(sqlite3.IntegrityError):
            self.db.conn.execute(
                "INSERT INTO finding_assignment(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, ?, 'stranger', '2026-04-19')",
                (fid, sid),
            )

    def test_check_constraint_blocks_bad_reason(self) -> None:
        _, fid = _add_source_with_finding(self.db, path="/a.jpg")
        with pytest.raises(sqlite3.IntegrityError):
            self.db.conn.execute(
                "INSERT INTO finding_assignment(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, NULL, 'bogus', '2026-04-19')",
                (fid,),
            )


class TestAssignmentMixin(TestCase):
    """New helpers for reading/writing finding_assignment and cluster_findings
    (the target API after the Apr 2026 refactor; existing call sites still
    use the old denormalized columns — they'll be switched in Commit C)."""

    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.db = db

    # finding_assignment

    def test_set_subject_then_get_curation(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        sid = self.db.create_subject("Alice")
        self.db.set_subject(fid, sid)
        assert self.db.get_curation(fid) == (sid, None)

    def test_set_exclusion_stranger(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        self.db.set_exclusion(fid, "stranger")
        assert self.db.get_curation(fid) == (None, "stranger")

    def test_set_exclusion_rejects_bad_reason(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        with pytest.raises(ValueError):
            self.db.set_exclusion(fid, "bogus")

    def test_set_subject_overwrites_exclusion(self) -> None:
        """Picking a name on a previously-marked stranger implicitly unmarks."""
        _, fid = _add_source_with_finding(self.db)
        sid = self.db.create_subject("Alice")
        self.db.set_exclusion(fid, "stranger")
        self.db.set_subject(fid, sid)
        assert self.db.get_curation(fid) == (sid, None)

    def test_set_exclusion_overwrites_subject(self) -> None:
        """Marking as stranger clears a prior assignment."""
        _, fid = _add_source_with_finding(self.db)
        sid = self.db.create_subject("Alice")
        self.db.set_subject(fid, sid)
        self.db.set_exclusion(fid, "stranger")
        assert self.db.get_curation(fid) == (None, "stranger")

    def test_get_curation_returns_none_when_uncurated(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        assert self.db.get_curation(fid) is None

    def test_clear_curation(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        sid = self.db.create_subject("Alice")
        self.db.set_subject(fid, sid)
        self.db.clear_curation(fid)
        assert self.db.get_curation(fid) is None

    def test_clear_curations_batch(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.set_exclusion(fid1, "stranger")
        self.db.set_exclusion(fid2, "not_a_face")
        self.db.clear_curations([fid1, fid2])
        assert self.db.get_curation(fid1) is None
        assert self.db.get_curation(fid2) is None

    def test_set_exclusions_batch(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.set_exclusions([fid1, fid2], "stranger")
        assert self.db.get_curation(fid1) == (None, "stranger")
        assert self.db.get_curation(fid2) == (None, "stranger")

    # cluster_findings

    def test_set_and_get_cluster_membership(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        self.db.set_cluster_memberships({fid: 42})
        assert self.db.get_cluster_membership(fid) == 42

    def test_set_cluster_memberships_bulk(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.set_cluster_memberships({fid1: 7, fid2: 9})
        assert self.db.get_cluster_membership(fid1) == 7
        assert self.db.get_cluster_membership(fid2) == 9

    def test_remove_cluster_memberships(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        self.db.set_cluster_memberships({fid: 42})
        self.db.remove_cluster_memberships([fid])
        assert self.db.get_cluster_membership(fid) is None

    def test_clear_species_cluster_memberships(self) -> None:
        """Clearing dogs leaves humans alone."""
        _, human_fid = _add_source_with_finding(self.db, path="/h.jpg", species="human")
        _, dog_fid = _add_source_with_finding(self.db, path="/d.jpg", species="dog")
        self.db.set_cluster_memberships({human_fid: 1, dog_fid: 2})
        self.db.clear_species_cluster_memberships("dog")
        assert self.db.get_cluster_membership(human_fid) == 1
        assert self.db.get_cluster_membership(dog_fid) is None

    def test_merge_cluster_memberships(self) -> None:
        _, fid1 = _add_source_with_finding(self.db, path="/a.jpg")
        _, fid2 = _add_source_with_finding(self.db, path="/b.jpg")
        self.db.set_cluster_memberships({fid1: 1, fid2: 1})
        moved = self.db.merge_cluster_memberships(source_cluster=1, target_cluster=5)
        assert moved == 2
        assert self.db.get_cluster_membership(fid1) == 5
        assert self.db.get_cluster_membership(fid2) == 5

    def test_merge_same_cluster_is_noop(self) -> None:
        _, fid = _add_source_with_finding(self.db)
        self.db.set_cluster_memberships({fid: 1})
        assert self.db.merge_cluster_memberships(1, 1) == 0
