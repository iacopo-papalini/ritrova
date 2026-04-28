"""Tests for ritrova.app routes and API endpoints."""

import io
from pathlib import Path
from unittest import TestCase
from zipfile import ZipFile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ritrova.app import create_app
from ritrova.db import FaceDB

from ._helpers import add_findings


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_finding(db: FaceDB, path: str, seed: int = 42, species: str = "human") -> tuple[int, int]:
    """Add source + finding, return (source_id, finding_id)."""
    from ._helpers import add_findings

    pid = db.add_source(path, width=100, height=100)
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    add_findings(db, [(pid, (10, 10, 50, 50), _emb(seed, dim=dim), 0.95)], species=species)
    findings = db.get_source_findings(pid)
    return pid, findings[0].id


def _add_video_finding(
    db: FaceDB, video_path: str, frame_path: str, seed: int = 42, species: str = "human"
) -> tuple[int, int]:
    """Add a video source + finding with a frame_path, return (source_id, finding_id).

    `frame_path` is stored relative to the DB directory (e.g., "tmp/frames/foo.jpg") —
    the caller is responsible for ensuring the file actually exists at db.db_path.parent / frame_path
    before exercising endpoints that read it.
    """
    from ._helpers import add_findings

    sid = db.add_source(video_path, source_type="video", width=100, height=100)
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    add_findings(
        db,
        [(sid, (10, 10, 50, 50), _emb(seed, dim=dim), 0.95)],
        species=species,
        frame_path=frame_path,
    )
    findings = db.get_source_findings(sid)
    return sid, findings[0].id


class TestImageEndpoints(TestCase):
    """Test thumbnail and source image API endpoints."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)
        self.tmp = tmp_path

        # Create a real JPEG image for testing
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="red")
        self.image_path = tmp_path / "test_photo.jpg"
        img.save(str(self.image_path), "JPEG")

    def test_finding_thumbnail_returns_jpeg(self) -> None:
        source_id, finding_id = _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    def test_finding_thumbnail_cached(self) -> None:
        source_id, finding_id = _add_finding(self.db, str(self.image_path), seed=1)
        # First request generates thumbnail
        resp1 = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert resp1.status_code == 200
        # Second request serves from cache
        resp2 = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert resp2.status_code == 200
        assert resp1.content == resp2.content

    def test_finding_thumbnail_404_unknown_finding(self) -> None:
        resp = self.client.get("/api/findings/99999/thumbnail")
        assert resp.status_code == 404

    def test_finding_thumbnail_404_when_finding_is_orphaned(self) -> None:
        """An orphaned finding (source row deleted out from under it) must 404,
        not 500. Previously ``resolve_finding_image`` raised ValueError which
        bubbled to a 500; the new contract returns None so endpoints 404 cleanly."""
        source_id, finding_id = _add_finding(self.db, str(self.image_path), seed=1)
        # Orphan the finding by deleting its source directly.
        self.db.conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
        self.db.conn.commit()

        resp = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=120")
        assert resp.status_code == 404
        # /frame must also 404 cleanly for the same finding.
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 404

    def test_image_endpoints_set_immutable_cache_control(self) -> None:
        """Images are content-addressed by immutable keys (finding_id, source_id);
        the browser must cache them aggressively or every page nav re-fetches
        hundreds of thumbnails."""
        source_id, finding_id = _add_finding(self.db, str(self.image_path), seed=1)
        expected = "public, max-age=31536000, immutable"

        # Thumbnail — hits the cache-generation path on first request...
        r1 = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert r1.headers.get("cache-control") == expected
        # ...and the on-disk-cache fast path on second.
        r2 = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert r2.headers.get("cache-control") == expected

        # Source image.
        assert (
            self.client.get(f"/api/sources/{source_id}/image").headers.get("cache-control")
            == expected
        )
        # Source original download.
        assert (
            self.client.get(f"/api/sources/{source_id}/original").headers.get("cache-control")
            == expected
        )
        # Finding frame.
        assert (
            self.client.get(f"/api/findings/{finding_id}/frame").headers.get("cache-control")
            == expected
        )

    def test_source_image_returns_jpeg(self) -> None:
        source_id, _ = _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/sources/{source_id}/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_source_image_404_unknown(self) -> None:
        resp = self.client.get("/api/sources/99999/image")
        assert resp.status_code == 404

    def test_source_image_for_video_falls_back_to_frame(self) -> None:
        """Video sources have no raw image; the endpoint should serve a representative
        finding's extracted frame so Together-grid thumbnails still render."""
        from PIL import Image

        frame_rel = "tmp/frames/fallback_frame.jpg"
        frame_abs = self.tmp / frame_rel
        frame_abs.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 90), color="purple").save(str(frame_abs), "JPEG")

        source_id, _ = _add_video_finding(self.db, "/some/video.mp4", frame_path=frame_rel, seed=9)
        resp = self.client.get(f"/api/sources/{source_id}/image?max_size=200")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    def test_source_image_for_video_with_no_frames_404s(self) -> None:
        """Edge case: video source with zero frame findings → 404 (no fallback)."""
        sid = self.db.add_source("/empty_vid.mp4", source_type="video", width=100, height=100)
        resp = self.client.get(f"/api/sources/{sid}/image")
        assert resp.status_code == 404

    def test_source_original_returns_file_with_attachment(self) -> None:
        source_id, _ = _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/sources/{source_id}/original")
        assert resp.status_code == 200
        # Content-Disposition forces a download with the original filename.
        disposition = resp.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert self.image_path.name in disposition
        # Body is the raw file bytes, not a re-encoded thumbnail.
        assert resp.content == self.image_path.read_bytes()

    def test_source_original_404_unknown_source(self) -> None:
        resp = self.client.get("/api/sources/99999/original")
        assert resp.status_code == 404

    def test_source_original_404_when_file_missing(self) -> None:
        source_id, _ = _add_finding(self.db, str(self.image_path), seed=1)
        self.image_path.unlink()
        resp = self.client.get(f"/api/sources/{source_id}/original")
        assert resp.status_code == 404

    # ── /api/findings/{id}/frame  (FEAT-14) ───────────────────────────

    def test_finding_frame_returns_jpeg_for_photo_finding(self) -> None:
        """Photo findings: frame endpoint serves the resized source image."""
        _, finding_id = _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    def test_finding_frame_returns_jpeg_for_video_finding(self) -> None:
        """Video findings: frame endpoint serves the extracted frame JPEG."""
        from PIL import Image

        # Create a real JPEG frame at db.db_path.parent/tmp/frames/test_frame.jpg.
        frame_rel = "tmp/frames/test_frame.jpg"
        frame_abs = self.tmp / frame_rel
        frame_abs.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 90), color="green").save(str(frame_abs), "JPEG")

        _, finding_id = _add_video_finding(self.db, "/some/video.mp4", frame_path=frame_rel, seed=2)
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    def test_finding_frame_404_unknown_finding(self) -> None:
        resp = self.client.get("/api/findings/99999/frame")
        assert resp.status_code == 404

    def test_finding_frame_404_when_frame_missing(self) -> None:
        """Video finding with a stored frame_path that no longer exists on disk → 404."""
        frame_rel = "tmp/frames/doomed.jpg"
        # Intentionally do NOT create the file.
        _, finding_id = _add_video_finding(
            self.db, "/some/other_video.mp4", frame_path=frame_rel, seed=3
        )
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 404

    # ── /api/findings/{id}/info  (FEAT-14) ───────────────────────────

    def test_finding_info_photo_returns_source_metadata(self) -> None:
        source_id, finding_id = _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/findings/{finding_id}/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_id"] == source_id
        assert body["type"] == "photo"
        assert body["file_path"].endswith("test_photo.jpg")

    def test_finding_info_video_returns_source_metadata(self) -> None:
        """`file_path` reflects the video's path, NOT the frame jpeg path."""
        source_id, finding_id = _add_video_finding(
            self.db, "/movies/wedding.mp4", frame_path="tmp/frames/x.jpg", seed=4
        )
        resp = self.client.get(f"/api/findings/{finding_id}/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_id"] == source_id
        assert body["type"] == "video"
        assert body["file_path"].endswith("wedding.mp4")
        assert "tmp/frames" not in body["file_path"]

    def test_finding_info_404_unknown(self) -> None:
        resp = self.client.get("/api/findings/99999/info")
        assert resp.status_code == 404


class TestMergeSuggestionsAPI(TestCase):
    """Test merge suggestions API endpoint."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_empty_suggestions(self) -> None:
        resp = self.client.get("/api/merge-suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["suggestions"] == []

    def test_cluster_pair_suggestion(self) -> None:
        """Two unnamed clusters with identical embeddings should be suggested."""
        _, fid1 = _add_finding(self.db, "/a1.jpg", seed=1)
        _, fid2 = _add_finding(self.db, "/a2.jpg", seed=1)
        _, fid3 = _add_finding(self.db, "/b1.jpg", seed=1)
        _, fid4 = _add_finding(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] >= 1
        s = data["suggestions"][0]
        assert s["similarity_pct"] > 90
        # Both are unnamed clusters — names should be None
        assert s["name_a"] is None
        assert s["name_b"] is None

    def test_subject_cluster_suggestion_shows_subject_name(self) -> None:
        """A subject-cluster pair should show the subject's name on the subject side only."""
        sid = self.db.create_subject("Alice")
        _, fid1 = _add_finding(self.db, "/a1.jpg", seed=1)
        self.db.assign_finding_to_subject(fid1, sid)

        _, fid2 = _add_finding(self.db, "/b1.jpg", seed=1)
        _, fid3 = _add_finding(self.db, "/b2.jpg", seed=1)
        self.db.update_cluster_ids({fid2: 10, fid3: 10})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] >= 1
        s = data["suggestions"][0]
        # Exactly one side should have a name
        names = [s["name_a"], s["name_b"]]
        assert "Alice" in names
        assert None in names

    def test_subject_to_subject_excluded(self) -> None:
        """Two named subjects should NOT appear in suggestions."""
        sid_a = self.db.create_subject("Alice")
        _, fid1 = _add_finding(self.db, "/a1.jpg", seed=1)
        self.db.assign_finding_to_subject(fid1, sid_a)

        sid_b = self.db.create_subject("Alice2")
        _, fid2 = _add_finding(self.db, "/b1.jpg", seed=1)
        self.db.assign_finding_to_subject(fid2, sid_b)

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] == 0


class TestClusterDetailPage(TestCase):
    """Cluster detail should orient the user within the unnamed queue."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_cluster_detail_shows_queue_context_and_navigation(self) -> None:
        _, old_a = _add_finding(self.db, "/archive/2023-01/old-a.jpg", seed=1)
        _, old_b = _add_finding(self.db, "/archive/2023-01/old-b.jpg", seed=2)
        _, new_a = _add_finding(self.db, "/archive/2024-01/new-a.jpg", seed=3)
        _, new_b = _add_finding(self.db, "/archive/2024-01/new-b.jpg", seed=4)
        self.db.update_cluster_ids({old_a: 10, old_b: 10, new_a: 20, new_b: 20})

        resp = self.client.get("/clusters/20")

        assert resp.status_code == 200
        assert "1 of 2 unnamed" in resp.text
        assert 'href="/clusters/10"' in resp.text
        assert "Dismiss entire cluster" in resp.text


class TestSearchConsolidation(TestCase):
    """Legacy /search should fold into the scoped subjects filter."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_legacy_search_redirects_to_people_filter(self) -> None:
        resp = self.client.get("/search?q=Alice Smith", follow_redirects=False)

        assert resp.status_code == 303
        assert resp.headers["location"] == "/people?filter=Alice+Smith"

    def test_subjects_page_seeds_filter_from_query_param(self) -> None:
        self.db.create_subject("Alice")

        resp = self.client.get("/people?filter=Ali")

        assert resp.status_code == 200
        assert 'action="/people"' in resp.text
        assert 'name="filter"' in resp.text
        assert 'value="Ali"' in resp.text
        assert 'data-filter-query="Ali"' in resp.text


class TestSubjectsAPI(TestCase):
    """Test /api/subjects/* endpoints used by the typeahead picker."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_all_subjects_empty(self) -> None:
        resp = self.client.get("/api/subjects/all")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_all_subjects_returns_finding_id(self) -> None:
        sid = self.db.create_subject("Alice")
        _, fid = _add_finding(self.db, "/a.jpg", seed=1)
        self.db.assign_finding_to_subject(fid, sid)

        resp = self.client.get("/api/subjects/all")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == sid
        assert data[0]["name"] == "Alice"
        assert data[0]["face_count"] == 1
        assert data[0]["face_id"] == fid

    def test_all_subjects_includes_pets(self) -> None:
        sid_human = self.db.create_subject("Alice", kind="person")
        _, fid_h = _add_finding(self.db, "/a.jpg", seed=1, species="human")
        self.db.assign_finding_to_subject(fid_h, sid_human)

        sid_pet = self.db.create_subject("Figaro", kind="pet")
        _, fid_p = _add_finding(self.db, "/b.jpg__pets", seed=2, species="dog")
        self.db.assign_finding_to_subject(fid_p, sid_pet)

        resp = self.client.get("/api/subjects/all")
        data = resp.json()
        names = {d["name"] for d in data}
        assert names == {"Alice", "Figaro"}

    def test_create_subject(self) -> None:
        resp = self.client.post("/api/subjects/create", json={"name": "Bob"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Bob"
        assert data["id"] > 0
        assert data["face_count"] == 0

    def test_create_subject_returns_existing(self) -> None:
        """Creating with an existing name returns the existing subject."""
        self.db.create_subject("Alice")
        resp = self.client.post("/api/subjects/create", json={"name": "Alice"})
        data = resp.json()
        assert data["name"] == "Alice"

    def test_source_info_includes_gps(self) -> None:
        pid = self.db.add_source(
            "/test.jpg", width=100, height=100, latitude=43.77, longitude=11.25
        )
        resp = self.client.get(f"/api/sources/{pid}/info")
        data = resp.json()
        assert data["latitude"] == pytest.approx(43.77)
        assert data["longitude"] == pytest.approx(11.25)

    def test_source_info_null_gps(self) -> None:
        pid = self.db.add_source("/test.jpg", width=100, height=100)
        resp = self.client.get(f"/api/sources/{pid}/info")
        data = resp.json()
        assert data["latitude"] is None
        assert data["longitude"] is None


class TestTogetherAPI(TestCase):
    """Test /api/together endpoint for multi-subject source search."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_empty_query(self) -> None:
        resp = self.client.get("/api/together")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_finds_source_with_both_subjects(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        # One source with both findings
        source_id = self.db.add_source("/group.jpg", width=100, height=100)
        add_findings(self.db, [(source_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        add_findings(self.db, [(source_id, (50, 50, 30, 30), _emb(2), 0.9)], species="human")
        findings = self.db.get_source_findings(source_id)
        self.db.assign_finding_to_subject(findings[0].id, sid_a)
        self.db.assign_finding_to_subject(findings[1].id, sid_b)

        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        data = resp.json()
        assert data["total"] == 1
        assert data["sources"][0]["id"] == source_id

    def test_excludes_source_with_only_one(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        # Source with only Alice
        source_id = self.db.add_source("/solo.jpg", width=100, height=100)
        add_findings(self.db, [(source_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        findings = self.db.get_source_findings(source_id)
        self.db.assign_finding_to_subject(findings[0].id, sid_a)

        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        assert resp.json()["total"] == 0

    def test_cross_kind_human_and_pet(self) -> None:
        sid_human = self.db.create_subject("Eva", kind="person")
        sid_pet = self.db.create_subject("Figaro", kind="pet")
        source_id = self.db.add_source("/family.jpg", width=100, height=100)
        add_findings(self.db, [(source_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        add_findings(self.db, [(source_id, (50, 50, 30, 30), _emb(2, dim=768), 0.9)], species="dog")
        findings = self.db.get_source_findings(source_id)
        self.db.assign_finding_to_subject(findings[0].id, sid_human)
        self.db.assign_finding_to_subject(findings[1].id, sid_pet)

        resp = self.client.get(f"/api/together?subject_ids={sid_human},{sid_pet}")
        data = resp.json()
        assert data["total"] == 1

    def test_alone_excludes_group_photos(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        sid_c = self.db.create_subject("Charlie")
        # Source with Alice + Bob (alone=true should include)
        s1 = self.db.add_source("/duo.jpg", width=100, height=100)
        add_findings(self.db, [(s1, (10, 10, 30, 30), _emb(10), 0.9)], species="human")
        add_findings(self.db, [(s1, (50, 50, 30, 30), _emb(11), 0.9)], species="human")
        f1 = self.db.get_source_findings(s1)
        self.db.assign_finding_to_subject(f1[0].id, sid_a)
        self.db.assign_finding_to_subject(f1[1].id, sid_b)
        # Source with Alice + Bob + Charlie (alone=true should exclude)
        s2 = self.db.add_source("/trio.jpg", width=100, height=100)
        add_findings(self.db, [(s2, (10, 10, 30, 30), _emb(12), 0.9)], species="human")
        add_findings(self.db, [(s2, (50, 50, 30, 30), _emb(13), 0.9)], species="human")
        add_findings(self.db, [(s2, (80, 80, 30, 30), _emb(14), 0.9)], species="human")
        f2 = self.db.get_source_findings(s2)
        self.db.assign_finding_to_subject(f2[0].id, sid_a)
        self.db.assign_finding_to_subject(f2[1].id, sid_b)
        self.db.assign_finding_to_subject(f2[2].id, sid_c)

        # Without alone: both sources match
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        assert resp.json()["total"] == 2
        # With alone: only the duo
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&alone=true")
        assert resp.json()["total"] == 1

    # ── FEAT-20: source_type filter ─────────────────────────────────

    def _seed_mixed_media(self, sid_a: int, sid_b: int) -> dict[str, int]:
        """Create one photo source and one video source, each containing both subjects."""
        photo_id = self.db.add_source("/vacation.jpg", source_type="photo", width=100, height=100)
        add_findings(self.db, [(photo_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        add_findings(self.db, [(photo_id, (50, 50, 30, 30), _emb(2), 0.9)], species="human")
        for f in self.db.get_source_findings(photo_id):
            self.db.assign_finding_to_subject(f.id, sid_a if f.bbox_x < 30 else sid_b)

        video_id = self.db.add_source("/vacation.mp4", source_type="video", width=100, height=100)
        add_findings(
            self.db,
            [(video_id, (10, 10, 30, 30), _emb(3), 0.9)],
            species="human",
            frame_path="tmp/frames/v1.jpg",
        )
        add_findings(
            self.db,
            [(video_id, (50, 50, 30, 30), _emb(4), 0.9)],
            species="human",
            frame_path="tmp/frames/v2.jpg",
        )
        for f in self.db.get_source_findings(video_id):
            self.db.assign_finding_to_subject(f.id, sid_a if f.bbox_x < 30 else sid_b)

        return {"photo": photo_id, "video": video_id}

    def test_source_type_either_returns_both(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        ids = self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        returned = {s["id"] for s in resp.json()["sources"]}
        assert returned == {ids["photo"], ids["video"]}

    def test_source_type_photo_excludes_videos(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        ids = self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&source_type=photo")
        data = resp.json()
        assert data["total"] == 1
        assert data["sources"][0]["id"] == ids["photo"]

    def test_source_type_video_excludes_photos(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        ids = self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&source_type=video")
        data = resp.json()
        assert data["total"] == 1
        assert data["sources"][0]["id"] == ids["video"]

    def test_source_type_invalid_falls_back_to_either(self) -> None:
        sid_a = self.db.create_subject("Alice")
        sid_b = self.db.create_subject("Bob")
        ids = self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&source_type=bogus")
        assert {s["id"] for s in resp.json()["sources"]} == {ids["photo"], ids["video"]}


class TestBrowsePage(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_browse_page_renders(self) -> None:
        resp = self.client.get("/browse")

        assert resp.status_code == 200
        assert "Path tags" in resp.text

    def test_browse_html_filters_by_path_tag(self) -> None:
        matching = self.db.add_source("2022/2022-10_LuccaComics/IMG_20221031.jpg")
        self.db.add_source("2022/2022-11_Roma/IMG_20221102.jpg")

        resp = self.client.get("/api/browse-html?path_tags=luccacomics")

        assert resp.status_code == 200
        assert f'data-source-id="{matching}"' in resp.text
        assert "1 source found" in resp.text

    def test_browse_html_backfills_missing_path_metadata(self) -> None:
        matching = self.db.add_source("2020/2020-12-29.Libera/20201229_165654.jpg")
        self.db.conn.execute("DELETE FROM source_path_metadata")
        self.db.conn.commit()

        resp = self.client.get("/api/browse-html?path_tags=libera")

        assert resp.status_code == 200
        assert f'data-source-id="{matching}"' in resp.text
        assert self.db.conn.execute("SELECT COUNT(*) FROM source_path_metadata").fetchone()[0] == 1

    def test_browse_html_combines_subject_and_path_tag(self) -> None:
        alice = self.db.create_subject("Alice")
        matching_source, matching_finding = _add_finding(
            self.db, "2023/2023-08_Amsterdam/IMG_20230801.jpg"
        )
        _, other_finding = _add_finding(self.db, "2023/2023-08_Roma/IMG_20230802.jpg")
        self.db.assign_finding_to_subject(matching_finding, alice)
        self.db.assign_finding_to_subject(other_finding, alice)

        resp = self.client.get(f"/api/browse-html?subject_ids={alice}&path_tags=amsterdam")

        assert resp.status_code == 200
        assert f'data-source-id="{matching_source}"' in resp.text
        assert "1 source found" in resp.text


class TestPrintSelectionPage(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db", base_dir=tmp_path)
        self.app = create_app(str(tmp_path / "test.db"), photos_dir=str(tmp_path))
        self.client = TestClient(self.app)
        self.tmp = tmp_path
        (tmp_path / "prints").mkdir()
        self.first_path = tmp_path / "prints" / "z first.jpg"
        self.second_path = tmp_path / "prints" / "a_second.jpg"
        self.first_path.write_bytes(b"first")
        self.second_path.write_bytes(b"second")

    def test_add_remove_and_list_print_selection(self) -> None:
        first = self.db.add_source("prints/z first.jpg")
        second = self.db.add_source("prints/a_second.jpg")

        resp = self.client.post(f"/api/print-selection/{first}")
        assert resp.status_code == 200
        assert resp.json()["source_ids"] == [first]
        assert "undo_token" not in resp.json()

        resp = self.client.post(f"/api/print-selection/{second}")
        assert resp.status_code == 200
        assert resp.json()["source_ids"] == [first, second]
        assert "undo_token" not in resp.json()

        resp = self.client.delete(f"/api/print-selection/{first}")
        assert resp.status_code == 200
        assert resp.json()["source_ids"] == [second]
        assert "undo_token" not in resp.json()

    def test_print_selection_view_removal_and_clear_are_undoable(self) -> None:
        first = self.db.add_source("prints/z first.jpg")
        second = self.db.add_source("prints/a_second.jpg")

        self.db.add_to_print_selection(first)
        self.db.add_to_print_selection(second)
        remove_resp = self.client.delete(f"/api/print-selection/{first}?undo=true")
        assert remove_resp.status_code == 200
        assert self.db.get_print_selection_ids() == [second]
        undo_resp = self.client.post(f"/api/undo/{remove_resp.json()['undo_token']}")
        assert undo_resp.status_code == 200
        assert self.db.get_print_selection_ids() == [first, second]

        clear_resp = self.client.post("/api/print-selection/clear")
        assert clear_resp.status_code == 200
        assert self.db.get_print_selection_ids() == []
        undo_resp = self.client.post(f"/api/undo/{clear_resp.json()['undo_token']}")
        assert undo_resp.status_code == 200
        assert self.db.get_print_selection_ids() == [first, second]

    def test_print_selection_rejects_video_sources(self) -> None:
        video = self.db.add_source("prints/video.mp4", source_type="video")

        resp = self.client.post(f"/api/print-selection/{video}")

        assert resp.status_code == 400
        assert "Only photos" in resp.json()["detail"]

    def test_export_print_selection_zip_preserves_order_and_original_bytes(self) -> None:
        first = self.db.add_source("prints/z first.jpg")
        second = self.db.add_source("prints/a_second.jpg")
        self.db.add_to_print_selection(first)
        self.db.add_to_print_selection(second)

        resp = self.client.post("/api/print-selection/export")

        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "attachment" in resp.headers["content-disposition"]
        with ZipFile(io.BytesIO(resp.content)) as archive:
            assert archive.namelist() == ["0001_z_first.jpg", "0002_a_second.jpg"]
            assert archive.read("0001_z_first.jpg") == b"first"
            assert archive.read("0002_a_second.jpg") == b"second"

    def test_export_empty_print_selection_400(self) -> None:
        resp = self.client.post("/api/print-selection/export")

        assert resp.status_code == 400


class TestNamespaceCollision(TestCase):
    """Verify subject IDs and cluster IDs don't collide in merge suggestions."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_cluster_id_matching_subject_id_not_mislabeled(self) -> None:
        """If cluster_id == subject_id, the cluster should NOT show the subject's name."""
        # Create subject with id=1
        sid = self.db.create_subject("Alice")
        _, fid_alice = _add_finding(self.db, "/alice.jpg", seed=1)
        self.db.assign_finding_to_subject(fid_alice, sid)

        # Create unnamed cluster with cluster_id=1 (same as subject id!)
        # Use different seed so it's a different embedding from Alice
        _, fid_c1 = _add_finding(self.db, "/c1.jpg", seed=99)
        _, fid_c2 = _add_finding(self.db, "/c2.jpg", seed=99)
        self.db.update_cluster_ids({fid_c1: sid, fid_c2: sid})  # cluster_id == subject_id

        # Create another unnamed cluster to get a suggestion pair
        _, fid_c3 = _add_finding(self.db, "/c3.jpg", seed=99)
        _, fid_c4 = _add_finding(self.db, "/c4.jpg", seed=99)
        self.db.update_cluster_ids({fid_c3: 999, fid_c4: 999})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()

        # Find the cluster-cluster suggestion (not the subject-cluster one)
        for s in data["suggestions"]:
            ids = {s["cluster_a"], s["cluster_b"]}
            if ids == {sid, 999}:
                # cluster_id=1 should NOT have name "Alice"
                if s["cluster_a"] == sid:
                    assert s["name_a"] is None, (
                        f"Cluster {sid} incorrectly labeled as '{s['name_a']}' "
                        f"due to subject-id/cluster-id collision"
                    )
                if s["cluster_b"] == sid:
                    assert s["name_b"] is None, (
                        f"Cluster {sid} incorrectly labeled as '{s['name_b']}' "
                        f"due to subject-id/cluster-id collision"
                    )
                break
        else:
            pytest.fail("Expected cluster-cluster suggestion not found")
