"""Tests for ritrova.app routes and API endpoints."""

import io
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
import pytest

from ritrova.db import FaceDB

from ._helpers import MISSING_ID, add_findings
from ._postgres import PostgresAppTestCase


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


async def _add_finding(
    db: FaceDB, path: str, seed: int = 42, species: str = "human"
) -> tuple[int, int]:
    """Add source + finding, return (source_id, finding_id)."""
    from ._helpers import add_findings

    pid = await db.add_source(path, width=100, height=100)
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    await add_findings(db, [(pid, (10, 10, 50, 50), _emb(seed, dim=dim), 0.95)], species=species)
    findings = await db.get_source_findings(pid)
    return pid, findings[0].id


async def _add_video_finding(
    db: FaceDB, video_path: str, frame_path: str, seed: int = 42, species: str = "human"
) -> tuple[int, int]:
    """Add a video source + finding with a frame_path, return (source_id, finding_id).

    `frame_path` is stored relative to the DB directory (e.g., "tmp/frames/foo.jpg") —
    the caller is responsible for ensuring the file actually exists at db.db_path.parent / frame_path
    before exercising endpoints that read it.
    """
    from ._helpers import add_findings

    sid = await db.add_source(video_path, source_type="video", width=100, height=100)
    dim = 768 if species in FaceDB.PET_SPECIES else 512
    await add_findings(
        db,
        [(sid, (10, 10, 50, 50), _emb(seed, dim=dim), 0.95)],
        species=species,
        frame_path=frame_path,
    )
    findings = await db.get_source_findings(sid)
    return sid, findings[0].id


class TestImageEndpoints(PostgresAppTestCase):
    """Test thumbnail and source image API endpoints."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()
        self.tmp = self.tmp_path

        # Create a real JPEG image for testing
        from PIL import Image

        img = Image.new("RGB", (200, 200), color="red")
        self.image_path = self.tmp_path / "test_photo.jpg"
        img.save(str(self.image_path), "JPEG")

    async def test_finding_thumbnail_returns_jpeg(self) -> None:
        source_id, finding_id = await _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    async def test_finding_thumbnail_cached(self) -> None:
        source_id, finding_id = await _add_finding(self.db, str(self.image_path), seed=1)
        # First request generates thumbnail
        resp1 = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert resp1.status_code == 200
        # Second request serves from cache
        resp2 = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=100")
        assert resp2.status_code == 200
        assert resp1.content == resp2.content

    def test_finding_thumbnail_404_unknown_finding(self) -> None:
        resp = self.client.get(f"/api/findings/{MISSING_ID}/thumbnail")
        assert resp.status_code == 404

    async def test_finding_thumbnail_404_when_finding_is_orphaned(self) -> None:
        """An orphaned finding (source row deleted out from under it) must 404,
        not 500. Previously ``resolve_finding_image`` raised ValueError which
        bubbled to a 500; the new contract returns None so endpoints 404 cleanly."""
        source_id, finding_id = await _add_finding(self.db, str(self.image_path), seed=1)
        # Orphan the finding by deleting its source directly.
        await self.db.execute("DELETE FROM sources WHERE id = ?", source_id)

        resp = self.client.get(f"/api/findings/{finding_id}/thumbnail?size=120")
        assert resp.status_code == 404
        # /frame must also 404 cleanly for the same finding.
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 404

    async def test_image_endpoints_set_immutable_cache_control(self) -> None:
        """Images are content-addressed by immutable keys (finding_id, source_id);
        the browser must cache them aggressively or every page nav re-fetches
        hundreds of thumbnails."""
        source_id, finding_id = await _add_finding(self.db, str(self.image_path), seed=1)
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

    async def test_source_image_returns_jpeg(self) -> None:
        source_id, _ = await _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/sources/{source_id}/image")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"

    def test_source_image_404_unknown(self) -> None:
        resp = self.client.get(f"/api/sources/{MISSING_ID}/image")
        assert resp.status_code == 404

    async def test_source_image_for_video_falls_back_to_frame(self) -> None:
        """Video sources have no raw image; the endpoint should serve a representative
        finding's extracted frame so Together-grid thumbnails still render."""
        from PIL import Image

        frame_rel = "tmp/frames/fallback_frame.jpg"
        frame_abs = self.tmp / frame_rel
        frame_abs.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 90), color="purple").save(str(frame_abs), "JPEG")

        source_id, _ = await _add_video_finding(
            self.db, "/some/video.mp4", frame_path=frame_rel, seed=9
        )
        resp = self.client.get(f"/api/sources/{source_id}/image?max_size=200")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    async def test_source_image_for_video_with_no_frames_404s(self) -> None:
        """Edge case: video source with zero frame findings → 404 (no fallback)."""
        sid = await self.db.add_source("/empty_vid.mp4", source_type="video", width=100, height=100)
        resp = self.client.get(f"/api/sources/{sid}/image")
        assert resp.status_code == 404

    async def test_source_original_returns_file_with_attachment(self) -> None:
        source_id, _ = await _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/sources/{source_id}/original")
        assert resp.status_code == 200
        # Content-Disposition forces a download with the original filename.
        disposition = resp.headers.get("content-disposition", "")
        assert "attachment" in disposition
        assert self.image_path.name in disposition
        # Body is the raw file bytes, not a re-encoded thumbnail.
        assert resp.content == self.image_path.read_bytes()

    def test_source_original_404_unknown_source(self) -> None:
        resp = self.client.get(f"/api/sources/{MISSING_ID}/original")
        assert resp.status_code == 404

    async def test_source_original_404_when_file_missing(self) -> None:
        source_id, _ = await _add_finding(self.db, str(self.image_path), seed=1)
        self.image_path.unlink()
        resp = self.client.get(f"/api/sources/{source_id}/original")
        assert resp.status_code == 404

    # ── /api/findings/{id}/frame  (FEAT-14) ───────────────────────────

    async def test_finding_frame_returns_jpeg_for_photo_finding(self) -> None:
        """Photo findings: frame endpoint serves the resized source image."""
        _, finding_id = await _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    async def test_finding_frame_returns_jpeg_for_video_finding(self) -> None:
        """Video findings: frame endpoint serves the extracted frame JPEG."""
        from PIL import Image

        # Create a real JPEG frame at db.db_path.parent/tmp/frames/test_frame.jpg.
        frame_rel = "tmp/frames/test_frame.jpg"
        frame_abs = self.tmp / frame_rel
        frame_abs.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 90), color="green").save(str(frame_abs), "JPEG")

        _, finding_id = await _add_video_finding(
            self.db, "/some/video.mp4", frame_path=frame_rel, seed=2
        )
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        assert len(resp.content) > 0

    def test_finding_frame_404_unknown_finding(self) -> None:
        resp = self.client.get(f"/api/findings/{MISSING_ID}/frame")
        assert resp.status_code == 404

    async def test_finding_frame_404_when_frame_missing(self) -> None:
        """Video finding with a stored frame_path that no longer exists on disk → 404."""
        frame_rel = "tmp/frames/doomed.jpg"
        # Intentionally do NOT create the file.
        _, finding_id = await _add_video_finding(
            self.db, "/some/other_video.mp4", frame_path=frame_rel, seed=3
        )
        resp = self.client.get(f"/api/findings/{finding_id}/frame")
        assert resp.status_code == 404

    # ── /api/findings/{id}/info  (FEAT-14) ───────────────────────────

    async def test_finding_info_photo_returns_source_metadata(self) -> None:
        source_id, finding_id = await _add_finding(self.db, str(self.image_path), seed=1)
        resp = self.client.get(f"/api/findings/{finding_id}/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_id"] == str(source_id)
        assert body["type"] == "photo"
        assert body["file_path"].endswith("test_photo.jpg")

    async def test_finding_info_video_returns_source_metadata(self) -> None:
        """`file_path` reflects the video's path, NOT the frame jpeg path."""
        source_id, finding_id = await _add_video_finding(
            self.db, "/movies/wedding.mp4", frame_path="tmp/frames/x.jpg", seed=4
        )
        resp = self.client.get(f"/api/findings/{finding_id}/info")
        assert resp.status_code == 200
        body = resp.json()
        assert body["source_id"] == str(source_id)
        assert body["type"] == "video"
        assert body["file_path"].endswith("wedding.mp4")
        assert "tmp/frames" not in body["file_path"]

    def test_finding_info_404_unknown(self) -> None:
        resp = self.client.get(f"/api/findings/{MISSING_ID}/info")
        assert resp.status_code == 404


class TestMergeSuggestionsAPI(PostgresAppTestCase):
    """Test merge suggestions API endpoint."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    def test_empty_suggestions(self) -> None:
        resp = self.client.get("/api/merge-suggestions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["suggestions"] == []

    async def test_cluster_pair_suggestion(self) -> None:
        """Two unnamed clusters with identical embeddings should be suggested."""
        _, fid1 = await _add_finding(self.db, "/a1.jpg", seed=1)
        _, fid2 = await _add_finding(self.db, "/a2.jpg", seed=1)
        _, fid3 = await _add_finding(self.db, "/b1.jpg", seed=1)
        _, fid4 = await _add_finding(self.db, "/b2.jpg", seed=1)
        await self.db.update_cluster_ids({fid1: 10, fid2: 10, fid3: 20, fid4: 20})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] >= 1
        s = data["suggestions"][0]
        assert s["similarity_pct"] > 90
        # Both are unnamed clusters — names should be None
        assert s["name_a"] is None
        assert s["name_b"] is None

    async def test_subject_cluster_suggestion_shows_subject_name(self) -> None:
        """A subject-cluster pair should show the subject's name on the subject side only."""
        sid = await self.db.create_subject("Alice")
        _, fid1 = await _add_finding(self.db, "/a1.jpg", seed=1)
        await self.db.assign_finding_to_subject(fid1, sid)

        _, fid2 = await _add_finding(self.db, "/b1.jpg", seed=1)
        _, fid3 = await _add_finding(self.db, "/b2.jpg", seed=1)
        await self.db.update_cluster_ids({fid2: 10, fid3: 10})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] >= 1
        s = data["suggestions"][0]
        # Exactly one side should have a name
        names = [s["name_a"], s["name_b"]]
        assert "Alice" in names
        assert None in names

    async def test_subject_to_subject_excluded(self) -> None:
        """Two named subjects should NOT appear in suggestions."""
        sid_a = await self.db.create_subject("Alice")
        _, fid1 = await _add_finding(self.db, "/a1.jpg", seed=1)
        await self.db.assign_finding_to_subject(fid1, sid_a)

        sid_b = await self.db.create_subject("Alice2")
        _, fid2 = await _add_finding(self.db, "/b1.jpg", seed=1)
        await self.db.assign_finding_to_subject(fid2, sid_b)

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()
        assert data["total"] == 0


class TestClusterDetailPage(PostgresAppTestCase):
    """Cluster detail should orient the user within the unnamed queue."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    async def test_cluster_detail_shows_queue_context_and_navigation(self) -> None:
        _, old_a = await _add_finding(self.db, "/archive/2023-01/old-a.jpg", seed=1)
        _, old_b = await _add_finding(self.db, "/archive/2023-01/old-b.jpg", seed=2)
        _, new_a = await _add_finding(self.db, "/archive/2024-01/new-a.jpg", seed=3)
        _, new_b = await _add_finding(self.db, "/archive/2024-01/new-b.jpg", seed=4)
        await self.db.update_cluster_ids({old_a: 10, old_b: 10, new_a: 20, new_b: 20})

        resp = self.client.get("/clusters/20")

        assert resp.status_code == 200
        assert "1 of 2 unnamed" in resp.text
        assert 'href="/clusters/10"' in resp.text
        assert "Dismiss entire cluster" in resp.text


class TestSearchConsolidation(PostgresAppTestCase):
    """Legacy /search should fold into the scoped subjects filter."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    def test_legacy_search_redirects_to_people_filter(self) -> None:
        resp = self.client.get("/search?q=Alice Smith", follow_redirects=False)

        assert resp.status_code == 303
        assert resp.headers["location"] == "/people?filter=Alice+Smith"

    async def test_subjects_page_seeds_filter_from_query_param(self) -> None:
        await self.db.create_subject("Alice")

        resp = self.client.get("/people?filter=Ali")

        assert resp.status_code == 200
        assert 'action="/people"' in resp.text
        assert 'name="filter"' in resp.text
        assert 'value="Ali"' in resp.text
        assert 'data-filter-query="Ali"' in resp.text


class TestSubjectsAPI(PostgresAppTestCase):
    """Test /api/subjects/* endpoints used by the typeahead picker."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    def test_all_subjects_empty(self) -> None:
        resp = self.client.get("/api/subjects/all")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_all_subjects_returns_finding_id(self) -> None:
        sid = await self.db.create_subject("Alice")
        _, fid = await _add_finding(self.db, "/a.jpg", seed=1)
        await self.db.assign_finding_to_subject(fid, sid)

        resp = self.client.get("/api/subjects/all")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["id"] == str(sid)
        assert data[0]["name"] == "Alice"
        assert data[0]["face_count"] == 1
        assert data[0]["face_id"] == str(fid)

    async def test_all_subjects_includes_pets(self) -> None:
        sid_human = await self.db.create_subject("Alice", kind="person")
        _, fid_h = await _add_finding(self.db, "/a.jpg", seed=1, species="human")
        await self.db.assign_finding_to_subject(fid_h, sid_human)

        sid_pet = await self.db.create_subject("Figaro", kind="pet")
        _, fid_p = await _add_finding(self.db, "/b.jpg__pets", seed=2, species="dog")
        await self.db.assign_finding_to_subject(fid_p, sid_pet)

        resp = self.client.get("/api/subjects/all")
        data = resp.json()
        names = {d["name"] for d in data}
        assert names == {"Alice", "Figaro"}

    def test_create_subject(self) -> None:
        resp = self.client.post("/api/subjects/create", json={"name": "Bob"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Bob"
        assert data["id"]
        assert data["face_count"] == 0

    async def test_create_subject_returns_existing(self) -> None:
        """Creating with an existing name returns the existing subject."""
        await self.db.create_subject("Alice")
        resp = self.client.post("/api/subjects/create", json={"name": "Alice"})
        data = resp.json()
        assert data["name"] == "Alice"

    async def test_source_info_includes_gps(self) -> None:
        pid = await self.db.add_source(
            "/test.jpg", width=100, height=100, latitude=43.77, longitude=11.25
        )
        resp = self.client.get(f"/api/sources/{pid}/info")
        data = resp.json()
        assert data["latitude"] == pytest.approx(43.77)
        assert data["longitude"] == pytest.approx(11.25)

    async def test_source_info_null_gps(self) -> None:
        pid = await self.db.add_source("/test.jpg", width=100, height=100)
        resp = self.client.get(f"/api/sources/{pid}/info")
        data = resp.json()
        assert data["latitude"] is None
        assert data["longitude"] is None


class TestTogetherAPI(PostgresAppTestCase):
    """Test /api/together endpoint for multi-subject source search."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    def test_empty_query(self) -> None:
        resp = self.client.get("/api/together")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    async def test_finds_source_with_both_subjects(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Bob")
        # One source with both findings
        source_id = await self.db.add_source("/group.jpg", width=100, height=100)
        await add_findings(self.db, [(source_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        await add_findings(self.db, [(source_id, (50, 50, 30, 30), _emb(2), 0.9)], species="human")
        findings = await self.db.get_source_findings(source_id)
        await self.db.assign_finding_to_subject(findings[0].id, sid_a)
        await self.db.assign_finding_to_subject(findings[1].id, sid_b)

        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        data = resp.json()
        assert data["total"] == 1
        assert data["sources"][0]["id"] == str(source_id)

    async def test_excludes_source_with_only_one(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Bob")
        # Source with only Alice
        source_id = await self.db.add_source("/solo.jpg", width=100, height=100)
        await add_findings(self.db, [(source_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        findings = await self.db.get_source_findings(source_id)
        await self.db.assign_finding_to_subject(findings[0].id, sid_a)

        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        assert resp.json()["total"] == 0

    async def test_cross_kind_human_and_pet(self) -> None:
        sid_human = await self.db.create_subject("Eva", kind="person")
        sid_pet = await self.db.create_subject("Figaro", kind="pet")
        source_id = await self.db.add_source("/family.jpg", width=100, height=100)
        await add_findings(self.db, [(source_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        await add_findings(
            self.db, [(source_id, (50, 50, 30, 30), _emb(2, dim=768), 0.9)], species="dog"
        )
        findings = await self.db.get_source_findings(source_id)
        await self.db.assign_finding_to_subject(findings[0].id, sid_human)
        await self.db.assign_finding_to_subject(findings[1].id, sid_pet)

        resp = self.client.get(f"/api/together?subject_ids={sid_human},{sid_pet}")
        data = resp.json()
        assert data["total"] == 1

    async def test_alone_excludes_group_photos(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Bob")
        sid_c = await self.db.create_subject("Charlie")
        # Source with Alice + Bob (alone=true should include)
        s1 = await self.db.add_source("/duo.jpg", width=100, height=100)
        await add_findings(self.db, [(s1, (10, 10, 30, 30), _emb(10), 0.9)], species="human")
        await add_findings(self.db, [(s1, (50, 50, 30, 30), _emb(11), 0.9)], species="human")
        f1 = await self.db.get_source_findings(s1)
        await self.db.assign_finding_to_subject(f1[0].id, sid_a)
        await self.db.assign_finding_to_subject(f1[1].id, sid_b)
        # Source with Alice + Bob + Charlie (alone=true should exclude)
        s2 = await self.db.add_source("/trio.jpg", width=100, height=100)
        await add_findings(self.db, [(s2, (10, 10, 30, 30), _emb(12), 0.9)], species="human")
        await add_findings(self.db, [(s2, (50, 50, 30, 30), _emb(13), 0.9)], species="human")
        await add_findings(self.db, [(s2, (80, 80, 30, 30), _emb(14), 0.9)], species="human")
        f2 = await self.db.get_source_findings(s2)
        await self.db.assign_finding_to_subject(f2[0].id, sid_a)
        await self.db.assign_finding_to_subject(f2[1].id, sid_b)
        await self.db.assign_finding_to_subject(f2[2].id, sid_c)

        # Without alone: both sources match
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        assert resp.json()["total"] == 2
        # With alone: only the duo
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&alone=true")
        assert resp.json()["total"] == 1

    # ── FEAT-20: source_type filter ─────────────────────────────────

    async def _seed_mixed_media(self, sid_a: int, sid_b: int) -> dict[str, int]:
        """Create one photo source and one video source, each containing both subjects."""
        photo_id = await self.db.add_source(
            "/vacation.jpg", source_type="photo", width=100, height=100
        )
        await add_findings(self.db, [(photo_id, (10, 10, 30, 30), _emb(1), 0.9)], species="human")
        await add_findings(self.db, [(photo_id, (50, 50, 30, 30), _emb(2), 0.9)], species="human")
        for f in await self.db.get_source_findings(photo_id):
            await self.db.assign_finding_to_subject(f.id, sid_a if f.bbox_x < 30 else sid_b)

        video_id = await self.db.add_source(
            "/vacation.mp4", source_type="video", width=100, height=100
        )
        await add_findings(
            self.db,
            [(video_id, (10, 10, 30, 30), _emb(3), 0.9)],
            species="human",
            frame_path="tmp/frames/v1.jpg",
        )
        await add_findings(
            self.db,
            [(video_id, (50, 50, 30, 30), _emb(4), 0.9)],
            species="human",
            frame_path="tmp/frames/v2.jpg",
        )
        for f in await self.db.get_source_findings(video_id):
            await self.db.assign_finding_to_subject(f.id, sid_a if f.bbox_x < 30 else sid_b)

        return {"photo": photo_id, "video": video_id}

    async def test_source_type_either_returns_both(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Bob")
        ids = await self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}")
        returned = {s["id"] for s in resp.json()["sources"]}
        assert returned == {str(ids["photo"]), str(ids["video"])}

    async def test_source_type_photo_excludes_videos(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Bob")
        ids = await self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&source_type=photo")
        data = resp.json()
        assert data["total"] == 1
        assert data["sources"][0]["id"] == str(ids["photo"])

    async def test_source_type_video_excludes_photos(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Bob")
        ids = await self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&source_type=video")
        data = resp.json()
        assert data["total"] == 1
        assert data["sources"][0]["id"] == str(ids["video"])

    async def test_source_type_invalid_falls_back_to_either(self) -> None:
        sid_a = await self.db.create_subject("Alice")
        sid_b = await self.db.create_subject("Bob")
        ids = await self._seed_mixed_media(sid_a, sid_b)
        resp = self.client.get(f"/api/together?subject_ids={sid_a},{sid_b}&source_type=bogus")
        assert {s["id"] for s in resp.json()["sources"]} == {str(ids["photo"]), str(ids["video"])}


class TestBrowsePage(PostgresAppTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    def test_browse_page_renders(self) -> None:
        resp = self.client.get("/browse")

        assert resp.status_code == 200
        assert "Path tags" in resp.text

    async def test_browse_html_filters_by_path_tag(self) -> None:
        matching = await self.db.add_source("2022/2022-10_LuccaComics/IMG_20221031.jpg")
        await self.db.add_source("2022/2022-11_Roma/IMG_20221102.jpg")

        resp = self.client.get("/api/browse-html?path_tags=luccacomics")

        assert resp.status_code == 200
        assert f'data-source-id="{matching}"' in resp.text
        assert "1 source found" in resp.text

    async def test_browse_html_backfills_missing_path_metadata(self) -> None:
        matching = await self.db.add_source("2020/2020-12-29.Libera/20201229_165654.jpg")
        await self.db.execute("DELETE FROM source_path_metadata")

        resp = self.client.get("/api/browse-html?path_tags=libera")

        assert resp.status_code == 200
        assert f'data-source-id="{matching}"' in resp.text
        assert (await self.db.fetchrow("SELECT COUNT(*) FROM source_path_metadata"))[0] == 1

    async def test_browse_html_combines_subject_and_path_tag(self) -> None:
        alice = await self.db.create_subject("Alice")
        matching_source, matching_finding = await _add_finding(
            self.db, "2023/2023-08_Amsterdam/IMG_20230801.jpg"
        )
        _, other_finding = await _add_finding(self.db, "2023/2023-08_Roma/IMG_20230802.jpg")
        await self.db.assign_finding_to_subject(matching_finding, alice)
        await self.db.assign_finding_to_subject(other_finding, alice)

        resp = self.client.get(f"/api/browse-html?subject_ids={alice}&path_tags=amsterdam")

        assert resp.status_code == 200
        assert f'data-source-id="{matching_source}"' in resp.text
        assert "1 source found" in resp.text


class TestPrintSelectionPage(PostgresAppTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.db.base_dir = self.tmp_path.resolve()
        self.client = self.make_client(str(self.tmp_path))
        self.tmp = self.tmp_path
        (self.tmp_path / "prints").mkdir()
        self.first_path = self.tmp_path / "prints" / "z first.jpg"
        self.second_path = self.tmp_path / "prints" / "a_second.jpg"
        self.first_path.write_bytes(b"first")
        self.second_path.write_bytes(b"second")

    async def test_add_remove_and_list_print_selection(self) -> None:
        first = await self.db.add_source("prints/z first.jpg")
        second = await self.db.add_source("prints/a_second.jpg")

        resp = self.client.post(f"/api/print-selection/{first}")
        assert resp.status_code == 200
        assert resp.json()["source_ids"] == [str(first)]
        assert "undo_token" not in resp.json()

        resp = self.client.post(f"/api/print-selection/{second}")
        assert resp.status_code == 200
        assert resp.json()["source_ids"] == [str(first), str(second)]
        assert "undo_token" not in resp.json()

        resp = self.client.delete(f"/api/print-selection/{first}")
        assert resp.status_code == 200
        assert resp.json()["source_ids"] == [str(second)]
        assert "undo_token" not in resp.json()

    async def test_print_selection_view_removal_and_clear_are_undoable(self) -> None:
        first = await self.db.add_source("prints/z first.jpg")
        second = await self.db.add_source("prints/a_second.jpg")

        await self.db.add_to_print_selection(first)
        await self.db.add_to_print_selection(second)
        remove_resp = self.client.post(f"/api/print-selection/{first}/remove-from-list")
        assert remove_resp.status_code == 200
        assert "undo_token" in remove_resp.json()
        assert await self.db.get_print_selection_ids() == [second]
        undo_resp = self.client.post(f"/api/undo/{remove_resp.json()['undo_token']}")
        assert undo_resp.status_code == 200
        assert await self.db.get_print_selection_ids() == [first, second]

        clear_resp = self.client.post("/api/print-selection/clear")
        assert clear_resp.status_code == 200
        assert await self.db.get_print_selection_ids() == []
        undo_resp = self.client.post(f"/api/undo/{clear_resp.json()['undo_token']}")
        assert undo_resp.status_code == 200
        assert await self.db.get_print_selection_ids() == [first, second]

    async def test_print_selection_rejects_video_sources(self) -> None:
        video = await self.db.add_source("prints/video.mp4", source_type="video")

        resp = self.client.post(f"/api/print-selection/{video}")

        assert resp.status_code == 400
        assert "Only photos" in resp.json()["detail"]

    async def test_export_print_selection_zip_preserves_order_and_original_bytes(self) -> None:
        first = await self.db.add_source("prints/z first.jpg")
        second = await self.db.add_source("prints/a_second.jpg")
        await self.db.add_to_print_selection(first)
        await self.db.add_to_print_selection(second)

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


class TestNamespaceCollision(PostgresAppTestCase):
    """Verify subject IDs and cluster IDs don't collide in merge suggestions."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    async def test_cluster_suggestion_is_not_labelled_with_a_subject_name(self) -> None:
        """A cluster-cluster suggestion carries no subject name.

        This used to be tested by giving a cluster the same id as a subject,
        which is how the original bug was found. That collision can no longer
        be built: subject ids are uuids and cluster ids are int32, so the two
        namespaces cannot overlap and the schema rules the bug out. What is
        still worth asserting is the visible behaviour.
        """
        sid = await self.db.create_subject("Alice")
        _, fid_alice = await _add_finding(self.db, "/alice.jpg", seed=1)
        await self.db.assign_finding_to_subject(fid_alice, sid)

        _, fid_c1 = await _add_finding(self.db, "/c1.jpg", seed=99)
        _, fid_c2 = await _add_finding(self.db, "/c2.jpg", seed=99)
        await self.db.update_cluster_ids({fid_c1: 1, fid_c2: 1})

        _, fid_c3 = await _add_finding(self.db, "/c3.jpg", seed=99)
        _, fid_c4 = await _add_finding(self.db, "/c4.jpg", seed=99)
        await self.db.update_cluster_ids({fid_c3: 999, fid_c4: 999})

        resp = self.client.get("/api/merge-suggestions?min_sim=50")
        data = resp.json()

        for s in data["suggestions"]:
            if {s["cluster_a"], s["cluster_b"]} == {1, 999}:
                assert s["name_a"] is None, f"cluster 1 labelled '{s['name_a']}'"
                assert s["name_b"] is None, f"cluster 999 labelled '{s['name_b']}'"
                break
        else:
            pytest.fail("Expected cluster-cluster suggestion not found")


class TestTemplateIdRendering(PostgresAppTestCase):
    """Ids reach Alpine through ``|tojson``, which json.dumps cannot render
    for a UUID — every such page 500s without the configured policy."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()

    async def test_subject_detail_renders_with_uuid_ids(self) -> None:
        sid = await self.db.create_subject("Alice")
        resp = self.client.get(f"/people/{sid}")
        assert resp.status_code == 200, resp.text

    async def test_subject_id_reaches_javascript_quoted(self) -> None:
        """A bare UUID in a JS expression is not valid JavaScript."""
        sid = await self.db.create_subject("Alice")
        body = self.client.get(f"/people/{sid}").text
        assert f'subjectId: "{sid}"' in body
        assert f"subjectId: {sid}," not in body


class TestServeOnlyMode(PostgresAppTestCase):
    """A base install has no detection models (W3). The app must still
    serve everything that does not need one, and refuse the one route
    that does with a message rather than a traceback."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.client = self.make_client()
        self.source_id = await self.db.add_source("/a.jpg", width=400, height=300)

    async def test_manual_finding_returns_503_without_models(self) -> None:
        with patch("ritrova.app.routers.findings.models_for_species", return_value=False):
            resp = self.client.post(
                f"/api/sources/{self.source_id}/findings",
                json={"bbox": [10, 10, 60, 60], "species": "human"},
            )
        assert resp.status_code == 503, resp.text
        assert "uv sync --extra scan" in resp.json()["error"]

    async def test_bad_bbox_still_422_not_503(self) -> None:
        """Validation comes first: a malformed request is the client's
        fault whether or not the models are installed."""
        with patch("ritrova.app.routers.findings.models_for_species", return_value=False):
            resp = self.client.post(
                f"/api/sources/{self.source_id}/findings",
                json={"bbox": [10, 10], "species": "human"},
            )
        assert resp.status_code == 422

    async def test_healthz_reports_ok(self) -> None:
        resp = self.client.get("/healthz")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "ok"

    async def test_healthz_unhealthy_when_the_photo_mount_is_gone(self) -> None:
        """A bind mount over a disk that did not come back leaves an empty
        directory, which otherwise serves 404s while reporting healthy."""
        empty = Path(self.tmp_path) / "no_mount"
        empty.mkdir()
        client = self.make_client(photos_dir=str(empty))
        resp = client.get("/healthz")
        assert resp.status_code == 503, resp.text
        assert resp.json()["photos"] == "archive empty or unreadable"

    async def test_healthz_ok_when_the_archive_has_content(self) -> None:
        mounted = Path(self.tmp_path) / "mounted"
        (mounted / "2002").mkdir(parents=True)
        client = self.make_client(photos_dir=str(mounted))
        assert client.get("/healthz").json()["status"] == "ok"

    async def test_healthz_is_not_shadowed_by_the_kind_catchall(self) -> None:
        """``/{kind}`` matches any single segment; /healthz must win."""
        assert self.client.get("/healthz").headers["content-type"].startswith("application/json")


class TestSourceImageCache(PostgresAppTestCase):
    """Resizing a multi-megapixel original on every view is the one image
    cost that never improves; thumbnails have always been cached."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        photos = Path(self.tmp_path) / "photos"
        photos.mkdir()
        from PIL import Image

        Image.new("RGB", (2400, 1800), "red").save(photos / "big.jpg")
        self.client = self.make_client(photos_dir=str(photos))
        self.source_id = await self.db.add_source("big.jpg", width=2400, height=1800)

    async def test_second_request_is_served_from_disk(self) -> None:
        first = self.client.get(f"/api/sources/{self.source_id}/image")
        assert first.status_code == 200, first.text
        cached = Path(self.tmp_path) / "tmp" / "views" / f"{self.source_id}_1600.jpg"
        assert cached.exists()

        # Remove the original: a second hit that still succeeds can only
        # have come from the cache.
        (Path(self.tmp_path) / "photos" / "big.jpg").unlink()
        second = self.client.get(f"/api/sources/{self.source_id}/image")
        assert second.status_code == 200
        assert second.content == first.content

    async def test_each_size_is_cached_separately(self) -> None:
        self.client.get(f"/api/sources/{self.source_id}/image?max_size=400")
        self.client.get(f"/api/sources/{self.source_id}/image?max_size=1600")
        views = Path(self.tmp_path) / "tmp" / "views"
        assert {p.name for p in views.iterdir()} == {
            f"{self.source_id}_400.jpg",
            f"{self.source_id}_1600.jpg",
        }
