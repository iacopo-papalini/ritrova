"""POST /api/sources/{id}/findings router tests (FEAT-29).

The service's ``_embed`` is monkey-patched so tests don't spin up the
real detector weights. The router covers: validation (422 branches),
the happy-path 200 with suggestion, and undo round-trip via the
``/api/undo/pop`` endpoint.
"""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ritrova.app import create_app
from ritrova.app.deps import get_findings_service
from ritrova.db import FaceDB
from tests._helpers import add_findings


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestCreateManualFindingRouter(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        db_path = tmp_path / "test.db"
        self.db = FaceDB(db_path)
        self.app = create_app(str(db_path))
        self.client = TestClient(self.app)

        self.image_path = tmp_path / "photo.jpg"
        Image.new("RGB", (400, 300), color="red").save(self.image_path, "JPEG")
        self.source_id = self.db.add_source(str(self.image_path), width=400, height=300)
        self.scan_id = self.db.record_scan(self.source_id, "subjects")

        # Stub the embedding path on the app's cached service.
        self.fake_emb = _unit(seed=3)
        svc = get_findings_service()
        svc._embed = lambda image, bbox, species: self.fake_emb  # type: ignore[assignment]

    def test_valid_bbox_returns_200_with_expected_keys(self) -> None:
        resp = self.client.post(
            f"/api/sources/{self.source_id}/findings",
            json={"bbox": [10, 10, 60, 60], "species": "human"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["ok"] is True
        assert isinstance(body["finding_id"], int)
        assert "undo_token" in body
        assert body["message"] == "Added manual face"
        # No subjects yet → no suggestion.
        assert body["suggestion"] is None

    def test_out_of_bounds_bbox_422(self) -> None:
        resp = self.client.post(
            f"/api/sources/{self.source_id}/findings",
            json={"bbox": [500, 500, 60, 60], "species": "human"},
        )
        assert resp.status_code == 422

    def test_too_small_bbox_422(self) -> None:
        resp = self.client.post(
            f"/api/sources/{self.source_id}/findings",
            json={"bbox": [10, 10, 10, 10], "species": "human"},
        )
        assert resp.status_code == 422

    def test_bad_species_422(self) -> None:
        resp = self.client.post(
            f"/api/sources/{self.source_id}/findings",
            json={"bbox": [10, 10, 60, 60], "species": "bird"},
        )
        assert resp.status_code == 422

    def test_bad_bbox_shape_422(self) -> None:
        resp = self.client.post(
            f"/api/sources/{self.source_id}/findings",
            json={"bbox": [10, 10, 60], "species": "human"},
        )
        assert resp.status_code == 422

    def test_video_source_422(self) -> None:
        vid = self.db.add_source("/movie.mp4", source_type="video", width=400, height=300)
        self.db.record_scan(vid, "subjects")
        resp = self.client.post(
            f"/api/sources/{vid}/findings",
            json={"bbox": [10, 10, 60, 60], "species": "human"},
        )
        assert resp.status_code == 422
        assert "video" in resp.json()["error"].lower()

    def test_missing_subjects_scan_422(self) -> None:
        unscanned_path = self.tmp / "noscan.jpg"
        Image.new("RGB", (400, 300), color="blue").save(unscanned_path, "JPEG")
        unscanned_id = self.db.add_source(str(unscanned_path), width=400, height=300)
        resp = self.client.post(
            f"/api/sources/{unscanned_id}/findings",
            json={"bbox": [10, 10, 60, 60], "species": "human"},
        )
        assert resp.status_code == 422

    def test_suggestion_returned_for_matching_subject(self) -> None:
        sid = self.db.create_subject("Caterina", kind="person")
        seed_src = self.db.add_source("/seed.jpg", width=100, height=100)
        add_findings(
            self.db,
            [(seed_src, (0, 0, 50, 50), self.fake_emb, 0.9) for _ in range(3)],
            species="human",
        )
        for f in self.db.get_source_findings(seed_src):
            self.db.assign_finding_to_subject(f.id, sid)

        resp = self.client.post(
            f"/api/sources/{self.source_id}/findings",
            json={"bbox": [10, 10, 60, 60], "species": "human"},
        )
        body = resp.json()
        assert body["suggestion"] is not None
        assert body["suggestion"]["subject_id"] == sid
        assert body["suggestion"]["name"] == "Caterina"

    def test_undo_deletes_created_finding(self) -> None:
        resp = self.client.post(
            f"/api/sources/{self.source_id}/findings",
            json={"bbox": [10, 10, 60, 60], "species": "human"},
        )
        body = resp.json()
        finding_id = body["finding_id"]
        token = body["undo_token"]
        assert self.db.get_finding(finding_id) is not None

        undo_resp = self.client.post(f"/api/undo/{token}")
        assert undo_resp.status_code == 200
        assert self.db.get_finding(finding_id) is None
