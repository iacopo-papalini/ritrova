"""FindingsService round-trip tests (FEAT-29).

The service's detector loading is heavy (InsightFace / SigLIP weights),
so we monkey-patch ``_embed`` to return a deterministic vector rather
than invoking the real models. Every test here exercises the plumbing
around the embedding — scan-id lookup, DB insert, suggestion ranking,
undo — not the model weights themselves.
"""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from PIL import Image

from ritrova.db import FaceDB
from ritrova.services_domain import FindingsService, ManualFindingError, UndoReceipt
from ritrova.undo import UndoStore
from tests._helpers import add_findings


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestCreateManualHappyPath(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, db: FaceDB) -> None:
        self.db = db
        self.undo = UndoStore()
        self.tmp = tmp_path
        # Real JPG on disk so the service can open it and walk EXIF.
        self.image_path = tmp_path / "photo.jpg"
        Image.new("RGB", (400, 300), color="red").save(self.image_path, "JPEG")
        self.source_id = db.add_source(str(self.image_path), width=400, height=300)
        # The manual-finding endpoint requires a 'subjects' scan on the source.
        self.scan_id = db.record_scan(self.source_id, "subjects")

        self.svc = FindingsService(db, self.undo)
        # Stub the model path — deterministic 512-d unit vector.
        self._fake_embedding = _unit(seed=7)
        self.svc._embed = lambda image, bbox, species: self._fake_embedding  # type: ignore[assignment]

    def test_create_inserts_finding_row_with_embedding(self) -> None:
        result = self.svc.create_manual(
            source_id=self.source_id, bbox=(10, 10, 60, 60), species="human"
        )
        findings = self.db.get_source_findings(self.source_id)
        assert len(findings) == 1
        f = findings[0]
        assert f.id == result.finding_id
        assert f.bbox_x == 10 and f.bbox_y == 10 and f.bbox_w == 60 and f.bbox_h == 60
        assert f.confidence == 1.0
        assert f.species == "human"
        assert f.scan_id == self.scan_id
        assert f.frame_path is None
        # Embedding is stored and L2-normalized.
        assert f.embedding.shape == (512,)
        assert abs(float(np.linalg.norm(f.embedding)) - 1.0) < 1e-5

    def test_no_subjects_yields_no_suggestion(self) -> None:
        result = self.svc.create_manual(
            source_id=self.source_id, bbox=(10, 10, 60, 60), species="human"
        )
        assert result.suggestion is None
        assert isinstance(result.receipt, UndoReceipt)
        assert result.receipt.message == "Added manual face"

    def test_suggestion_points_to_nearest_named_subject(self) -> None:
        # Seed a named subject whose centroid matches the faked embedding.
        sid = self.db.create_subject("Caterina", kind="person")
        seed_src = self.db.add_source("/seed.jpg", width=100, height=100)
        # Three findings with the same embedding → centroid == that embedding.
        add_findings(
            self.db,
            [
                (seed_src, (0, 0, 50, 50), self._fake_embedding, 0.9),
                (seed_src, (5, 5, 50, 50), self._fake_embedding, 0.9),
                (seed_src, (10, 10, 50, 50), self._fake_embedding, 0.9),
            ],
            species="human",
        )
        for f in self.db.get_source_findings(seed_src):
            self.db.assign_finding_to_subject(f.id, sid)

        result = self.svc.create_manual(
            source_id=self.source_id, bbox=(10, 10, 60, 60), species="human"
        )
        assert result.suggestion is not None
        assert result.suggestion.subject_id == sid
        assert result.suggestion.name == "Caterina"
        # Cosine similarity of a vector with itself is 1.0 → 100%.
        assert result.suggestion.similarity_pct > 55.0

    def test_undo_deletes_the_finding(self) -> None:
        result = self.svc.create_manual(
            source_id=self.source_id, bbox=(10, 10, 60, 60), species="human"
        )
        assert self.db.get_finding(result.finding_id) is not None
        entry = self.undo.pop(result.receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_finding(result.finding_id) is None


class TestCreateManualValidation(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path, db: FaceDB) -> None:
        self.db = db
        self.undo = UndoStore()
        self.tmp = tmp_path
        self.image_path = tmp_path / "photo.jpg"
        Image.new("RGB", (400, 300), color="blue").save(self.image_path, "JPEG")
        self.source_id = db.add_source(str(self.image_path), width=400, height=300)
        self.scan_id = db.record_scan(self.source_id, "subjects")
        self.svc = FindingsService(db, self.undo)
        self.svc._embed = lambda image, bbox, species: _unit(seed=1)  # type: ignore[assignment]

    def test_rejects_bad_species(self) -> None:
        with pytest.raises(ManualFindingError):
            self.svc.create_manual(source_id=self.source_id, bbox=(10, 10, 60, 60), species="bird")

    def test_rejects_too_small_bbox(self) -> None:
        with pytest.raises(ManualFindingError):
            self.svc.create_manual(source_id=self.source_id, bbox=(10, 10, 10, 10), species="human")

    def test_rejects_bbox_outside_source(self) -> None:
        with pytest.raises(ManualFindingError):
            self.svc.create_manual(
                source_id=self.source_id, bbox=(390, 290, 40, 40), species="human"
            )

    def test_rejects_missing_source(self) -> None:
        with pytest.raises(ManualFindingError):
            self.svc.create_manual(source_id=999999, bbox=(10, 10, 60, 60), species="human")

    def test_rejects_video_source(self) -> None:
        vid = self.db.add_source("/movie.mp4", source_type="video", width=400, height=300)
        self.db.record_scan(vid, "subjects")
        with pytest.raises(ManualFindingError):
            self.svc.create_manual(source_id=vid, bbox=(10, 10, 60, 60), species="human")

    def test_rejects_source_with_no_subjects_scan(self) -> None:
        unscanned_path = self.tmp / "unscanned.jpg"
        Image.new("RGB", (400, 300), color="green").save(unscanned_path, "JPEG")
        unscanned_id = self.db.add_source(str(unscanned_path), width=400, height=300)
        with pytest.raises(ManualFindingError):
            self.svc.create_manual(source_id=unscanned_id, bbox=(10, 10, 60, 60), species="human")
