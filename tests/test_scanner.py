"""Tests for face_recog.scanner module."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pytest

from face_recog.db import FaceDB
from face_recog.scanner import _is_duplicate, find_images, find_videos, scan_pets, scan_photos


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_jpeg(path: Path) -> None:
    """Create a minimal valid JPEG file."""
    from PIL import Image

    img = Image.new("RGB", (200, 200), color="blue")
    img.save(str(path), "JPEG")


class TestFindImages(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_tmp(self, tmp_path: Path) -> None:
        self.tmp = tmp_path

    def test_discovers_jpg(self) -> None:
        (self.tmp / "photo.jpg").touch()
        (self.tmp / "photo.png").touch()
        result = find_images(self.tmp)
        assert len(result) == 1
        assert result[0].name == "photo.jpg"

    def test_case_insensitive(self) -> None:
        (self.tmp / "photo.JPG").touch()
        (self.tmp / "photo2.jpeg").touch()
        result = find_images(self.tmp)
        assert len(result) == 2

    def test_deduplicates_by_case(self) -> None:
        (self.tmp / "photo.jpg").touch()
        result = find_images(self.tmp)
        assert len(result) == 1


class TestFindVideos(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_tmp(self, tmp_path: Path) -> None:
        self.tmp = tmp_path

    def test_discovers_mp4(self) -> None:
        (self.tmp / "clip.mp4").touch()
        (self.tmp / "photo.jpg").touch()
        result = find_videos(self.tmp)
        assert len(result) == 1
        assert result[0].name == "clip.mp4"

    def test_discovers_multiple_extensions(self) -> None:
        (self.tmp / "a.mp4").touch()
        (self.tmp / "b.mov").touch()
        (self.tmp / "c.avi").touch()
        result = find_videos(self.tmp)
        assert len(result) == 3


class TestIsDuplicate(TestCase):
    def test_above_threshold(self) -> None:
        emb = np.array([1.0, 0.0], dtype=np.float32)
        seen = [np.array([1.0, 0.0], dtype=np.float32)]
        assert _is_duplicate(emb, seen, threshold=0.5)

    def test_below_threshold(self) -> None:
        emb = np.array([1.0, 0.0], dtype=np.float32)
        seen = [np.array([0.0, 1.0], dtype=np.float32)]
        assert not _is_duplicate(emb, seen, threshold=0.5)

    def test_empty_seen(self) -> None:
        emb = np.array([1.0, 0.0], dtype=np.float32)
        assert not _is_duplicate(emb, [], threshold=0.5)


class TestScanPhotos(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.photos_dir = tmp_path / "photos"
        self.photos_dir.mkdir()

    def _mock_detector(
        self,
        faces: list[dict[str, object]] | None = None,
        w: int = 200,
        h: int = 200,
    ) -> MagicMock:
        detector = MagicMock()
        if faces is None:
            faces = [
                {"bbox": (10, 10, 50, 50), "embedding": _emb(1), "confidence": 0.95},
            ]
        detector.detect.return_value = (faces, w, h)
        return detector

    def test_scans_new_photo(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_detector()

        result = scan_photos(self.db, self.photos_dir, detector)
        assert result["processed"] == 1
        assert result["faces_found"] == 1
        assert result["skipped"] == 0

    def test_skips_already_scanned(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_detector()

        scan_photos(self.db, self.photos_dir, detector)
        result = scan_photos(self.db, self.photos_dir, detector)
        assert result["skipped"] == 1
        assert result["processed"] == 0

    def test_filters_by_confidence(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        low_conf_faces: list[dict[str, object]] = [
            {"bbox": (10, 10, 50, 50), "embedding": _emb(1), "confidence": 0.3},
        ]
        detector = self._mock_detector(faces=low_conf_faces)

        result = scan_photos(self.db, self.photos_dir, detector, min_confidence=0.65)
        assert result["processed"] == 1
        assert result["faces_found"] == 0

    def test_handles_detection_error(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_detector(faces=[], w=0, h=0)

        result = scan_photos(self.db, self.photos_dir, detector)
        assert result["errors"] == 1
        assert result["processed"] == 0

    def test_stores_faces_in_db(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_detector()

        scan_photos(self.db, self.photos_dir, detector)
        assert self.db.get_face_count() == 1
        assert self.db.get_photo_count() == 1

    def test_multiple_faces_per_photo(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        faces: list[dict[str, object]] = [
            {"bbox": (10, 10, 50, 50), "embedding": _emb(1), "confidence": 0.95},
            {"bbox": (70, 10, 50, 50), "embedding": _emb(2), "confidence": 0.90},
        ]
        detector = self._mock_detector(faces=faces)

        result = scan_photos(self.db, self.photos_dir, detector)
        assert result["faces_found"] == 2
        assert self.db.get_face_count() == 2


class TestScanPets(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.photos_dir = tmp_path / "photos"
        self.photos_dir.mkdir()

    def _mock_pet_detector(self, detections: list[dict[str, object]] | None = None) -> MagicMock:
        detector = MagicMock()
        if detections is None:
            detections = [
                {
                    "species": "dog",
                    "bbox": (10, 10, 50, 50),
                    "embedding": _emb(1, dim=768),
                    "confidence": 0.9,
                },
            ]
        detector.detect.return_value = detections
        return detector

    def test_scans_pets(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_pet_detector()

        result = scan_pets(self.db, self.photos_dir, detector)
        assert result["processed"] == 1
        assert result["pets_found"] == 1

    def test_stores_species_correctly(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_pet_detector(
            [
                {
                    "species": "cat",
                    "bbox": (10, 10, 50, 50),
                    "embedding": _emb(1, dim=768),
                    "confidence": 0.9,
                },
            ]
        )

        scan_pets(self.db, self.photos_dir, detector)
        embs = self.db.get_all_embeddings(species="cat")
        assert len(embs) == 1

    def test_skips_already_scanned(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_pet_detector(detections=[])

        scan_pets(self.db, self.photos_dir, detector)
        result = scan_pets(self.db, self.photos_dir, detector)
        assert result["skipped"] == 1

    def test_filters_by_confidence(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_pet_detector(
            [
                {
                    "species": "dog",
                    "bbox": (10, 10, 50, 50),
                    "embedding": _emb(1, dim=768),
                    "confidence": 0.1,
                },
            ]
        )

        result = scan_pets(self.db, self.photos_dir, detector, min_confidence=0.5)
        assert result["pets_found"] == 0
