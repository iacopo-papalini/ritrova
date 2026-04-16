"""Tests for ritrova.scanner module."""

from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
import pytest

from ritrova.db import FaceDB
from ritrova.detection import Detection, DetectionResult
from ritrova.scanner import _is_duplicate, find_images, find_videos, scan_pets, scan_photos


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_jpeg(path: Path) -> None:
    """Create a minimal valid JPEG file."""
    from PIL import Image

    img = Image.new("RGB", (200, 200), color="blue")
    img.save(str(path), "JPEG")


def _stub_file(path: Path) -> None:
    """Create a 1-byte file (find_* skip 0-byte files defensively)."""
    path.write_bytes(b"x")


class TestFindImages(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_tmp(self, tmp_path: Path) -> None:
        self.tmp = tmp_path

    def test_discovers_jpg(self) -> None:
        _stub_file(self.tmp / "photo.jpg")
        _stub_file(self.tmp / "photo.png")
        result = find_images(self.tmp)
        assert len(result) == 1
        assert result[0].name == "photo.jpg"

    def test_case_insensitive(self) -> None:
        _stub_file(self.tmp / "photo.JPG")
        _stub_file(self.tmp / "photo2.jpeg")
        result = find_images(self.tmp)
        assert len(result) == 2

    def test_deduplicates_by_case(self) -> None:
        _stub_file(self.tmp / "photo.jpg")
        result = find_images(self.tmp)
        assert len(result) == 1

    def test_skips_zero_byte_files(self) -> None:
        """Defensive: 0-byte uploads from corrupted backups (real case from
        2026-04-13 benchmark on `Foto Eva/2015/20151212_183725.jpg`) should
        never reach the detector — they crash YOLO and waste an error slot."""
        _stub_file(self.tmp / "good.jpg")
        (self.tmp / "empty.jpg").touch()  # 0 bytes
        result = find_images(self.tmp)
        assert [p.name for p in result] == ["good.jpg"]


class TestFindVideos(TestCase):
    @pytest.fixture(autouse=True)
    def _setup_tmp(self, tmp_path: Path) -> None:
        self.tmp = tmp_path

    def test_discovers_mp4(self) -> None:
        _stub_file(self.tmp / "clip.mp4")
        _stub_file(self.tmp / "photo.jpg")
        result = find_videos(self.tmp)
        assert len(result) == 1
        assert result[0].name == "clip.mp4"

    def test_discovers_multiple_extensions(self) -> None:
        _stub_file(self.tmp / "a.mp4")
        _stub_file(self.tmp / "b.mov")
        _stub_file(self.tmp / "c.avi")
        result = find_videos(self.tmp)
        assert len(result) == 3

    def test_skips_zero_byte_videos(self) -> None:
        _stub_file(self.tmp / "good.mp4")
        (self.tmp / "empty.mp4").touch()
        result = find_videos(self.tmp)
        assert [p.name for p in result] == ["good.mp4"]


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


class TestGetExifGps(TestCase):
    """get_exif_gps must never raise on malformed EXIF — one bad photo would
    otherwise abort the whole scan-photos pass (regression: 2026-04-13 benchmark)."""

    def test_returns_none_on_zero_denominator(self) -> None:
        """Some cameras write IFDRational(0, 0) in GPS tags; PIL only blows
        up when we cast to float deep inside the per-image scan loop."""
        from unittest.mock import MagicMock, patch

        from ritrova.scanner import get_exif_gps

        class _BadRational:
            def __float__(self) -> float:
                msg = "division by zero"
                raise ZeroDivisionError(msg)

            def __bool__(self) -> bool:
                return True

        fake_exif = MagicMock()
        # GPS IFD with malformed lat/lon (3-tuple of bad rationals each).
        fake_exif.get_ifd.return_value = {
            1: "N",
            2: (_BadRational(), _BadRational(), _BadRational()),
            3: "E",
            4: (_BadRational(), _BadRational(), _BadRational()),
        }
        fake_img_cm = MagicMock()
        fake_img_cm.__enter__.return_value.getexif.return_value = fake_exif

        with patch("ritrova.scanner.Image.open", return_value=fake_img_cm):
            assert get_exif_gps(Path("/fake.jpg")) is None  # must not raise


class TestScanPhotos(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.photos_dir = tmp_path / "photos"
        self.photos_dir.mkdir()

    def _mock_detector(
        self,
        detections: list[Detection] | None = None,
        w: int = 200,
        h: int = 200,
    ) -> MagicMock:
        detector = MagicMock()
        if detections is None:
            detections = [
                Detection(bbox=(10, 10, 50, 50), embedding=_emb(1), confidence=0.95),
            ]
        detector.detect.return_value = DetectionResult(detections=detections, width=w, height=h)
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
        low_conf = [
            Detection(bbox=(10, 10, 50, 50), embedding=_emb(1), confidence=0.3),
        ]
        detector = self._mock_detector(detections=low_conf)

        result = scan_photos(self.db, self.photos_dir, detector, min_confidence=0.65)
        assert result["processed"] == 1
        assert result["faces_found"] == 0

    def test_handles_detection_error(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_detector(detections=[], w=0, h=0)

        result = scan_photos(self.db, self.photos_dir, detector)
        assert result["errors"] == 1
        assert result["processed"] == 0

    def test_stores_findings_in_db(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        detector = self._mock_detector()

        scan_photos(self.db, self.photos_dir, detector)
        assert self.db.get_finding_count() == 1
        assert self.db.get_source_count() == 1

    def test_multiple_findings_per_source(self) -> None:
        _make_jpeg(self.photos_dir / "a.jpg")
        multi = [
            Detection(bbox=(10, 10, 50, 50), embedding=_emb(1), confidence=0.95),
            Detection(bbox=(70, 10, 50, 50), embedding=_emb(2), confidence=0.90),
        ]
        detector = self._mock_detector(detections=multi)

        result = scan_photos(self.db, self.photos_dir, detector)
        assert result["faces_found"] == 2
        assert self.db.get_finding_count() == 2


class TestScanPets(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.photos_dir = tmp_path / "photos"
        self.photos_dir.mkdir()

    def _mock_pet_detector(
        self,
        detections: list[Detection] | None = None,
        w: int = 200,
        h: int = 200,
    ) -> MagicMock:
        detector = MagicMock()
        if detections is None:
            detections = [
                Detection(
                    bbox=(10, 10, 50, 50),
                    embedding=_emb(1, dim=768),
                    confidence=0.9,
                    species="dog",
                ),
            ]
        detector.detect.return_value = DetectionResult(detections=detections, width=w, height=h)
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
                Detection(
                    bbox=(10, 10, 50, 50),
                    embedding=_emb(1, dim=768),
                    confidence=0.9,
                    species="cat",
                ),
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
                Detection(
                    bbox=(10, 10, 50, 50),
                    embedding=_emb(1, dim=768),
                    confidence=0.1,
                    species="dog",
                ),
            ]
        )

        result = scan_pets(self.db, self.photos_dir, detector, min_confidence=0.5)
        assert result["pets_found"] == 0
