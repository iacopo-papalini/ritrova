"""Tests for face_recog.scanner module."""

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from face_recog.scanner import _is_duplicate, find_images, find_videos


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
        # On case-insensitive fs, only one file can exist
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
