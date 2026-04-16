"""Tests for concrete analysis steps (ADR-009 Phase B)."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from ritrova.analysis import AnalysisFinding, FrameRef, SourceAnalysis
from ritrova.analysis_steps import (
    CaptionStep,
    DeduplicationStep,
    FaceDetectionStep,
    PetDetectionStep,
)
from ritrova.detection import Detection, DetectionResult


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _frame(frame_number: int = 0) -> FrameRef:
    return FrameRef(
        source_path=Path("/fake.jpg"),
        frame_number=frame_number,
        image=Image.new("RGB", (200, 200)),
        width=200,
        height=200,
    )


def _state() -> SourceAnalysis:
    return SourceAnalysis(source_path="/fake.jpg", source_type="photo")


# ── FaceDetectionStep ────────────────────────────────────────────────────


class TestFaceDetectionStep(TestCase):
    def _mock_detector(self, detections: list[Detection] | None = None) -> MagicMock:
        detector = MagicMock()
        if detections is None:
            detections = [
                Detection(bbox=(10, 10, 50, 50), embedding=_emb(1), confidence=0.95),
            ]
        detector.detect.return_value = DetectionResult(detections=detections, width=200, height=200)
        return detector

    def test_appends_findings(self) -> None:
        step = FaceDetectionStep(self._mock_detector())
        result = step.analyse(_frame(), _state())
        assert len(result.findings) == 1
        assert result.findings[0].species == "human"
        assert result.findings[0].confidence == 0.95

    def test_filters_low_confidence(self) -> None:
        low = [Detection(bbox=(10, 10, 50, 50), embedding=_emb(1), confidence=0.3)]
        step = FaceDetectionStep(self._mock_detector(low), min_confidence=0.65)
        result = step.analyse(_frame(), _state())
        assert result.findings == []

    def test_preserves_existing_findings(self) -> None:
        state = _state()
        state.findings.append(
            AnalysisFinding(bbox=(0, 0, 10, 10), embedding=_emb(99), confidence=0.5, species="dog")
        )
        step = FaceDetectionStep(self._mock_detector())
        result = step.analyse(_frame(), state)
        assert len(result.findings) == 2
        assert result.findings[0].species == "dog"
        assert result.findings[1].species == "human"

    def test_sets_frame_number(self) -> None:
        step = FaceDetectionStep(self._mock_detector())
        result = step.analyse(_frame(frame_number=7), _state())
        assert result.findings[0].frame_number == 7


# ── PetDetectionStep ────────────────────────────────────────────────────


class TestPetDetectionStep(TestCase):
    def _mock_detector(self, detections: list[Detection] | None = None) -> MagicMock:
        detector = MagicMock()
        if detections is None:
            detections = [
                Detection(
                    bbox=(20, 20, 60, 60),
                    embedding=_emb(1, dim=768),
                    confidence=0.9,
                    species="dog",
                ),
            ]
        detector.detect.return_value = DetectionResult(detections=detections, width=200, height=200)
        return detector

    def test_appends_pet_findings(self) -> None:
        step = PetDetectionStep(self._mock_detector())
        result = step.analyse(_frame(), _state())
        assert len(result.findings) == 1
        assert result.findings[0].species == "dog"

    def test_preserves_species_from_detection(self) -> None:
        cat = [
            Detection(
                bbox=(10, 10, 40, 40),
                embedding=_emb(2, dim=768),
                confidence=0.85,
                species="cat",
            )
        ]
        step = PetDetectionStep(self._mock_detector(cat))
        result = step.analyse(_frame(), _state())
        assert result.findings[0].species == "cat"

    def test_filters_low_confidence(self) -> None:
        low = [
            Detection(
                bbox=(10, 10, 50, 50),
                embedding=_emb(1, dim=768),
                confidence=0.1,
                species="dog",
            )
        ]
        step = PetDetectionStep(self._mock_detector(low), min_confidence=0.5)
        result = step.analyse(_frame(), _state())
        assert result.findings == []


# ── CaptionStep ──────────────────────────────────────────────────────────


class TestCaptionStep(TestCase):
    def _mock_describer(
        self, caption: str = "A dog in a park", tags: set[str] | None = None
    ) -> MagicMock:
        describer = MagicMock()
        describer.model_id = "fake-vlm"
        describer.describe.return_value = (caption, tags or {"dog", "park"})
        return describer

    def _mock_translator(self) -> MagicMock:
        translator = MagicMock()
        translator.model_id = "fake-translator"
        translator.translate.return_value = ("Un cane nel parco", {"cane", "parco"})
        return translator

    def test_sets_caption_and_tags(self) -> None:
        step = CaptionStep(self._mock_describer())
        result = step.analyse(_frame(), _state())
        assert result.caption == "A dog in a park"
        assert result.tags == {"dog", "park"}

    def test_with_translator(self) -> None:
        step = CaptionStep(self._mock_describer(), translator=self._mock_translator())
        result = step.analyse(_frame(), _state())
        assert result.caption == "Un cane nel parco"
        assert result.tags == {"cane", "parco"}

    def test_skips_non_zero_frames(self) -> None:
        step = CaptionStep(self._mock_describer())
        result = step.analyse(_frame(frame_number=5), _state())
        assert result.caption == ""

    def test_empty_caption_leaves_state_unchanged(self) -> None:
        step = CaptionStep(self._mock_describer(caption=""))
        result = step.analyse(_frame(), _state())
        assert result.caption == ""
        assert result.tags == set()

    def test_name_includes_translator(self) -> None:
        step = CaptionStep(self._mock_describer(), translator=self._mock_translator())
        assert step.name == "fake-vlm+fake-translator"

    def test_name_without_translator(self) -> None:
        step = CaptionStep(self._mock_describer())
        assert step.name == "fake-vlm"


# ── DeduplicationStep ────────────────────────────────────────────────────


class TestDeduplicationStep(TestCase):
    def test_keeps_unique_findings(self) -> None:
        state = _state()
        state.findings = [
            AnalysisFinding(bbox=(0, 0, 10, 10), embedding=_emb(1), confidence=0.9),
            AnalysisFinding(bbox=(50, 50, 10, 10), embedding=_emb(2), confidence=0.8),
        ]
        step = DeduplicationStep(threshold=0.6)
        result = step.analyse(_frame(), state)
        assert len(result.findings) == 2

    def test_merges_similar_keeps_best_confidence(self) -> None:
        emb = _emb(1)
        state = _state()
        state.findings = [
            AnalysisFinding(
                bbox=(0, 0, 10, 10), embedding=emb.copy(), confidence=0.7, frame_number=0
            ),
            AnalysisFinding(
                bbox=(5, 5, 12, 12), embedding=emb.copy(), confidence=0.95, frame_number=5
            ),
        ]
        step = DeduplicationStep(threshold=0.6)
        result = step.analyse(_frame(), state)
        assert len(result.findings) == 1
        assert result.findings[0].confidence == 0.95
        assert result.findings[0].frame_number == 5

    def test_dedup_per_species(self) -> None:
        """Same embedding but different species should not merge."""
        emb = _emb(1)
        state = _state()
        state.findings = [
            AnalysisFinding(
                bbox=(0, 0, 10, 10), embedding=emb.copy(), confidence=0.9, species="human"
            ),
            AnalysisFinding(
                bbox=(0, 0, 10, 10), embedding=emb.copy(), confidence=0.9, species="dog"
            ),
        ]
        step = DeduplicationStep(threshold=0.6)
        result = step.analyse(_frame(), state)
        assert len(result.findings) == 2

    def test_empty_findings(self) -> None:
        step = DeduplicationStep()
        result = step.analyse(_frame(), _state())
        assert result.findings == []
