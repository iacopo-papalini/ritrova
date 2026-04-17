"""Concrete analysis steps for the unified pipeline (ADR-009).

Each step wraps an existing detector or describer, converting its output
into ``AnalysisFinding`` / caption+tags on the ``SourceAnalysis`` value object.
Steps never touch the database.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from .analysis import AnalysisFinding, AnalysisStep, FrameRef, SourceAnalysis

if TYPE_CHECKING:
    from .describer import Describer, Translator
    from .detector import FaceDetector
    from .pet_detector import PetDetector

logger = logging.getLogger(__name__)

# Serialise GPU/ANE inference across threads. Metal and CoreML don't support
# concurrent command buffer submissions from multiple threads safely.
_inference_lock = threading.Lock()


class FaceDetectionStep(AnalysisStep):
    """Detect human faces in a frame using ArcFace.

    When ``prefilter_enabled`` is True, detection is skipped on frames where
    the caption step previously set ``state.has_people`` to False. The default
    is False so prefilter stays off unless an upstream opts in — see ADR-010.
    """

    def __init__(
        self,
        detector: FaceDetector,
        min_confidence: float = 0.65,
        *,
        prefilter_enabled: bool = False,
    ) -> None:
        self._detector = detector
        self._min_confidence = min_confidence
        self._prefilter_enabled = prefilter_enabled

    @property
    def name(self) -> str:
        return "arcface"

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        if self._prefilter_enabled and not state.has_people:
            return state
        with _inference_lock:
            result = self._detector.detect_image(frame.image)
        for d in result.detections:
            if d.confidence >= self._min_confidence:
                state.findings.append(
                    AnalysisFinding(
                        bbox=d.bbox,
                        embedding=d.embedding,
                        confidence=d.confidence,
                        species="human",
                        frame_number=frame.frame_number,
                    )
                )
        return state


class PetDetectionStep(AnalysisStep):
    """Detect dogs and cats in a frame using YOLO + SigLIP.

    When ``prefilter_enabled`` is True, detection is skipped on frames where
    the caption step previously set ``state.has_animals`` to False.
    """

    def __init__(
        self,
        detector: PetDetector,
        min_confidence: float = 0.5,
        *,
        prefilter_enabled: bool = False,
    ) -> None:
        self._detector = detector
        self._min_confidence = min_confidence
        self._prefilter_enabled = prefilter_enabled

    @property
    def name(self) -> str:
        return "siglip"

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        if self._prefilter_enabled and not state.has_animals:
            return state
        with _inference_lock:
            result = self._detector.detect_image(frame.image)
        for d in result.detections:
            if d.confidence >= self._min_confidence:
                state.findings.append(
                    AnalysisFinding(
                        bbox=d.bbox,
                        embedding=d.embedding,
                        confidence=d.confidence,
                        species=d.species,
                        frame_number=frame.frame_number,
                    )
                )
        return state


class CaptionStep(AnalysisStep):
    """Generate caption and tags via VLM (no translation — see TranslationStep)."""

    def __init__(
        self,
        describer: Describer,
        vocab_hint: str | None = None,
    ) -> None:
        self._describer = describer
        self.vocab_hint = vocab_hint

    @property
    def name(self) -> str:
        return self._describer.model_id

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        # Only caption the first frame (frame 0 for photos; first frame of a video)
        if frame.frame_number != 0:
            return state

        with _inference_lock:
            output = self._describer.describe_image(frame.image, vocab_hint=self.vocab_hint)
        if not output.caption:
            return state

        state.has_people = output.has_people
        state.has_animals = output.has_animals
        state.caption = output.caption
        state.tags = output.tags
        return state


class TranslationStep(AnalysisStep):
    """Translate an already-generated caption/tags into the target language.

    Runs on CPU (MarianMT), so it does not contend with the Metal
    _inference_lock. Placing it as a separate step lets --profile attribute
    translation cost to a distinct row instead of hiding it inside the VLM
    bucket.
    """

    def __init__(self, translator: Translator) -> None:
        self._translator = translator

    @property
    def name(self) -> str:
        return self._translator.model_id

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        if frame.frame_number != 0 or not state.caption:
            return state
        state.caption, state.tags = self._translator.translate(state.caption, state.tags)
        return state


class DeduplicationStep(AnalysisStep):
    """Collapse duplicate findings across frames (same person/pet in multiple frames).

    Keeps the highest-confidence finding for each cluster of similar embeddings.
    Typically the last step before persistence.
    """

    def __init__(self, threshold: float = 0.6) -> None:
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "dedup"

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        # Dedup is a post-processing step — only run after all frames are done.
        # The pipeline calls it per-frame, but the real work happens when we
        # see the last frame. For single-frame sources (photos), that's frame 0.
        # For multi-frame, the caller should run dedup as a separate pass.
        # Here we implement per-species dedup over the accumulated findings.
        if not state.findings:
            return state

        by_species: dict[str, list[AnalysisFinding]] = {}
        for f in state.findings:
            by_species.setdefault(f.species, []).append(f)

        deduped: list[AnalysisFinding] = []
        for species_findings in by_species.values():
            deduped.extend(self._dedup_group(species_findings))

        state.findings = deduped
        return state

    def _dedup_group(self, findings: list[AnalysisFinding]) -> list[AnalysisFinding]:
        """Keep best-confidence finding per cluster of similar embeddings."""
        unique: list[AnalysisFinding] = []
        for f in findings:
            merged = False
            for i, u in enumerate(unique):
                similarity = float(f.embedding @ u.embedding)
                if similarity > self._threshold:
                    if f.confidence > u.confidence:
                        unique[i] = f
                    merged = True
                    break
            if not merged:
                unique.append(f)
        return unique
