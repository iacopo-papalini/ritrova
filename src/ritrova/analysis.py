"""Unified source analysis pipeline (ADR-009).

A single pipeline iterates sources once, applying N composable analysis steps.
Each step receives a ``FrameRef`` (one image from the source) and the
accumulated ``SourceAnalysis``, and returns an enriched ``SourceAnalysis``.

The pipeline never touches the database — persistence is a separate concern
handled by ``AnalysisPersister``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
from PIL import Image, ImageFile, ImageOps

if TYPE_CHECKING:
    from .db import FaceDB

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


# ── Value objects ────────────────────────────────────────────────────────


@dataclass
class FrameRef:
    """A single frame from a source (photo = 1 frame, video = many)."""

    source_path: Path
    frame_number: int  # 0 for photos
    image: Image.Image  # already loaded, EXIF-corrected, RGB
    width: int
    height: int


@dataclass
class AnalysisFinding:
    """A detection not yet persisted — no DB ids."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    embedding: np.ndarray
    confidence: float
    species: str = "human"
    frame_number: int = 0


@dataclass
class SourceAnalysis:
    """Accumulated knowledge about a source, flowing through the pipeline."""

    source_path: str  # relative path (DB key)
    source_type: str  # "photo" | "video"
    width: int = 0
    height: int = 0
    taken_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    findings: list[AnalysisFinding] = field(default_factory=list)
    caption: str = ""
    tags: set[str] = field(default_factory=set)
    # Caption pre-filter flags. Default True so that, when the caption step
    # never ran or parsing failed, every detection step still runs — accuracy
    # trumps the speedup and missed findings cannot be recovered later.
    has_people: bool = True
    has_animals: bool = True
    # Video frame cache: frame_number → PIL Image for frames that produced findings.
    # The persister saves these as JPEGs and stores the path as frame_path.
    # Empty for photo sources (frame 0 = the source file itself).
    frame_images: dict[int, Image.Image] = field(default_factory=dict)


# ── Step ABC ─────────────────────────────────────────────────────────────


class AnalysisStep(ABC):
    """One stage in the analysis pipeline.

    Receives a frame and the current accumulated state, returns the
    (possibly enriched) state.  Steps must not touch the database.
    """

    @abstractmethod
    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        """Process one frame and return the updated analysis state."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable step name (for logging and strategy ids)."""


# ── Frame iterators ──────────────────────────────────────────────────────


def photo_frames(source_path: Path) -> Iterator[FrameRef]:
    """Yield a single FrameRef for a still image."""
    try:
        raw = Image.open(source_path)
        img = ImageOps.exif_transpose(raw)
        if img is None:
            img = raw
        img = img.convert("RGB")
    except OSError:
        logger.warning("Could not read image: %s", source_path)
        return
    w, h = img.size
    yield FrameRef(
        source_path=source_path,
        frame_number=0,
        image=img,
        width=w,
        height=h,
    )


def video_frames(source_path: Path, interval_sec: float = 5.0) -> Iterator[FrameRef]:
    """Yield one ``FrameRef`` per sampled frame from a video.

    Samples every ``interval_sec`` seconds.  Each yielded frame is an RGB
    PIL Image with its original frame index as ``frame_number``.
    """
    import cv2

    cap = cv2.VideoCapture(str(source_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.warning("Could not read video (fps=0): %s", source_path)
        cap.release()
        return

    frame_interval = max(1, int(fps * interval_sec))
    frame_idx = 0

    try:
        while True:
            ret, bgr_frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                h, w = bgr_frame.shape[:2]
                rgb = bgr_frame[:, :, ::-1]  # BGR → RGB
                img = Image.fromarray(rgb)
                yield FrameRef(
                    source_path=source_path,
                    frame_number=frame_idx,
                    image=img,
                    width=w,
                    height=h,
                )
            frame_idx += 1
    finally:
        cap.release()


# ── Pipeline ─────────────────────────────────────────────────────────────


class AnalysisPipeline:
    """Iterates frames from a source, runs all steps on each frame.

    Returns the accumulated ``SourceAnalysis`` for the entire source.
    """

    def __init__(
        self,
        steps: list[AnalysisStep],
        frame_iterator: type[Iterator[FrameRef]] | None = None,
    ) -> None:
        if not steps:
            msg = "AnalysisPipeline requires at least one step"
            raise ValueError(msg)
        self.steps = steps
        self._frame_iterator = frame_iterator

    def analyse_source(
        self,
        source_path: Path,
        source_type: str = "photo",
        *,
        frames: Iterator[FrameRef] | None = None,
        initial_state: SourceAnalysis | None = None,
        step_times: dict[str, float] | None = None,
    ) -> SourceAnalysis:
        """Run the full pipeline on a source.

        If ``frames`` is not provided, uses ``photo_frames`` for photos.
        If ``initial_state`` is provided, enriches it; otherwise creates a new one.
        If ``step_times`` is provided, per-step elapsed wall time (seconds,
        summed across frames) is accumulated into it keyed by step name.
        """
        if frames is None:
            frames = photo_frames(source_path)

        state = initial_state or SourceAnalysis(
            source_path=str(source_path),
            source_type=source_type,
        )

        for frame_count, frame in enumerate(frames):
            if frame_count == 0:
                state.width = frame.width
                state.height = frame.height
            findings_before = len(state.findings)
            if step_times is None:
                for step in self.steps:
                    state = step.analyse(frame, state)
            else:
                import time

                for step in self.steps:
                    t0 = time.monotonic()
                    state = step.analyse(frame, state)
                    step_times[step.name] = step_times.get(step.name, 0.0) + time.monotonic() - t0
            # Cache the frame image if new findings were produced on a video frame
            if (
                len(state.findings) > findings_before
                and frame.frame_number > 0
                and frame.frame_number not in state.frame_images
            ):
                state.frame_images[frame.frame_number] = frame.image

        return state

    @property
    def strategy_id(self) -> str:
        """Combined step names for the scan record."""
        return "+".join(s.name for s in self.steps)


# ── Builder ──────────────────────────────────────────────────────────────


class AnalysisPipelineBuilder:
    """Fluent builder for composing an analysis pipeline."""

    def __init__(self) -> None:
        self._steps: list[AnalysisStep] = []

    def add_step(self, step: AnalysisStep) -> Self:
        """Add an arbitrary analysis step."""
        self._steps.append(step)
        return self

    def build(self) -> AnalysisPipeline:
        """Build the pipeline. Raises ValueError if no steps were added."""
        return AnalysisPipeline(steps=list(self._steps))


# ── Persistence ──────────────────────────────────────────────────────────


class AnalysisPersister:
    """Flush a ``SourceAnalysis`` to the database.

    Not an ``AnalysisStep`` — runs after the pipeline completes.
    Handles source creation, scan recording, finding insertion,
    frame cache saving (for videos), and description storage.
    """

    def __init__(self, db: FaceDB, frames_dir: Path | None = None) -> None:
        self._db = db
        self._frames_dir = frames_dir

    def persist(
        self,
        analysis: SourceAnalysis,
        strategy_id: str,
        scan_type: str = "composite",
    ) -> None:
        """Write analysis results to the database.

        Creates or updates the source row, records a scan, inserts findings
        grouped by species, and stores the description if present.
        For video findings, saves cached frame JPEGs to ``frames_dir``.
        """
        source_id = self._db.get_or_create_source(
            analysis.source_path,
            source_type=analysis.source_type,
            width=analysis.width,
            height=analysis.height,
            taken_at=analysis.taken_at,
            latitude=analysis.latitude,
            longitude=analysis.longitude,
        )
        scan_id = self._db.record_scan(source_id, scan_type, detection_strategy=strategy_id)

        # Save frame cache JPEGs for video findings
        frame_paths: dict[int, str] = {}
        if analysis.source_type == "video" and analysis.frame_images and self._frames_dir:
            self._frames_dir.mkdir(parents=True, exist_ok=True)
            import hashlib

            vid_hash = hashlib.md5(analysis.source_path.encode()).hexdigest()[:10]  # noqa: S324
            for frame_number, img in analysis.frame_images.items():
                frame_file = self._frames_dir / f"vid_{vid_hash}_{frame_number}.jpg"
                img.save(str(frame_file), "JPEG", quality=85)
                frame_paths[frame_number] = str(frame_file.relative_to(self._db.db_path.parent))

        # Group findings by (species, frame_number) for batch insert
        by_key: dict[
            tuple[str, int], list[tuple[int, tuple[int, int, int, int], np.ndarray, float]]
        ] = {}
        for f in analysis.findings:
            key = (f.species, f.frame_number)
            by_key.setdefault(key, []).append((source_id, f.bbox, f.embedding, f.confidence))

        for (species, frame_number), batch in by_key.items():
            self._db.add_findings_batch(
                batch,
                scan_id=scan_id,
                species=species,
                frame_number=frame_number,
                frame_path=frame_paths.get(frame_number),
            )

        if analysis.caption:
            self._db.add_description(source_id, scan_id, analysis.caption, analysis.tags)
