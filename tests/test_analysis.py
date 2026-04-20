"""Tests for the unified analysis pipeline (ADR-009 Phase A)."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from PIL import Image

from ritrova.analysis import (
    AnalysisFinding,
    AnalysisPersister,
    AnalysisPipeline,
    AnalysisPipelineBuilder,
    AnalysisStep,
    FrameRef,
    SourceAnalysis,
    photo_frames,
    video_frames,
)
from ritrova.db import FaceDB


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_frame(w: int = 200, h: int = 200, frame_number: int = 0) -> FrameRef:
    return FrameRef(
        source_path=Path("/fake.jpg"),
        frame_number=frame_number,
        image=Image.new("RGB", (w, h)),
        width=w,
        height=h,
    )


# ── Fake steps for testing ───────────────────────────────────────────────


class _AddFindingStep(AnalysisStep):
    """Appends one fake finding per frame."""

    def __init__(self, species: str = "human") -> None:
        self._species = species

    @property
    def name(self) -> str:
        return f"fake-{self._species}"

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        state.findings.append(
            AnalysisFinding(
                bbox=(10, 10, 50, 50),
                embedding=_emb(frame.frame_number),
                confidence=0.95,
                species=self._species,
                frame_number=frame.frame_number,
            )
        )
        return state


class _SetCaptionStep(AnalysisStep):
    """Sets a fixed caption and tags."""

    @property
    def name(self) -> str:
        return "fake-caption"

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        state.caption = "A dog in a park"
        state.tags = {"dog", "park"}
        return state


class _RemoveSpeciesStep(AnalysisStep):
    """Removes all findings of a given species (filter step)."""

    def __init__(self, species: str) -> None:
        self._species = species

    @property
    def name(self) -> str:
        return f"remove-{self._species}"

    def analyse(self, frame: FrameRef, state: SourceAnalysis) -> SourceAnalysis:
        state.findings = [f for f in state.findings if f.species != self._species]
        return state


# ── SourceAnalysis tests ─────────────────────────────────────────────────


class TestSourceAnalysis(TestCase):
    def test_defaults(self) -> None:
        sa = SourceAnalysis(source_path="/photo.jpg", source_type="photo")
        assert sa.findings == []
        assert sa.caption == ""
        assert sa.tags == set()
        assert sa.width == 0

    def test_findings_are_independent(self) -> None:
        """Two SourceAnalysis instances don't share findings lists."""
        a = SourceAnalysis(source_path="/a.jpg", source_type="photo")
        b = SourceAnalysis(source_path="/b.jpg", source_type="photo")
        a.findings.append(AnalysisFinding(bbox=(0, 0, 1, 1), embedding=_emb(), confidence=0.5))
        assert b.findings == []


# ── Pipeline tests ───────────────────────────────────────────────────────


class TestAnalysisPipeline(TestCase):
    def test_single_step_single_frame(self) -> None:
        pipeline = AnalysisPipeline(steps=[_AddFindingStep()])
        frames = iter([_make_frame()])
        result = pipeline.analyse_source(Path("/fake.jpg"), frames=frames)
        assert len(result.findings) == 1
        assert result.findings[0].species == "human"
        assert result.width == 200

    def test_multiple_steps_compose(self) -> None:
        pipeline = AnalysisPipeline(
            steps=[_AddFindingStep("human"), _AddFindingStep("dog"), _SetCaptionStep()]
        )
        frames = iter([_make_frame()])
        result = pipeline.analyse_source(Path("/fake.jpg"), frames=frames)
        assert len(result.findings) == 2
        assert {f.species for f in result.findings} == {"human", "dog"}
        assert result.caption == "A dog in a park"

    def test_filter_step_removes_findings(self) -> None:
        pipeline = AnalysisPipeline(
            steps=[
                _AddFindingStep("human"),
                _AddFindingStep("dog"),
                _RemoveSpeciesStep("human"),
            ]
        )
        frames = iter([_make_frame()])
        result = pipeline.analyse_source(Path("/fake.jpg"), frames=frames)
        assert len(result.findings) == 1
        assert result.findings[0].species == "dog"

    def test_multiple_frames_accumulate(self) -> None:
        pipeline = AnalysisPipeline(steps=[_AddFindingStep()])
        frames = iter([_make_frame(frame_number=0), _make_frame(frame_number=1)])
        result = pipeline.analyse_source(Path("/fake.mp4"), source_type="video", frames=frames)
        assert len(result.findings) == 2
        assert result.findings[0].frame_number == 0
        assert result.findings[1].frame_number == 1

    def test_empty_frames_returns_empty_state(self) -> None:
        pipeline = AnalysisPipeline(steps=[_AddFindingStep()])
        frames: list[FrameRef] = []
        result = pipeline.analyse_source(Path("/fake.jpg"), frames=iter(frames))
        assert result.findings == []
        assert result.width == 0

    def test_initial_state_is_enriched(self) -> None:
        pipeline = AnalysisPipeline(steps=[_AddFindingStep()])
        existing = SourceAnalysis(
            source_path="/photo.jpg",
            source_type="photo",
            caption="Existing caption",
        )
        result = pipeline.analyse_source(
            Path("/photo.jpg"), frames=iter([_make_frame()]), initial_state=existing
        )
        assert result.caption == "Existing caption"
        assert len(result.findings) == 1

    def test_strategy_id(self) -> None:
        pipeline = AnalysisPipeline(steps=[_AddFindingStep("human"), _SetCaptionStep()])
        assert pipeline.strategy_id == "fake-human+fake-caption"

    def test_requires_at_least_one_step(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            AnalysisPipeline(steps=[])


# ── Builder tests ────────────────────────────────────────────────────────


class TestAnalysisPipelineBuilder(TestCase):
    def test_builds_pipeline_with_steps(self) -> None:
        pipeline = (
            AnalysisPipelineBuilder()
            .add_step(_AddFindingStep())
            .add_step(_SetCaptionStep())
            .build()
        )
        assert len(pipeline.steps) == 2

    def test_empty_builder_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one step"):
            AnalysisPipelineBuilder().build()

    def test_builder_copies_steps(self) -> None:
        """Building doesn't share the internal list."""
        builder = AnalysisPipelineBuilder().add_step(_AddFindingStep())
        p1 = builder.build()
        builder.add_step(_SetCaptionStep())
        p2 = builder.build()
        assert len(p1.steps) == 1
        assert len(p2.steps) == 2


# ── Frame iterator tests ────────────────────────────────────────────────


class TestPhotoFrames(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp = tmp_path

    def test_yields_single_frame_for_jpeg(self) -> None:
        path = self.tmp / "photo.jpg"
        Image.new("RGB", (100, 80)).save(str(path), "JPEG")
        frames = list(photo_frames(path))
        assert len(frames) == 1
        assert frames[0].frame_number == 0
        assert frames[0].width == 100
        assert frames[0].height == 80
        assert frames[0].image.mode == "RGB"

    def test_yields_nothing_for_unreadable(self) -> None:
        path = self.tmp / "bad.jpg"
        path.write_bytes(b"not a jpeg")
        frames = list(photo_frames(path))
        assert frames == []

    def test_yields_nothing_for_missing(self) -> None:
        frames = list(photo_frames(self.tmp / "nonexistent.jpg"))
        assert frames == []


# ── Persister tests ──────────────────────────────────────────────────────


class TestAnalysisPersister(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.persister = AnalysisPersister(self.db)

    def test_persists_findings(self) -> None:
        analysis = SourceAnalysis(
            source_path="/photo.jpg",
            source_type="photo",
            width=200,
            height=200,
            findings=[
                AnalysisFinding(
                    bbox=(10, 10, 50, 50),
                    embedding=_emb(1),
                    confidence=0.95,
                ),
            ],
        )
        self.persister.persist(analysis, strategy_id="arcface")
        assert self.db.get_finding_count() == 1
        assert self.db.get_source_count() == 1

    def test_persists_description(self) -> None:
        analysis = SourceAnalysis(
            source_path="/photo.jpg",
            source_type="photo",
            width=200,
            height=200,
            caption="A dog in a park",
            tags={"dog", "park"},
        )
        self.persister.persist(analysis, strategy_id="vlm+translator")
        desc = self.db.get_description(
            self.db.get_source_by_path("/photo.jpg").id  # type: ignore[union-attr]
        )
        assert desc is not None
        assert desc.caption == "A dog in a park"
        assert desc.tags == {"dog", "park"}

    def test_persists_mixed_species(self) -> None:
        analysis = SourceAnalysis(
            source_path="/photo.jpg",
            source_type="photo",
            width=200,
            height=200,
            findings=[
                AnalysisFinding(
                    bbox=(10, 10, 50, 50),
                    embedding=_emb(1),
                    confidence=0.95,
                    species="human",
                ),
                AnalysisFinding(
                    bbox=(80, 80, 40, 40),
                    embedding=_emb(2, dim=768),
                    confidence=0.9,
                    species="dog",
                ),
            ],
        )
        self.persister.persist(analysis, strategy_id="composite")
        assert self.db.get_finding_count() == 2
        human_embs = self.db.get_all_embeddings(species="human")
        dog_embs = self.db.get_all_embeddings(species="dog")
        assert len(human_embs) == 1
        assert len(dog_embs) == 1

    def test_empty_analysis_creates_source_and_scan(self) -> None:
        analysis = SourceAnalysis(
            source_path="/empty.jpg",
            source_type="photo",
            width=100,
            height=100,
        )
        self.persister.persist(analysis, strategy_id="test")
        assert self.db.get_source_by_path("/empty.jpg") is not None
        assert self.db.get_finding_count() == 0

    def test_scan_type_defaults_to_subjects(self) -> None:
        analysis = SourceAnalysis(
            source_path="/photo.jpg",
            source_type="photo",
            width=200,
            height=200,
        )
        self.persister.persist(analysis, strategy_id="test")
        scans = self.db.find_scans()
        assert len(scans) == 1
        assert scans[0]["scan_type"] == "subjects"

    def test_persists_frame_number(self) -> None:
        analysis = SourceAnalysis(
            source_path="/video.mp4",
            source_type="video",
            width=640,
            height=480,
            findings=[
                AnalysisFinding(
                    bbox=(10, 10, 50, 50),
                    embedding=_emb(1),
                    confidence=0.95,
                    frame_number=30,
                ),
            ],
        )
        self.persister.persist(analysis, strategy_id="composite")
        source = self.db.get_source_by_path("/video.mp4")
        assert source is not None
        findings = self.db.get_source_findings(source.id)
        assert len(findings) == 1
        assert findings[0].frame_number == 30

    def test_saves_video_frame_cache(self) -> None:
        """Persister saves frame images as JPEGs for video findings."""
        frames_dir = self.db.db_path.parent / "tmp" / "frames"
        persister = AnalysisPersister(self.db, frames_dir=frames_dir)

        frame_img = Image.new("RGB", (640, 480), color="red")
        analysis = SourceAnalysis(
            source_path="/video.mp4",
            source_type="video",
            width=640,
            height=480,
            findings=[
                AnalysisFinding(
                    bbox=(10, 10, 50, 50),
                    embedding=_emb(1),
                    confidence=0.95,
                    frame_number=60,
                ),
            ],
            frame_images={60: frame_img},
        )
        persister.persist(analysis, strategy_id="composite")

        # Frame JPEG should exist
        assert frames_dir.exists()
        jpgs = list(frames_dir.glob("vid_*.jpg"))
        assert len(jpgs) == 1
        assert "60" in jpgs[0].name

        # Finding should have frame_path set
        source = self.db.get_source_by_path("/video.mp4")
        assert source is not None
        findings = self.db.get_source_findings(source.id)
        assert len(findings) == 1
        assert findings[0].frame_path is not None
        assert "60" in findings[0].frame_path


# ── Video frames tests ───────────────────────────────────────────────────


class TestVideoFrames(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.tmp = tmp_path

    def test_yields_nothing_for_missing_file(self) -> None:
        frames = list(video_frames(Path("/nonexistent.mp4")))
        assert frames == []

    def test_yields_nothing_for_invalid_file(self) -> None:
        bad = self.tmp / "bad.mp4"
        bad.write_bytes(b"not a video")
        frames = list(video_frames(bad))
        assert frames == []
