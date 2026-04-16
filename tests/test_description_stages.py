"""Tests for the composable description pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import pytest

from ritrova.description_stages import (
    DescriptionPipeline,
    DescriptionResult,
    DescriptionStage,
)


class _FakeVLM(DescriptionStage):
    """Stub that returns a fixed caption + tags without loading any model."""

    def __init__(self, caption: str = "A dog in a park", tags: set[str] | None = None) -> None:
        self._caption = caption
        self._tags = tags or {"dog", "park"}

    @property
    def model_id(self) -> str:
        return "fake-vlm"

    def process(self, image_path: Path, current: DescriptionResult) -> DescriptionResult:
        return DescriptionResult(caption=self._caption, tags=self._tags)


class _FakeTranslator(DescriptionStage):
    """Stub translator that uppercases caption and prefixes tags with 'it_'."""

    @property
    def model_id(self) -> str:
        return "fake-translator"

    def process(self, image_path: Path, current: DescriptionResult) -> DescriptionResult:
        return DescriptionResult(
            caption=current.caption.upper(),
            tags={f"it_{t}" for t in current.tags},
        )


class _FailingVLM(DescriptionStage):
    """Stub that returns empty caption (simulating unreadable image)."""

    @property
    def model_id(self) -> str:
        return "fail-vlm"

    def process(self, image_path: Path, current: DescriptionResult) -> DescriptionResult:
        return DescriptionResult(caption="", tags=set())


class TestDescriptionPipeline(TestCase):
    def test_single_stage(self) -> None:
        pipeline = DescriptionPipeline([_FakeVLM()])
        result = pipeline.describe(Path("/fake.jpg"))
        assert result.caption == "A dog in a park"
        assert result.tags == {"dog", "park"}

    def test_two_stage_composition(self) -> None:
        pipeline = DescriptionPipeline([_FakeVLM(), _FakeTranslator()])
        result = pipeline.describe(Path("/fake.jpg"))
        assert result.caption == "A DOG IN A PARK"
        assert result.tags == {"it_dog", "it_park"}

    def test_short_circuits_on_empty_caption(self) -> None:
        pipeline = DescriptionPipeline([_FailingVLM(), _FakeTranslator()])
        result = pipeline.describe(Path("/fake.jpg"))
        assert result.caption == ""
        assert result.tags == set()

    def test_strategy_id_joins_model_ids(self) -> None:
        pipeline = DescriptionPipeline([_FakeVLM(), _FakeTranslator()])
        assert pipeline.strategy_id == "fake-vlm+fake-translator"

    def test_strategy_id_single(self) -> None:
        pipeline = DescriptionPipeline([_FakeVLM()])
        assert pipeline.strategy_id == "fake-vlm"

    def test_requires_at_least_one_stage(self) -> None:
        with pytest.raises(ValueError, match="at least one stage"):
            DescriptionPipeline([])

    def test_vlm_stage_accessor(self) -> None:

        vlm = _FakeVLM()
        pipeline = DescriptionPipeline([vlm, _FakeTranslator()])
        # _FakeVLM is not a VLMCaptionStage, so should be None
        assert pipeline.vlm_stage is None

    def test_result_defaults(self) -> None:
        result = DescriptionResult()
        assert result.caption == ""
        assert result.tags == set()
