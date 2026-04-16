"""Composable description pipeline: stages that transform (caption, tags).

Each ``DescriptionStage`` receives the image path and the current result from
prior stages, and returns a transformed result.  The ``DescriptionPipeline``
composes stages sequentially.

This module wraps the existing ``Describer`` and ``Translator`` classes —
those remain the inference adapters; stages are the pipeline participants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DescriptionResult:
    """Intermediate/final result flowing through the pipeline."""

    caption: str = ""
    tags: set[str] = field(default_factory=set)


class DescriptionStage(ABC):
    """One step in the description pipeline."""

    @abstractmethod
    def process(self, image_path: Path, current: DescriptionResult) -> DescriptionResult:
        """Transform the current description.

        The first stage typically ignores ``current`` and generates from the image.
        Later stages transform ``current`` (e.g. translation).
        """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Identifier for the strategy field in the scan record."""


class VLMCaptionStage(DescriptionStage):
    """Stage 1: generate English caption + tags from the image via VLM."""

    def __init__(
        self,
        model_id: str | None = None,
        system_prompt: str | None = None,
        user_prompt_with_vocab: str | None = None,
        user_prompt_no_vocab: str | None = None,
    ) -> None:
        from .describer import (
            DEFAULT_SYSTEM_PROMPT,
            DEFAULT_USER_PROMPT_NO_VOCAB,
            DEFAULT_USER_PROMPT_WITH_VOCAB,
            DEFAULT_VLM_MODEL,
            Describer,
        )

        self._describer = Describer(
            model_id=model_id or DEFAULT_VLM_MODEL,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
            user_prompt_with_vocab=user_prompt_with_vocab or DEFAULT_USER_PROMPT_WITH_VOCAB,
            user_prompt_no_vocab=user_prompt_no_vocab or DEFAULT_USER_PROMPT_NO_VOCAB,
        )
        self._vocab_hint: str | None = None

    @property
    def model_id(self) -> str:
        return self._describer.model_id

    @property
    def vocab_hint(self) -> str | None:
        return self._vocab_hint

    @vocab_hint.setter
    def vocab_hint(self, value: str | None) -> None:
        self._vocab_hint = value

    def process(self, image_path: Path, current: DescriptionResult) -> DescriptionResult:
        caption, tags = self._describer.describe(image_path, vocab_hint=self._vocab_hint)
        return DescriptionResult(caption=caption, tags=tags)


class TranslationStage(DescriptionStage):
    """Stage 2: translate caption + tags from English to target language."""

    def __init__(self, model_id: str | None = None) -> None:
        from .describer import DEFAULT_TRANSLATOR_MODEL, Translator

        self._translator = Translator(model_id=model_id or DEFAULT_TRANSLATOR_MODEL)

    @property
    def model_id(self) -> str:
        return self._translator.model_id

    def process(self, image_path: Path, current: DescriptionResult) -> DescriptionResult:
        caption, tags = self._translator.translate(current.caption, current.tags)
        return DescriptionResult(caption=caption, tags=tags)


class DescriptionPipeline:
    """Composes N stages into a single callable."""

    def __init__(self, stages: list[DescriptionStage]) -> None:
        if not stages:
            msg = "DescriptionPipeline requires at least one stage"
            raise ValueError(msg)
        self.stages = stages

    def describe(self, image_path: Path) -> DescriptionResult:
        """Run all stages in sequence. Short-circuits on empty caption."""
        result = DescriptionResult()
        for stage in self.stages:
            result = stage.process(image_path, result)
            if not result.caption:
                break
        return result

    @property
    def strategy_id(self) -> str:
        """Combined model identifiers for the scan record."""
        return "+".join(s.model_id for s in self.stages)

    @property
    def vlm_stage(self) -> VLMCaptionStage | None:
        """Return the VLM stage if present (for vocab hint management)."""
        for stage in self.stages:
            if isinstance(stage, VLMCaptionStage):
                return stage
        return None
