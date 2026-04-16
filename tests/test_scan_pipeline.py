"""Tests for the ScanPipeline abstraction."""

from __future__ import annotations

from unittest import TestCase

from ritrova.scan_pipeline import ScanPipeline, ScanStats


class _CountingPipeline(ScanPipeline[int]):
    """Minimal concrete pipeline for testing the template method."""

    def __init__(self, items: list[int], done: set[int], fail: set[int]) -> None:
        # no db needed for abstract tests
        self._items = items
        self._done = done
        self._fail = fail

    def discover(self) -> list[int]:
        return self._items

    def is_already_done(self, candidate: int) -> bool:
        return candidate in self._done

    def process_one(self, candidate: int) -> int:
        if candidate in self._fail:
            return -1
        return candidate  # use value as "found count"

    @property
    def found_label(self) -> str:
        return "items"


class TestScanPipelineTemplate(TestCase):
    def test_empty_pipeline(self) -> None:
        stats = _CountingPipeline(items=[], done=set(), fail=set()).run()
        assert stats == ScanStats()

    def test_all_processed(self) -> None:
        stats = _CountingPipeline(items=[1, 2, 3], done=set(), fail=set()).run()
        assert stats.processed == 3
        assert stats.found == 6  # 1 + 2 + 3
        assert stats.skipped == 0
        assert stats.errors == 0

    def test_skips_done(self) -> None:
        stats = _CountingPipeline(items=[1, 2, 3], done={2}, fail=set()).run()
        assert stats.processed == 2
        assert stats.skipped == 1
        assert stats.found == 4  # 1 + 3

    def test_errors_counted(self) -> None:
        stats = _CountingPipeline(items=[1, 2, 3], done=set(), fail={2}).run()
        assert stats.processed == 2
        assert stats.errors == 1
        assert stats.found == 4  # 1 + 3

    def test_mixed_skip_error_success(self) -> None:
        stats = _CountingPipeline(items=[1, 2, 3, 4], done={1}, fail={3}).run()
        assert stats.skipped == 1
        assert stats.errors == 1
        assert stats.processed == 2
        assert stats.found == 6  # 2 + 4
