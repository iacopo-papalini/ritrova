"""Tests for FEAT-8: VLM-generated scene descriptions and tags."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import pytest

from ritrova.db import FaceDB
from ritrova.describer import _parse_vlm_response


class TestDescriptionsDB(TestCase):
    """DB-level tests for the descriptions table and helpers."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.source_id = self.db.add_source("/photo1.jpg", width=100, height=100)
        self.scan_id = self.db.record_scan(
            self.source_id, "describe", detection_strategy="qwen2.5-vl-3b"
        )

    def test_add_and_get_description(self) -> None:
        desc_id = self.db.add_description(
            self.source_id,
            self.scan_id,
            caption="Un gruppo di persone a tavola in giardino",
            tags={"giardino", "cena", "gruppo", "estate"},
        )
        assert desc_id > 0

        desc = self.db.get_description(self.source_id)
        assert desc is not None
        assert desc.source_id == self.source_id
        assert desc.scan_id == self.scan_id
        assert desc.caption == "Un gruppo di persone a tavola in giardino"
        assert desc.tags == {"giardino", "cena", "gruppo", "estate"}

    def test_get_description_returns_none_for_missing(self) -> None:
        assert self.db.get_description(999) is None

    def test_get_description_returns_most_recent(self) -> None:
        """When multiple descriptions exist for a source, return the latest."""
        self.db.add_description(self.source_id, self.scan_id, caption="Old caption", tags={"old"})
        scan2 = self.db.record_scan(
            self.source_id, "describe_v2", detection_strategy="qwen2.5-vl-7b"
        )
        self.db.add_description(self.source_id, scan2, caption="New caption", tags={"new"})
        desc = self.db.get_description(self.source_id)
        assert desc is not None
        assert desc.caption == "New caption"
        assert desc.tags == {"new"}

    def test_get_all_tags(self) -> None:
        self.db.add_description(
            self.source_id,
            self.scan_id,
            caption="test",
            tags={"mare", "estate", "spiaggia"},
        )
        src2 = self.db.add_source("/photo2.jpg", width=100, height=100)
        scan2 = self.db.record_scan(src2, "describe", detection_strategy="qwen2.5-vl-3b")
        self.db.add_description(src2, scan2, caption="test", tags={"montagna", "estate", "neve"})
        tags = self.db.get_all_tags()
        assert tags == {"estate", "mare", "montagna", "neve", "spiaggia"}

    def test_get_all_tags_empty(self) -> None:
        assert self.db.get_all_tags() == set()

    def test_search_by_single_tag(self) -> None:
        self.db.add_description(
            self.source_id, self.scan_id, caption="test", tags={"mare", "estate"}
        )
        src2 = self.db.add_source("/photo2.jpg", width=100, height=100)
        scan2 = self.db.record_scan(src2, "describe", detection_strategy="qwen2.5-vl-3b")
        self.db.add_description(src2, scan2, caption="test", tags={"montagna", "neve"})
        results = self.db.search_sources_by_tags_and_caption(tag_filters={"mare"})
        assert results == [self.source_id]

    def test_search_by_multiple_tags_is_and(self) -> None:
        """Multiple tags must ALL match (AND semantics)."""
        self.db.add_description(
            self.source_id,
            self.scan_id,
            caption="test",
            tags={"mare", "estate", "cena"},
        )
        # mare AND estate -> match
        assert self.db.search_sources_by_tags_and_caption(tag_filters={"mare", "estate"}) == [
            self.source_id
        ]
        # mare AND neve -> no match
        assert self.db.search_sources_by_tags_and_caption(tag_filters={"mare", "neve"}) == []

    def test_search_by_caption_keyword(self) -> None:
        self.db.add_description(
            self.source_id,
            self.scan_id,
            caption="Un gruppo di persone a tavola in giardino",
            tags={"giardino"},
        )
        results = self.db.search_sources_by_tags_and_caption(caption_query="giardino")
        assert results == [self.source_id]
        assert self.db.search_sources_by_tags_and_caption(caption_query="spiaggia") == []

    def test_search_combined_tags_and_caption(self) -> None:
        self.db.add_description(
            self.source_id,
            self.scan_id,
            caption="Cena estiva in giardino",
            tags={"giardino", "cena", "estate"},
        )
        # Both match -> returned
        assert self.db.search_sources_by_tags_and_caption(
            tag_filters={"cena"}, caption_query="giardino"
        ) == [self.source_id]
        # Tag matches but caption doesn't -> not returned
        assert (
            self.db.search_sources_by_tags_and_caption(
                tag_filters={"cena"}, caption_query="spiaggia"
            )
            == []
        )

    def test_search_no_filters_returns_empty(self) -> None:
        self.db.add_description(self.source_id, self.scan_id, caption="test", tags={"test"})
        assert self.db.search_sources_by_tags_and_caption() == []

    def test_tag_exact_match_no_substring(self) -> None:
        """Searching for 'mare' must not match 'tramonto' or 'maremma'."""
        self.db.add_description(
            self.source_id,
            self.scan_id,
            caption="test",
            tags={"tramonto", "maremma"},
        )
        assert self.db.search_sources_by_tags_and_caption(tag_filters={"mare"}) == []

    def test_get_undescribed_source_ids(self) -> None:
        src2 = self.db.add_source("/photo2.jpg", width=100, height=100)
        undescribed = self.db.get_undescribed_source_ids()
        assert self.source_id in undescribed
        assert src2 in undescribed

        self.db.add_description(self.source_id, self.scan_id, caption="test", tags={"test"})
        undescribed = self.db.get_undescribed_source_ids()
        assert self.source_id not in undescribed
        assert src2 in undescribed

    def test_description_cascade_on_scan_delete(self) -> None:
        """Deleting a scan must cascade-delete its descriptions."""
        self.db.add_description(self.source_id, self.scan_id, caption="test", tags={"test"})
        assert self.db.get_description(self.source_id) is not None
        self.db.delete_scan(self.scan_id)
        assert self.db.get_description(self.source_id) is None

    def test_description_cascade_on_source_delete(self) -> None:
        """Deleting a source must cascade-delete its descriptions."""
        self.db.add_description(self.source_id, self.scan_id, caption="test", tags={"test"})
        self.db.conn.execute("DELETE FROM sources WHERE id = ?", (self.source_id,))
        self.db.conn.commit()
        assert self.db.get_description(self.source_id) is None

    def test_tags_roundtrip_preserves_content(self) -> None:
        """Tags with special characters survive encode/decode."""
        original = {"san valentino", "mare", "2024"}
        self.db.add_description(self.source_id, self.scan_id, caption="test", tags=original)
        desc = self.db.get_description(self.source_id)
        assert desc is not None
        assert desc.tags == original

    def test_empty_tags_set(self) -> None:
        self.db.add_description(self.source_id, self.scan_id, caption="No tags", tags=set())
        desc = self.db.get_description(self.source_id)
        assert desc is not None
        assert desc.tags == set()


class TestDescriberParsing(TestCase):
    """Tests for VLM response parsing (no model needed)."""

    def test_parse_clean_json(self) -> None:
        text = '{"caption": "Un gruppo a tavola", "tags": ["cena", "giardino"]}'
        caption, tags = _parse_vlm_response(text)
        assert caption == "Un gruppo a tavola"
        assert tags == {"cena", "giardino"}

    def test_parse_json_with_markdown_fences(self) -> None:
        text = '```json\n{"caption": "Spiaggia", "tags": ["mare", "estate"]}\n```'
        caption, tags = _parse_vlm_response(text)
        assert caption == "Spiaggia"
        assert tags == {"mare", "estate"}

    def test_parse_tags_lowercased(self) -> None:
        text = '{"caption": "Test", "tags": ["Mare", "ESTATE", "Cena"]}'
        caption, tags = _parse_vlm_response(text)
        assert tags == {"mare", "estate", "cena"}

    def test_parse_empty_tags(self) -> None:
        text = '{"caption": "Niente di speciale", "tags": []}'
        caption, tags = _parse_vlm_response(text)
        assert caption == "Niente di speciale"
        assert tags == set()

    def test_parse_invalid_json_falls_back(self) -> None:
        text = "This is not JSON at all"
        caption, tags = _parse_vlm_response(text)
        assert caption == "This is not JSON at all"
        assert tags == set()

    def test_parse_strips_whitespace(self) -> None:
        text = '  {"caption": "  Ciao  ", "tags": ["  mare  ", "neve"]}  '
        caption, tags = _parse_vlm_response(text)
        assert caption == "Ciao"
        assert tags == {"mare", "neve"}

    def test_parse_missing_caption_key(self) -> None:
        text = '{"tags": ["mare"]}'
        caption, tags = _parse_vlm_response(text)
        assert caption == ""
        assert tags == {"mare"}

    def test_parse_missing_tags_key(self) -> None:
        text = '{"caption": "Una foto"}'
        caption, tags = _parse_vlm_response(text)
        assert caption == "Una foto"
        assert tags == set()

    def test_parse_multiword_tags_kept_in_json(self) -> None:
        """JSON format: multi-word tags are kept intact (lowercased)."""
        text = '{"caption": "Test", "tags": ["bird of prey", "stone wall"]}'
        _, tags = _parse_vlm_response(text)
        assert tags == {"bird of prey", "stone wall"}

    def test_parse_line_based_format(self) -> None:
        """Line-based format: first line is caption, rest are single-word tags."""
        text = "A dog sits on a beach at sunset.\ndog\nbeach\nsunset\nsand"
        caption, tags = _parse_vlm_response(text)
        assert caption == "A dog sits on a beach at sunset."
        assert tags == {"dog", "beach", "sunset", "sand"}

    def test_parse_line_based_skips_multiword_lines(self) -> None:
        """Line-based: lines with multiple words after the caption are ignored."""
        text = "A man holds a bird.\neagle\nthis is not a tag\ncrowd"
        caption, tags = _parse_vlm_response(text)
        assert caption == "A man holds a bird."
        assert tags == {"eagle", "crowd"}

    def test_parse_line_based_strips_dots(self) -> None:
        text = "A photo.\ndog.\ncat."
        caption, tags = _parse_vlm_response(text)
        assert caption == "A photo."
        assert tags == {"dog", "cat"}

    def test_parse_prefers_json_over_lines(self) -> None:
        """If the output is valid JSON, use that even if it has newlines."""
        text = '{\n"caption": "Test",\n"tags": ["a", "b"]\n}'
        caption, tags = _parse_vlm_response(text)
        assert caption == "Test"
        assert tags == {"a", "b"}
