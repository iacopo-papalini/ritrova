"""Tests for FEAT-8: VLM-generated scene descriptions and tags."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import pytest

from ritrova.db import FaceDB
from ritrova.describer import Describer, _parse_vlm_response


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
        out = _parse_vlm_response(text)
        assert out.caption == "Un gruppo a tavola"
        assert out.tags == {"cena", "giardino"}

    def test_parse_json_with_markdown_fences(self) -> None:
        text = '```json\n{"caption": "Spiaggia", "tags": ["mare", "estate"]}\n```'
        out = _parse_vlm_response(text)
        assert out.caption == "Spiaggia"
        assert out.tags == {"mare", "estate"}

    def test_parse_tags_lowercased(self) -> None:
        text = '{"caption": "Test", "tags": ["Mare", "ESTATE", "Cena"]}'
        out = _parse_vlm_response(text)
        assert out.tags == {"mare", "estate", "cena"}

    def test_parse_empty_tags(self) -> None:
        text = '{"caption": "Niente di speciale", "tags": []}'
        out = _parse_vlm_response(text)
        assert out.caption == "Niente di speciale"
        assert out.tags == set()

    def test_parse_invalid_json_falls_back(self) -> None:
        text = "This is not JSON at all"
        out = _parse_vlm_response(text)
        assert out.caption == "This is not JSON at all"
        assert out.tags == set()

    def test_parse_strips_whitespace(self) -> None:
        text = '  {"caption": "  Ciao  ", "tags": ["  mare  ", "neve"]}  '
        out = _parse_vlm_response(text)
        assert out.caption == "Ciao"
        assert out.tags == {"mare", "neve"}

    def test_parse_missing_caption_key(self) -> None:
        text = '{"tags": ["mare"]}'
        out = _parse_vlm_response(text)
        assert out.caption == ""
        assert out.tags == {"mare"}

    def test_parse_missing_tags_key(self) -> None:
        text = '{"caption": "Una foto"}'
        out = _parse_vlm_response(text)
        assert out.caption == "Una foto"
        assert out.tags == set()

    def test_parse_multiword_tags_kept_in_json(self) -> None:
        """JSON format: multi-word tags are kept intact (lowercased)."""
        text = '{"caption": "Test", "tags": ["bird of prey", "stone wall"]}'
        out = _parse_vlm_response(text)
        assert out.tags == {"bird of prey", "stone wall"}

    def test_parse_line_based_format(self) -> None:
        """Line-based format: first line is caption, rest are single-word tags."""
        text = "A dog sits on a beach at sunset.\ndog\nbeach\nsunset\nsand"
        out = _parse_vlm_response(text)
        assert out.caption == "A dog sits on a beach at sunset."
        assert out.tags == {"dog", "beach", "sunset", "sand"}

    def test_parse_line_based_skips_multiword_lines(self) -> None:
        """Line-based: lines with multiple words after the caption are ignored."""
        text = "A man holds a bird.\neagle\nthis is not a tag\ncrowd"
        out = _parse_vlm_response(text)
        assert out.caption == "A man holds a bird."
        assert out.tags == {"eagle", "crowd"}

    def test_parse_line_based_strips_dots(self) -> None:
        text = "A photo.\ndog.\ncat."
        out = _parse_vlm_response(text)
        assert out.caption == "A photo."
        assert out.tags == {"dog", "cat"}

    def test_parse_prefers_json_over_lines(self) -> None:
        """If the output is valid JSON, use that even if it has newlines."""
        text = '{\n"caption": "Test",\n"tags": ["a", "b"]\n}'
        out = _parse_vlm_response(text)
        assert out.caption == "Test"
        assert out.tags == {"a", "b"}

    # ── Prefilter booleans (ADR-010 step 2a) ─────────────────────────

    def test_parse_reads_subject_booleans(self) -> None:
        text = '{"caption": "A cat", "tags": ["cat"], "has_people": false, "has_animals": true}'
        out = _parse_vlm_response(text)
        assert out.has_people is False
        assert out.has_animals is True

    def test_parse_subject_booleans_default_true_when_missing(self) -> None:
        """Accuracy-first: unknown fields fail open so detection still runs."""
        text = '{"caption": "A scene", "tags": []}'
        out = _parse_vlm_response(text)
        assert out.has_people is True
        assert out.has_animals is True

    def test_parse_subject_booleans_default_true_on_line_format(self) -> None:
        """Line-based replies never supply the booleans — default to True."""
        text = "A photo.\nthing\nstuff"
        out = _parse_vlm_response(text)
        assert out.has_people is True
        assert out.has_animals is True

    def test_parse_subject_booleans_accept_string_false(self) -> None:
        """Some VLMs emit string 'false' instead of JSON boolean — treat as false."""
        text = (
            '{"caption": "Empty room", "tags": ["room"], '
            '"has_people": "false", "has_animals": "no"}'
        )
        out = _parse_vlm_response(text)
        assert out.has_people is False
        assert out.has_animals is False

    def test_parse_subject_booleans_unrecognised_value_is_true(self) -> None:
        """Anything we can't interpret as a clear 'false' stays True (fail-open)."""
        text = '{"caption": "Maybe", "tags": ["thing"], "has_people": "maybe", "has_animals": null}'
        out = _parse_vlm_response(text)
        assert out.has_people is True
        assert out.has_animals is True

    def test_parse_caption_mentions_override_false_booleans(self) -> None:
        """Qwen sometimes emits has_animals=false while describing a cat.

        The caption/tag keywords force the boolean back to true so detection
        is not skipped.
        """
        text = (
            '{"caption": "A person sitting with a cat on their lap.", '
            '"tags": ["person", "cat", "tablet"], '
            '"has_people": false, "has_animals": false}'
        )
        out = _parse_vlm_response(text)
        assert out.has_people is True  # caption mentions "person"
        assert out.has_animals is True  # tags include "cat"

    def test_parse_booleans_stay_false_when_nothing_mentioned(self) -> None:
        """Override only fires when keywords appear — empty scenes stay false."""
        text = (
            '{"caption": "A snowy mountain ridge under clear skies.", '
            '"tags": ["mountain", "snow", "sky"], '
            '"has_people": false, "has_animals": false}'
        )
        out = _parse_vlm_response(text)
        assert out.has_people is False
        assert out.has_animals is False


class TestDescriberKwargs(TestCase):
    """VLM generation tuning knobs — defaults and propagation."""

    def test_default_max_tokens_is_128(self) -> None:
        """Lower cap than the old 256: captions fit in <100 tokens anyway."""
        d = Describer(model_id="dummy")
        assert d.max_tokens == 128

    def test_default_max_side_is_896(self) -> None:
        """Phase B A/B: 896 matches 1024 quality on high-complexity photos."""
        d = Describer(model_id="dummy")
        assert d.max_side == 896

    def test_kwargs_override_defaults(self) -> None:
        d = Describer(model_id="dummy", max_tokens=64, max_side=640)
        assert d.max_tokens == 64
        assert d.max_side == 640
