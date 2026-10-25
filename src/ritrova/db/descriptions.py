"""Description (caption + tags) mixin."""

from __future__ import annotations

from uuid import UUID

from uuid6 import uuid7

from ._base import _DBAccessor
from .models import Description


class DescriptionMixin(_DBAccessor):
    @staticmethod
    def _encode_tags(tags: set[str]) -> str:
        """Encode a set of tags into colon-delimited storage format."""
        return ":" + ":".join(sorted(tags)) + ":" if tags else ""

    @staticmethod
    def _decode_tags(raw: str) -> set[str]:
        """Decode colon-delimited storage format into a set of tags."""
        return {t for t in raw.split(":") if t}

    async def add_description(
        self, source_id: UUID, scan_id: UUID, caption: str, tags: set[str]
    ) -> UUID:
        """Insert a VLM-generated description for a source."""
        description_id = uuid7()
        await self.execute(
            "INSERT INTO descriptions (id, source_id, scan_id, caption, tags, generated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            description_id,
            source_id,
            scan_id,
            caption,
            self._encode_tags(tags),
            self._now(),
        )
        return description_id

    async def get_description(self, source_id: UUID) -> Description | None:
        """Return the most recent description for a source, or None."""
        row = await self.fetchrow(
            "SELECT * FROM descriptions WHERE source_id = ? ORDER BY generated_at DESC, id DESC "
            "LIMIT 1",
            source_id,
        )
        if not row:
            return None
        d = dict(row)
        d["tags"] = self._decode_tags(d["tags"])
        return Description(**d)

    async def get_all_tags(self) -> set[str]:
        """Return all distinct tags across all descriptions.

        Used to build the vocabulary for VLM prompt guidance and for tag
        autocomplete in the UI.
        """
        rows = await self.fetch("SELECT DISTINCT tags FROM descriptions")
        result: set[str] = set()
        for row in rows:
            result |= self._decode_tags(row["tags"])
        return result

    async def search_sources_by_tags_and_caption(
        self,
        tag_filters: set[str] | None = None,
        caption_query: str | None = None,
    ) -> list[UUID]:
        """Return source_ids matching all given tags and/or caption keyword.

        Tags use exact match, caption uses ILIKE.
        Returns source_ids sorted descending (newest first by id).
        """
        clauses: list[str] = []
        params: list[str] = []
        if tag_filters:
            for tag in sorted(tag_filters):
                # ILIKE, not LIKE: SQLite's LIKE is case-insensitive for ASCII
                # and Postgres's is not, so a literal port would quietly stop
                # matching tags that differ in case.
                clauses.append("d.tags ILIKE ?")
                params.append(f"%:{tag}:%")
        if caption_query:
            clauses.append("d.caption ILIKE ?")
            params.append(f"%{caption_query}%")
        if not clauses:
            return []
        where = " AND ".join(clauses)
        rows = await self.fetch(
            f"SELECT DISTINCT d.source_id FROM descriptions d "
            f"WHERE {where} ORDER BY d.source_id DESC",
            *params,
        )
        return [r[0] for r in rows]

    async def get_undescribed_source_ids(self) -> list[UUID]:
        """Return source_ids that have no description yet."""
        rows = await self.fetch(
            "SELECT s.id FROM sources s "
            "LEFT JOIN descriptions d ON d.source_id = s.id "
            "WHERE d.id IS NULL "
            "ORDER BY s.id"
        )
        return [r[0] for r in rows]
