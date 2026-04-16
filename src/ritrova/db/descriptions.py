"""Description (caption + tags) mixin."""

from __future__ import annotations

from ._base import _DBAccessor, _locked
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

    @_locked
    def add_description(self, source_id: int, scan_id: int, caption: str, tags: set[str]) -> int:
        """Insert a VLM-generated description for a source."""
        cur = self.conn.execute(
            "INSERT INTO descriptions (source_id, scan_id, caption, tags, generated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, scan_id, caption, self._encode_tags(tags), self._now()),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    @_locked
    def get_description(self, source_id: int) -> Description | None:
        """Return the most recent description for a source, or None."""
        row = self.conn.execute(
            "SELECT * FROM descriptions WHERE source_id = ? ORDER BY id DESC LIMIT 1",
            (source_id,),
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["tags"] = self._decode_tags(d["tags"])
        return Description(**d)

    @_locked
    def get_all_tags(self) -> set[str]:
        """Return all distinct tags across all descriptions.

        Used to build the vocabulary for VLM prompt guidance and for tag
        autocomplete in the UI.
        """
        rows = self.conn.execute("SELECT DISTINCT tags FROM descriptions").fetchall()
        result: set[str] = set()
        for row in rows:
            result |= self._decode_tags(row["tags"])
        return result

    @_locked
    def search_sources_by_tags_and_caption(
        self,
        tag_filters: set[str] | None = None,
        caption_query: str | None = None,
    ) -> list[int]:
        """Return source_ids matching all given tags and/or caption keyword.

        Tags use exact match, caption uses LIKE.
        Returns source_ids sorted descending (newest first by id).
        """
        clauses: list[str] = []
        params: list[str] = []
        if tag_filters:
            for tag in sorted(tag_filters):
                clauses.append("d.tags LIKE ?")
                params.append(f"%:{tag}:%")
        if caption_query:
            clauses.append("d.caption LIKE ?")
            params.append(f"%{caption_query}%")
        if not clauses:
            return []
        where = " AND ".join(clauses)
        rows = self.conn.execute(
            f"SELECT DISTINCT d.source_id FROM descriptions d "
            f"WHERE {where} ORDER BY d.source_id DESC",
            tuple(params),
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def get_undescribed_source_ids(self) -> list[int]:
        """Return source_ids that have no description yet."""
        rows = self.conn.execute(
            "SELECT s.id FROM sources s "
            "LEFT JOIN descriptions d ON d.source_id = s.id "
            "WHERE d.id IS NULL "
            "ORDER BY s.id"
        ).fetchall()
        return [r[0] for r in rows]
