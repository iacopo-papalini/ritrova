"""Source browsing filters.

This mixin is the shared query surface for "show me photos with these
people, path tags, and dates".  `/together` can remain a narrow UI while the
new Browse page grows on this abstraction instead of reimplementing each
filter combination.
"""

from __future__ import annotations

from ._base import _DBAccessor, _locked
from .models import Source
from .path_metadata import normalize_path_tag


class SourceSearchMixin(_DBAccessor):
    def _source_search_parts(
        self,
        *,
        subject_ids: list[int] | None = None,
        path_tags: set[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        source_type: str | None = "photo",
        alone: bool = False,
    ) -> tuple[list[str], list[str], list[int | str]]:
        joins = ["LEFT JOIN source_path_metadata spm ON spm.source_id = s.id"]
        clauses: list[str] = []
        params: list[int | str] = []

        if subject_ids:
            inner, inner_params = self._together_query(subject_ids, alone)
            joins.append(f"JOIN ({inner}) matched ON matched.source_id = s.id")
            params.extend(inner_params)

        if source_type:
            clauses.append("s.type = ?")
            params.append(source_type)

        if date_from:
            clauses.append("spm.date_text IS NOT NULL AND spm.date_text >= ?")
            params.append(date_from)

        if date_to:
            clauses.append("spm.date_text IS NOT NULL AND spm.date_text <= ?")
            params.append(date_to)

        for tag in sorted(path_tags or set()):
            normalized = normalize_path_tag(tag)
            if normalized:
                clauses.append("spm.path_tags LIKE ?")
                params.append(f"%:{normalized}:%")

        return joins, clauses, params

    @_locked
    def search_sources(
        self,
        *,
        subject_ids: list[int] | None = None,
        path_tags: set[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        source_type: str | None = "photo",
        alone: bool = False,
        limit: int = 60,
        offset: int = 0,
    ) -> list[Source]:
        joins, clauses, params = self._source_search_parts(
            subject_ids=subject_ids,
            path_tags=path_tags,
            date_from=date_from,
            date_to=date_to,
            source_type=source_type,
            alone=alone,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        pagination = ""
        if limit > 0:
            pagination = " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        rows = self.conn.execute(
            f"""
            SELECT s.* FROM sources s
            {" ".join(joins)}
            {where}
            ORDER BY
                CASE WHEN spm.date_text IS NULL THEN 1 ELSE 0 END,
                spm.date_text DESC,
                s.file_path DESC,
                s.id DESC
            {pagination}
            """,
            tuple(params),
        ).fetchall()
        return [Source(**dict(r)) for r in rows]

    @_locked
    def count_search_sources(
        self,
        *,
        subject_ids: list[int] | None = None,
        path_tags: set[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        source_type: str | None = "photo",
        alone: bool = False,
    ) -> int:
        joins, clauses, params = self._source_search_parts(
            subject_ids=subject_ids,
            path_tags=path_tags,
            date_from=date_from,
            date_to=date_to,
            source_type=source_type,
            alone=alone,
        )
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM sources s
            {" ".join(joins)}
            {where}
            """,
            tuple(params),
        ).fetchone()
        return int(row[0]) if row else 0
