"""Curation mutations and together-query mixin.

Apr 2026 refactor: dismissals + subject unassignments route through
AssignmentMixin; all state lives on `finding_assignment` and
`cluster_findings`.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from ._base import _DBAccessor
from .findings import _FINDING_COLUMNS, _row_to_finding
from .models import Finding, Source


@dataclass
class PruneReport:
    """Result of a prune operation."""

    by_subject: int  # duplicate (source_id, subject_id) findings removed
    by_cluster: int  # duplicate (source_id, cluster_id) findings removed

    @property
    def total(self) -> int:
        return self.by_subject + self.by_cluster


class CurationMixin(_DBAccessor):
    async def dismiss_findings(self, finding_ids: list[UUID]) -> None:
        """Mark findings as non-faces (statues, paintings, etc.).

        Writes `exclusion_reason='not_a_face'` on each (overwriting any
        prior assignment) and drops their cluster membership so the Merge
        and curation UIs stop offering them.
        """
        if not finding_ids:
            return
        async with self.transaction():
            await self.set_exclusions(finding_ids, "not_a_face")  # type: ignore[attr-defined]
            await self.remove_cluster_memberships(finding_ids)  # type: ignore[attr-defined]

    async def unassign_finding(self, finding_id: UUID) -> None:
        """Clear a finding's assignment — returns to 'uncurated'."""
        await self.clear_curation(finding_id)  # type: ignore[attr-defined]

    async def unassign_findings(self, finding_ids: list[UUID]) -> None:
        """Batch variant of unassign_finding."""
        await self.clear_curations(finding_ids)  # type: ignore[attr-defined]

    async def exclude_findings(self, finding_ids: list[UUID], cluster_id: int) -> None:
        """Remove findings from a cluster. ``cluster_id`` is kept in the
        signature for clarity / validation; any findings not currently in
        that cluster are left alone (no schema-level constraint to rely on,
        so we filter explicitly)."""
        if not finding_ids:
            return
        await self.execute(
            "DELETE FROM cluster_findings WHERE finding_id = ANY(?) AND cluster_id = ?",
            finding_ids,
            cluster_id,
        )

    def _together_query(
        self, subject_ids: list[UUID], alone: bool = False
    ) -> tuple[str, tuple[object, ...]]:
        """Build the inner subquery for together (shared by get and count)."""
        n = len(subject_ids)
        if alone:
            return (
                """SELECT f.source_id FROM findings f
                    JOIN finding_assignment fa ON fa.finding_id = f.id
                    WHERE fa.subject_id IS NOT NULL
                    GROUP BY f.source_id
                    HAVING COUNT(DISTINCT CASE WHEN fa.subject_id = ANY(?)
                                               THEN fa.subject_id END) = ?
                    AND COUNT(DISTINCT fa.subject_id) = ?""",
                (subject_ids, n, n),
            )
        return (
            """SELECT f.source_id FROM findings f
                JOIN finding_assignment fa ON fa.finding_id = f.id
                WHERE fa.subject_id = ANY(?)
                GROUP BY f.source_id
                HAVING COUNT(DISTINCT fa.subject_id) = ?""",
            (subject_ids, n),
        )

    async def get_sources_with_all_subjects(
        self,
        subject_ids: list[UUID],
        limit: int = 0,
        offset: int = 0,
        alone: bool = False,
        source_type: str | None = None,
    ) -> list[Source]:
        """Find sources that contain ALL given subjects."""
        if not subject_ids:
            return []
        inner, params = self._together_query(subject_ids, alone)
        type_filter = ""
        full_params: tuple[object, ...] = params
        if source_type:
            type_filter = " WHERE s.type = ?"
            full_params = (*full_params, source_type)
        pagination = ""
        if limit > 0:
            pagination = " LIMIT ? OFFSET ?"
            full_params = (*full_params, limit, offset)
        rows = await self.fetch(
            f"""
            SELECT s.* FROM sources s
            JOIN ({inner}) matched ON matched.source_id = s.id
            {type_filter}
            ORDER BY s.file_path DESC{pagination}
            """,
            *full_params,
        )
        return [Source(**dict(r)) for r in rows]

    async def count_sources_with_all_subjects(
        self,
        subject_ids: list[UUID],
        alone: bool = False,
        source_type: str | None = None,
    ) -> int:
        """Count sources containing ALL given subjects, optionally filtered by type."""
        if not subject_ids:
            return 0
        inner, params = self._together_query(subject_ids, alone)
        if source_type:
            return await self._count(
                f"SELECT COUNT(*) FROM sources s JOIN ({inner}) m ON m.source_id = s.id "
                f"WHERE s.type = ?",
                *params,
                source_type,
            )
        # Postgres requires a name for a derived table; SQLite did not.
        return await self._count(f"SELECT COUNT(*) FROM ({inner}) matched", *params)

    async def get_subject_sources(self, subject_id: UUID) -> list[Source]:
        """All unique sources containing a subject, newest first (by path)."""
        rows = await self.fetch(
            """
            SELECT DISTINCT s.* FROM sources s
            JOIN findings f ON f.source_id = s.id
            JOIN finding_assignment fa ON fa.finding_id = f.id
            WHERE fa.subject_id = ?
            ORDER BY s.file_path DESC
            """,
            subject_id,
        )
        return [Source(**dict(r)) for r in rows]

    async def get_subject_sources_with_findings(
        self, subject_id: UUID, source_type: str
    ) -> list[tuple[Source, list[Finding]]]:
        """Sources of a given type containing the subject, paired with the
        subject's findings on each source. Used by the Videos tab on the
        subject detail page."""
        rows = await self.fetch(
            f"""
            SELECT
                s.id        AS s_id,
                s.file_path AS s_file_path,
                s.type      AS s_type,
                s.width     AS s_width,
                s.height    AS s_height,
                s.taken_at  AS s_taken_at,
                s.latitude  AS s_latitude,
                s.longitude AS s_longitude,
                {_FINDING_COLUMNS}
            FROM sources s
            JOIN findings f ON f.source_id = s.id
            LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
            LEFT JOIN cluster_findings cf ON cf.finding_id = f.id
            WHERE fa.subject_id = ? AND s.type = ?
            ORDER BY s.file_path DESC, f.id
            """,
            subject_id,
            source_type,
        )

        grouped: dict[UUID, tuple[Source, list[Finding]]] = {}
        for row in rows:
            finding = _row_to_finding(row)
            if finding is None:
                continue
            source_id = row["s_id"]
            if source_id not in grouped:
                source = Source(
                    id=source_id,
                    file_path=row["s_file_path"],
                    type=row["s_type"],
                    width=row["s_width"],
                    height=row["s_height"],
                    taken_at=row["s_taken_at"],
                    latitude=row["s_latitude"],
                    longitude=row["s_longitude"],
                )
                grouped[source_id] = (source, [])
            grouped[source_id][1].append(finding)

        return list(grouped.values())

    async def prune_duplicate_findings(self, *, dry_run: bool = False) -> PruneReport:
        """Remove duplicate findings per (source_id, subject_id) and per
        (source_id, cluster_id). Keeps the newest finding. Dismissed /
        excluded findings are skipped.

        "Newest" still means the greatest id: UUIDv7 keys are time-ordered,
        and the migration stamped each one with the row's own creation time
        rather than the migration's. Postgres has no ``max(uuid)`` aggregate,
        so the comparison goes through the canonical text form — which sorts
        the same way, the timestamp being the leading hex digits.
        """
        subject_dupes = await self.fetch(
            """
            WITH assigned AS (
                SELECT f.id, f.source_id, fa.subject_id
                FROM findings f
                JOIN finding_assignment fa ON fa.finding_id = f.id
                WHERE fa.subject_id IS NOT NULL
            )
            SELECT id FROM assigned
            WHERE id NOT IN (
                SELECT MAX(id::text)::uuid FROM assigned GROUP BY source_id, subject_id
            )
            """
        )
        subject_ids = [r[0] for r in subject_dupes]

        # Duplicates by cluster: same cluster on same source, and the
        # finding has no subject assignment (if it did, the subject-level
        # dedup above already handles it).
        cluster_dupes = await self.fetch(
            """
            WITH clustered_uncurated AS (
                SELECT f.id, f.source_id, cf.cluster_id
                FROM findings f
                JOIN cluster_findings cf ON cf.finding_id = f.id
                LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
                WHERE fa.finding_id IS NULL
            )
            SELECT id FROM clustered_uncurated
            WHERE id NOT IN (
                SELECT MAX(id::text)::uuid FROM clustered_uncurated GROUP BY source_id, cluster_id
            )
            """
        )
        cluster_ids = [r[0] for r in cluster_dupes]

        if not dry_run and (subject_ids or cluster_ids):
            async with self.transaction():
                if subject_ids:
                    await self.execute("DELETE FROM findings WHERE id = ANY(?)", subject_ids)
                if cluster_ids:
                    await self.execute("DELETE FROM findings WHERE id = ANY(?)", cluster_ids)

        return PruneReport(by_subject=len(subject_ids), by_cluster=len(cluster_ids))
