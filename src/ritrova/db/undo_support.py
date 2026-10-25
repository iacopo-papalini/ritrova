"""Undo snapshot/restore mixin (FEAT-5).

Apr 2026 refactor: snapshots + restores operate on `finding_assignment`
and `cluster_findings`; the old denormalized columns on `findings` are
gone.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from ._base import _DBAccessor
from .assignments import _UPSERT_ASSIGNMENT

_UPSERT_CLUSTER = (
    "INSERT INTO cluster_findings(finding_id, cluster_id) VALUES (?, ?) "
    "ON CONFLICT (finding_id) DO UPDATE SET cluster_id = EXCLUDED.cluster_id"
)


class UndoMixin(_DBAccessor):
    async def snapshot_findings_fields(
        self, finding_ids: list[UUID]
    ) -> list[tuple[UUID, UUID | None, int | None]]:
        """Return ``[(finding_id, subject_id, cluster_id), ...]`` for each id.

        Pulled from the new tables; missing rows materialise as (id, None, None).
        """
        if not finding_ids:
            return []
        rows = await self.fetch(
            """
            SELECT f.id,
                   fa.subject_id AS subject_id,
                   cf.cluster_id AS cluster_id
            FROM findings f
            LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
            LEFT JOIN cluster_findings cf ON cf.finding_id = f.id
            WHERE f.id = ANY(?)
            """,
            finding_ids,
        )
        return [(r["id"], r["subject_id"], r["cluster_id"]) for r in rows]

    async def restore_dismissed_findings(
        self, snapshots: list[tuple[UUID, UUID | None, int | None]]
    ) -> None:
        """Undo ``dismiss_findings``: clear the 'not_a_face' row and restore
        the prior (subject_id, cluster_id) state.

        Each snapshot row says "this finding, before the dismiss, had
        subject_id=X and cluster_id=Y". Restore both: delete the exclusion
        row, reinstate the subject assignment (if any), reinstate the
        cluster membership (if any).
        """
        if not snapshots:
            return
        now = self._now()
        finding_ids = [s[0] for s in snapshots]
        async with self.transaction():
            await self.execute(
                "DELETE FROM finding_assignment WHERE finding_id = ANY(?) "
                "AND exclusion_reason = 'not_a_face'",
                finding_ids,
            )
            await self.executemany(
                _UPSERT_ASSIGNMENT,
                [(fid, sid, None, now) for fid, sid, _cid in snapshots if sid is not None],
            )
            await self.executemany(
                _UPSERT_CLUSTER,
                [(fid, cid) for fid, _sid, cid in snapshots if cid is not None],
            )

    async def restore_cluster_id(self, finding_ids: list[UUID], cluster_id: int) -> None:
        """Set the cluster for the given findings (unwinds merge_clusters)."""
        if not finding_ids:
            return
        await self.executemany(_UPSERT_CLUSTER, [(fid, cluster_id) for fid in finding_ids])

    async def get_finding_subject_id(self, finding_id: UUID) -> UUID | None:
        """Read just the subject_id for a finding."""
        subject_id: UUID | None = await self.fetchval(
            "SELECT subject_id FROM finding_assignment WHERE finding_id = ?", finding_id
        )
        return subject_id

    async def get_unassigned_cluster_finding_ids(self, cluster_id: int) -> list[UUID]:
        """Finding IDs in ``cluster_id`` that have no assignment row yet.

        Snapshot for ``assign_cluster_to_subject`` undo — only these
        findings will actually get a new assignment row, so the inverse
        is to delete exactly them.
        """
        rows = await self.fetch(
            """
            SELECT cf.finding_id FROM cluster_findings cf
            LEFT JOIN finding_assignment fa ON fa.finding_id = cf.finding_id
            WHERE cf.cluster_id = ? AND fa.finding_id IS NULL
            """,
            cluster_id,
        )
        return [r[0] for r in rows]

    async def get_subject_row(self, subject_id: UUID) -> tuple[UUID, str, str, datetime] | None:
        """Return ``(id, name, kind, created_at)`` for a subject, or None."""
        row = await self.fetchrow(
            "SELECT id, name, kind, created_at FROM subjects WHERE id = ?", subject_id
        )
        if not row:
            return None
        return (row["id"], row["name"], row["kind"], row["created_at"])

    async def get_subject_finding_ids(self, subject_id: UUID) -> list[UUID]:
        """Finding IDs currently assigned to ``subject_id``."""
        rows = await self.fetch(
            "SELECT finding_id FROM finding_assignment WHERE subject_id = ?", subject_id
        )
        return [r[0] for r in rows]

    async def recreate_subject(
        self, subject_id: UUID, name: str, kind: str, created_at: datetime
    ) -> None:
        """INSERT a subject row with an explicit id.

        Used by undo to resurrect a subject that was destroyed by
        ``delete_subject`` or ``merge_subjects``. The id is a UUID and was
        never reissued to anything else, so it is safe to bring back.
        """
        await self.execute(
            "INSERT INTO subjects (id, name, kind, created_at) VALUES (?, ?, ?, ?)",
            subject_id,
            name,
            kind,
            created_at,
        )

    async def restore_subject_ids(self, snapshots: list[tuple[UUID, UUID | None]]) -> None:
        """Set per-finding subject_id from a snapshot map.

        Each element is ``(finding_id, prior_subject_id)``. A None means
        "no assignment" — delete the assignment row. A value means
        "set the subject" — upsert.
        """
        if not snapshots:
            return
        now = self._now()
        to_delete = [fid for fid, sid in snapshots if sid is None]
        to_upsert = [(fid, sid, None, now) for fid, sid in snapshots if sid is not None]
        async with self.transaction():
            if to_delete:
                await self.execute(
                    "DELETE FROM finding_assignment WHERE finding_id = ANY(?)", to_delete
                )
            await self.executemany(_UPSERT_ASSIGNMENT, to_upsert)
