"""Undo snapshot/restore mixin (FEAT-5)."""

from __future__ import annotations

from ._base import _DBAccessor, _locked


class UndoMixin(_DBAccessor):
    @_locked
    def snapshot_findings_fields(
        self, finding_ids: list[int]
    ) -> list[tuple[int, int | None, int | None]]:
        """Return ``[(finding_id, person_id, cluster_id), ...]`` for each id.

        Used to capture pre-mutation state before destructive bulk ops so undo
        can restore both assignment fields.
        """
        if not finding_ids:
            return []
        placeholders = ",".join("?" * len(finding_ids))
        rows = self.conn.execute(
            f"SELECT id, person_id, cluster_id FROM findings WHERE id IN ({placeholders})",
            tuple(finding_ids),
        ).fetchall()
        return [(r["id"], r["person_id"], r["cluster_id"]) for r in rows]

    @_locked
    def restore_dismissed_findings(
        self, snapshots: list[tuple[int, int | None, int | None]]
    ) -> None:
        """Undo ``dismiss_findings``: remove dismissed_findings rows and restore
        the prior ``(person_id, cluster_id)`` for each finding."""
        if not snapshots:
            return
        finding_ids = [s[0] for s in snapshots]
        placeholders = ",".join("?" * len(finding_ids))
        self.conn.execute(
            f"DELETE FROM dismissed_findings WHERE finding_id IN ({placeholders})",
            tuple(finding_ids),
        )
        self.conn.executemany(
            "UPDATE findings SET person_id = ?, cluster_id = ? WHERE id = ?",
            [(person_id, cluster_id, fid) for fid, person_id, cluster_id in snapshots],
        )
        self.conn.commit()

    @_locked
    def restore_cluster_id(self, finding_ids: list[int], cluster_id: int) -> None:
        """Set cluster_id on the given findings (used to unwind merge_clusters)."""
        if not finding_ids:
            return
        self.conn.executemany(
            "UPDATE findings SET cluster_id = ? WHERE id = ?",
            [(cluster_id, fid) for fid in finding_ids],
        )
        self.conn.commit()

    @_locked
    def get_finding_person_id(self, finding_id: int) -> int | None:
        """Read just the person_id for a finding, without deserializing the embedding."""
        row = self.conn.execute(
            "SELECT person_id FROM findings WHERE id = ?", (finding_id,)
        ).fetchone()
        return row["person_id"] if row else None

    @_locked
    def get_unassigned_cluster_finding_ids(self, cluster_id: int) -> list[int]:
        """Return finding IDs in ``cluster_id`` with no person assigned yet.

        Snapshot for ``assign_cluster_to_subject`` undo — only these rows will
        actually be updated by the assignment (WHERE person_id IS NULL)."""
        rows = self.conn.execute(
            "SELECT id FROM findings WHERE cluster_id = ? AND person_id IS NULL",
            (cluster_id,),
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def get_subject_row(self, subject_id: int) -> tuple[int, str, str, str] | None:
        """Return ``(id, name, kind, created_at)`` for a subject, or None.

        Used to snapshot a subject before delete/merge so undo can resurrect
        the row verbatim.
        """
        row = self.conn.execute(
            "SELECT id, name, kind, created_at FROM subjects WHERE id = ?",
            (subject_id,),
        ).fetchone()
        if not row:
            return None
        return (row["id"], row["name"], row["kind"], row["created_at"])

    @_locked
    def get_subject_finding_ids(self, subject_id: int) -> list[int]:
        """Return finding IDs currently assigned to ``subject_id``.

        Light-weight alternative to ``get_subject_findings`` when undo only
        needs ids, not full Finding rows + embeddings.
        """
        rows = self.conn.execute(
            "SELECT id FROM findings WHERE person_id = ?", (subject_id,)
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def recreate_subject(self, subject_id: int, name: str, kind: str, created_at: str) -> None:
        """INSERT a subject row with an explicit id.

        Used by undo to resurrect a subject that was destroyed by
        ``delete_subject`` or ``merge_subjects``. AUTOINCREMENT only ever
        advances, so the freed id is safe to reuse — no live row can collide.
        """
        self.conn.execute(
            "INSERT INTO subjects (id, name, kind, created_at) VALUES (?, ?, ?, ?)",
            (subject_id, name, kind, created_at),
        )
        self.conn.commit()

    @_locked
    def restore_person_ids(self, snapshots: list[tuple[int, int | None]]) -> None:
        """Set per-finding person_id from a snapshot map.

        Each element is ``(finding_id, prior_person_id)`` — handles mixed
        NULLs, uniform values, and different subjects in one batch.
        """
        if not snapshots:
            return
        self.conn.executemany(
            "UPDATE findings SET person_id = ? WHERE id = ?",
            [(pid, fid) for fid, pid in snapshots],
        )
        self.conn.commit()
