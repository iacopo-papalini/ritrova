"""Undo snapshot/restore mixin (FEAT-5).

Apr 2026 refactor: snapshots + restores operate on `finding_assignment`
and `cluster_findings`; the old denormalized columns on `findings` are
gone.
"""

from __future__ import annotations

from ._base import _DBAccessor, _locked


class UndoMixin(_DBAccessor):
    @_locked
    def snapshot_findings_fields(
        self, finding_ids: list[int]
    ) -> list[tuple[int, int | None, int | None]]:
        """Return ``[(finding_id, subject_id, cluster_id), ...]`` for each id.

        Pulled from the new tables; missing rows materialise as (id, None, None).
        """
        if not finding_ids:
            return []
        placeholders = ",".join("?" * len(finding_ids))
        rows = self.conn.execute(
            f"""
            SELECT f.id,
                   fa.subject_id AS subject_id,
                   cf.cluster_id AS cluster_id
            FROM findings f
            LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
            LEFT JOIN cluster_findings cf ON cf.finding_id = f.id
            WHERE f.id IN ({placeholders})
            """,
            tuple(finding_ids),
        ).fetchall()
        return [(r["id"], r["subject_id"], r["cluster_id"]) for r in rows]

    @_locked
    def restore_dismissed_findings(
        self, snapshots: list[tuple[int, int | None, int | None]]
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
        placeholders = ",".join("?" * len(finding_ids))
        # Drop the 'not_a_face' exclusion rows for these findings.
        self.conn.execute(
            f"DELETE FROM finding_assignment WHERE finding_id IN ({placeholders}) "
            f"AND exclusion_reason = 'not_a_face'",
            tuple(finding_ids),
        )
        # Re-insert subject assignments for those that had one.
        subject_rows = [(fid, sid, now) for fid, sid, _cid in snapshots if sid is not None]
        if subject_rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO finding_assignment"
                "(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, ?, NULL, ?)",
                subject_rows,
            )
        # Re-insert cluster memberships for those that had one.
        cluster_rows = [(fid, cid) for fid, _sid, cid in snapshots if cid is not None]
        if cluster_rows:
            self.conn.executemany(
                "INSERT OR REPLACE INTO cluster_findings(finding_id, cluster_id) VALUES (?, ?)",
                cluster_rows,
            )
        self.conn.commit()

    @_locked
    def restore_cluster_id(self, finding_ids: list[int], cluster_id: int) -> None:
        """Set the cluster for the given findings (unwinds merge_clusters)."""
        if not finding_ids:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO cluster_findings(finding_id, cluster_id) VALUES (?, ?)",
            [(fid, cluster_id) for fid in finding_ids],
        )
        self.conn.commit()

    @_locked
    def get_finding_subject_id(self, finding_id: int) -> int | None:
        """Read just the subject_id for a finding."""
        row = self.conn.execute(
            "SELECT subject_id FROM finding_assignment WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        return row["subject_id"] if row else None

    @_locked
    def get_unassigned_cluster_finding_ids(self, cluster_id: int) -> list[int]:
        """Finding IDs in ``cluster_id`` that have no assignment row yet.

        Snapshot for ``assign_cluster_to_subject`` undo — only these
        findings will actually get a new assignment row, so the inverse
        is to delete exactly them.
        """
        rows = self.conn.execute(
            """
            SELECT cf.finding_id FROM cluster_findings cf
            LEFT JOIN finding_assignment fa ON fa.finding_id = cf.finding_id
            WHERE cf.cluster_id = ? AND fa.finding_id IS NULL
            """,
            (cluster_id,),
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def get_subject_row(self, subject_id: int) -> tuple[int, str, str, str] | None:
        """Return ``(id, name, kind, created_at)`` for a subject, or None."""
        row = self.conn.execute(
            "SELECT id, name, kind, created_at FROM subjects WHERE id = ?",
            (subject_id,),
        ).fetchone()
        if not row:
            return None
        return (row["id"], row["name"], row["kind"], row["created_at"])

    @_locked
    def get_subject_finding_ids(self, subject_id: int) -> list[int]:
        """Finding IDs currently assigned to ``subject_id``."""
        rows = self.conn.execute(
            "SELECT finding_id FROM finding_assignment WHERE subject_id = ?",
            (subject_id,),
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
    def restore_subject_ids(self, snapshots: list[tuple[int, int | None]]) -> None:
        """Set per-finding subject_id from a snapshot map.

        Each element is ``(finding_id, prior_subject_id)``. A None means
        "no assignment" — delete the assignment row. A value means
        "set the subject" — upsert.
        """
        if not snapshots:
            return
        now = self._now()
        to_delete = [fid for fid, sid in snapshots if sid is None]
        to_upsert = [(fid, sid, now) for fid, sid in snapshots if sid is not None]
        if to_delete:
            placeholders = ",".join("?" * len(to_delete))
            self.conn.execute(
                f"DELETE FROM finding_assignment WHERE finding_id IN ({placeholders})",
                tuple(to_delete),
            )
        if to_upsert:
            self.conn.executemany(
                "INSERT OR REPLACE INTO finding_assignment"
                "(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, ?, NULL, ?)",
                to_upsert,
            )
        self.conn.commit()
