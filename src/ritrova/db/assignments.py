"""Assignment + cluster-membership helpers (Apr 2026 refactor).

These read/write the new `finding_assignment` and `cluster_findings`
tables. The old denormalized columns on `findings` are still populated
at this commit — callers haven't been switched over yet (Commit C does
that). The helpers here are the target API.

Design rules:
* `finding_assignment` is sparse. A finding is "uncurated" iff no row
  exists. Uncurated is the *default*, not something you insert a row
  for with NULLs.
* `cluster_findings` is sparse. Unclustered = no row.
* All writes are `INSERT OR REPLACE` — a finding can only be in one
  assignment state at a time, and the CHECK constraint enforces the
  subject-id-XOR-exclusion-reason invariant on insert.
"""

from __future__ import annotations

from ._base import _DBAccessor, _locked

VALID_EXCLUSION_REASONS = ("stranger", "not_a_face")


class AssignmentMixin(_DBAccessor):
    # ── finding_assignment ───────────────────────────────────────────────

    @_locked
    def set_subject(self, finding_id: int, subject_id: int) -> None:
        """Assign a finding to a subject. Overwrites any prior curation state
        (including an exclusion_reason), which is the right behavior: picking
        a name on a previously-marked stranger implicitly unmarks it."""
        self.conn.execute(
            "INSERT OR REPLACE INTO finding_assignment"
            "(finding_id, subject_id, exclusion_reason, curated_at) "
            "VALUES (?, ?, NULL, ?)",
            (finding_id, subject_id, self._now()),
        )
        self.conn.commit()

    @_locked
    def set_exclusion(self, finding_id: int, reason: str) -> None:
        """Mark a finding excluded (``stranger`` or ``not_a_face``). Overwrites
        any prior subject assignment."""
        if reason not in VALID_EXCLUSION_REASONS:
            msg = f"Invalid exclusion reason: {reason!r}"
            raise ValueError(msg)
        self.conn.execute(
            "INSERT OR REPLACE INTO finding_assignment"
            "(finding_id, subject_id, exclusion_reason, curated_at) "
            "VALUES (?, NULL, ?, ?)",
            (finding_id, reason, self._now()),
        )
        self.conn.commit()

    @_locked
    def set_exclusions(self, finding_ids: list[int], reason: str) -> None:
        """Batch set_exclusion — used by mark-stranger on a whole cluster
        and by dismiss on selected faces."""
        if reason not in VALID_EXCLUSION_REASONS:
            msg = f"Invalid exclusion reason: {reason!r}"
            raise ValueError(msg)
        if not finding_ids:
            return
        now = self._now()
        self.conn.executemany(
            "INSERT OR REPLACE INTO finding_assignment"
            "(finding_id, subject_id, exclusion_reason, curated_at) "
            "VALUES (?, NULL, ?, ?)",
            [(fid, reason, now) for fid in finding_ids],
        )
        self.conn.commit()

    @_locked
    def clear_curation(self, finding_id: int) -> None:
        """Drop a finding's assignment row — returns it to 'uncurated'."""
        self.conn.execute("DELETE FROM finding_assignment WHERE finding_id = ?", (finding_id,))
        self.conn.commit()

    @_locked
    def clear_curations(self, finding_ids: list[int]) -> None:
        if not finding_ids:
            return
        placeholders = ",".join("?" * len(finding_ids))
        self.conn.execute(
            f"DELETE FROM finding_assignment WHERE finding_id IN ({placeholders})",
            tuple(finding_ids),
        )
        self.conn.commit()

    @_locked
    def get_curation(self, finding_id: int) -> tuple[int | None, str | None] | None:
        """Return ``(subject_id, exclusion_reason)`` for a finding, or ``None``
        if the finding is uncurated (no assignment row). Exactly one of the
        two elements is non-None when a row exists (enforced by CHECK)."""
        row = self.conn.execute(
            "SELECT subject_id, exclusion_reason FROM finding_assignment WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    # ── cluster_findings ─────────────────────────────────────────────────

    @_locked
    def set_cluster_memberships(self, face_cluster_map: dict[int, int]) -> None:
        """Bulk upsert: each ``{finding_id: cluster_id}`` entry places that
        finding in that cluster. Used by cluster_faces to write the whole
        result set in one pass."""
        if not face_cluster_map:
            return
        self.conn.executemany(
            "INSERT OR REPLACE INTO cluster_findings(finding_id, cluster_id) VALUES (?, ?)",
            list(face_cluster_map.items()),
        )
        self.conn.commit()

    @_locked
    def remove_cluster_memberships(self, finding_ids: list[int]) -> None:
        """Unset cluster membership for the given findings."""
        if not finding_ids:
            return
        placeholders = ",".join("?" * len(finding_ids))
        self.conn.execute(
            f"DELETE FROM cluster_findings WHERE finding_id IN ({placeholders})",
            tuple(finding_ids),
        )
        self.conn.commit()

    @_locked
    def clear_species_cluster_memberships(self, species: str) -> None:
        """Drop every cluster membership for findings of one species —
        called at the start of cluster_faces so a fresh run of the
        algorithm can write new assignments without any stale rows."""
        clause, params = self.species_filter(species)
        self.conn.execute(
            f"DELETE FROM cluster_findings WHERE finding_id IN "
            f"(SELECT id FROM findings WHERE {clause})",
            params,
        )
        self.conn.commit()

    @_locked
    def get_cluster_membership(self, finding_id: int) -> int | None:
        row = self.conn.execute(
            "SELECT cluster_id FROM cluster_findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        return int(row[0]) if row else None

    @_locked
    def merge_cluster_memberships(self, source_cluster: int, target_cluster: int) -> int:
        """Move every finding from ``source_cluster`` into ``target_cluster``.
        Returns the number of rows moved. Idempotent if source == target."""
        if source_cluster == target_cluster:
            return 0
        cur = self.conn.execute(
            "UPDATE cluster_findings SET cluster_id = ? WHERE cluster_id = ?",
            (target_cluster, source_cluster),
        )
        self.conn.commit()
        return cur.rowcount
