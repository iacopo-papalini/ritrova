"""Assignment + cluster-membership helpers (Apr 2026 refactor).

These read/write the `finding_assignment` and `cluster_findings` tables.

Design rules:
* `finding_assignment` is sparse. A finding is "uncurated" iff no row
  exists. Uncurated is the *default*, not something you insert a row
  for with NULLs.
* `cluster_findings` is sparse. Unclustered = no row.
* All writes upsert on the primary key — a finding can only be in one
  assignment state at a time, and the CHECK constraint enforces the
  subject-id-XOR-exclusion-reason invariant on insert.

``ON CONFLICT ... DO UPDATE`` replaces SQLite's ``INSERT OR REPLACE``, and the
difference is not cosmetic: ``INSERT OR REPLACE`` deletes the conflicting row
and inserts a new one, firing ``ON DELETE CASCADE`` on anything referencing it.
``DO UPDATE`` edits in place.
"""

from __future__ import annotations

from uuid import UUID

from ._base import _DBAccessor

VALID_EXCLUSION_REASONS = ("stranger", "not_a_face")

_UPSERT_ASSIGNMENT = (
    "INSERT INTO finding_assignment"
    "(finding_id, subject_id, exclusion_reason, curated_at) "
    "VALUES (?, ?, ?, ?) "
    "ON CONFLICT (finding_id) DO UPDATE SET "
    "subject_id = EXCLUDED.subject_id, "
    "exclusion_reason = EXCLUDED.exclusion_reason, "
    "curated_at = EXCLUDED.curated_at"
)


class AssignmentMixin(_DBAccessor):
    # ── finding_assignment ───────────────────────────────────────────────

    async def set_subject(self, finding_id: UUID, subject_id: UUID) -> None:
        """Assign a finding to a subject. Overwrites any prior curation state
        (including an exclusion_reason), which is the right behavior: picking
        a name on a previously-marked stranger implicitly unmarks it."""
        await self.execute(_UPSERT_ASSIGNMENT, finding_id, subject_id, None, self._now())

    async def set_exclusion(self, finding_id: UUID, reason: str) -> None:
        """Mark a finding excluded (``stranger`` or ``not_a_face``). Overwrites
        any prior subject assignment."""
        if reason not in VALID_EXCLUSION_REASONS:
            msg = f"Invalid exclusion reason: {reason!r}"
            raise ValueError(msg)
        await self.execute(_UPSERT_ASSIGNMENT, finding_id, None, reason, self._now())

    async def set_exclusions(self, finding_ids: list[UUID], reason: str) -> None:
        """Batch set_exclusion — used by mark-stranger on a whole cluster
        and by dismiss on selected faces."""
        if reason not in VALID_EXCLUSION_REASONS:
            msg = f"Invalid exclusion reason: {reason!r}"
            raise ValueError(msg)
        if not finding_ids:
            return
        now = self._now()
        await self.executemany(
            _UPSERT_ASSIGNMENT, [(fid, None, reason, now) for fid in finding_ids]
        )

    async def clear_curation(self, finding_id: UUID) -> None:
        """Drop a finding's assignment row — returns it to 'uncurated'."""
        await self.execute("DELETE FROM finding_assignment WHERE finding_id = ?", finding_id)

    async def clear_curations(self, finding_ids: list[UUID]) -> None:
        if not finding_ids:
            return
        await self.execute("DELETE FROM finding_assignment WHERE finding_id = ANY(?)", finding_ids)

    async def get_curation(self, finding_id: UUID) -> tuple[UUID | None, str | None] | None:
        """Return ``(subject_id, exclusion_reason)`` for a finding, or ``None``
        if the finding is uncurated (no assignment row). Exactly one of the
        two elements is non-None when a row exists (enforced by CHECK)."""
        row = await self.fetchrow(
            "SELECT subject_id, exclusion_reason FROM finding_assignment WHERE finding_id = ?",
            finding_id,
        )
        return (row[0], row[1]) if row else None

    # ── cluster_findings ─────────────────────────────────────────────────

    async def set_cluster_memberships(self, face_cluster_map: dict[UUID, int]) -> None:
        """Bulk upsert: each ``{finding_id: cluster_id}`` entry places that
        finding in that cluster. Used by cluster_faces to write the whole
        result set in one pass."""
        if not face_cluster_map:
            return
        await self.executemany(
            "INSERT INTO cluster_findings(finding_id, cluster_id) VALUES (?, ?) "
            "ON CONFLICT (finding_id) DO UPDATE SET cluster_id = EXCLUDED.cluster_id",
            list(face_cluster_map.items()),
        )

    async def remove_cluster_memberships(self, finding_ids: list[UUID]) -> None:
        """Unset cluster membership for the given findings."""
        if not finding_ids:
            return
        await self.execute("DELETE FROM cluster_findings WHERE finding_id = ANY(?)", finding_ids)

    async def clear_species_cluster_memberships(self, species: str) -> None:
        """Drop every cluster membership for findings of one species —
        called at the start of cluster_faces so a fresh run of the
        algorithm can write new assignments without any stale rows."""
        clause, params = self.species_filter(species)
        await self.execute(
            f"DELETE FROM cluster_findings WHERE finding_id IN "
            f"(SELECT id FROM findings WHERE {clause})",
            *params,
        )

    async def get_cluster_membership(self, finding_id: UUID) -> int | None:
        value = await self.fetchval(
            "SELECT cluster_id FROM cluster_findings WHERE finding_id = ?", finding_id
        )
        return int(value) if value is not None else None

    async def merge_cluster_memberships(self, source_cluster: int, target_cluster: int) -> int:
        """Move every finding from ``source_cluster`` into ``target_cluster``.
        Returns the number of rows moved. Idempotent if source == target."""
        if source_cluster == target_cluster:
            return 0
        moved = await self.fetch(
            "UPDATE cluster_findings SET cluster_id = ? WHERE cluster_id = ? RETURNING finding_id",
            target_cluster,
            source_cluster,
        )
        return len(moved)
