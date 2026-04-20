"""Curation coordinator: dismiss / mark-stranger / unassign / exclude.

Every method snapshots the prior state, performs the mutation, and
registers the undo payload — the ADR-012 §M3 replacement for the
``snapshot_findings_fields + mutate + undo_store.put`` block that was
previously inlined in each router.
"""

from __future__ import annotations

from ..db import FaceDB
from ..undo import (
    DismissPayload,
    FindingFieldsSnapshot,
    FindingSubjectSnapshot,
    RestoreClusterPayload,
    RestoreFromStrangerBatchPayload,
    RestoreFromStrangerPayload,
    RestoreSubjectIdsPayload,
    UndoStore,
)
from .receipts import UndoReceipt


def _noun(n: int) -> str:
    return "face" if n == 1 else "faces"


class CurationService:
    """Finding-level curation actions that produce an undoable result.

    Construct once per app via ``create_app`` and share via
    ``ritrova.app.deps.get_curation_service``.
    """

    def __init__(self, db: FaceDB, undo: UndoStore) -> None:
        self._db = db
        self._undo = undo

    # ── Dismiss (whole cluster) ──────────────────────────────────────────

    def dismiss_findings_as_cluster(self, cluster_id: int) -> UndoReceipt | None:
        """Dismiss every finding in a cluster as a non-face.

        Returns ``None`` when the cluster is already empty — the router
        still emits a 200, just without an undo toast.
        """
        finding_ids = self._db.get_cluster_finding_ids(cluster_id)
        if not finding_ids:
            return None
        snapshots = self._snapshot_finding_fields(finding_ids)
        self._db.dismiss_findings(finding_ids)
        message = f"Dismissed {len(finding_ids)} {_noun(len(finding_ids))} in cluster #{cluster_id}"
        return self._put(message, DismissPayload(snapshots=snapshots))

    # ── Dismiss (arbitrary batch) ────────────────────────────────────────

    def dismiss_findings(self, finding_ids: list[int]) -> UndoReceipt | None:
        """Dismiss an arbitrary batch — used by the selection-bar action."""
        if not finding_ids:
            return None
        snapshots = self._snapshot_finding_fields(finding_ids)
        self._db.dismiss_findings(finding_ids)
        message = f"Dismissed {len(finding_ids)} {_noun(len(finding_ids))}"
        return self._put(message, DismissPayload(snapshots=snapshots))

    # ── Mark strangers ──────────────────────────────────────────────────

    def mark_strangers(self, finding_ids: list[int]) -> UndoReceipt | None:
        """Ad-hoc batch mark-stranger (the "I don't know this person/pet" button)."""
        if not finding_ids:
            return None
        snapshots = self._snapshot_finding_fields(finding_ids)
        self._db.set_exclusions(finding_ids, "stranger")
        self._db.remove_cluster_memberships(finding_ids)
        message = f"Marked {len(finding_ids)} {_noun(len(finding_ids))} as stranger"
        return self._put(message, RestoreFromStrangerBatchPayload(snapshots=snapshots))

    def mark_cluster_stranger(self, cluster_id: int) -> UndoReceipt | None:
        """Flag every uncurated finding in a cluster as stranger.

        Only touches findings with no assignment row — keeps previously
        named / dismissed findings as-is. Returns ``None`` when every
        finding in the cluster is already curated.
        """
        pending_ids = self._db.get_unassigned_cluster_finding_ids(cluster_id)
        if not pending_ids:
            return None
        self._db.set_exclusions(pending_ids, "stranger")
        self._db.remove_cluster_memberships(pending_ids)
        message = f"Marked {len(pending_ids)} {_noun(len(pending_ids))} as stranger"
        return self._put(
            message,
            RestoreFromStrangerPayload(cluster_id=cluster_id, finding_ids=pending_ids),
        )

    # ── Unassign ────────────────────────────────────────────────────────

    def unassign_findings(self, finding_ids: list[int]) -> UndoReceipt | None:
        """Bulk unassign: clear each finding's subject, snapshot the prior one."""
        if not finding_ids:
            return None
        prior = self._db.snapshot_findings_fields(finding_ids)
        snapshots = [
            FindingSubjectSnapshot(finding_id=fid, subject_id=sid)
            for fid, sid, _cid in prior
            if sid is not None
        ]
        self._db.unassign_findings(finding_ids)
        message = f"Removed {len(finding_ids)} {_noun(len(finding_ids))}"
        return self._put(message, RestoreSubjectIdsPayload(snapshots=snapshots))

    def unassign_finding(self, finding_id: int) -> UndoReceipt | None:
        """Single-finding unassign.

        Emits the "Removed face from {subject_name}" message the photo
        page expects (the batch variant uses a simpler "Removed N
        face(s)" phrasing). Returns ``None`` when the finding had no
        subject assignment.
        """
        prior_subject_id = self._db.get_finding_subject_id(finding_id)
        if prior_subject_id is None:
            self._db.unassign_finding(finding_id)  # idempotent — keeps state clean
            return None
        self._db.unassign_finding(finding_id)
        subject = self._db.get_subject(prior_subject_id)
        name = subject.name if subject else f"#{prior_subject_id}"
        message = f"Removed face from {name}"
        return self._put(
            message,
            RestoreSubjectIdsPayload(
                snapshots=[
                    FindingSubjectSnapshot(finding_id=finding_id, subject_id=prior_subject_id)
                ]
            ),
        )

    # ── Exclude-from-cluster ────────────────────────────────────────────

    def exclude_findings_from_cluster(
        self, cluster_id: int, finding_ids: list[int]
    ) -> UndoReceipt | None:
        """Per-cluster "remove these faces" — drops only the cluster_findings
        rows, leaves any subject assignments alone."""
        if not finding_ids:
            return None
        self._db.exclude_findings(finding_ids, cluster_id=cluster_id)
        message = (
            f"Excluded {len(finding_ids)} {_noun(len(finding_ids))} from cluster #{cluster_id}"
        )
        return self._put(
            message,
            RestoreClusterPayload(cluster_id=cluster_id, finding_ids=finding_ids),
        )

    # ── Restore-from-stranger ───────────────────────────────────────────

    def restore_from_stranger(
        self, cluster_id: int, finding_ids: list[int] | None = None
    ) -> UndoReceipt | None:
        """Inverse of ``mark_cluster_stranger``. ADR-012 specified
        ``(cluster_id: int)``; we accept optional ``finding_ids`` because
        the DB retains no memory of which cluster a stranger-marked
        finding used to belong to. When omitted we restore every current
        stranger — only safe immediately after a ``mark_cluster_stranger``.
        """
        if finding_ids is None:
            rows = self._db.conn.execute(
                "SELECT finding_id FROM finding_assignment WHERE exclusion_reason = 'stranger'"
            ).fetchall()
            finding_ids = [int(r[0]) for r in rows]
        if not finding_ids:
            return None
        snapshots = self._snapshot_finding_fields(finding_ids)
        self._db.clear_curations(finding_ids)
        self._db.set_cluster_memberships({fid: cluster_id for fid in finding_ids})
        message = f"Restored {len(finding_ids)} {_noun(len(finding_ids))} from stranger"
        return self._put(message, RestoreFromStrangerBatchPayload(snapshots=snapshots))

    # ── Helpers ─────────────────────────────────────────────────────────

    def _snapshot_finding_fields(self, finding_ids: list[int]) -> list[FindingFieldsSnapshot]:
        rows = self._db.snapshot_findings_fields(finding_ids)
        return [
            FindingFieldsSnapshot(finding_id=fid, subject_id=sid, cluster_id=cid)
            for fid, sid, cid in rows
        ]

    def _put(
        self,
        message: str,
        payload: DismissPayload
        | RestoreFromStrangerPayload
        | RestoreFromStrangerBatchPayload
        | RestoreSubjectIdsPayload
        | RestoreClusterPayload,
    ) -> UndoReceipt:
        token = self._undo.put(description=message, payload=payload)
        return UndoReceipt(token=token, message=message)
