"""Cluster coordinator: merge + assign.

``merge_clusters`` and ``assign_cluster`` are the two cluster-level
mutations that need snapshot+mutate+undo orchestration. Other cluster
reads stay on ``FaceDB`` and are called directly by routers.
"""

from __future__ import annotations

from ..db import FaceDB
from ..undo import (
    FindingSubjectSnapshot,
    RestoreClusterPayload,
    RestoreSubjectIdsPayload,
    UndoStore,
)
from .receipts import SpeciesMismatch, UndoReceipt


def _noun(n: int) -> str:
    return "face" if n == 1 else "faces"


class ClusterService:
    """Cluster-aggregate mutations with undo orchestration."""

    def __init__(self, db: FaceDB, undo: UndoStore) -> None:
        self._db = db
        self._undo = undo

    def merge_clusters(self, source_id: int, target_id: int) -> UndoReceipt:
        """Move every finding from ``source_id`` into ``target_id``.

        Snapshots the finding ids in the source cluster first so the
        undo can flip them back.
        """
        moved_ids = self._db.get_cluster_finding_ids(source_id)
        self._db.merge_clusters(source_id, target_id)
        message = (
            f"Merged cluster #{source_id} into #{target_id} "
            f"({len(moved_ids)} {_noun(len(moved_ids))})"
        )
        token = self._undo.put(
            description=message,
            payload=RestoreClusterPayload(cluster_id=source_id, finding_ids=moved_ids),
        )
        return UndoReceipt(token=token, message=message)

    def assign_cluster(
        self, cluster_id: int, subject_id: int, *, force: bool = False
    ) -> UndoReceipt:
        """Assign every uncurated finding in ``cluster_id`` to ``subject_id``.

        Raises ``SpeciesMismatch`` when the cluster's species is
        incompatible with the subject's kind and ``force=False`` — the
        router translates to 409 ``{error, needs_confirm: true}``.
        """
        subject = self._db.get_subject(subject_id)
        if not subject:
            msg = "Subject not found"
            raise ValueError(msg)
        # Snapshot the findings that assign_cluster_to_subject will mutate
        # (uncurated only) so the undo deletes exactly those assignment rows.
        pending_ids = self._db.get_unassigned_cluster_finding_ids(cluster_id)
        try:
            self._db.assign_cluster_to_subject(cluster_id, subject_id, force=force)
        except ValueError as e:
            raise SpeciesMismatch(str(e)) from e
        message = f"Assigned {len(pending_ids)} {_noun(len(pending_ids))} to {subject.name}"
        token = self._undo.put(
            description=message,
            payload=RestoreSubjectIdsPayload(
                snapshots=[
                    FindingSubjectSnapshot(finding_id=fid, subject_id=None) for fid in pending_ids
                ]
            ),
        )
        return UndoReceipt(token=token, message=message)
