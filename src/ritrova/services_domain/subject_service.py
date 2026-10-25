"""Subject coordinator: CRUD + claim-faces + swap + merge.

Methods that mutate state snapshot prior state, mutate, and register
an undo payload — the ADR-012 §M3 replacement for the corresponding
inline blocks in ``app/routers/subjects.py`` and ``findings.py``.

``claim_faces`` is the single server-side home of the 409-needs-confirm
check — routers translate ``SpeciesMismatch`` into the HTTP response.
"""

from __future__ import annotations

from uuid import UUID

from ..db import FaceDB
from ..undo import (
    DeleteSubjectPayload,
    FindingSubjectSnapshot,
    RestoreSubjectIdsPayload,
    ResurrectSubjectPayload,
    SubjectSnapshot,
    UndoStore,
)
from .receipts import SpeciesMismatch, UndoReceipt


def _noun(n: int) -> str:
    return "face" if n == 1 else "faces"


class SubjectService:
    """Subject-aggregate mutations (create / rename / delete / merge /
    claim-faces / swap-finding).

    Construct once per app via ``create_app``; share via
    ``ritrova.app.deps.get_subject_service``.
    """

    def __init__(self, db: FaceDB, undo: UndoStore) -> None:
        self._db = db
        self._undo = undo

    # ── Claim / swap (finding-level mutations) ─────────────────────────

    async def claim_faces(
        self, subject_id: UUID, finding_ids: list[UUID], *, force: bool = False
    ) -> UndoReceipt | None:
        """Assign every finding in ``finding_ids`` to ``subject_id``.

        Raises ``SpeciesMismatch`` when any finding's species is
        incompatible with the subject's kind and ``force=False``.
        Returns ``None`` when ``finding_ids`` is empty.
        """
        if not finding_ids:
            return None
        async with self._db.transaction():
            prior = await self._db.snapshot_findings_fields(finding_ids)
            snapshots = [
                FindingSubjectSnapshot(finding_id=fid, subject_id=sid) for fid, sid, _cid in prior
            ]
            try:
                for fid in finding_ids:
                    await self._db.assign_finding_to_subject(fid, subject_id, force=force)
            except ValueError as e:
                raise SpeciesMismatch(str(e)) from e
        subject = await self._db.get_subject(subject_id)
        name = subject.name if subject else f"#{subject_id}"
        message = f"Claimed {len(finding_ids)} {_noun(len(finding_ids))} for {name}"
        token = self._undo.put(
            description=message,
            payload=RestoreSubjectIdsPayload(snapshots=snapshots),
        )
        return UndoReceipt(token=token, message=message)

    async def swap_finding(self, finding_id: UUID, target_subject_id: UUID) -> UndoReceipt:
        """Reassign a single finding to a different subject.

        Used by the per-face picker. Snapshots the prior subject_id so
        the undo reverses the swap (restoring the original assignment,
        or clearing it if the finding was previously uncurated).
        """
        return await self._swap_batch([finding_id], target_subject_id)

    async def swap_findings(
        self, finding_ids: list[UUID], target_subject_id: UUID
    ) -> UndoReceipt | None:
        """Batch variant of ``swap_finding`` — moves every finding in
        ``finding_ids`` onto ``target_subject_id`` under one undo slot.

        Returns ``None`` when ``finding_ids`` is empty.
        """
        if not finding_ids:
            return None
        return await self._swap_batch(finding_ids, target_subject_id)

    async def _swap_batch(self, finding_ids: list[UUID], target_subject_id: UUID) -> UndoReceipt:
        async with self._db.transaction():
            prior = await self._db.snapshot_findings_fields(finding_ids)
            snapshots = [
                FindingSubjectSnapshot(finding_id=fid, subject_id=sid) for fid, sid, _cid in prior
            ]
            for fid in finding_ids:
                await self._db.assign_finding_to_subject(fid, target_subject_id)
        subject = await self._db.get_subject(target_subject_id)
        name = subject.name if subject else f"#{target_subject_id}"
        message = f"Swapped {len(finding_ids)} {_noun(len(finding_ids))} for {name}"
        token = self._undo.put(
            description=message, payload=RestoreSubjectIdsPayload(snapshots=snapshots)
        )
        return UndoReceipt(token=token, message=message)

    # ── CRUD ────────────────────────────────────────────────────────────

    async def create_subject(self, name: str, kind: str) -> tuple[UUID, UndoReceipt | None]:
        """Create a subject (or return the existing id for a name+kind match).

        No undo for create — delete is the obvious reverse and the UI has
        that button. The ``UndoReceipt | None`` slot is here so a future
        M5 persistent-undo pass can add one without a signature change.
        """
        subject_id = await self._db.create_subject(name, kind=kind)
        return subject_id, None

    async def rename_subject(self, subject_id: UUID, new_name: str) -> UndoReceipt | None:
        """Rename; no undo (trivially reversible by the user editing again)."""
        await self._db.rename_subject(subject_id, new_name)
        return None

    async def merge_subjects(self, source_id: UUID, target_id: UUID) -> UndoReceipt:
        """Merge the source subject into the target — reassigns every
        finding, then deletes the source.

        Raises ``ValueError`` when ``source_id == target_id`` or either
        subject is missing.
        """
        if source_id == target_id:
            msg = "Cannot merge subject with themselves"
            raise ValueError(msg)
        async with self._db.transaction():
            target = await self._db.get_subject(target_id)
            if not target:
                msg = "Target subject not found"
                raise ValueError(msg)
            source_row = await self._db.get_subject_row(source_id)
            if not source_row:
                msg = "Source subject not found"
                raise ValueError(msg)
            moved_ids = await self._db.get_subject_finding_ids(source_id)
            source_snapshot = SubjectSnapshot(
                id=source_row[0],
                name=source_row[1],
                kind=source_row[2],
                created_at=source_row[3],
            )
            await self._db.merge_subjects(source_id, target_id)
        message = (
            f"Merged {source_snapshot.name} into {target.name} "
            f"({len(moved_ids)} {_noun(len(moved_ids))})"
        )
        token = self._undo.put(
            description=message,
            payload=ResurrectSubjectPayload(subject=source_snapshot, finding_ids=moved_ids),
        )
        return UndoReceipt(token=token, message=message)

    async def delete_subject(self, subject_id: UUID) -> UndoReceipt:
        """Delete a subject (FK cascade drops its finding_assignment rows).

        Raises ``ValueError`` when the subject is missing.
        """
        async with self._db.transaction():
            subject = await self._db.get_subject(subject_id)
            if not subject:
                msg = "Subject not found"
                raise ValueError(msg)
            row = await self._db.get_subject_row(subject_id)
            assert row is not None
            finding_ids = await self._db.get_subject_finding_ids(subject_id)
            snapshot = SubjectSnapshot(id=row[0], name=row[1], kind=row[2], created_at=row[3])
            await self._db.delete_subject(subject_id)
        message = (
            f"Deleted {snapshot.name} ({len(finding_ids)} {_noun(len(finding_ids))} unassigned)"
        )
        token = self._undo.put(
            description=message,
            payload=ResurrectSubjectPayload(subject=snapshot, finding_ids=finding_ids),
        )
        return UndoReceipt(token=token, message=message)

    # ── Internal helper used by the cluster-level "name this cluster"
    # mutation: one subject is created and findings are attached, so the
    # undo has to re-delete the new subject.
    async def create_subject_and_register_delete(
        self, name: str, species: str, n_findings: int
    ) -> tuple[UUID, UndoReceipt]:
        """Create a subject keyed by finding species, and register an
        undo that deletes it (plus FK-cascading its assignment rows).
        """
        subject_id = await self._db.create_subject_for_species(name, species=species)
        message = f"Created {name} and assigned {n_findings} {_noun(n_findings)}"
        token = self._undo.put(
            description=message,
            payload=DeleteSubjectPayload(subject_id=subject_id),
        )
        return subject_id, UndoReceipt(token=token, message=message)
