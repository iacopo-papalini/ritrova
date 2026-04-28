"""In-memory single-step undo store for FEAT-5.

One pending undo at a time across the whole app — a second write clobbers the
previous token. Entries expire after ``ttl`` seconds so stale tokens can never
resurrect old state. Thread-safe: all mutations hold ``_lock``.

The store itself is transport-agnostic. Each endpoint that wants to be undoable
builds a payload (an ``UndoPayload`` subclass), calls ``put()`` to get a token,
and returns that token to the client. On undo, ``pop()`` hands the payload
back and calls ``payload.undo(db)`` to invert the mutation.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import FaceDB

# ── Snapshot types ──────────────────────────────────────────────────


@dataclass
class FindingFieldsSnapshot:
    """Prior (subject_id, cluster_id) for a single finding."""

    finding_id: int
    subject_id: int | None
    cluster_id: int | None


@dataclass
class FindingSubjectSnapshot:
    """Prior subject_id for a single finding."""

    finding_id: int
    subject_id: int | None


@dataclass
class SubjectSnapshot:
    """Enough of a ``subjects`` row to resurrect it verbatim."""

    id: int
    name: str
    kind: str
    created_at: str


# ── Payload ABC ─────────────────────────────────────────────────────


class UndoPayload(ABC):
    """Base for all undo payloads. Each subclass carries the prior-state
    information needed to invert one kind of mutation."""

    @abstractmethod
    def undo(self, db: FaceDB) -> None:
        """Apply the inverse mutation to restore prior state."""


# ── Concrete payloads ───────────────────────────────────────────────


@dataclass
class DismissPayload(UndoPayload):
    """Undo ``dismiss_findings``: remove dismissed_findings rows and restore
    the prior ``(subject_id, cluster_id)`` for each finding.

    Used for both whole-cluster dismiss and partial (selected faces) dismiss.
    """

    snapshots: list[FindingFieldsSnapshot]

    def undo(self, db: FaceDB) -> None:
        db.restore_dismissed_findings(
            [(s.finding_id, s.subject_id, s.cluster_id) for s in self.snapshots]
        )


@dataclass
class RestoreClusterPayload(UndoPayload):
    """Undo cluster merge or findings exclude: restore cluster_id on findings."""

    cluster_id: int
    finding_ids: list[int]

    def undo(self, db: FaceDB) -> None:
        db.restore_cluster_id(self.finding_ids, self.cluster_id)


@dataclass
class RestoreSubjectIdsPayload(UndoPayload):
    """Undo claim-faces, swap, unassign, or cluster assign: restore
    per-finding subject_id (including NULL for previously-unassigned)."""

    snapshots: list[FindingSubjectSnapshot]

    def undo(self, db: FaceDB) -> None:
        db.restore_subject_ids([(s.finding_id, s.subject_id) for s in self.snapshots])


@dataclass
class DeleteSubjectPayload(UndoPayload):
    """Undo cluster naming: delete the subject that was created.

    The subjects FK ``ON DELETE CASCADE`` on ``finding_assignment.subject_id``
    drops every assignment row that pointed at this subject, returning those
    findings to the uncurated state.
    """

    subject_id: int

    def undo(self, db: FaceDB) -> None:
        db.delete_subject(self.subject_id)


@dataclass
class ResurrectSubjectPayload(UndoPayload):
    """Undo subject delete or merge: recreate the destroyed subject row
    and reassign its findings.

    NOTE: the two ``db.`` calls below run as independent transactions
    (the payload's ``undo`` path does not enter a ``db.transaction()``
    block). If the process crashes between them the subject exists but
    its findings are not yet reassigned — acceptable for a single-user
    desktop app, since the user can just re-run undo and the second
    call is idempotent.
    """

    subject: SubjectSnapshot
    finding_ids: list[int]

    def undo(self, db: FaceDB) -> None:
        db.recreate_subject(
            self.subject.id,
            self.subject.name,
            self.subject.kind,
            self.subject.created_at,
        )
        db.restore_subject_ids([(fid, self.subject.id) for fid in self.finding_ids])


@dataclass
class RestoreFromStrangerPayload(UndoPayload):
    """Undo mark-stranger: remove the 'stranger' exclusion rows and
    restore each finding's prior cluster membership."""

    cluster_id: int
    finding_ids: list[int]

    def undo(self, db: FaceDB) -> None:
        db.clear_curations(self.finding_ids)
        db.set_cluster_memberships({fid: self.cluster_id for fid in self.finding_ids})


@dataclass
class RestoreFromStrangerBatchPayload(UndoPayload):
    """Undo mark-stranger on an ad-hoc batch of findings: clear exclusion
    rows and restore each finding's prior cluster (or leave uncluttered
    if the finding had none — e.g. singletons)."""

    snapshots: list[FindingFieldsSnapshot]

    def undo(self, db: FaceDB) -> None:
        fids = [s.finding_id for s in self.snapshots]
        db.clear_curations(fids)
        restore = {s.finding_id: s.cluster_id for s in self.snapshots if s.cluster_id is not None}
        if restore:
            db.set_cluster_memberships(restore)


@dataclass
class AddSubjectToCirclePayload(UndoPayload):
    """Undo a circle membership removal: re-add the subject."""

    subject_id: int
    circle_id: int

    def undo(self, db: FaceDB) -> None:
        db.add_subject_to_circle(self.subject_id, self.circle_id)


@dataclass
class RemoveSubjectFromCirclePayload(UndoPayload):
    """Undo a circle membership add: remove the subject again."""

    subject_id: int
    circle_id: int

    def undo(self, db: FaceDB) -> None:
        db.remove_subject_from_circle(self.subject_id, self.circle_id)


@dataclass
class DeleteCirclePayload(UndoPayload):
    """Undo circle creation: delete the circle (cascades to memberships)."""

    circle_id: int

    def undo(self, db: FaceDB) -> None:
        db.delete_circle(self.circle_id)


@dataclass
class DeleteManualFindingPayload(UndoPayload):
    """Undo a manual-finding create (FEAT-29): delete the freshly-inserted row.

    ``ON DELETE CASCADE`` on the ``finding_assignment`` / ``cluster_findings``
    child tables takes care of any side rows the user may have created between
    the manual insert and the undo (e.g. assigned the finding to a subject
    before hitting undo).
    """

    finding_id: int

    def undo(self, db: FaceDB) -> None:
        db.delete_finding(self.finding_id)


@dataclass
class RestorePrintSelectionPayload(UndoPayload):
    """Undo print-selection membership changes by restoring the prior worklist."""

    source_ids: list[int]

    def undo(self, db: FaceDB) -> None:
        db.restore_print_selection_ids(self.source_ids)


@dataclass
class RecreateCirclePayload(UndoPayload):
    """Undo circle deletion: recreate the circle and all its memberships.

    The new circle gets a fresh id (old id is gone); the subject_circles
    rows are re-inserted for every subject that was a member. Subjects are
    unchanged — they never got deleted, only the membership rows did.
    """

    name: str
    description: str | None
    member_subject_ids: list[int]

    def undo(self, db: FaceDB) -> None:
        circle_id = db.create_circle(self.name, description=self.description)
        for sid in self.member_subject_ids:
            db.add_subject_to_circle(sid, circle_id)


# ── Store ────────────────────────────────────────────────────────────


@dataclass
class UndoEntry:
    token: str
    description: str
    payload: UndoPayload
    created_at: float = field(default_factory=time.monotonic)


class UndoStore:
    """Single-slot, TTL-bounded undo store.

    Thread-safety: callers may hit this from any request thread; all methods
    serialise on ``_lock``. The DB mutations guarded by an undo entry are
    themselves serialised by FaceDB's own lock, so the happens-before chain
    (snapshot → mutate → put) is preserved per-request. Services wrap the
    snapshot+mutate pair in ``FaceDB.transaction()`` (ADR-012 §M4) so the
    two DB calls are atomic under one lock acquisition — no interleave
    window remains between them.
    """

    def __init__(self, ttl: float = 60.0) -> None:
        self._lock = Lock()
        self._current: UndoEntry | None = None
        self._ttl = ttl

    def put(self, description: str, payload: UndoPayload) -> str:
        """Register a new undoable action. Clobbers any prior pending entry."""
        token = uuid.uuid4().hex
        entry = UndoEntry(token=token, description=description, payload=payload)
        with self._lock:
            self._current = entry
        return token

    def pop(self, token: str) -> UndoEntry | None:
        """Atomically claim and remove the entry matching ``token``.

        Returns None if no pending entry, token mismatch, or the entry is past
        its TTL. One-shot: a successful pop clears the slot.
        """
        with self._lock:
            entry = self._current
            if entry is None or entry.token != token:
                return None
            if time.monotonic() - entry.created_at > self._ttl:
                self._current = None
                return None
            self._current = None
            return entry

    def peek(self) -> UndoEntry | None:
        """Return the current pending entry without consuming it.

        Used by tests and by clients that want to display the pending undo
        description without triggering it.
        """
        with self._lock:
            entry = self._current
            if entry is None:
                return None
            if time.monotonic() - entry.created_at > self._ttl:
                self._current = None
                return None
            return entry

    def clear(self) -> None:
        """Drop any pending entry. Used by tests and on server shutdown."""
        with self._lock:
            self._current = None
