"""Circles coordinator: create / rename / delete + subject membership.

Replaces the inline undo-orchestration in ``app/routers/circles.py``.
Circle creation gets no undo (delete is the obvious reverse, per the
decision already baked into the router); deletion snapshots the circle
row and its membership set so the full circle can be resurrected.
"""

from __future__ import annotations

from ..db import FaceDB
from ..undo import (
    AddSubjectToCirclePayload,
    RecreateCirclePayload,
    RemoveSubjectFromCirclePayload,
    UndoStore,
)
from .receipts import UndoReceipt


class CirclesService:
    """Circle-aggregate mutations with undo orchestration."""

    def __init__(self, db: FaceDB, undo: UndoStore) -> None:
        self._db = db
        self._undo = undo

    # ── Membership ──────────────────────────────────────────────────────

    def add_subject(self, circle_id: int, subject_id: int) -> UndoReceipt | None:
        """Add a subject to a circle.

        Returns ``None`` when the subject was already a member — the
        no-op is idempotent and doesn't deserve an undo slot. Raises
        ``ValueError`` when subject or circle is missing.
        """
        subject = self._db.get_subject(subject_id)
        circle = self._db.get_circle(circle_id)
        if subject is None or circle is None:
            msg = "Subject or circle not found"
            raise ValueError(msg)
        added = self._db.add_subject_to_circle(subject_id, circle_id)
        if not added:
            return None
        message = f"Added {subject.name} to {circle.name}"
        token = self._undo.put(
            description=message,
            payload=RemoveSubjectFromCirclePayload(subject_id=subject_id, circle_id=circle_id),
        )
        return UndoReceipt(token=token, message=message)

    def remove_subject(self, circle_id: int, subject_id: int) -> UndoReceipt | None:
        """Remove a subject from a circle.

        Returns ``None`` when the subject wasn't a member. Raises
        ``ValueError`` when subject or circle is missing.
        """
        subject = self._db.get_subject(subject_id)
        circle = self._db.get_circle(circle_id)
        if subject is None or circle is None:
            msg = "Subject or circle not found"
            raise ValueError(msg)
        removed = self._db.remove_subject_from_circle(subject_id, circle_id)
        if not removed:
            return None
        message = f"Removed {subject.name} from {circle.name}"
        token = self._undo.put(
            description=message,
            payload=AddSubjectToCirclePayload(subject_id=subject_id, circle_id=circle_id),
        )
        return UndoReceipt(token=token, message=message)

    # ── CRUD ────────────────────────────────────────────────────────────

    def create_circle(
        self, name: str, description: str | None = None
    ) -> tuple[int, UndoReceipt | None]:
        """Create a circle (idempotent on name). No undo — the user can
        click delete; see router comment history."""
        circle_id = self._db.create_circle(name, description=description)
        return circle_id, None

    def delete_circle(self, circle_id: int) -> UndoReceipt:
        """Delete a circle, snapshotting its row and members first so the
        undo resurrects the whole thing (new id; old FK-cascade deletions
        of the subject_circles rows are re-inserted).

        Raises ``ValueError`` when the circle doesn't exist.
        """
        circle = self._db.get_circle(circle_id)
        if circle is None:
            msg = "Circle not found"
            raise ValueError(msg)
        member_ids = self._db.get_circle_subject_ids(circle_id)
        self._db.delete_circle(circle_id)
        message = f"Deleted circle '{circle.name}' ({len(member_ids)} members)"
        token = self._undo.put(
            description=message,
            payload=RecreateCirclePayload(
                name=circle.name,
                description=circle.description,
                member_subject_ids=member_ids,
            ),
        )
        return UndoReceipt(token=token, message=message)

    def rename_circle(self, circle_id: int, new_name: str) -> UndoReceipt | None:
        """Rename; no undo slot (trivially reversible)."""
        self._db.rename_circle(circle_id, new_name)
        return None
