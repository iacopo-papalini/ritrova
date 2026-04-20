"""CirclesService round-trip tests (ADR-012 §M3 step 5)."""

from __future__ import annotations

from unittest import TestCase

import pytest

from ritrova.db import FaceDB
from ritrova.services_domain import CirclesService
from ritrova.undo import UndoStore


class TestCirclesMembership(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.db = db
        self.undo = UndoStore()
        self.svc = CirclesService(db, self.undo)
        self.sid = db.create_subject("Alice")
        self.cid = db.create_circle("Family")

    def test_add_registers_undo(self) -> None:
        receipt = self.svc.add_subject(self.cid, self.sid)
        assert receipt is not None
        assert "Added Alice to Family" in receipt.message
        assert self.sid in self.db.get_circle_subject_ids(self.cid)

    def test_add_undo_removes(self) -> None:
        receipt = self.svc.add_subject(self.cid, self.sid)
        assert receipt is not None
        entry = self.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.sid not in self.db.get_circle_subject_ids(self.cid)

    def test_add_idempotent_returns_none(self) -> None:
        self.svc.add_subject(self.cid, self.sid)
        assert self.svc.add_subject(self.cid, self.sid) is None

    def test_remove_registers_undo_and_undo_re_adds(self) -> None:
        self.svc.add_subject(self.cid, self.sid)
        # clear the add undo so the remove undo is the pending one
        self.undo.clear()
        receipt = self.svc.remove_subject(self.cid, self.sid)
        assert receipt is not None
        assert self.sid not in self.db.get_circle_subject_ids(self.cid)
        entry = self.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.sid in self.db.get_circle_subject_ids(self.cid)


class TestDeleteCircle(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.db = db
        self.undo = UndoStore()
        self.svc = CirclesService(db, self.undo)
        self.cid = db.create_circle("Buddies", description="Old friends")
        self.sid = db.create_subject("Bob")
        db.add_subject_to_circle(self.sid, self.cid)

    def test_delete_removes_circle_and_membership(self) -> None:
        receipt = self.svc.delete_circle(self.cid)
        assert receipt.message == "Deleted circle 'Buddies' (1 members)"
        assert self.db.get_circle(self.cid) is None

    def test_delete_undo_recreates(self) -> None:
        receipt = self.svc.delete_circle(self.cid)
        entry = self.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        circle = self.db.get_circle_by_name("Buddies")
        assert circle is not None
        assert circle.description == "Old friends"
        assert self.sid in self.db.get_circle_subject_ids(circle.id)

    def test_delete_missing_raises(self) -> None:
        with pytest.raises(ValueError):
            self.svc.delete_circle(99999)
