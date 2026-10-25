"""CirclesService round-trip tests (ADR-012 §M3 step 5)."""

from __future__ import annotations

import pytest

from ritrova.services_domain import CirclesService
from ritrova.undo import UndoStore

from .._helpers import MISSING_ID
from .._postgres import PostgresTestCase


class TestCirclesMembership(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.undo = UndoStore()
        self.svc = CirclesService(self.db, self.undo)
        self.sid = await self.db.create_subject("Alice")
        self.cid = await self.db.create_circle("Family")

    async def test_add_registers_undo(self) -> None:
        receipt = await self.svc.add_subject(self.cid, self.sid)
        assert receipt is not None
        assert "Added Alice to Family" in receipt.message
        assert self.sid in await self.db.get_circle_subject_ids(self.cid)

    async def test_add_undo_removes(self) -> None:
        receipt = await self.svc.add_subject(self.cid, self.sid)
        assert receipt is not None
        entry = self.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert self.sid not in await self.db.get_circle_subject_ids(self.cid)

    async def test_add_idempotent_returns_none(self) -> None:
        await self.svc.add_subject(self.cid, self.sid)
        assert await self.svc.add_subject(self.cid, self.sid) is None

    async def test_remove_registers_undo_and_undo_re_adds(self) -> None:
        await self.svc.add_subject(self.cid, self.sid)
        # clear the add undo so the remove undo is the pending one
        self.undo.clear()
        receipt = await self.svc.remove_subject(self.cid, self.sid)
        assert receipt is not None
        assert self.sid not in await self.db.get_circle_subject_ids(self.cid)
        entry = self.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert self.sid in await self.db.get_circle_subject_ids(self.cid)


class TestDeleteCircle(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.undo = UndoStore()
        self.svc = CirclesService(self.db, self.undo)
        self.cid = await self.db.create_circle("Buddies", description="Old friends")
        self.sid = await self.db.create_subject("Bob")
        await self.db.add_subject_to_circle(self.sid, self.cid)

    async def test_delete_removes_circle_and_membership(self) -> None:
        receipt = await self.svc.delete_circle(self.cid)
        assert receipt.message == "Deleted circle 'Buddies' (1 members)"
        assert await self.db.get_circle(self.cid) is None

    async def test_delete_undo_recreates(self) -> None:
        receipt = await self.svc.delete_circle(self.cid)
        entry = self.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        circle = await self.db.get_circle_by_name("Buddies")
        assert circle is not None
        assert circle.description == "Old friends"
        assert self.sid in await self.db.get_circle_subject_ids(circle.id)

    async def test_delete_missing_raises(self) -> None:
        with pytest.raises(ValueError):
            await self.svc.delete_circle(MISSING_ID)
