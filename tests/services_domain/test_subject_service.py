"""SubjectService round-trip tests (ADR-012 §M3 step 5)."""

from __future__ import annotations

import pytest

from ritrova.services_domain import SpeciesMismatch, SubjectService

from .._postgres import PostgresTestCase
from ._fixtures import Fixture, build_fixture


class TestClaimFaces(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = SubjectService(self.db, self.fx.undo)
        self.sid = await self.db.create_subject("Alice")

    async def test_claim_assigns_each_finding(self) -> None:
        receipt = await self.svc.claim_faces(self.sid, [self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        assert receipt.message == "Claimed 2 faces for Alice"
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid
        assert await self.db.get_finding_subject_id(self.fx.finding_b_id) == self.sid

    async def test_claim_undo_restores_prior_state(self) -> None:
        receipt = await self.svc.claim_faces(self.sid, [self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) is None
        assert await self.db.get_finding_subject_id(self.fx.finding_b_id) is None

    async def test_claim_empty_returns_none(self) -> None:
        assert await self.svc.claim_faces(self.sid, []) is None


class TestClaimFacesSpeciesMismatch(PostgresTestCase):
    """Cross-species claim raises SpeciesMismatch unless force=True."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        # Seed with pets.
        self.fx: Fixture = await build_fixture(self.db, species="dog")
        self.svc = SubjectService(self.db, self.fx.undo)
        self.person_sid = await self.db.create_subject("Alice", kind="person")

    async def test_mismatch_raises(self) -> None:
        with pytest.raises(SpeciesMismatch):
            await self.svc.claim_faces(self.person_sid, [self.fx.finding_a_id])

    async def test_force_overrides(self) -> None:
        receipt = await self.svc.claim_faces(self.person_sid, [self.fx.finding_a_id], force=True)
        assert receipt is not None
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.person_sid


class TestMergeSubjects(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = SubjectService(self.db, self.fx.undo)
        self.src_sid = await self.db.create_subject("Alice")
        self.dst_sid = await self.db.create_subject("Alexa")
        await self.db.assign_finding_to_subject(self.fx.finding_a_id, self.src_sid)

    async def test_merge_moves_findings_and_deletes_source(self) -> None:
        receipt = await self.svc.merge_subjects(self.src_sid, self.dst_sid)
        assert "Merged Alice into Alexa" in receipt.message
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.dst_sid
        assert await self.db.get_subject(self.src_sid) is None

    async def test_merge_undo_resurrects_source(self) -> None:
        receipt = await self.svc.merge_subjects(self.src_sid, self.dst_sid)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        restored = await self.db.get_subject(self.src_sid)
        assert restored is not None and restored.name == "Alice"
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.src_sid

    async def test_merge_self_raises(self) -> None:
        with pytest.raises(ValueError):
            await self.svc.merge_subjects(self.src_sid, self.src_sid)


class TestDeleteSubject(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = SubjectService(self.db, self.fx.undo)
        self.sid = await self.db.create_subject("Alice")
        await self.db.assign_finding_to_subject(self.fx.finding_a_id, self.sid)

    async def test_delete_removes_subject_and_unassigns(self) -> None:
        receipt = await self.svc.delete_subject(self.sid)
        assert "Deleted Alice" in receipt.message
        assert await self.db.get_subject(self.sid) is None
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) is None

    async def test_delete_undo_resurrects(self) -> None:
        receipt = await self.svc.delete_subject(self.sid)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        restored = await self.db.get_subject(self.sid)
        assert restored is not None and restored.name == "Alice"
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid


class TestSwapFindings(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = SubjectService(self.db, self.fx.undo)
        self.from_sid = await self.db.create_subject("From")
        self.to_sid = await self.db.create_subject("To")
        await self.db.assign_finding_to_subject(self.fx.finding_a_id, self.from_sid)

    async def test_swap_moves_finding(self) -> None:
        receipt = await self.svc.swap_findings([self.fx.finding_a_id], self.to_sid)
        assert receipt is not None
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.to_sid
        assert receipt.message == "Swapped 1 face for To"

    async def test_swap_undo_restores_original_subject(self) -> None:
        receipt = await self.svc.swap_findings([self.fx.finding_a_id], self.to_sid)
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.from_sid
