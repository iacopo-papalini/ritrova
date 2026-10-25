"""CurationService round-trip tests (ADR-012 §M3 step 5).

Each test seeds a DB, asks the service to mutate, checks the new state,
then pops the undo token and asserts the original state is restored.
"""

from __future__ import annotations

from ritrova.services_domain import CurationService

from .._postgres import PostgresTestCase
from ._fixtures import Fixture, build_fixture


class TestDismissCluster(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = CurationService(self.db, self.fx.undo)
        # Put both findings into the same cluster.
        await self.db.update_cluster_ids({self.fx.finding_a_id: 1, self.fx.finding_b_id: 1})

    async def test_dismiss_marks_findings_not_a_face(self) -> None:
        receipt = await self.svc.dismiss_findings_as_cluster(1)
        assert receipt is not None
        assert "Dismissed 2" in receipt.message
        for fid in (self.fx.finding_a_id, self.fx.finding_b_id):
            curation = await self.db.get_curation(fid)
            assert curation is not None
            assert curation[1] == "not_a_face"

    async def test_undo_restores_cluster_membership(self) -> None:
        receipt = await self.svc.dismiss_findings_as_cluster(1)
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        # Findings are back in cluster 1, no exclusion row.
        for fid in (self.fx.finding_a_id, self.fx.finding_b_id):
            assert await self.db.get_curation(fid) is None
            assert await self.db.get_cluster_membership(fid) == 1

    async def test_empty_cluster_returns_none(self) -> None:
        assert await self.svc.dismiss_findings_as_cluster(999) is None


class TestMarkStrangers(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = CurationService(self.db, self.fx.undo)
        await self.db.update_cluster_ids({self.fx.finding_a_id: 1, self.fx.finding_b_id: 1})

    async def test_mark_batch_writes_stranger_and_drops_cluster(self) -> None:
        receipt = await self.svc.mark_strangers([self.fx.finding_a_id])
        assert receipt is not None
        curation = await self.db.get_curation(self.fx.finding_a_id)
        assert curation is not None
        assert curation[1] == "stranger"
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) is None

    async def test_undo_restores_cluster_and_clears_exclusion(self) -> None:
        receipt = await self.svc.mark_strangers([self.fx.finding_a_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_curation(self.fx.finding_a_id) is None
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) == 1


class TestUnassignFindings(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = CurationService(self.db, self.fx.undo)
        self.subject_id = await self.db.create_subject("Alice")
        await self.db.assign_finding_to_subject(self.fx.finding_a_id, self.subject_id)
        await self.db.assign_finding_to_subject(self.fx.finding_b_id, self.subject_id)

    async def test_unassign_clears_subject_rows(self) -> None:
        receipt = await self.svc.unassign_findings([self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) is None
        assert await self.db.get_finding_subject_id(self.fx.finding_b_id) is None

    async def test_undo_restores_subject_assignments(self) -> None:
        receipt = await self.svc.unassign_findings([self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.subject_id
        assert await self.db.get_finding_subject_id(self.fx.finding_b_id) == self.subject_id


class TestExcludeFindingsFromCluster(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = CurationService(self.db, self.fx.undo)
        await self.db.update_cluster_ids({self.fx.finding_a_id: 7, self.fx.finding_b_id: 7})

    async def test_exclude_drops_cluster_row_for_target_only(self) -> None:
        receipt = await self.svc.exclude_findings_from_cluster(7, [self.fx.finding_a_id])
        assert receipt is not None
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) is None
        assert await self.db.get_cluster_membership(self.fx.finding_b_id) == 7

    async def test_undo_restores_cluster_membership(self) -> None:
        receipt = await self.svc.exclude_findings_from_cluster(7, [self.fx.finding_a_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) == 7


class TestUnassignSingleFinding(PostgresTestCase):
    """The photo-page single-face unassign preserves the "Removed face from X" phrasing."""

    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = CurationService(self.db, self.fx.undo)
        self.sid = await self.db.create_subject("Alice")
        await self.db.assign_finding_to_subject(self.fx.finding_a_id, self.sid)

    async def test_single_unassign_message(self) -> None:
        receipt = await self.svc.unassign_finding(self.fx.finding_a_id)
        assert receipt is not None
        assert receipt.message == "Removed face from Alice"

    async def test_single_unassign_undo_restores(self) -> None:
        receipt = await self.svc.unassign_finding(self.fx.finding_a_id)
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid

    async def test_single_unassign_already_unassigned_returns_none(self) -> None:
        assert await self.svc.unassign_finding(self.fx.finding_b_id) is None
