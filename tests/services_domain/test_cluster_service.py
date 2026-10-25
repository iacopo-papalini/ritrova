"""ClusterService round-trip tests (ADR-012 §M3 step 5)."""

from __future__ import annotations

import pytest

from ritrova.services_domain import ClusterService, SpeciesMismatch

from .._postgres import PostgresTestCase
from ._fixtures import Fixture, build_fixture


class TestMergeClusters(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = ClusterService(self.db, self.fx.undo)
        await self.db.update_cluster_ids({self.fx.finding_a_id: 5, self.fx.finding_b_id: 9})

    async def test_merge_moves_findings_into_target(self) -> None:
        receipt = await self.svc.merge_clusters(source_id=5, target_id=9)
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) == 9
        assert await self.db.get_cluster_membership(self.fx.finding_b_id) == 9
        assert "Merged cluster #5 into #9" in receipt.message

    async def test_merge_undo_restores_source_cluster(self) -> None:
        receipt = await self.svc.merge_clusters(source_id=5, target_id=9)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) == 5
        assert await self.db.get_cluster_membership(self.fx.finding_b_id) == 9


class TestSplitCluster(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = ClusterService(self.db, self.fx.undo)
        await self.db.update_cluster_ids({self.fx.finding_a_id: 5, self.fx.finding_b_id: 5})

    async def test_split_moves_selected_findings_into_new_cluster(self) -> None:
        result = await self.svc.split_cluster(5, [self.fx.finding_a_id])
        assert result is not None
        new_cluster_id, moved_count, receipt = result
        assert moved_count == 1
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) == new_cluster_id
        assert await self.db.get_cluster_membership(self.fx.finding_b_id) == 5
        assert f"into #{new_cluster_id}" in receipt.message

    async def test_split_undo_restores_source_cluster(self) -> None:
        result = await self.svc.split_cluster(5, [self.fx.finding_a_id])
        assert result is not None
        _new_cluster_id, _moved_count, receipt = result
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) == 5
        assert await self.db.get_cluster_membership(self.fx.finding_b_id) == 5

    async def test_split_ignores_stale_selection_outside_source_cluster(self) -> None:
        assert await self.svc.split_cluster(99, [self.fx.finding_a_id]) is None
        assert await self.db.get_cluster_membership(self.fx.finding_a_id) == 5


class TestAssignCluster(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.fx: Fixture = await build_fixture(self.db)
        self.svc = ClusterService(self.db, self.fx.undo)
        await self.db.update_cluster_ids({self.fx.finding_a_id: 1, self.fx.finding_b_id: 1})
        self.sid = await self.db.create_subject("Alice")

    async def test_assign_puts_findings_on_subject(self) -> None:
        receipt = await self.svc.assign_cluster(cluster_id=1, subject_id=self.sid)
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid
        assert await self.db.get_finding_subject_id(self.fx.finding_b_id) == self.sid
        assert receipt.message == "Assigned 2 faces to Alice"

    async def test_assign_undo_clears_assignment(self) -> None:
        receipt = await self.svc.assign_cluster(cluster_id=1, subject_id=self.sid)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        await entry.payload.undo(self.db)
        assert await self.db.get_finding_subject_id(self.fx.finding_a_id) is None
        assert await self.db.get_finding_subject_id(self.fx.finding_b_id) is None

    async def test_species_mismatch_raises(self) -> None:
        pet_sid = await self.db.create_subject("Figaro", kind="pet")
        with pytest.raises(SpeciesMismatch):
            await self.svc.assign_cluster(cluster_id=1, subject_id=pet_sid)
