"""ClusterService round-trip tests (ADR-012 §M3 step 5)."""

from __future__ import annotations

from unittest import TestCase

import pytest

from ritrova.db import FaceDB
from ritrova.services_domain import ClusterService, SpeciesMismatch

from ._fixtures import Fixture, build_fixture


class TestMergeClusters(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = ClusterService(db, self.fx.undo)
        db.update_cluster_ids({self.fx.finding_a_id: 5, self.fx.finding_b_id: 9})

    def test_merge_moves_findings_into_target(self) -> None:
        receipt = self.svc.merge_clusters(source_id=5, target_id=9)
        assert self.db.get_cluster_membership(self.fx.finding_a_id) == 9
        assert self.db.get_cluster_membership(self.fx.finding_b_id) == 9
        assert "Merged cluster #5 into #9" in receipt.message

    def test_merge_undo_restores_source_cluster(self) -> None:
        receipt = self.svc.merge_clusters(source_id=5, target_id=9)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_cluster_membership(self.fx.finding_a_id) == 5
        assert self.db.get_cluster_membership(self.fx.finding_b_id) == 9


class TestAssignCluster(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = ClusterService(db, self.fx.undo)
        db.update_cluster_ids({self.fx.finding_a_id: 1, self.fx.finding_b_id: 1})
        self.sid = db.create_subject("Alice")

    def test_assign_puts_findings_on_subject(self) -> None:
        receipt = self.svc.assign_cluster(cluster_id=1, subject_id=self.sid)
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid
        assert self.db.get_finding_subject_id(self.fx.finding_b_id) == self.sid
        assert receipt.message == "Assigned 2 faces to Alice"

    def test_assign_undo_clears_assignment(self) -> None:
        receipt = self.svc.assign_cluster(cluster_id=1, subject_id=self.sid)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) is None
        assert self.db.get_finding_subject_id(self.fx.finding_b_id) is None

    def test_species_mismatch_raises(self) -> None:
        pet_sid = self.db.create_subject("Figaro", kind="pet")
        with pytest.raises(SpeciesMismatch):
            self.svc.assign_cluster(cluster_id=1, subject_id=pet_sid)
