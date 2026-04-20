"""CurationService round-trip tests (ADR-012 §M3 step 5).

Each test seeds a DB, asks the service to mutate, checks the new state,
then pops the undo token and asserts the original state is restored.
"""

from __future__ import annotations

from unittest import TestCase

import pytest

from ritrova.db import FaceDB
from ritrova.services_domain import CurationService

from ._fixtures import Fixture, build_fixture


class TestDismissCluster(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = CurationService(db, self.fx.undo)
        # Put both findings into the same cluster.
        db.update_cluster_ids({self.fx.finding_a_id: 1, self.fx.finding_b_id: 1})

    def test_dismiss_marks_findings_not_a_face(self) -> None:
        receipt = self.svc.dismiss_findings_as_cluster(1)
        assert receipt is not None
        assert "Dismissed 2" in receipt.message
        for fid in (self.fx.finding_a_id, self.fx.finding_b_id):
            curation = self.db.get_curation(fid)
            assert curation is not None
            assert curation[1] == "not_a_face"

    def test_undo_restores_cluster_membership(self) -> None:
        receipt = self.svc.dismiss_findings_as_cluster(1)
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        # Findings are back in cluster 1, no exclusion row.
        for fid in (self.fx.finding_a_id, self.fx.finding_b_id):
            assert self.db.get_curation(fid) is None
            assert self.db.get_cluster_membership(fid) == 1

    def test_empty_cluster_returns_none(self) -> None:
        assert self.svc.dismiss_findings_as_cluster(999) is None


class TestMarkStrangers(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = CurationService(db, self.fx.undo)
        db.update_cluster_ids({self.fx.finding_a_id: 1, self.fx.finding_b_id: 1})

    def test_mark_batch_writes_stranger_and_drops_cluster(self) -> None:
        receipt = self.svc.mark_strangers([self.fx.finding_a_id])
        assert receipt is not None
        curation = self.db.get_curation(self.fx.finding_a_id)
        assert curation is not None
        assert curation[1] == "stranger"
        assert self.db.get_cluster_membership(self.fx.finding_a_id) is None

    def test_undo_restores_cluster_and_clears_exclusion(self) -> None:
        receipt = self.svc.mark_strangers([self.fx.finding_a_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_curation(self.fx.finding_a_id) is None
        assert self.db.get_cluster_membership(self.fx.finding_a_id) == 1


class TestUnassignFindings(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = CurationService(db, self.fx.undo)
        self.subject_id = db.create_subject("Alice")
        db.assign_finding_to_subject(self.fx.finding_a_id, self.subject_id)
        db.assign_finding_to_subject(self.fx.finding_b_id, self.subject_id)

    def test_unassign_clears_subject_rows(self) -> None:
        receipt = self.svc.unassign_findings([self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) is None
        assert self.db.get_finding_subject_id(self.fx.finding_b_id) is None

    def test_undo_restores_subject_assignments(self) -> None:
        receipt = self.svc.unassign_findings([self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.subject_id
        assert self.db.get_finding_subject_id(self.fx.finding_b_id) == self.subject_id


class TestExcludeFindingsFromCluster(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = CurationService(db, self.fx.undo)
        db.update_cluster_ids({self.fx.finding_a_id: 7, self.fx.finding_b_id: 7})

    def test_exclude_drops_cluster_row_for_target_only(self) -> None:
        receipt = self.svc.exclude_findings_from_cluster(7, [self.fx.finding_a_id])
        assert receipt is not None
        assert self.db.get_cluster_membership(self.fx.finding_a_id) is None
        assert self.db.get_cluster_membership(self.fx.finding_b_id) == 7

    def test_undo_restores_cluster_membership(self) -> None:
        receipt = self.svc.exclude_findings_from_cluster(7, [self.fx.finding_a_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_cluster_membership(self.fx.finding_a_id) == 7


class TestUnassignSingleFinding(TestCase):
    """The photo-page single-face unassign preserves the "Removed face from X" phrasing."""

    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = CurationService(db, self.fx.undo)
        self.sid = db.create_subject("Alice")
        db.assign_finding_to_subject(self.fx.finding_a_id, self.sid)

    def test_single_unassign_message(self) -> None:
        receipt = self.svc.unassign_finding(self.fx.finding_a_id)
        assert receipt is not None
        assert receipt.message == "Removed face from Alice"

    def test_single_unassign_undo_restores(self) -> None:
        receipt = self.svc.unassign_finding(self.fx.finding_a_id)
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid

    def test_single_unassign_already_unassigned_returns_none(self) -> None:
        assert self.svc.unassign_finding(self.fx.finding_b_id) is None
