"""SubjectService round-trip tests (ADR-012 §M3 step 5)."""

from __future__ import annotations

from unittest import TestCase

import pytest

from ritrova.db import FaceDB
from ritrova.services_domain import SpeciesMismatch, SubjectService

from ._fixtures import Fixture, build_fixture


class TestClaimFaces(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = SubjectService(db, self.fx.undo)
        self.sid = db.create_subject("Alice")

    def test_claim_assigns_each_finding(self) -> None:
        receipt = self.svc.claim_faces(self.sid, [self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        assert receipt.message == "Claimed 2 faces for Alice"
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid
        assert self.db.get_finding_subject_id(self.fx.finding_b_id) == self.sid

    def test_claim_undo_restores_prior_state(self) -> None:
        receipt = self.svc.claim_faces(self.sid, [self.fx.finding_a_id, self.fx.finding_b_id])
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) is None
        assert self.db.get_finding_subject_id(self.fx.finding_b_id) is None

    def test_claim_empty_returns_none(self) -> None:
        assert self.svc.claim_faces(self.sid, []) is None


class TestClaimFacesSpeciesMismatch(TestCase):
    """Cross-species claim raises SpeciesMismatch unless force=True."""

    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        # Seed with pets.
        self.fx: Fixture = build_fixture(db, species="dog")
        self.db = db
        self.svc = SubjectService(db, self.fx.undo)
        self.person_sid = db.create_subject("Alice", kind="person")

    def test_mismatch_raises(self) -> None:
        with pytest.raises(SpeciesMismatch):
            self.svc.claim_faces(self.person_sid, [self.fx.finding_a_id])

    def test_force_overrides(self) -> None:
        receipt = self.svc.claim_faces(self.person_sid, [self.fx.finding_a_id], force=True)
        assert receipt is not None
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.person_sid


class TestMergeSubjects(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = SubjectService(db, self.fx.undo)
        self.src_sid = db.create_subject("Alice")
        self.dst_sid = db.create_subject("Alexa")
        db.assign_finding_to_subject(self.fx.finding_a_id, self.src_sid)

    def test_merge_moves_findings_and_deletes_source(self) -> None:
        receipt = self.svc.merge_subjects(self.src_sid, self.dst_sid)
        assert "Merged Alice into Alexa" in receipt.message
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.dst_sid
        assert self.db.get_subject(self.src_sid) is None

    def test_merge_undo_resurrects_source(self) -> None:
        receipt = self.svc.merge_subjects(self.src_sid, self.dst_sid)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        restored = self.db.get_subject(self.src_sid)
        assert restored is not None and restored.name == "Alice"
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.src_sid

    def test_merge_self_raises(self) -> None:
        with pytest.raises(ValueError):
            self.svc.merge_subjects(self.src_sid, self.src_sid)


class TestDeleteSubject(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = SubjectService(db, self.fx.undo)
        self.sid = db.create_subject("Alice")
        db.assign_finding_to_subject(self.fx.finding_a_id, self.sid)

    def test_delete_removes_subject_and_unassigns(self) -> None:
        receipt = self.svc.delete_subject(self.sid)
        assert "Deleted Alice" in receipt.message
        assert self.db.get_subject(self.sid) is None
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) is None

    def test_delete_undo_resurrects(self) -> None:
        receipt = self.svc.delete_subject(self.sid)
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        restored = self.db.get_subject(self.sid)
        assert restored is not None and restored.name == "Alice"
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.sid


class TestSwapFindings(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, db: FaceDB) -> None:
        self.fx: Fixture = build_fixture(db)
        self.db = db
        self.svc = SubjectService(db, self.fx.undo)
        self.from_sid = db.create_subject("From")
        self.to_sid = db.create_subject("To")
        db.assign_finding_to_subject(self.fx.finding_a_id, self.from_sid)

    def test_swap_moves_finding(self) -> None:
        receipt = self.svc.swap_findings([self.fx.finding_a_id], self.to_sid)
        assert receipt is not None
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.to_sid
        assert receipt.message == "Swapped 1 face for To"

    def test_swap_undo_restores_original_subject(self) -> None:
        receipt = self.svc.swap_findings([self.fx.finding_a_id], self.to_sid)
        assert receipt is not None
        entry = self.fx.undo.pop(receipt.token)
        assert entry is not None
        entry.payload.undo(self.db)
        assert self.db.get_finding_subject_id(self.fx.finding_a_id) == self.from_sid
