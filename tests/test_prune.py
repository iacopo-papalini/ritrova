"""Tests for duplicate finding pruning."""

from __future__ import annotations

from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest

from ritrova.db import FaceDB


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestPruneDuplicateFindings(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.source_id = self.db.add_source("/photo.jpg", width=200, height=200)

    def _add_finding(
        self,
        scan_id: int,
        species: str = "human",
        person_id: int | None = None,
        cluster_id: int | None = None,
    ) -> int:
        self.db.add_findings_batch(
            [(self.source_id, (10, 10, 50, 50), _emb(), 0.95)],
            scan_id=scan_id,
            species=species,
        )
        row = self.db.conn.execute("SELECT id FROM findings ORDER BY id DESC LIMIT 1").fetchone()
        fid = row[0]
        if person_id is not None:
            self.db.set_subject(fid, person_id)
        if cluster_id is not None:
            self.db.set_cluster_memberships({fid: cluster_id})
        return fid

    def test_no_duplicates_returns_zero(self) -> None:
        subject_id = self.db.create_subject("Alice")
        scan_id = self.db.record_scan(self.source_id, "human")
        self._add_finding(scan_id, person_id=subject_id)
        report = self.db.prune_duplicate_findings()
        assert report.total == 0

    def test_prunes_subject_duplicates(self) -> None:
        subject_id = self.db.create_subject("Alice")
        # Old scan (human) and new scan (composite) both have a finding for Alice
        scan1 = self.db.record_scan(self.source_id, "human")
        old_fid = self._add_finding(scan1, person_id=subject_id)
        scan2 = self.db.record_scan(self.source_id, "composite")
        new_fid = self._add_finding(scan2, person_id=subject_id)

        assert self.db.get_finding_count() == 2

        report = self.db.prune_duplicate_findings()
        assert report.by_subject == 1
        assert self.db.get_finding_count() == 1
        # Kept the newer one (higher id)
        assert self.db.get_finding(new_fid) is not None
        assert self.db.get_finding(old_fid) is None

    def test_prunes_cluster_duplicates(self) -> None:
        scan1 = self.db.record_scan(self.source_id, "human")
        old_fid = self._add_finding(scan1, cluster_id=42)
        scan2 = self.db.record_scan(self.source_id, "composite")
        new_fid = self._add_finding(scan2, cluster_id=42)

        report = self.db.prune_duplicate_findings()
        assert report.by_cluster == 1
        assert self.db.get_finding(new_fid) is not None
        assert self.db.get_finding(old_fid) is None

    def test_dry_run_does_not_delete(self) -> None:
        subject_id = self.db.create_subject("Bob")
        scan1 = self.db.record_scan(self.source_id, "human")
        self._add_finding(scan1, person_id=subject_id)
        scan2 = self.db.record_scan(self.source_id, "composite")
        self._add_finding(scan2, person_id=subject_id)

        report = self.db.prune_duplicate_findings(dry_run=True)
        assert report.by_subject == 1
        assert self.db.get_finding_count() == 2  # nothing deleted

    def test_different_subjects_not_pruned(self) -> None:
        alice = self.db.create_subject("Alice")
        bob = self.db.create_subject("Bob")
        scan = self.db.record_scan(self.source_id, "composite")
        self._add_finding(scan, person_id=alice)
        self._add_finding(scan, person_id=bob)

        report = self.db.prune_duplicate_findings()
        assert report.total == 0
        assert self.db.get_finding_count() == 2

    def test_different_sources_not_pruned(self) -> None:
        src2 = self.db.add_source("/photo2.jpg", width=200, height=200)
        subject_id = self.db.create_subject("Alice")

        scan1 = self.db.record_scan(self.source_id, "composite")
        self._add_finding(scan1, person_id=subject_id)

        scan2 = self.db.record_scan(src2, "composite")
        self.db.add_findings_batch(
            [(src2, (10, 10, 50, 50), _emb(), 0.95)],
            scan_id=scan2,
        )
        fid = self.db.conn.execute(
            "SELECT id FROM findings WHERE source_id = ?", (src2,)
        ).fetchone()[0]
        self.db.set_subject(fid, subject_id)

        report = self.db.prune_duplicate_findings()
        assert report.total == 0
