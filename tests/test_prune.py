"""Tests for duplicate finding pruning."""

from __future__ import annotations

import numpy as np

from ._postgres import PostgresTestCase


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


class TestPruneDuplicateFindings(PostgresTestCase):
    async def asyncSetUp(self) -> None:
        await super().asyncSetUp()
        self.source_id = await self.db.add_source("/photo.jpg", width=200, height=200)

    async def _add_finding(
        self,
        scan_id: int,
        species: str = "human",
        subject_id: int | None = None,
        cluster_id: int | None = None,
    ) -> int:
        await self.db.add_findings_batch(
            [(self.source_id, (10, 10, 50, 50), _emb(), 0.95)],
            scan_id=scan_id,
            species=species,
        )
        row = await self.db.fetchrow("SELECT id FROM findings ORDER BY id DESC LIMIT 1")
        fid = row[0]
        if subject_id is not None:
            await self.db.set_subject(fid, subject_id)
        if cluster_id is not None:
            await self.db.set_cluster_memberships({fid: cluster_id})
        return fid

    async def test_no_duplicates_returns_zero(self) -> None:
        subject_id = await self.db.create_subject("Alice")
        scan_id = await self.db.record_scan(self.source_id, "human")
        await self._add_finding(scan_id, subject_id=subject_id)
        report = await self.db.prune_duplicate_findings()
        assert report.total == 0

    async def test_prunes_subject_duplicates(self) -> None:
        subject_id = await self.db.create_subject("Alice")
        # Old scan (human) and new scan (composite) both have a finding for Alice
        scan1 = await self.db.record_scan(self.source_id, "human")
        old_fid = await self._add_finding(scan1, subject_id=subject_id)
        scan2 = await self.db.record_scan(self.source_id, "composite")
        new_fid = await self._add_finding(scan2, subject_id=subject_id)

        assert await self.db.get_finding_count() == 2

        report = await self.db.prune_duplicate_findings()
        assert report.by_subject == 1
        assert await self.db.get_finding_count() == 1
        # Kept the newer one (higher id)
        assert await self.db.get_finding(new_fid) is not None
        assert await self.db.get_finding(old_fid) is None

    async def test_prunes_cluster_duplicates(self) -> None:
        scan1 = await self.db.record_scan(self.source_id, "human")
        old_fid = await self._add_finding(scan1, cluster_id=42)
        scan2 = await self.db.record_scan(self.source_id, "composite")
        new_fid = await self._add_finding(scan2, cluster_id=42)

        report = await self.db.prune_duplicate_findings()
        assert report.by_cluster == 1
        assert await self.db.get_finding(new_fid) is not None
        assert await self.db.get_finding(old_fid) is None

    async def test_dry_run_does_not_delete(self) -> None:
        subject_id = await self.db.create_subject("Bob")
        scan1 = await self.db.record_scan(self.source_id, "human")
        await self._add_finding(scan1, subject_id=subject_id)
        scan2 = await self.db.record_scan(self.source_id, "composite")
        await self._add_finding(scan2, subject_id=subject_id)

        report = await self.db.prune_duplicate_findings(dry_run=True)
        assert report.by_subject == 1
        assert await self.db.get_finding_count() == 2  # nothing deleted

    async def test_different_subjects_not_pruned(self) -> None:
        alice = await self.db.create_subject("Alice")
        bob = await self.db.create_subject("Bob")
        scan = await self.db.record_scan(self.source_id, "composite")
        await self._add_finding(scan, subject_id=alice)
        await self._add_finding(scan, subject_id=bob)

        report = await self.db.prune_duplicate_findings()
        assert report.total == 0
        assert await self.db.get_finding_count() == 2

    async def test_different_sources_not_pruned(self) -> None:
        src2 = await self.db.add_source("/photo2.jpg", width=200, height=200)
        subject_id = await self.db.create_subject("Alice")

        scan1 = await self.db.record_scan(self.source_id, "composite")
        await self._add_finding(scan1, subject_id=subject_id)

        scan2 = await self.db.record_scan(src2, "composite")
        await self.db.add_findings_batch(
            [(src2, (10, 10, 50, 50), _emb(), 0.95)],
            scan_id=scan2,
        )
        fid = (await self.db.fetchrow("SELECT id FROM findings WHERE source_id = ?", src2))[0]
        await self.db.set_subject(fid, subject_id)

        report = await self.db.prune_duplicate_findings()
        assert report.total == 0
