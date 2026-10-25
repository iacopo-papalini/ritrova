"""Concurrency tests for ``FaceDB.transaction()`` (ADR-012 §M4).

The SQLite version of this file tested a process-wide ``threading.RLock``
and its reentrancy. Both are gone with W1.7, and so is the deadlock test
that existed only to prove the RLock was reentrant — Postgres does the
serialising now, per row rather than per process.

What survives is the property that mattered: concurrent dismissals of one
cluster must not leave it half-dismissed. The receipt count is no longer
asserted. Under the old global lock exactly one caller could ever see a
populated cluster; two transactions in Postgres both read under READ
COMMITTED before either writes, so both can legitimately return a receipt.
That is a change in what the callers observe, not in what lands in the DB.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import numpy as np
import pytest

from ritrova.db import FaceDB
from ritrova.services_domain import CurationService
from ritrova.undo import UndoStore
from tests._helpers import add_findings


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
async def seeded(db: FaceDB) -> AsyncIterator[tuple[FaceDB, list, int]]:
    """Seed N findings all in cluster 1, return (db, finding_ids, cluster)."""
    n = 12
    source_ids = [await db.add_source(f"/t/{i}.jpg", width=100, height=100) for i in range(n)]
    for i, sid in enumerate(source_ids):
        await add_findings(db, [(sid, (0, 0, 10, 10), _unit(i), 0.9)])
    finding_ids = [f.id for src in source_ids for f in await db.get_source_findings(src)]
    cluster_id = 1
    await db.update_cluster_ids(dict.fromkeys(finding_ids, cluster_id))
    yield db, finding_ids, cluster_id


async def test_concurrent_dismiss_same_cluster_leaves_no_half_state(
    seeded: tuple[FaceDB, list, int],
) -> None:
    """Two concurrent dismissals of one cluster: every finding ends up
    dismissed and out of the cluster, with no mixture of the two."""
    db, finding_ids, cluster_id = seeded
    svc = CurationService(db, UndoStore())

    results = await asyncio.gather(
        svc.dismiss_findings_as_cluster(cluster_id),
        svc.dismiss_findings_as_cluster(cluster_id),
        return_exceptions=True,
    )
    for r in results:
        assert not isinstance(r, BaseException), f"dismissal raised: {r!r}"

    for fid in finding_ids:
        curation = await db.get_curation(fid)
        assert curation is not None, f"finding {fid} not curated"
        assert curation[1] == "not_a_face", f"finding {fid} has {curation}"
        assert await db.get_cluster_membership(fid) is None


async def test_nested_transaction_reuses_outer(seeded: tuple[FaceDB, list, int]) -> None:
    """A nested ``transaction()`` joins the open one instead of opening a
    second: Postgres has no nested BEGIN, and the outer owns the commit."""
    db, finding_ids, _cluster_id = seeded
    async with db.transaction():
        async with db.transaction():
            await db.snapshot_findings_fields(finding_ids[:2])
        await db.clear_curations(finding_ids[:2])

    for fid in finding_ids[:2]:
        assert await db.get_curation(fid) is None


async def test_exception_inside_transaction_rolls_back(
    seeded: tuple[FaceDB, list, int],
) -> None:
    """If the body raises, nothing it wrote persists."""
    db, finding_ids, cluster_id = seeded
    assert await db.get_cluster_membership(finding_ids[0]) == cluster_id

    class Sentinel(Exception):
        pass

    with pytest.raises(Sentinel):
        async with db.transaction():
            await db.remove_cluster_memberships(finding_ids)
            assert await db.get_cluster_membership(finding_ids[0]) is None
            raise Sentinel

    assert await db.get_cluster_membership(finding_ids[0]) == cluster_id
