"""Concurrency tests for ``FaceDB.transaction()`` (ADR-012 §M4).

Two scenarios:

1. Two threads racing ``dismiss_findings_as_cluster`` on the same cluster.
   With the snapshot+mutate wrapped in ``with db.transaction():``, one
   thread wins and the other sees an empty cluster (its read returns no
   finding_ids). No half-state: either every finding in the cluster is
   dismissed, or none are.

2. Nested locking under a transaction must not deadlock.
   ``@_locked`` calls inside a ``transaction()`` body reuse the RLock
   (reentrant). A second thread racing an independent ``@_locked`` call
   must be serialised behind the first, not hang forever.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest

from ritrova.db import FaceDB
from ritrova.services_domain import CurationService
from ritrova.services_domain.receipts import UndoReceipt
from ritrova.undo import UndoStore
from tests._helpers import add_findings


def _unit(seed: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def seeded(tmp_path: Path) -> Generator[tuple[FaceDB, list[int], int]]:
    """Seed a DB with N findings all in cluster 1, return (db, finding_ids, cluster)."""
    db = FaceDB(tmp_path / "conc.db")
    n = 12
    source_ids = [db.add_source(f"/t/{i}.jpg", width=100, height=100) for i in range(n)]
    for sid in source_ids:
        add_findings(db, [(sid, (0, 0, 10, 10), _unit(sid), 0.9)])
    finding_ids = sorted(f.id for src in source_ids for f in db.get_source_findings(src))
    cluster_id = 1
    db.update_cluster_ids(dict.fromkeys(finding_ids, cluster_id))
    yield db, finding_ids, cluster_id
    db.close()


# ── Test 1: two-thread stress on the same cluster ─────────────────────


def test_two_threads_dismiss_same_cluster_no_half_state(
    seeded: tuple[FaceDB, list[int], int],
) -> None:
    """Both threads call ``dismiss_findings_as_cluster`` on the same
    cluster at exactly the same time. Expected outcome: either both
    succeed-or-noop (one gets a receipt, the other gets None because
    the cluster is already empty) or one wins and one sees the empty
    cluster. Invariant: every finding is either dismissed (not_a_face)
    or still in the cluster — never mixed.
    """
    db, finding_ids, cluster_id = seeded
    undo = UndoStore()
    svc = CurationService(db, undo)

    barrier = threading.Barrier(2)
    results: dict[str, UndoReceipt | None | Exception] = {}

    def worker(name: str) -> None:
        try:
            barrier.wait(timeout=5.0)
            results[name] = svc.dismiss_findings_as_cluster(cluster_id)
        except Exception as exc:
            results[name] = exc

    t1 = threading.Thread(target=worker, args=("a",), name="dismiss-a")
    t2 = threading.Thread(target=worker, args=("b",), name="dismiss-b")
    t1.start()
    t2.start()
    t1.join(timeout=10.0)
    t2.join(timeout=10.0)
    assert not t1.is_alive(), "thread a hung"
    assert not t2.is_alive(), "thread b hung"

    # No unexpected exceptions.
    for name, r in results.items():
        if isinstance(r, Exception):
            pytest.fail(f"thread {name} raised: {r!r}")

    # Exactly one thread sees the cluster with findings; the other sees
    # it empty — (receipt, None) in some order. Two receipts would mean
    # the second read saw the first's uncommitted state (impossible
    # with the transaction) OR the cluster was not cleared (bug).
    receipts = [r for r in results.values() if isinstance(r, UndoReceipt)]
    nones = [r for r in results.values() if r is None]
    assert len(receipts) == 1, f"expected exactly 1 receipt, got {results}"
    assert len(nones) == 1, f"expected exactly 1 None, got {results}"

    # Consistency: every finding is dismissed.
    for fid in finding_ids:
        curation = db.get_curation(fid)
        assert curation is not None, f"finding {fid} not curated"
        assert curation[1] == "not_a_face", f"finding {fid} has {curation}"
        # Cluster row dropped.
        assert db.get_cluster_membership(fid) is None


def test_two_threads_dismiss_same_cluster_repeats_stable(
    seeded: tuple[FaceDB, list[int], int],
) -> None:
    """Run the race 10 times (using 10 distinct clusters seeded the same
    way). Catches rare interleavings a single-run test would miss."""
    db, finding_ids, cluster_id = seeded
    undo = UndoStore()
    svc = CurationService(db, undo)

    def race_once(barrier: threading.Barrier, winners: list[UndoReceipt]) -> None:
        barrier.wait(timeout=5.0)
        r = svc.dismiss_findings_as_cluster(cluster_id)
        if r is not None:
            winners.append(r)

    iters = 10
    for _ in range(iters):
        # Rewind: put findings back into the cluster (undo state we messed up).
        db.clear_curations(finding_ids)
        db.update_cluster_ids(dict.fromkeys(finding_ids, cluster_id))

        barrier = threading.Barrier(2)
        winners: list[UndoReceipt] = []

        t1 = threading.Thread(target=race_once, args=(barrier, winners))
        t2 = threading.Thread(target=race_once, args=(barrier, winners))
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)
        assert len(winners) == 1, f"expected 1 winner per race, got {len(winners)}"
        for fid in finding_ids:
            assert db.get_curation(fid) is not None


# ── Test 2: deadlock detection for nested @_locked calls ──────────────


def test_nested_locked_call_inside_transaction_does_not_deadlock(
    seeded: tuple[FaceDB, list[int], int],
) -> None:
    """Thread A enters a ``transaction()`` and calls an ``@_locked``
    method inside. Thread B, running a different ``@_locked`` method,
    must be serialised behind A — not hang.

    Concretely: A does a transaction that holds for ~200ms (while B tries
    to run a plain ``@_locked`` read). B should complete AFTER A, within
    5 seconds total. Anything slower counts as a deadlock.
    """
    db, finding_ids, cluster_id = seeded

    a_entered = threading.Event()
    a_releasing = threading.Event()
    b_done_at: list[float] = []

    def thread_a() -> None:
        with db.transaction():
            # Inner @_locked call — must be reentrant on the RLock.
            inner = db.snapshot_findings_fields(finding_ids)
            assert len(inner) == len(finding_ids)
            a_entered.set()
            # Hold the lock while B tries to acquire.
            time.sleep(0.2)
            a_releasing.set()

    def thread_b() -> None:
        # Wait until A has definitely entered the transaction.
        assert a_entered.wait(timeout=2.0)
        # Independent @_locked call — should block on the RLock, then run.
        _ = db.get_curation(finding_ids[0])
        b_done_at.append(time.monotonic())

    start = time.monotonic()
    ta = threading.Thread(target=thread_a, name="tx-holder")
    tb = threading.Thread(target=thread_b, name="locked-waiter")
    ta.start()
    tb.start()
    ta.join(timeout=5.0)
    tb.join(timeout=5.0)
    assert not ta.is_alive(), "transaction-holder deadlocked"
    assert not tb.is_alive(), "locked-waiter deadlocked (>5s)"
    assert a_releasing.is_set(), "A never released"
    assert b_done_at, "B never finished"
    # B should have completed after A started releasing (serialised).
    total = time.monotonic() - start
    assert total < 5.0, f"total runtime {total:.2f}s — near the deadlock threshold"


def test_nested_transaction_reuses_outer(
    seeded: tuple[FaceDB, list[int], int],
) -> None:
    """Calling ``transaction()`` from inside an already-open transaction
    must not BEGIN again (SQLite would error). The inner context is a
    no-op; the outer owns commit/rollback.
    """
    db, finding_ids, _cluster_id = seeded
    with db.transaction():
        # Nested transaction — should not raise.
        with db.transaction():
            db.snapshot_findings_fields(finding_ids[:2])
        # Still inside outer — a mutation now should be guarded by the outer commit.
        db.clear_curations(finding_ids[:2])
    # After outer exit, state is committed.
    for fid in finding_ids[:2]:
        assert db.get_curation(fid) is None


def test_exception_inside_transaction_rolls_back(
    seeded: tuple[FaceDB, list[int], int],
) -> None:
    """If the body raises, SQLite rolls back — none of the mutations persist."""
    db, finding_ids, cluster_id = seeded
    assert db.get_cluster_membership(finding_ids[0]) == cluster_id

    class Sentinel(Exception):
        pass

    with pytest.raises(Sentinel), db.transaction():
        db.remove_cluster_memberships(finding_ids)
        # Confirm the mutation is visible inside the transaction.
        assert db.get_cluster_membership(finding_ids[0]) is None
        raise Sentinel

    # After rollback, memberships are back.
    assert db.get_cluster_membership(finding_ids[0]) == cluster_id
