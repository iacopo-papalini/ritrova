"""Tests for FEAT-5: global single-step undo.

Two layers:

* ``TestUndoStore`` — unit tests for the in-memory store: put/pop/peek/ttl/clobber.
* ``TestUndoEndpoints`` — integration: each undoable endpoint returns a token
  and posting to ``/api/undo/{token}`` fully reverses the DB state.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest import TestCase

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ritrova.app import create_app
from ritrova.db import FaceDB
from ritrova.undo import (
    DismissPayload,
    FindingFieldsSnapshot,
    UndoStore,
)

from ._helpers import add_findings


def _emb(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _add_finding(db: FaceDB, path: str, seed: int = 42) -> tuple[int, int]:
    """Add source + one human finding, return (source_id, finding_id)."""
    pid = db.add_source(path, width=100, height=100)
    add_findings(db, [(pid, (10, 10, 50, 50), _emb(seed), 0.95)])
    findings = db.get_source_findings(pid)
    return pid, findings[0].id


# ── Unit tests for UndoStore ──────────────────────────────────────────


class TestUndoStore(TestCase):
    def setUp(self) -> None:
        self.store = UndoStore(ttl=60.0)

    def _sample_payload(self, cluster_id: int = 1) -> DismissPayload:
        return DismissPayload(
            snapshots=[FindingFieldsSnapshot(finding_id=10, person_id=5, cluster_id=cluster_id)],
        )

    def test_put_returns_unique_token(self) -> None:
        t1 = self.store.put("one", self._sample_payload(1))
        t2 = self.store.put("two", self._sample_payload(2))
        assert t1 != t2
        # Second put clobbers first — single-slot semantics.
        assert self.store.pop(t1) is None
        entry = self.store.pop(t2)
        assert entry is not None
        assert entry.description == "two"

    def test_pop_is_one_shot(self) -> None:
        token = self.store.put("x", self._sample_payload())
        first = self.store.pop(token)
        assert first is not None
        assert self.store.pop(token) is None

    def test_peek_does_not_consume(self) -> None:
        token = self.store.put("x", self._sample_payload())
        e1 = self.store.peek()
        e2 = self.store.peek()
        assert e1 is not None and e2 is not None
        assert e1.token == e2.token == token
        # Pop still works after peeks.
        assert self.store.pop(token) is not None

    def test_expired_entry_is_not_returned(self) -> None:
        store = UndoStore(ttl=0.0)  # everything is instantly stale
        token = store.put("x", self._sample_payload())
        # Tiny sleep so monotonic() advances past created_at on fast clocks.
        time.sleep(0.001)
        assert store.peek() is None
        assert store.pop(token) is None

    def test_clear_drops_pending(self) -> None:
        token = self.store.put("x", self._sample_payload())
        self.store.clear()
        assert self.store.pop(token) is None

    def test_pop_wrong_token(self) -> None:
        self.store.put("x", self._sample_payload())
        assert self.store.pop("not-a-real-token") is None


# ── Integration tests: the three undoable cluster endpoints ──────────


class TestUndoEndpoints(TestCase):
    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path: Path) -> None:
        self.db = FaceDB(tmp_path / "test.db")
        self.app = create_app(str(tmp_path / "test.db"))
        self.client = TestClient(self.app)

    def test_peek_empty(self) -> None:
        resp = self.client.get("/api/undo/peek")
        assert resp.status_code == 200
        assert resp.json() == {"pending": False}

    def test_undo_unknown_token_returns_404(self) -> None:
        resp = self.client.post("/api/undo/does-not-exist")
        assert resp.status_code == 404

    # ── cluster dismiss ─────────────────────────

    def test_undo_cluster_dismiss_restores_assignments(self) -> None:
        """Dismissing a cluster must be fully reversible: findings return to
        their prior ``(person_id, cluster_id)`` and dismissed_findings is
        cleared for them."""
        sid = self.db.create_subject("Alice")
        _, fid1 = _add_finding(self.db, "/a.jpg", seed=1)
        _, fid2 = _add_finding(self.db, "/b.jpg", seed=2)
        self.db.update_cluster_ids({fid1: 7, fid2: 7})
        # fid1 is both clustered AND assigned; fid2 only clustered. Undo must
        # restore both fields exactly.
        self.db.assign_finding_to_subject(fid1, sid)

        resp = self.client.post("/api/clusters/7/dismiss")
        assert resp.status_code == 200
        body = resp.json()
        assert body["dismissed"] == 2
        assert "undo_token" in body
        assert body["message"].startswith("Dismissed 2 faces")

        # Mid-flight state: both findings NULLed and in dismissed_findings.
        for fid in (fid1, fid2):
            f = self.db.get_finding(fid)
            assert f is not None
            assert f.cluster_id is None
            assert f.person_id is None

        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 200

        f1 = self.db.get_finding(fid1)
        f2 = self.db.get_finding(fid2)
        assert f1 is not None and f2 is not None
        assert f1.cluster_id == 7
        assert f1.person_id == sid
        assert f2.cluster_id == 7
        assert f2.person_id is None

        # Exclusion rows cleared for those findings (the 'not_a_face' row
        # inserted by dismiss should be gone after undo).
        row = self.db.conn.execute(
            "SELECT COUNT(*) FROM finding_assignment "
            "WHERE finding_id IN (?, ?) AND exclusion_reason = 'not_a_face'",
            (fid1, fid2),
        ).fetchone()
        assert row[0] == 0

        # One-shot: replaying undo fails.
        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 404

    # ── cluster merge ───────────────────────────

    def test_undo_cluster_merge_flips_cluster_id_back(self) -> None:
        _, fid_s1 = _add_finding(self.db, "/s1.jpg", seed=1)
        _, fid_s2 = _add_finding(self.db, "/s2.jpg", seed=2)
        _, fid_t = _add_finding(self.db, "/t1.jpg", seed=3)
        self.db.update_cluster_ids({fid_s1: 2, fid_s2: 2, fid_t: 1})

        resp = self.client.post(
            "/api/clusters/merge",
            data={"source_cluster": "2", "target_cluster": "1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "undo_token" in body
        # Mid-flight: all three findings now in cluster 1.
        for fid in (fid_s1, fid_s2, fid_t):
            f = self.db.get_finding(fid)
            assert f is not None
            assert f.cluster_id == 1

        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 200

        # Source findings back in cluster 2; target untouched in 1.
        assert self.db.get_finding(fid_s1).cluster_id == 2  # type: ignore[union-attr]
        assert self.db.get_finding(fid_s2).cluster_id == 2  # type: ignore[union-attr]
        assert self.db.get_finding(fid_t).cluster_id == 1  # type: ignore[union-attr]

    # ── cluster assign ──────────────────────────

    def test_undo_cluster_assign_only_reverts_newly_assigned(self) -> None:
        """Undo must NULL out only the rows the assign actually touched (those
        that were NULL before). Findings already assigned to another subject
        should not be affected by the undo."""
        alice = self.db.create_subject("Alice")
        bob = self.db.create_subject("Bob")
        _, fid_unassigned1 = _add_finding(self.db, "/u1.jpg", seed=1)
        _, fid_unassigned2 = _add_finding(self.db, "/u2.jpg", seed=2)
        _, fid_pre_bob = _add_finding(self.db, "/pb.jpg", seed=3)
        self.db.update_cluster_ids({fid_unassigned1: 5, fid_unassigned2: 5, fid_pre_bob: 5})
        # One finding in the cluster is already Bob's — assign_cluster_to_subject
        # leaves it alone (WHERE person_id IS NULL) and undo must not touch it.
        self.db.assign_finding_to_subject(fid_pre_bob, bob)

        resp = self.client.post(
            "/api/clusters/5/assign",
            data={"person_id": str(alice), "force": "false"},
            headers={"HX-Request": "true"},
        )
        assert resp.status_code == 200
        trigger = resp.headers.get("HX-Trigger")
        assert trigger is not None
        import json

        payload = json.loads(trigger)
        assert "undoToast" in payload
        token = payload["undoToast"]["token"]

        # Mid-flight: the two previously-unassigned findings now point at Alice;
        # Bob's untouched.
        assert self.db.get_finding(fid_unassigned1).person_id == alice  # type: ignore[union-attr]
        assert self.db.get_finding(fid_unassigned2).person_id == alice  # type: ignore[union-attr]
        assert self.db.get_finding(fid_pre_bob).person_id == bob  # type: ignore[union-attr]

        resp = self.client.post(f"/api/undo/{token}")
        assert resp.status_code == 200

        assert self.db.get_finding(fid_unassigned1).person_id is None  # type: ignore[union-attr]
        assert self.db.get_finding(fid_unassigned2).person_id is None  # type: ignore[union-attr]
        # Critically, Bob's finding is still Bob's — undo didn't over-reach.
        assert self.db.get_finding(fid_pre_bob).person_id == bob  # type: ignore[union-attr]

    # ── peek + clobbering ───────────────────────

    def test_peek_returns_latest_pending_after_clobber(self) -> None:
        _, fid_a = _add_finding(self.db, "/a.jpg", seed=1)
        _, fid_b = _add_finding(self.db, "/b.jpg", seed=2)
        self.db.update_cluster_ids({fid_a: 10, fid_b: 20})

        self.client.post("/api/clusters/10/dismiss")
        second = self.client.post("/api/clusters/20/dismiss").json()

        resp = self.client.get("/api/undo/peek")
        data = resp.json()
        assert data["pending"] is True
        # Single-slot: peek reflects the *second* write.
        assert data["token"] == second["undo_token"]

    # ── Round 2A: destructive subject ops ──────

    def _pending_token(self) -> str:
        """Helper: fetch the currently-pending undo token via peek."""
        data = self.client.get("/api/undo/peek").json()
        assert data["pending"] is True, "expected a pending undo but /peek says none"
        token = data["token"]
        assert isinstance(token, str)
        return token

    def test_undo_subject_delete_restores_row_and_assignments(self) -> None:
        """Delete subject + undo: subject row is recreated with the original id,
        and every finding previously assigned to it gets its person_id back."""
        alice = self.db.create_subject("Alice")
        _, fid1 = _add_finding(self.db, "/a1.jpg", seed=1)
        _, fid2 = _add_finding(self.db, "/a2.jpg", seed=2)
        self.db.assign_finding_to_subject(fid1, alice)
        self.db.assign_finding_to_subject(fid2, alice)
        original_row = self.db.get_subject_row(alice)
        assert original_row is not None

        resp = self.client.post(f"/api/subjects/{alice}/delete", follow_redirects=False)
        # The route redirects to /{kind}; that's fine — we just need it to
        # have fired and the undo to be registered.
        assert resp.status_code in (303, 307)

        # Mid-flight: subject gone, findings unassigned.
        assert self.db.get_subject(alice) is None
        assert self.db.get_finding(fid1).person_id is None  # type: ignore[union-attr]
        assert self.db.get_finding(fid2).person_id is None  # type: ignore[union-attr]

        token = self._pending_token()
        resp = self.client.post(f"/api/undo/{token}")
        assert resp.status_code == 200

        # Subject row restored verbatim — same id, name, kind, created_at.
        restored = self.db.get_subject_row(alice)
        assert restored == original_row
        # Findings reassigned.
        assert self.db.get_finding(fid1).person_id == alice  # type: ignore[union-attr]
        assert self.db.get_finding(fid2).person_id == alice  # type: ignore[union-attr]

    def test_undo_subject_delete_with_zero_findings(self) -> None:
        """Deleting an unused subject must still be undoable (no findings to
        restore — just the row)."""
        orphan = self.db.create_subject("NobodyCares")
        original_row = self.db.get_subject_row(orphan)
        self.client.post(f"/api/subjects/{orphan}/delete", follow_redirects=False)
        assert self.db.get_subject(orphan) is None

        self.client.post(f"/api/undo/{self._pending_token()}")
        assert self.db.get_subject_row(orphan) == original_row

    def test_undo_subject_merge_recreates_source_and_unswaps_findings(self) -> None:
        """Source subject is re-INSERTed with its original id; findings that
        moved source -> target are flipped back. Target subject is untouched."""
        source = self.db.create_subject("Bob")
        target = self.db.create_subject("Uncle Bob")
        _, fid_s = _add_finding(self.db, "/s.jpg", seed=1)
        _, fid_t = _add_finding(self.db, "/t.jpg", seed=2)
        self.db.assign_finding_to_subject(fid_s, source)
        self.db.assign_finding_to_subject(fid_t, target)
        source_row = self.db.get_subject_row(source)
        target_row = self.db.get_subject_row(target)

        resp = self.client.post(
            "/api/subjects/merge",
            data={"source_id": str(source), "target_id": str(target)},
            follow_redirects=False,
        )
        assert resp.status_code in (303, 307)
        # Mid-flight: source gone, its finding re-pointed at target.
        assert self.db.get_subject(source) is None
        assert self.db.get_finding(fid_s).person_id == target  # type: ignore[union-attr]
        assert self.db.get_finding(fid_t).person_id == target  # type: ignore[union-attr]

        self.client.post(f"/api/undo/{self._pending_token()}")

        # Source restored with the same row, source's finding back on source,
        # target's own finding untouched.
        assert self.db.get_subject_row(source) == source_row
        assert self.db.get_subject_row(target) == target_row
        assert self.db.get_finding(fid_s).person_id == source  # type: ignore[union-attr]
        assert self.db.get_finding(fid_t).person_id == target  # type: ignore[union-attr]

    # ── Round 2B: mass-update ops ────────────

    def test_undo_findings_dismiss_restores_assignments(self) -> None:
        """Dismissing individual findings (not a whole cluster) is reversible:
        findings return to their prior person_id/cluster_id and are removed
        from dismissed_findings."""
        alice = self.db.create_subject("Alice")
        _, fid1 = _add_finding(self.db, "/d1.jpg", seed=10)
        _, fid2 = _add_finding(self.db, "/d2.jpg", seed=11)
        self.db.update_cluster_ids({fid1: 3, fid2: 3})
        self.db.assign_finding_to_subject(fid1, alice)

        resp = self.client.post("/api/findings/dismiss", json={"face_ids": [fid1, fid2]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["dismissed"] == 2
        assert "undo_token" in body

        # Mid-flight: dismissed, NULLed.
        for fid in (fid1, fid2):
            f = self.db.get_finding(fid)
            assert f is not None
            assert f.cluster_id is None
            assert f.person_id is None

        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 200

        f1 = self.db.get_finding(fid1)
        f2 = self.db.get_finding(fid2)
        assert f1 is not None and f2 is not None
        assert f1.cluster_id == 3 and f1.person_id == alice
        assert f2.cluster_id == 3 and f2.person_id is None

    def test_undo_findings_exclude_restores_cluster_id(self) -> None:
        """Excluding faces from a cluster sets cluster_id to NULL; undo
        restores it."""
        _, fid1 = _add_finding(self.db, "/e1.jpg", seed=20)
        _, fid2 = _add_finding(self.db, "/e2.jpg", seed=21)
        _, fid_stay = _add_finding(self.db, "/e3.jpg", seed=22)
        self.db.update_cluster_ids({fid1: 8, fid2: 8, fid_stay: 8})

        resp = self.client.post(
            "/api/findings/exclude", json={"face_ids": [fid1, fid2], "cluster_id": 8}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["excluded"] == 2
        assert "undo_token" in body

        # Mid-flight: excluded findings have NULL cluster_id; fid_stay untouched.
        assert self.db.get_finding(fid1).cluster_id is None  # type: ignore[union-attr]
        assert self.db.get_finding(fid2).cluster_id is None  # type: ignore[union-attr]
        assert self.db.get_finding(fid_stay).cluster_id == 8  # type: ignore[union-attr]

        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 200

        assert self.db.get_finding(fid1).cluster_id == 8  # type: ignore[union-attr]
        assert self.db.get_finding(fid2).cluster_id == 8  # type: ignore[union-attr]

    def test_undo_claim_faces_restores_prior_person_ids(self) -> None:
        """claim-faces overwrites person_id; undo restores each finding's
        prior value (including NULL for unassigned and a different subject)."""
        alice = self.db.create_subject("Alice")
        bob = self.db.create_subject("Bob")
        _, fid_none = _add_finding(self.db, "/c1.jpg", seed=30)
        _, fid_bob = _add_finding(self.db, "/c2.jpg", seed=31)
        self.db.assign_finding_to_subject(fid_bob, bob)

        resp = self.client.post(
            f"/api/subjects/{alice}/claim-faces",
            json={"face_ids": [fid_none, fid_bob], "force": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["claimed"] == 2
        assert "undo_token" in body

        # Mid-flight: both assigned to Alice.
        assert self.db.get_finding(fid_none).person_id == alice  # type: ignore[union-attr]
        assert self.db.get_finding(fid_bob).person_id == alice  # type: ignore[union-attr]

        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 200

        # Restored: fid_none back to NULL, fid_bob back to Bob.
        assert self.db.get_finding(fid_none).person_id is None  # type: ignore[union-attr]
        assert self.db.get_finding(fid_bob).person_id == bob  # type: ignore[union-attr]

    def test_undo_swap_findings_restores_prior_person_ids(self) -> None:
        """swap reassigns findings to a target subject; undo restores each
        finding's prior person_id."""
        alice = self.db.create_subject("Alice")
        bob = self.db.create_subject("Bob")
        _, fid1 = _add_finding(self.db, "/w1.jpg", seed=40)
        _, fid2 = _add_finding(self.db, "/w2.jpg", seed=41)
        self.db.assign_finding_to_subject(fid1, alice)
        self.db.assign_finding_to_subject(fid2, alice)

        resp = self.client.post(
            "/api/findings/swap",
            json={"face_ids": [fid1, fid2], "target_person_id": bob},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["swapped"] == 2
        assert "undo_token" in body

        # Mid-flight: both now Bob's.
        assert self.db.get_finding(fid1).person_id == bob  # type: ignore[union-attr]
        assert self.db.get_finding(fid2).person_id == bob  # type: ignore[union-attr]

        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 200

        # Restored back to Alice.
        assert self.db.get_finding(fid1).person_id == alice  # type: ignore[union-attr]
        assert self.db.get_finding(fid2).person_id == alice  # type: ignore[union-attr]

    def test_undo_finding_unassign_restores_person_id(self) -> None:
        """Removing a face from a person makes it vanish from the grid;
        undo must restore the assignment."""
        alice = self.db.create_subject("Alice")
        _, fid = _add_finding(self.db, "/un1.jpg", seed=50)
        self.db.assign_finding_to_subject(fid, alice)

        resp = self.client.post(f"/api/findings/{fid}/unassign")
        assert resp.status_code == 200
        body = resp.json()
        assert "undo_token" in body
        assert self.db.get_finding(fid).person_id is None  # type: ignore[union-attr]

        resp = self.client.post(f"/api/undo/{body['undo_token']}")
        assert resp.status_code == 200
        assert self.db.get_finding(fid).person_id == alice  # type: ignore[union-attr]

    # ── cluster name ───────────────────────────

    def test_undo_cluster_name_deletes_subject_and_frees_findings(self) -> None:
        """cluster/name creates a subject + assigns the cluster in one shot.
        Undo must remove the subject entirely AND free the findings it grabbed."""
        _, fid1 = _add_finding(self.db, "/c1.jpg", seed=1)
        _, fid2 = _add_finding(self.db, "/c2.jpg", seed=2)
        self.db.update_cluster_ids({fid1: 42, fid2: 42})

        # Pre-state: no subject with this name.
        assert not any(s.name == "Freshly Named" for s in self.db.get_subjects())

        resp = self.client.post(
            "/api/clusters/42/name",
            data={"name": "Freshly Named"},
            follow_redirects=False,
        )
        assert resp.status_code in (303, 307)

        created = next(s for s in self.db.get_subjects() if s.name == "Freshly Named")
        assert self.db.get_finding(fid1).person_id == created.id  # type: ignore[union-attr]

        self.client.post(f"/api/undo/{self._pending_token()}")

        # Subject gone, findings back to NULL person_id but still in the cluster.
        assert not any(s.name == "Freshly Named" for s in self.db.get_subjects())
        for fid in (fid1, fid2):
            f = self.db.get_finding(fid)
            assert f is not None
            assert f.person_id is None
            assert f.cluster_id == 42

    # mark-stranger tests are deleted in this commit — the endpoint is being
    # rewritten against the new finding_assignment.exclusion_reason='stranger'
    # model in Commit C. New tests for that model will land then.
    #
    # test_subjects_page_hide_strangers_default_on is also gone: the
    # hide-strangers pill itself goes away (subjects don't carry the stranger
    # state anymore; findings do).
