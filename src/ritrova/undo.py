"""In-memory single-step undo store for FEAT-5.

One pending undo at a time across the whole app — a second write clobbers the
previous token. Entries expire after ``ttl`` seconds so stale tokens can never
resurrect old state. Thread-safe: all mutations hold ``_lock``.

The store itself is transport-agnostic. Each endpoint that wants to be undoable
builds a kind-specific payload (see dataclasses below), calls ``put()`` to get
a token, and returns that token to the client. On undo, ``pop()`` hands the
payload back to the endpoint that knows how to invert it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

# ── Payload dataclasses ──────────────────────────────────────────────
# One per undoable operation kind. Each carries exactly the prior-state
# information needed to invert the mutation.


@dataclass
class FindingFieldsSnapshot:
    """Prior (person_id, cluster_id) for a single finding."""

    finding_id: int
    person_id: int | None
    cluster_id: int | None


@dataclass
class ClusterDismissPayload:
    """Undo a ``dismiss_findings`` call on every finding in a cluster.

    ``snapshots`` holds the pre-dismiss person_id/cluster_id per finding so we
    can restore both fields. Undo also deletes the dismissed_findings rows.
    """

    cluster_id: int
    snapshots: list[FindingFieldsSnapshot]


@dataclass
class ClusterMergePayload:
    """Undo a cluster merge: flip cluster_id back from target to source."""

    source_cluster_id: int
    target_cluster_id: int
    moved_finding_ids: list[int]


@dataclass
class ClusterAssignPayload:
    """Undo ``assign_cluster_to_subject``: NULL the person_id on findings that
    were actually updated (those that had person_id IS NULL pre-assign)."""

    cluster_id: int
    subject_id: int
    assigned_finding_ids: list[int]


# ── Store ────────────────────────────────────────────────────────────


@dataclass
class UndoEntry:
    token: str
    kind: str
    description: str
    payload: Any
    created_at: float = field(default_factory=time.monotonic)


class UndoStore:
    """Single-slot, TTL-bounded undo store.

    Thread-safety: callers may hit this from any request thread; all methods
    serialise on ``_lock``. The DB mutations guarded by an undo entry are
    themselves serialised by FaceDB's own lock, so the happens-before chain
    (snapshot → mutate → put) is preserved per-request.
    """

    def __init__(self, ttl: float = 60.0) -> None:
        self._lock = Lock()
        self._current: UndoEntry | None = None
        self._ttl = ttl

    def put(self, kind: str, description: str, payload: Any) -> str:
        """Register a new undoable action. Clobbers any prior pending entry."""
        token = uuid.uuid4().hex
        entry = UndoEntry(token=token, kind=kind, description=description, payload=payload)
        with self._lock:
            self._current = entry
        return token

    def pop(self, token: str) -> UndoEntry | None:
        """Atomically claim and remove the entry matching ``token``.

        Returns None if no pending entry, token mismatch, or the entry is past
        its TTL. One-shot: a successful pop clears the slot.
        """
        with self._lock:
            entry = self._current
            if entry is None or entry.token != token:
                return None
            if time.monotonic() - entry.created_at > self._ttl:
                self._current = None
                return None
            self._current = None
            return entry

    def peek(self) -> UndoEntry | None:
        """Return the current pending entry without consuming it.

        Used by tests and by clients that want to display the pending undo
        description without triggering it.
        """
        with self._lock:
            entry = self._current
            if entry is None:
                return None
            if time.monotonic() - entry.created_at > self._ttl:
                self._current = None
                return None
            return entry

    def clear(self) -> None:
        """Drop any pending entry. Used by tests and on server shutdown."""
        with self._lock:
            self._current = None
