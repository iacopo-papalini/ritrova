"""Shared infrastructure for db mixins: lock decorator and accessor protocol.

``_DBAccessor`` declares the shared state (conn, db_path, ...) plus stubs for
every method that is called across mixin boundaries.  This lets each mixin call
``self.species_filter(...)`` or ``self.get_source(...)`` without type: ignore --
the real implementation lives in the mixin that owns the method, and FaceDB
inherits them all.
"""

from __future__ import annotations

import functools
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

if TYPE_CHECKING:
    from .models import Finding, Source, Subject

P = ParamSpec("P")
R = TypeVar("R")


def _locked(method: Callable[..., R]) -> Callable[..., R]:
    """Decorator: hold the DB lock for the entire method call."""

    @functools.wraps(method)
    def wrapper(self: _DBAccessor, *args: Any, **kwargs: Any) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class _DBAccessor:
    """Shared state that all mixins expect on ``self``.

    FaceDB inherits this (via the mixin chain) so the attributes are real.
    Mixins type ``self`` against this class so static analysers see the
    fields without importing the fully-assembled FaceDB.

    Stub methods raise ``NotImplementedError`` — they exist purely so that
    cross-mixin calls type-check. The real implementations live in the
    mixin that owns each method; FaceDB inherits them all and the stubs
    are never reached at runtime.
    """

    conn: sqlite3.Connection
    db_path: Path
    base_dir: Path | None
    _lock: threading.RLock

    # ── helpers (implemented on FaceDB / connection.py) ──────────────

    def _now(self) -> str:
        raise NotImplementedError

    def _count(self, sql: str, params: tuple[str, ...] = ()) -> int:
        raise NotImplementedError

    # ── PathMixin stubs ──────────────────────────────────────────────

    def species_filter(self, species: str) -> tuple[str, tuple[str, ...]]:
        raise NotImplementedError

    def _dim_filter(self, embedding_dim: int | None) -> tuple[str, tuple[int, ...]]:
        raise NotImplementedError

    def _species_for_kind(self, kind: str) -> str:
        raise NotImplementedError

    def _kind_for_species(self, species: str) -> str:
        raise NotImplementedError

    def _is_species_kind_compatible(self, finding_species: str, subject_kind: str) -> bool:
        raise NotImplementedError

    # ── SourceMixin stubs ────────────────────────────────────────────

    def get_source(self, source_id: int) -> Source | None:
        raise NotImplementedError

    # ── FindingMixin stubs ───────────────────────────────────────────

    def get_finding(self, finding_id: int) -> Finding | None:
        raise NotImplementedError

    # ── ClusterMixin stubs ───────────────────────────────────────────

    def get_cluster_findings(self, cluster_id: int, limit: int = 200) -> list[Finding]:
        raise NotImplementedError

    def get_unnamed_cluster_count(self, species: str = "human") -> int:
        raise NotImplementedError

    # ── SubjectMixin stubs ───────────────────────────────────────────

    def get_subject(self, subject_id: int) -> Subject | None:
        raise NotImplementedError

    def get_subjects(self) -> list[Subject]:
        raise NotImplementedError

    def get_subject_findings(
        self, subject_id: int, limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        raise NotImplementedError
