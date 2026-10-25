"""Shared infrastructure for db mixins: query helpers and the accessor protocol.

``_DBAccessor`` declares the shared state (pool, data_dir, ...) plus stubs for
every method that is called across mixin boundaries.  This lets each mixin call
``self.species_filter(...)`` or ``await self.get_source(...)`` without
type: ignore -- the real implementation lives in the mixin that owns the method,
and FaceDB inherits them all.

Mixin SQL is written with ``?`` placeholders and translated to ``$1``-style on
the way to asyncpg. The queries here are composed from fragments --
``species_filter``, ``_dim_filter``, ``_together_query`` and a dozen
``",".join("?" * n)`` IN-lists -- so a positional placeholder would have to know
its index at the point it is written, which the fragment does not. Translating
once at the boundary keeps the composition readable and puts the numbering in a
single tested place.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID

import asyncpg

if TYPE_CHECKING:
    from .models import Finding, Source, SourcePathMetadata, Subject

_PLACEHOLDER = re.compile(r"\?")


@lru_cache(maxsize=2048)
def to_pg(sql: str) -> str:
    """Rewrite ``?`` placeholders as ``$1``, ``$2``, ... for asyncpg."""
    counter = 0

    def numbered(_: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        return f"${counter}"

    return _PLACEHOLDER.sub(numbered, sql)


def glob_to_like(pattern: str) -> str:
    """Translate a SQLite GLOB pattern to a LIKE pattern.

    ``*`` and ``?`` are the glob wildcards; ``%`` and ``_`` are LIKE's and must
    be escaped so a literal underscore in a filename stays literal.
    """
    escaped = pattern.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    return escaped.replace("*", "%").replace("?", "_")


class _DBAccessor:
    """Shared state that all mixins expect on ``self``.

    FaceDB inherits this (via the mixin chain) so the attributes are real.
    Mixins type ``self`` against this class so static analysers see the
    fields without importing the fully-assembled FaceDB.

    Stub methods raise ``NotImplementedError`` -- they exist purely so that
    cross-mixin calls type-check. The real implementations live in the
    mixin that owns each method; FaceDB inherits them all and the stubs
    are never reached at runtime.
    """

    pool: asyncpg.Pool
    data_dir: Path
    base_dir: Path | None
    _pinned: ContextVar[asyncpg.Connection | None]

    # ── connection plumbing ──────────────────────────────────────────

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        """Yield a connection, reusing the one pinned by ``transaction()``.

        Mixin methods call each other freely, so without the pin a nested call
        inside a transaction would take a second connection from the pool and
        run outside that transaction.
        """
        pinned = self._pinned.get()
        if pinned is not None:
            yield pinned
            return
        async with self.pool.acquire() as connection:
            token = self._pinned.set(connection)
            try:
                yield connection
            finally:
                self._pinned.reset(token)

    async def fetch(self, sql: str, *args: Any) -> list[asyncpg.Record]:
        async with self.acquire() as connection:
            return cast("list[asyncpg.Record]", await connection.fetch(to_pg(sql), *args))

    async def fetchrow(self, sql: str, *args: Any) -> asyncpg.Record | None:
        async with self.acquire() as connection:
            return await connection.fetchrow(to_pg(sql), *args)

    async def fetchval(self, sql: str, *args: Any) -> Any:
        async with self.acquire() as connection:
            return await connection.fetchval(to_pg(sql), *args)

    async def execute(self, sql: str, *args: Any) -> str:
        async with self.acquire() as connection:
            return cast("str", await connection.execute(to_pg(sql), *args))

    async def executemany(self, sql: str, args: Iterable[Sequence[Any]]) -> None:
        rows = list(args)
        if not rows:
            return
        async with self.acquire() as connection:
            await connection.executemany(to_pg(sql), rows)

    def transaction(self) -> Any:
        raise NotImplementedError

    def _now(self) -> datetime:
        return datetime.now(UTC)

    async def _count(self, sql: str, *args: Any) -> int:
        value = await self.fetchval(sql, *args)
        return int(value) if value is not None else 0

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

    async def get_source(self, source_id: UUID) -> Source | None:
        raise NotImplementedError

    async def upsert_source_path_metadata(self, source_id: UUID) -> SourcePathMetadata | None:
        raise NotImplementedError

    # ── FindingMixin stubs ───────────────────────────────────────────

    async def get_finding(self, finding_id: UUID) -> Finding | None:
        raise NotImplementedError

    # ── ClusterMixin stubs ───────────────────────────────────────────

    async def get_cluster_findings(self, cluster_id: int, limit: int = 200) -> list[Finding]:
        raise NotImplementedError

    async def get_unnamed_cluster_count(self, species: str = "human") -> int:
        raise NotImplementedError

    # ── CurationMixin stubs ─────────────────────────────────────────

    def _together_query(
        self, subject_ids: list[UUID], alone: bool = False
    ) -> tuple[str, tuple[object, ...]]:
        raise NotImplementedError

    # ── SubjectMixin stubs ───────────────────────────────────────────

    async def get_subject(self, subject_id: UUID) -> Subject | None:
        raise NotImplementedError

    async def get_subjects(self) -> list[Subject]:
        raise NotImplementedError

    async def get_subject_findings(
        self, subject_id: UUID, limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        raise NotImplementedError
