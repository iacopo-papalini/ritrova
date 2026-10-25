"""FaceDB: the assembled mixin class and its asyncpg connection pool.

Schema creation lives in Alembic (D4), not here: the on-open ``executescript``
and the one-shot SQLite migrations it used to run are gone, along with the
shared connection, the re-entrant lock and ``_TransactionalConnection``. Those
existed because one SQLite connection was shared by every request thread; a
pool and an event loop replace all of it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Self

import asyncpg

from .assignments import AssignmentMixin
from .circles import CirclesMixin
from .clusters import ClusterMixin
from .curation import CurationMixin
from .descriptions import DescriptionMixin
from .findings import FindingMixin
from .maintenance import MaintenanceMixin
from .path_metadata import PathMetadataMixin
from .paths import PathMixin
from .print_selection import PrintSelectionMixin
from .scans import ScanMixin
from .source_search import SourceSearchMixin
from .sources import SourceMixin
from .subjects import SubjectMixin
from .undo_support import UndoMixin

# Logical FK metadata for scans.scan_type — kept in sync with the shipping
# pipelines via ``sync_scan_types``. Retire entries by setting ``retired_at``
# rather than deleting them: historic scans keep pointing at the row for
# audit / A/B lineage.
KNOWN_SCAN_TYPES: list[dict[str, str | None]] = [
    {
        "name": "human",
        "pipeline": "arcface(buffalo_l, det_size=640)",
        "outputs": "findings(species=human, 512-dim embedding, bbox, confidence)",
        "introduced_at": "2026-04",
        "retired_at": "2026-04-20",
        "notes": (
            "Retired in ADR-012 §M3. Pre-composite face-only pipeline. "
            "Historic `human` scan rows stay put (logical FK only); new "
            "writes are `subjects`. Rescan no longer produces this type."
        ),
    },
    {
        "name": "pet",
        "pipeline": "yolo(yolo11m, classes={cat=15,dog=16}) -> siglip(base-patch16-224)",
        "outputs": "findings(species=dog|cat|other_pet, 768-dim embedding, bbox, confidence)",
        "introduced_at": "2026-04",
        "retired_at": "2026-04-20",
        "notes": (
            "Retired in ADR-012 §M3. Pre-composite pet-only pipeline. "
            "Historic `pet` scan rows stay put; new writes are `subjects`."
        ),
    },
    {
        "name": "composite",
        "pipeline": (
            "qwen2.5-vl-7b(mlx,4-bit) prefilter -> arcface(buffalo_l, min_face_size=50, "
            "sharpness+edge filters) -> yolo+siglip -> opus-mt-tc-big-en-it -> dedup"
        ),
        "outputs": "mixed findings + descriptions(it caption + it tags)",
        "introduced_at": "2026-04-14",
        "retired_at": "2026-04-18",
        "notes": "Retired; see ADR-011. Italian translation quality + face recall loss.",
    },
    {
        "name": "subjects",
        "pipeline": (
            "arcface(buffalo_l, det_size=640, min_face_size=50, sharpness+edge filters) "
            "-> yolo11m+siglip-base-patch16-224 -> dedup"
        ),
        "outputs": "findings(species=human|dog|cat|other_pet, embedding, bbox, confidence)",
        "introduced_at": "2026-04-18",
        "retired_at": None,
        "notes": "Default `ritrova analyse`. Unifies human+pet under one scan.",
    },
    {
        "name": "subjects+captions",
        "pipeline": (
            "qwen2.5-vl-7b(mlx,4-bit) prefilter -> arcface(min_face_size=50) "
            "-> yolo+siglip -> opus-mt-tc-big-en-it -> dedup"
        ),
        "outputs": "subjects findings + descriptions(it caption + it tags)",
        "introduced_at": "2026-04-18",
        "retired_at": None,
        "notes": "Opt-in via `ritrova analyse --caption`. Apple Silicon only (MLX).",
    },
]


class FaceDB(
    PathMixin,
    SourceMixin,
    ScanMixin,
    FindingMixin,
    SubjectMixin,
    ClusterMixin,
    CurationMixin,
    DescriptionMixin,
    PathMetadataMixin,
    SourceSearchMixin,
    PrintSelectionMixin,
    CirclesMixin,
    AssignmentMixin,
    UndoMixin,
    MaintenanceMixin,
):
    """Query layer over Postgres. Build one with ``await FaceDB.connect(...)``."""

    def __init__(self, pool: asyncpg.Pool, data_dir: Path, base_dir: Path | None) -> None:
        self.pool = pool
        self.data_dir = data_dir
        self.base_dir = base_dir
        self._pinned: ContextVar[asyncpg.Connection | None] = ContextVar("pinned", default=None)

    @classmethod
    async def connect(
        cls,
        dsn: str,
        data_dir: str | Path,
        base_dir: str | Path | None = None,
        *,
        min_size: int = 2,
        max_size: int = 10,
    ) -> Self:
        """Open the pool. ``data_dir`` holds the generated artefacts — video
        frames under ``tmp/``, thumbnails, analysis reports — that used to sit
        next to the SQLite file."""
        resolved_data_dir = Path(data_dir)
        resolved_data_dir.mkdir(parents=True, exist_ok=True)
        pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
        if pool is None:
            msg = f"could not open a connection pool to {dsn}"
            raise RuntimeError(msg)
        return cls(
            pool,
            resolved_data_dir,
            Path(base_dir).resolve() if base_dir else None,
        )

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        """Atomic snapshot+mutate over one pinned connection.

        Nested calls join the outer transaction rather than opening a second
        one, which is what the old ``_skip_commit`` flag was for. On exception
        asyncpg rolls back and re-raises.
        """
        async with self.acquire() as connection:
            if connection.is_in_transaction():
                yield
                return
            async with connection.transaction():
                yield

    async def sync_scan_types(self) -> None:
        """Upsert the known scan_types entries.

        Idempotent — keeps the catalog aligned with the code's current
        understanding of pipelines and their outputs. When a new scan_type
        ships, add it to ``KNOWN_SCAN_TYPES`` and call this.
        """
        await self.executemany(
            "INSERT INTO scan_types "
            "(name, pipeline, outputs, introduced_at, retired_at, notes) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (name) DO UPDATE SET "
            "pipeline = EXCLUDED.pipeline, outputs = EXCLUDED.outputs, "
            "introduced_at = EXCLUDED.introduced_at, retired_at = EXCLUDED.retired_at, "
            "notes = EXCLUDED.notes",
            [
                (
                    row["name"],
                    row["pipeline"],
                    row["outputs"],
                    row.get("introduced_at"),
                    row.get("retired_at"),
                    row.get("notes"),
                )
                for row in KNOWN_SCAN_TYPES
            ],
        )

    async def run(self, sql: str, *args: object) -> None:
        """Execute SQL. For ad-hoc queries from outside."""
        await self.execute(sql, *args)

    async def query(self, sql: str, *args: object) -> list[asyncpg.Record]:
        """Execute SQL and return all rows."""
        return await self.fetch(sql, *args)

    async def close(self) -> None:
        await self.pool.close()
