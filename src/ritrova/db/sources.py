"""Source CRUD mixin."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from uuid6 import uuid7

from ._base import _DBAccessor
from .models import Source


class SourceMixin(_DBAccessor):
    async def add_source(
        self,
        file_path: str,
        source_type: str = "photo",
        width: int = 0,
        height: int = 0,
        taken_at: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> UUID:
        source_id = uuid7()
        await self.execute(
            "INSERT INTO sources (id, file_path, type, width, height, taken_at, latitude, longitude) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            source_id,
            file_path,
            source_type,
            width,
            height,
            taken_at,
            latitude,
            longitude,
        )
        await self.upsert_source_path_metadata(source_id)
        return source_id

    async def get_or_create_source(
        self,
        file_path: str,
        source_type: str = "photo",
        width: int = 0,
        height: int = 0,
        taken_at: datetime | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> UUID:
        """Return existing source ID or create a new one."""
        existing: UUID | None = await self.fetchval(
            "SELECT id FROM sources WHERE file_path = ?", file_path
        )
        if existing:
            return existing
        return await self.add_source(
            file_path, source_type, width, height, taken_at, latitude, longitude
        )

    async def get_source(self, source_id: UUID) -> Source | None:
        row = await self.fetchrow("SELECT * FROM sources WHERE id = ?", source_id)
        if not row:
            return None
        return Source(**dict(row))

    async def get_source_by_path(self, file_path: str) -> Source | None:
        row = await self.fetchrow("SELECT * FROM sources WHERE file_path = ?", file_path)
        if not row:
            return None
        return Source(**dict(row))

    async def get_sources_batch(self, source_ids: list[UUID]) -> dict[UUID, Source]:
        """Return {source_id: Source} for multiple sources in one query."""
        if not source_ids:
            return {}
        rows = await self.fetch("SELECT * FROM sources WHERE id = ANY(?)", source_ids)
        return {r["id"]: Source(**dict(r)) for r in rows}

    async def get_source_count(self) -> int:
        return await self._count("SELECT COUNT(*) FROM sources WHERE type = 'photo'")

    async def get_all_source_ids(self) -> list[UUID]:
        """Return all source_ids, ordered by id."""
        rows = await self.fetch("SELECT id FROM sources ORDER BY id")
        return [r["id"] for r in rows]
