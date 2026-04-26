"""Source CRUD mixin."""

from __future__ import annotations

from ._base import _DBAccessor, _locked
from .models import Source


class SourceMixin(_DBAccessor):
    @_locked
    def add_source(
        self,
        file_path: str,
        source_type: str = "photo",
        width: int = 0,
        height: int = 0,
        taken_at: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO sources (file_path, type, width, height, taken_at, latitude, longitude) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (file_path, source_type, width, height, taken_at, latitude, longitude),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        source_id = int(cur.lastrowid)
        self.upsert_source_path_metadata(source_id)
        return source_id

    @_locked
    def get_or_create_source(
        self,
        file_path: str,
        source_type: str = "photo",
        width: int = 0,
        height: int = 0,
        taken_at: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> int:
        """Return existing source ID or create a new one."""
        row = self.conn.execute(
            "SELECT id FROM sources WHERE file_path = ?", (file_path,)
        ).fetchone()
        if row:
            return int(row[0])
        return self.add_source(file_path, source_type, width, height, taken_at, latitude, longitude)

    @_locked
    def get_source(self, source_id: int) -> Source | None:
        row = self.conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if not row:
            return None
        return Source(**dict(row))

    @_locked
    def get_source_by_path(self, file_path: str) -> Source | None:
        row = self.conn.execute(
            "SELECT * FROM sources WHERE file_path = ?", (file_path,)
        ).fetchone()
        if not row:
            return None
        return Source(**dict(row))

    @_locked
    def get_sources_batch(self, source_ids: list[int]) -> dict[int, Source]:
        """Return {source_id: Source} for multiple sources in one query."""
        if not source_ids:
            return {}
        placeholders = ",".join("?" * len(source_ids))
        rows = self.conn.execute(
            f"SELECT * FROM sources WHERE id IN ({placeholders})", source_ids
        ).fetchall()
        return {r["id"]: Source(**dict(r)) for r in rows}

    @_locked
    def get_source_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM sources WHERE type = 'photo'").fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_all_source_ids(self) -> list[int]:
        """Return all source_ids, ordered by id."""
        rows = self.conn.execute("SELECT id FROM sources ORDER BY id").fetchall()
        return [r[0] for r in rows]
