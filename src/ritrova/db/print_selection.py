"""Persistent ordered print-selection worklist."""

from __future__ import annotations

from uuid import UUID

from ._base import _DBAccessor
from .models import PrintSelectionItem, Source

_ORDERED = "ORDER BY position, added_at, source_id"


class PrintSelectionMixin(_DBAccessor):
    async def list_print_selection(self) -> list[PrintSelectionItem]:
        rows = await self.fetch(
            """
            SELECT s.*, ps.position, ps.added_at
            FROM print_selection ps
            JOIN sources s ON s.id = ps.source_id
            ORDER BY ps.position, ps.added_at, ps.source_id
            """
        )
        return [
            PrintSelectionItem(
                source=Source(
                    id=r["id"],
                    file_path=r["file_path"],
                    type=r["type"],
                    width=r["width"],
                    height=r["height"],
                    taken_at=r["taken_at"],
                    latitude=r["latitude"],
                    longitude=r["longitude"],
                ),
                position=int(r["position"]),
                added_at=r["added_at"],
            )
            for r in rows
        ]

    async def get_print_selection_ids(self) -> list[UUID]:
        rows = await self.fetch(f"SELECT source_id FROM print_selection {_ORDERED}")
        return [r["source_id"] for r in rows]

    async def add_to_print_selection(self, source_id: UUID) -> int:
        source = await self.get_source(source_id)
        if source is None:
            raise ValueError("Source not found")
        if source.type != "photo":
            raise ValueError("Only photos can be selected for print")
        existing = await self.fetchval(
            "SELECT position FROM print_selection WHERE source_id = ?", source_id
        )
        if existing is not None:
            return int(existing)
        position = int(
            await self.fetchval("SELECT COALESCE(MAX(position), 0) + 1 FROM print_selection")
        )
        await self.execute(
            "INSERT INTO print_selection (source_id, position, added_at) VALUES (?, ?, ?)",
            source_id,
            position,
            self._now(),
        )
        return position

    async def remove_from_print_selection(self, source_id: UUID) -> None:
        async with self.transaction():
            await self.execute("DELETE FROM print_selection WHERE source_id = ?", source_id)
            await self._compact_print_selection_positions()

    async def clear_print_selection(self) -> None:
        await self.execute("DELETE FROM print_selection")

    async def restore_print_selection_ids(self, source_ids: list[UUID]) -> None:
        """Replace the print worklist with ``source_ids`` in the given order."""
        async with self.transaction():
            await self.execute("DELETE FROM print_selection")
            now = self._now()
            await self.executemany(
                "INSERT INTO print_selection (source_id, position, added_at) VALUES (?, ?, ?)",
                [
                    (source_id, position, now)
                    for position, source_id in enumerate(source_ids, start=1)
                ],
            )

    async def reorder_print_selection(self, source_ids: list[UUID]) -> None:
        current = set(await self.get_print_selection_ids())
        ordered: list[UUID] = []
        seen: set[UUID] = set()
        for source_id in source_ids:
            if source_id in current and source_id not in seen:
                ordered.append(source_id)
                seen.add(source_id)
        for source_id in await self.get_print_selection_ids():
            if source_id not in seen:
                ordered.append(source_id)
        # Two passes through negative positions: the unique index on `position`
        # is checked per statement, so renumbering in place would collide with
        # a row that has not moved yet.
        async with self.transaction():
            await self.executemany(
                "UPDATE print_selection SET position = ? WHERE source_id = ?",
                [(-position, sid) for position, sid in enumerate(ordered, start=1)],
            )
            await self.executemany(
                "UPDATE print_selection SET position = ? WHERE source_id = ?",
                [(position, sid) for position, sid in enumerate(ordered, start=1)],
            )

    async def _compact_print_selection_positions(self) -> None:
        rows = await self.fetch(f"SELECT source_id FROM print_selection {_ORDERED}")
        await self.executemany(
            "UPDATE print_selection SET position = ? WHERE source_id = ?",
            [(position, row["source_id"]) for position, row in enumerate(rows, start=1)],
        )
