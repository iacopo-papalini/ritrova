"""Persistent ordered print-selection worklist."""

from __future__ import annotations

from ._base import _DBAccessor, _locked
from .models import PrintSelectionItem, Source


class PrintSelectionMixin(_DBAccessor):
    @_locked
    def list_print_selection(self) -> list[PrintSelectionItem]:
        rows = self.conn.execute(
            """
            SELECT s.*, ps.position, ps.added_at
            FROM print_selection ps
            JOIN sources s ON s.id = ps.source_id
            ORDER BY ps.position, ps.added_at, ps.source_id
            """
        ).fetchall()
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

    @_locked
    def get_print_selection_ids(self) -> list[int]:
        rows = self.conn.execute(
            "SELECT source_id FROM print_selection ORDER BY position, added_at, source_id"
        ).fetchall()
        return [int(r["source_id"]) for r in rows]

    @_locked
    def add_to_print_selection(self, source_id: int) -> int:
        source = self.get_source(source_id)
        if source is None:
            raise ValueError("Source not found")
        if source.type != "photo":
            raise ValueError("Only photos can be selected for print")
        existing = self.conn.execute(
            "SELECT position FROM print_selection WHERE source_id = ?", (source_id,)
        ).fetchone()
        if existing:
            return int(existing["position"])
        row = self.conn.execute(
            "SELECT COALESCE(MAX(position), 0) + 1 FROM print_selection"
        ).fetchone()
        position = int(row[0]) if row else 1
        self.conn.execute(
            "INSERT INTO print_selection (source_id, position, added_at) VALUES (?, ?, ?)",
            (source_id, position, self._now()),
        )
        self.conn.commit()
        return position

    @_locked
    def remove_from_print_selection(self, source_id: int) -> None:
        self.conn.execute("DELETE FROM print_selection WHERE source_id = ?", (source_id,))
        self._compact_print_selection_positions()
        self.conn.commit()

    @_locked
    def clear_print_selection(self) -> None:
        self.conn.execute("DELETE FROM print_selection")
        self.conn.commit()

    @_locked
    def restore_print_selection_ids(self, source_ids: list[int]) -> None:
        """Replace the print worklist with ``source_ids`` in the given order."""
        with self.transaction():
            self.conn.execute("DELETE FROM print_selection")
            now = self._now()
            for position, source_id in enumerate(source_ids, start=1):
                self.conn.execute(
                    "INSERT INTO print_selection (source_id, position, added_at) VALUES (?, ?, ?)",
                    (source_id, position, now),
                )

    @_locked
    def reorder_print_selection(self, source_ids: list[int]) -> None:
        current = set(self.get_print_selection_ids())
        ordered: list[int] = []
        seen: set[int] = set()
        for source_id in source_ids:
            if source_id in current and source_id not in seen:
                ordered.append(source_id)
                seen.add(source_id)
        for source_id in self.get_print_selection_ids():
            if source_id not in seen:
                ordered.append(source_id)
        with self.transaction():
            for position, source_id in enumerate(ordered, start=1):
                self.conn.execute(
                    "UPDATE print_selection SET position = ? WHERE source_id = ?",
                    (-position, source_id),
                )
            for position, source_id in enumerate(ordered, start=1):
                self.conn.execute(
                    "UPDATE print_selection SET position = ? WHERE source_id = ?",
                    (position, source_id),
                )

    def _compact_print_selection_positions(self) -> None:
        rows = self.conn.execute(
            "SELECT source_id FROM print_selection ORDER BY position, added_at, source_id"
        ).fetchall()
        for position, row in enumerate(rows, start=1):
            self.conn.execute(
                "UPDATE print_selection SET position = ? WHERE source_id = ?",
                (position, int(row["source_id"])),
            )
