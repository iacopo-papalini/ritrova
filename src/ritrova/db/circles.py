"""Circles: user-defined labelled groups of subjects (FEAT-27).

A subject can belong to zero, one, or many circles (`family`, `acquaintances`,
`strangers`, …). The primary use case is *exclusion* in photo views —
"hide photos whose only named subjects are in `acquaintances`".

No hard FK from subjects to circles; the join table `subject_circles` is
the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from uuid6 import uuid7

from ._base import _DBAccessor


@dataclass(frozen=True)
class Circle:
    id: UUID
    name: str
    description: str | None
    created_at: datetime
    member_count: int = 0


# SQLite's NOCASE collation has no Postgres equivalent on a C.UTF-8 database,
# and the cluster is deliberately C.UTF-8 so the rest of the ordering matches
# what it was. lower() reproduces the case-insensitive sort these three
# listings had.
_BY_NAME = "ORDER BY lower(c.name)"

_CIRCLE_SELECT = """
    SELECT c.id, c.name, c.description, c.created_at,
           (SELECT COUNT(*) FROM subject_circles WHERE circle_id = c.id) AS n
    FROM circles c
"""


def _to_circle(row: object) -> Circle:
    return Circle(
        id=row[0],  # type: ignore[index]
        name=row[1],  # type: ignore[index]
        description=row[2],  # type: ignore[index]
        created_at=row[3],  # type: ignore[index]
        member_count=row[4],  # type: ignore[index]
    )


class CirclesMixin(_DBAccessor):
    # ── circle CRUD ──────────────────────────────────────────────────────

    async def create_circle(self, name: str, description: str | None = None) -> UUID:
        """Create a circle (idempotent on name). Returns its id."""
        name = name.strip()
        if not name:
            msg = "Circle name cannot be empty."
            raise ValueError(msg)
        existing: UUID | None = await self.fetchval("SELECT id FROM circles WHERE name = ?", name)
        if existing:
            return existing
        circle_id = uuid7()
        await self.execute(
            "INSERT INTO circles (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            circle_id,
            name,
            description,
            self._now(),
        )
        return circle_id

    async def rename_circle(self, circle_id: UUID, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            msg = "Circle name cannot be empty."
            raise ValueError(msg)
        await self.execute("UPDATE circles SET name = ? WHERE id = ?", new_name, circle_id)

    async def delete_circle(self, circle_id: UUID) -> None:
        """Delete a circle. Relies on FK cascade to clear subject_circles."""
        await self.execute("DELETE FROM circles WHERE id = ?", circle_id)

    async def get_circle(self, circle_id: UUID) -> Circle | None:
        row = await self.fetchrow(f"{_CIRCLE_SELECT} WHERE c.id = ?", circle_id)
        return _to_circle(row) if row else None

    async def get_circle_by_name(self, name: str) -> Circle | None:
        circle_id = await self.fetchval("SELECT id FROM circles WHERE name = ?", name.strip())
        return await self.get_circle(circle_id) if circle_id else None

    async def list_circles(self) -> list[Circle]:
        rows = await self.fetch(f"{_CIRCLE_SELECT} {_BY_NAME}")
        return [_to_circle(r) for r in rows]

    # ── membership ───────────────────────────────────────────────────────

    async def add_subject_to_circle(self, subject_id: UUID, circle_id: UUID) -> bool:
        """Add a subject to a circle. Returns True if newly added, False if already member."""
        added = await self.fetchval(
            "INSERT INTO subject_circles (subject_id, circle_id, added_at) "
            "VALUES (?, ?, ?) ON CONFLICT DO NOTHING RETURNING subject_id",
            subject_id,
            circle_id,
            self._now(),
        )
        return added is not None

    async def remove_subject_from_circle(self, subject_id: UUID, circle_id: UUID) -> bool:
        removed = await self.fetchval(
            "DELETE FROM subject_circles WHERE subject_id = ? AND circle_id = ? "
            "RETURNING subject_id",
            subject_id,
            circle_id,
        )
        return removed is not None

    async def get_subject_circles(self, subject_id: UUID) -> list[Circle]:
        rows = await self.fetch(
            f"""
            SELECT c.id, c.name, c.description, c.created_at,
                   (SELECT COUNT(*) FROM subject_circles WHERE circle_id = c.id) AS n
            FROM subject_circles sc JOIN circles c ON c.id = sc.circle_id
            WHERE sc.subject_id = ?
            {_BY_NAME}
            """,
            subject_id,
        )
        return [_to_circle(r) for r in rows]

    async def get_circle_subject_ids(self, circle_id: UUID) -> list[UUID]:
        rows = await self.fetch(
            "SELECT subject_id FROM subject_circles WHERE circle_id = ?", circle_id
        )
        return [r[0] for r in rows]

    async def get_circle_members(self, circle_id: UUID) -> list[tuple[UUID, str, str]]:
        """(subject_id, name, kind) for every member of a circle, name-sorted."""
        rows = await self.fetch(
            """
            SELECT s.id, s.name, s.kind
            FROM subject_circles sc JOIN subjects s ON s.id = sc.subject_id
            WHERE sc.circle_id = ?
            ORDER BY lower(s.name)
            """,
            circle_id,
        )
        return [(r[0], r[1], r[2]) for r in rows]

    async def subjects_in_any_circle(self, circle_ids: list[UUID]) -> set[UUID]:
        """Return the set of subject_ids in any of the given circles.

        Used for exclusion filters: a photo is hidden when every named
        subject on it is in this set.
        """
        if not circle_ids:
            return set()
        rows = await self.fetch(
            "SELECT DISTINCT subject_id FROM subject_circles WHERE circle_id = ANY(?)",
            circle_ids,
        )
        return {r[0] for r in rows}
