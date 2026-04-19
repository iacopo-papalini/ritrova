"""Circles: user-defined labelled groups of subjects (FEAT-27).

A subject can belong to zero, one, or many circles (`family`, `acquaintances`,
`strangers`, …). The primary use case is *exclusion* in photo views —
"hide photos whose only named subjects are in `acquaintances`".

No hard FK from subjects to circles; the join table `subject_circles` is
the source of truth. All three tables are created in ``connection.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._base import _DBAccessor, _locked


@dataclass(frozen=True)
class Circle:
    id: int
    name: str
    description: str | None
    created_at: str
    member_count: int = 0


class CirclesMixin(_DBAccessor):
    # ── circle CRUD ──────────────────────────────────────────────────────

    @_locked
    def create_circle(self, name: str, description: str | None = None) -> int:
        """Create a circle (idempotent on name). Returns its id."""
        name = name.strip()
        if not name:
            msg = "Circle name cannot be empty."
            raise ValueError(msg)
        row = self.conn.execute("SELECT id FROM circles WHERE name = ?", (name,)).fetchone()
        if row:
            return int(row[0])
        cur = self.conn.execute(
            "INSERT INTO circles (name, description, created_at) VALUES (?, ?, ?)",
            (name, description, self._now()),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    @_locked
    def rename_circle(self, circle_id: int, new_name: str) -> None:
        new_name = new_name.strip()
        if not new_name:
            msg = "Circle name cannot be empty."
            raise ValueError(msg)
        self.conn.execute("UPDATE circles SET name = ? WHERE id = ?", (new_name, circle_id))
        self.conn.commit()

    @_locked
    def delete_circle(self, circle_id: int) -> None:
        """Delete a circle. Relies on FK cascade to clear subject_circles."""
        self.conn.execute("DELETE FROM circles WHERE id = ?", (circle_id,))
        self.conn.commit()

    @_locked
    def get_circle(self, circle_id: int) -> Circle | None:
        row = self.conn.execute(
            """
            SELECT c.id, c.name, c.description, c.created_at,
                   (SELECT COUNT(*) FROM subject_circles WHERE circle_id = c.id) AS n
            FROM circles c WHERE c.id = ?
            """,
            (circle_id,),
        ).fetchone()
        if not row:
            return None
        return Circle(
            id=row[0], name=row[1], description=row[2], created_at=row[3], member_count=row[4]
        )

    @_locked
    def get_circle_by_name(self, name: str) -> Circle | None:
        row = self.conn.execute("SELECT id FROM circles WHERE name = ?", (name.strip(),)).fetchone()
        return self.get_circle(int(row[0])) if row else None

    @_locked
    def list_circles(self) -> list[Circle]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.name, c.description, c.created_at,
                   (SELECT COUNT(*) FROM subject_circles WHERE circle_id = c.id) AS n
            FROM circles c
            ORDER BY c.name COLLATE NOCASE
            """
        ).fetchall()
        return [
            Circle(id=r[0], name=r[1], description=r[2], created_at=r[3], member_count=r[4])
            for r in rows
        ]

    # ── membership ───────────────────────────────────────────────────────

    @_locked
    def add_subject_to_circle(self, subject_id: int, circle_id: int) -> bool:
        """Add a subject to a circle. Returns True if newly added, False if already member."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO subject_circles (subject_id, circle_id, added_at) "
            "VALUES (?, ?, ?)",
            (subject_id, circle_id, self._now()),
        )
        self.conn.commit()
        return cur.rowcount > 0

    @_locked
    def remove_subject_from_circle(self, subject_id: int, circle_id: int) -> bool:
        cur = self.conn.execute(
            "DELETE FROM subject_circles WHERE subject_id = ? AND circle_id = ?",
            (subject_id, circle_id),
        )
        self.conn.commit()
        return cur.rowcount > 0

    @_locked
    def get_subject_circles(self, subject_id: int) -> list[Circle]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.name, c.description, c.created_at,
                   (SELECT COUNT(*) FROM subject_circles WHERE circle_id = c.id) AS n
            FROM subject_circles sc JOIN circles c ON c.id = sc.circle_id
            WHERE sc.subject_id = ?
            ORDER BY c.name COLLATE NOCASE
            """,
            (subject_id,),
        ).fetchall()
        return [
            Circle(id=r[0], name=r[1], description=r[2], created_at=r[3], member_count=r[4])
            for r in rows
        ]

    @_locked
    def get_circle_subject_ids(self, circle_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT subject_id FROM subject_circles WHERE circle_id = ?",
            (circle_id,),
        ).fetchall()
        return [int(r[0]) for r in rows]

    @_locked
    def get_circle_members(self, circle_id: int) -> list[tuple[int, str, str]]:
        """(subject_id, name, kind) for every member of a circle, name-sorted."""
        rows = self.conn.execute(
            """
            SELECT s.id, s.name, s.kind
            FROM subject_circles sc JOIN subjects s ON s.id = sc.subject_id
            WHERE sc.circle_id = ?
            ORDER BY s.name COLLATE NOCASE
            """,
            (circle_id,),
        ).fetchall()
        return [(int(r[0]), r[1], r[2]) for r in rows]

    @_locked
    def find_stranger_bucket(self, circle_id: int, species: str) -> int | None:
        """Return the subject_id in `circle_id` with the most findings of `species`.

        Used by the "I don't know this {cat|dog|person}" action to route
        marks into an existing catch-all subject (e.g. `cani a caso`,
        `gatti a caso`) rather than creating yet another `Stranger #N`.
        The user designates a bucket simply by adding the subject to the
        Strangers circle — no new config required. Returns ``None`` when no
        matching subject exists (caller then falls back to creating one).
        """
        row = self.conn.execute(
            """
            SELECT sc.subject_id, COUNT(f.id) AS n
            FROM subject_circles sc
            JOIN findings f ON f.person_id = sc.subject_id
            WHERE sc.circle_id = ? AND f.species = ?
            GROUP BY sc.subject_id
            ORDER BY n DESC
            LIMIT 1
            """,
            (circle_id, species),
        ).fetchone()
        return int(row[0]) if row else None

    def get_stranger_subject_ids(self) -> set[int]:
        """Subjects in the Strangers circle.

        These are catch-all buckets (``cani a caso``, ``Stranger #N``) whose
        embedding centroid averages wildly different faces and therefore
        isn't a reliable anchor for similarity-based suggestions or
        auto-assign. Consumers filter these out before computing or using
        subject centroids.
        """
        strangers = self.get_circle_by_name("Strangers")
        if strangers is None:
            return set()
        return self.subjects_in_any_circle([strangers.id])

    @_locked
    def subjects_in_any_circle(self, circle_ids: list[int]) -> set[int]:
        """Return the set of subject_ids in any of the given circles.

        Used for exclusion filters: a photo is hidden when every named
        subject on it is in this set.
        """
        if not circle_ids:
            return set()
        placeholders = ",".join("?" * len(circle_ids))
        rows = self.conn.execute(
            f"SELECT DISTINCT subject_id FROM subject_circles WHERE circle_id IN ({placeholders})",
            tuple(circle_ids),
        ).fetchall()
        return {int(r[0]) for r in rows}
