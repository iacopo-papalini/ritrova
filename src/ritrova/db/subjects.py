"""Subject CRUD, assignment, and centroid mixin.

Apr 2026 refactor: subject assignment lives on `finding_assignment.subject_id`
and all writes go via AssignmentMixin (`self.set_subject`,
`self.clear_curation`, etc.).
"""

from __future__ import annotations

from uuid import UUID

import numpy as np
from uuid6 import uuid7

from ._base import _DBAccessor
from .findings import _FINDING_COLUMNS, _FINDING_FROM, _row_to_finding
from .models import Finding, Subject

_SUBJECT_SELECT = """
    SELECT s.id, s.name, s.kind, COUNT(fa.finding_id) as face_count
    FROM subjects s
    LEFT JOIN finding_assignment fa ON fa.subject_id = s.id
"""


def _to_subject(row: object) -> Subject:
    return Subject(
        id=row["id"],  # type: ignore[index]
        name=row["name"],  # type: ignore[index]
        kind=row["kind"],  # type: ignore[index]
        face_count=row["face_count"],  # type: ignore[index]
    )


class SubjectMixin(_DBAccessor):
    async def create_subject(self, name: str, kind: str = "person") -> UUID:
        """Create a subject or return existing one if name+kind matches."""
        existing: UUID | None = await self.fetchval(
            "SELECT id FROM subjects WHERE name = ? AND kind = ?", name, kind
        )
        if existing:
            return existing
        subject_id = uuid7()
        await self.execute(
            "INSERT INTO subjects (id, name, kind, created_at) VALUES (?, ?, ?, ?)",
            subject_id,
            name,
            kind,
            self._now(),
        )
        return subject_id

    async def create_subject_for_species(self, name: str, species: str) -> UUID:
        """Species-keyed wrapper around ``create_subject``.

        HTTP code carries finding ``species`` (a DB→HTTP crossing value) and
        has no business computing the singular ``subject.kind`` itself —
        this helper does the translation inside the DB layer.
        """
        return await self.create_subject(name, kind=self._kind_for_species(species))

    async def get_subject(self, subject_id: UUID) -> Subject | None:
        row = await self.fetchrow(f"{_SUBJECT_SELECT} WHERE s.id = ? GROUP BY s.id", subject_id)
        return _to_subject(row) if row else None

    async def get_subjects(self) -> list[Subject]:
        rows = await self.fetch(f"{_SUBJECT_SELECT} GROUP BY s.id ORDER BY s.name")
        return [_to_subject(r) for r in rows]

    async def get_subjects_by_kind(self, kind: str) -> list[Subject]:
        """Return subjects of the given kind with their finding counts."""
        rows = await self.fetch(
            f"{_SUBJECT_SELECT} WHERE s.kind = ? GROUP BY s.id ORDER BY s.name", kind
        )
        return [_to_subject(r) for r in rows]

    async def get_subjects_by_species(self, species: str) -> list[Subject]:
        """Species-keyed wrapper around ``get_subjects_by_kind``.

        HTTP code speaks species (a DB→HTTP crossing value) so the
        singular subject-kind translation stays inside the DB layer.
        """
        return await self.get_subjects_by_kind(self._kind_for_species(species))

    async def get_subject_centroids(
        self, kind: str = "person", embedding_dim: int | None = None
    ) -> list[tuple[UUID, str, np.ndarray]]:
        """Return [(subject_id, name, centroid)] for all subjects of a kind."""
        species = self._species_for_kind(kind)
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = await self.fetch(
            f"""
            SELECT fa.subject_id, s.name, f.embedding
            FROM finding_assignment fa
            JOIN subjects s ON s.id = fa.subject_id
            JOIN findings f ON f.id = fa.finding_id
            WHERE fa.subject_id IS NOT NULL AND s.kind = ?
              AND {clause} AND {dim_clause}
            ORDER BY fa.subject_id
            """,
            kind,
            *params,
            *dim_params,
        )

        from ritrova.embeddings import compute_centroid

        subject_embs: dict[UUID, tuple[str, list[np.ndarray]]] = {}
        for r in rows:
            sid = r["subject_id"]
            if sid not in subject_embs:
                subject_embs[sid] = (r["name"], [])
            subject_embs[sid][1].append(np.frombuffer(r["embedding"], dtype=np.float32))

        result: list[tuple[UUID, str, np.ndarray]] = []
        for sid, (name, embs) in subject_embs.items():
            centroid = compute_centroid(np.array(embs))
            result.append((sid, name, centroid))
        return result

    async def rename_subject(self, subject_id: UUID, name: str) -> None:
        await self.execute("UPDATE subjects SET name = ? WHERE id = ?", name, subject_id)

    async def assign_finding_to_subject(
        self, finding_id: UUID, subject_id: UUID, *, force: bool = False
    ) -> None:
        """Assign a finding to a subject — delegates to AssignmentMixin.set_subject.

        Raises ValueError on species/kind mismatch unless force=True.
        Species is never mutated; the finding keeps its original species
        and embedding space. Overwrites any prior exclusion (stranger /
        not_a_face) — picking a name implicitly un-marks.
        """
        finding = await self.get_finding(finding_id)
        subject = await self.get_subject(subject_id)
        if (
            finding
            and subject
            and not force
            and not self._is_species_kind_compatible(finding.species, subject.kind)
        ):
            raise ValueError(f"Cannot assign {finding.species} finding to a {subject.kind} subject")
        await self.set_subject(finding_id, subject_id)  # type: ignore[attr-defined]

    async def assign_cluster_to_subject(
        self, cluster_id: int, subject_id: UUID, *, force: bool = False
    ) -> None:
        """Assign every uncurated finding in a cluster to a subject.

        "Uncurated" = no finding_assignment row. Previously-excluded
        findings (stranger / not_a_face) are preserved — we don't silently
        pull them back into a subject. If the user wants those reassigned,
        they go through the per-face pickers on the photo page.
        """
        subject = await self.get_subject(subject_id)
        if subject and not force:
            findings = await self.get_cluster_findings(cluster_id, limit=1)
            if findings and not self._is_species_kind_compatible(findings[0].species, subject.kind):
                raise ValueError(
                    f"Cannot assign {findings[0].species} finding to a {subject.kind} subject"
                )
        async with self.transaction():
            # Snapshot the uncurated findings and write one assignment row each.
            uncurated = await self.fetch(
                """
                SELECT f.id FROM findings f
                JOIN cluster_findings cf ON cf.finding_id = f.id
                LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
                WHERE cf.cluster_id = ? AND fa.finding_id IS NULL
                """,
                cluster_id,
            )
            now = self._now()
            await self.executemany(
                "INSERT INTO finding_assignment"
                "(finding_id, subject_id, exclusion_reason, curated_at) "
                "VALUES (?, ?, NULL, ?) "
                "ON CONFLICT (finding_id) DO UPDATE SET "
                "subject_id = EXCLUDED.subject_id, "
                "exclusion_reason = EXCLUDED.exclusion_reason, "
                "curated_at = EXCLUDED.curated_at",
                [(row[0], subject_id, now) for row in uncurated],
            )

    async def merge_subjects(self, source_id: UUID, target_id: UUID) -> None:
        """Merge source subject into target: reassign findings, delete source."""
        async with self.transaction():
            await self.execute(
                "UPDATE finding_assignment SET subject_id = ? WHERE subject_id = ?",
                target_id,
                source_id,
            )
            await self.execute("DELETE FROM subjects WHERE id = ?", source_id)

    async def get_subject_face_stubs(
        self, subject_id: UUID, limit: int = 200, offset: int = 0
    ) -> list[tuple[UUID, UUID]]:
        """Paginated ``(finding_id, source_id)`` for a subject's assigned findings.

        Used by face-grid endpoints that render thumbnails + link to the
        source photo and don't need full ``Finding`` rows. Goes through
        ``_FINDING_FROM`` so a change to the curation-state JOIN convention
        stays local to ``findings.py``.
        """
        rows = await self.fetch(
            f"SELECT f.id, f.source_id {_FINDING_FROM} WHERE fa.subject_id = ? LIMIT ? OFFSET ?",
            subject_id,
            limit,
            offset,
        )
        return [(r[0], r[1]) for r in rows]

    async def get_subject_findings(
        self, subject_id: UUID, limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        rows = await self.fetch(
            f"""SELECT {_FINDING_COLUMNS}
                {_FINDING_FROM}
                LEFT JOIN sources src ON f.source_id = src.id
                WHERE fa.subject_id = ?
                ORDER BY src.taken_at DESC NULLS LAST, f.id
                LIMIT ? OFFSET ?""",
            subject_id,
            limit,
            offset,
        )
        return [f for f in (_row_to_finding(r) for r in rows) if f is not None]

    async def get_subject_findings_with_paths(
        self, subject_id: UUID, limit: int = 200, offset: int = 0
    ) -> list[tuple[Finding, str]]:
        """Return findings with their source's file_path, sorted by path."""
        rows = await self.fetch(
            f"""SELECT {_FINDING_COLUMNS}, src.file_path AS source_path
                {_FINDING_FROM}
                LEFT JOIN sources src ON f.source_id = src.id
                WHERE fa.subject_id = ?
                ORDER BY src.file_path DESC NULLS LAST, f.id
                LIMIT ? OFFSET ?""",
            subject_id,
            limit,
            offset,
        )
        result = []
        for row in rows:
            finding = _row_to_finding(row)
            if finding is None:
                continue
            path = row["source_path"] or ""
            result.append((finding, path))
        return result

    async def get_random_avatars(self, subject_ids: list[UUID]) -> dict[UUID, UUID]:
        """Pick one random finding ID per subject for avatar thumbnails."""
        if not subject_ids:
            return {}
        rows = await self.fetch(
            """SELECT subject_id, finding_id FROM (
                    SELECT fa.subject_id, fa.finding_id, ROW_NUMBER() OVER (
                        PARTITION BY fa.subject_id ORDER BY random()
                    ) AS rn FROM finding_assignment fa
                    WHERE fa.subject_id = ANY(?)
                ) ranked WHERE rn = 1""",
            subject_ids,
        )
        return {r[0]: r[1] for r in rows}

    async def delete_subject(self, subject_id: UUID) -> None:
        """Delete a subject. FK cascade clears finding_assignment rows
        (leaving those findings uncurated — no longer 'orphans' with
        subject_id=NULL, they just vanish from the assignment table)."""
        await self.execute("DELETE FROM subjects WHERE id = ?", subject_id)

    async def search_subjects(self, query: str, kind: str | None = None) -> list[Subject]:
        # ILIKE, not LIKE: this is the subject typeahead, and SQLite's LIKE was
        # case-insensitive for ASCII where Postgres's is not. A literal port
        # stops matching "marco" against "Marco".
        if kind:
            rows = await self.fetch(
                f"{_SUBJECT_SELECT} WHERE s.name ILIKE ? AND s.kind = ? "
                "GROUP BY s.id ORDER BY s.name",
                f"%{query}%",
                kind,
            )
        else:
            rows = await self.fetch(
                f"{_SUBJECT_SELECT} WHERE s.name ILIKE ? GROUP BY s.id ORDER BY s.name",
                f"%{query}%",
            )
        return [_to_subject(r) for r in rows]
