"""Subject CRUD, assignment, and centroid mixin.

Apr 2026 refactor: all reads of `findings.person_id` are routed through
`finding_assignment.subject_id`, and all writes go via AssignmentMixin
(`self.set_subject`, `self.clear_curation`, etc.). The old column is no
longer touched; Commit D drops it.
"""

from __future__ import annotations

import numpy as np

from ._base import _DBAccessor, _locked
from .findings import _FINDING_COLUMNS, _FINDING_FROM, _row_to_finding
from .models import Finding, Subject


class SubjectMixin(_DBAccessor):
    @_locked
    def create_subject(self, name: str, kind: str = "person") -> int:
        """Create a subject or return existing one if name+kind matches."""
        row = self.conn.execute(
            "SELECT id FROM subjects WHERE name = ? AND kind = ?", (name, kind)
        ).fetchone()
        if row:
            return int(row[0])
        cur = self.conn.execute(
            "INSERT INTO subjects (name, kind, created_at) VALUES (?, ?, ?)",
            (name, kind, self._now()),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    def create_subject_for_species(self, name: str, species: str) -> int:
        """Species-keyed wrapper around ``create_subject``.

        HTTP code carries finding ``species`` (a DB→HTTP crossing value) and
        has no business computing the singular ``subject.kind`` itself —
        this helper does the translation inside the DB layer.
        """
        return self.create_subject(name, kind=self._kind_for_species(species))

    @_locked
    def get_subject(self, subject_id: int) -> Subject | None:
        row = self.conn.execute(
            """
            SELECT s.id, s.name, s.kind, COUNT(fa.finding_id) as face_count
            FROM subjects s
            LEFT JOIN finding_assignment fa ON fa.subject_id = s.id
            WHERE s.id = ? GROUP BY s.id
            """,
            (subject_id,),
        ).fetchone()
        if not row:
            return None
        return Subject(
            id=row["id"], name=row["name"], kind=row["kind"], face_count=row["face_count"]
        )

    @_locked
    def get_subjects(self) -> list[Subject]:
        rows = self.conn.execute("""
            SELECT s.id, s.name, s.kind, COUNT(fa.finding_id) as face_count
            FROM subjects s
            LEFT JOIN finding_assignment fa ON fa.subject_id = s.id
            GROUP BY s.id ORDER BY s.name
        """).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]

    @_locked
    def get_subjects_by_kind(self, kind: str) -> list[Subject]:
        """Return subjects of the given kind with their finding counts."""
        rows = self.conn.execute(
            """
            SELECT s.id, s.name, s.kind, COUNT(fa.finding_id) as face_count
            FROM subjects s
            LEFT JOIN finding_assignment fa ON fa.subject_id = s.id
            WHERE s.kind = ?
            GROUP BY s.id ORDER BY s.name
            """,
            (kind,),
        ).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]

    def get_subjects_by_species(self, species: str) -> list[Subject]:
        """Species-keyed wrapper around ``get_subjects_by_kind``.

        HTTP code speaks species (a DB→HTTP crossing value) so the
        singular subject-kind translation stays inside the DB layer.
        """
        return self.get_subjects_by_kind(self._kind_for_species(species))

    @_locked
    def get_subject_centroids(
        self, kind: str = "person", embedding_dim: int | None = None
    ) -> list[tuple[int, str, np.ndarray]]:
        """Return [(subject_id, name, centroid)] for all subjects of a kind."""
        species = self._species_for_kind(kind)
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"""
            SELECT fa.subject_id, s.name, f.embedding
            FROM finding_assignment fa
            JOIN subjects s ON s.id = fa.subject_id
            JOIN findings f ON f.id = fa.finding_id
            WHERE fa.subject_id IS NOT NULL AND s.kind = ?
              AND {clause} AND {dim_clause}
            ORDER BY fa.subject_id
            """,
            (kind, *params, *dim_params),
        ).fetchall()

        from ritrova.embeddings import compute_centroid

        subject_embs: dict[int, tuple[str, list[np.ndarray]]] = {}
        for r in rows:
            sid = r["subject_id"]
            if sid not in subject_embs:
                subject_embs[sid] = (r["name"], [])
            subject_embs[sid][1].append(np.frombuffer(r["embedding"], dtype=np.float32))

        result: list[tuple[int, str, np.ndarray]] = []
        for sid, (name, embs) in subject_embs.items():
            centroid = compute_centroid(np.array(embs))
            result.append((sid, name, centroid))
        return result

    @_locked
    def rename_subject(self, subject_id: int, name: str) -> None:
        self.conn.execute("UPDATE subjects SET name = ? WHERE id = ?", (name, subject_id))
        self.conn.commit()

    def assign_finding_to_subject(
        self, finding_id: int, subject_id: int, *, force: bool = False
    ) -> None:
        """Assign a finding to a subject — delegates to AssignmentMixin.set_subject.

        Raises ValueError on species/kind mismatch unless force=True.
        Species is never mutated; the finding keeps its original species
        and embedding space. Overwrites any prior exclusion (stranger /
        not_a_face) — picking a name implicitly un-marks.
        """
        finding = self.get_finding(finding_id)
        subject = self.get_subject(subject_id)
        if (
            finding
            and subject
            and not force
            and not self._is_species_kind_compatible(finding.species, subject.kind)
        ):
            raise ValueError(f"Cannot assign {finding.species} finding to a {subject.kind} subject")
        self.set_subject(finding_id, subject_id)  # type: ignore[attr-defined]

    def assign_cluster_to_subject(
        self, cluster_id: int, subject_id: int, *, force: bool = False
    ) -> None:
        """Assign every uncurated finding in a cluster to a subject.

        "Uncurated" = no finding_assignment row. Previously-excluded
        findings (stranger / not_a_face) are preserved — we don't silently
        pull them back into a subject. If the user wants those reassigned,
        they go through the per-face pickers on the photo page.
        """
        subject = self.get_subject(subject_id)
        if subject and not force:
            findings = self.get_cluster_findings(cluster_id, limit=1)
            if findings and not self._is_species_kind_compatible(findings[0].species, subject.kind):
                raise ValueError(
                    f"Cannot assign {findings[0].species} finding to a {subject.kind} subject"
                )
        # Snapshot the uncurated findings and write one assignment row each.
        uncurated_ids = [
            row[0]
            for row in self.conn.execute(
                """
                SELECT f.id FROM findings f
                JOIN cluster_findings cf ON cf.finding_id = f.id
                LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
                WHERE cf.cluster_id = ? AND fa.finding_id IS NULL
                """,
                (cluster_id,),
            ).fetchall()
        ]
        now = self._now()
        self.conn.executemany(
            "INSERT OR REPLACE INTO finding_assignment"
            "(finding_id, subject_id, exclusion_reason, curated_at) "
            "VALUES (?, ?, NULL, ?)",
            [(fid, subject_id, now) for fid in uncurated_ids],
        )
        self.conn.commit()

    @_locked
    def merge_subjects(self, source_id: int, target_id: int) -> None:
        """Merge source subject into target: reassign findings, delete source."""
        self.conn.execute(
            "UPDATE finding_assignment SET subject_id = ? WHERE subject_id = ?",
            (target_id, source_id),
        )
        self.conn.execute("DELETE FROM subjects WHERE id = ?", (source_id,))
        self.conn.commit()

    @_locked
    def get_subject_findings(
        self, subject_id: int, limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        rows = self.conn.execute(
            f"""SELECT {_FINDING_COLUMNS}
                {_FINDING_FROM}
                LEFT JOIN sources src ON f.source_id = src.id
                WHERE fa.subject_id = ?
                ORDER BY src.taken_at DESC, f.id
                LIMIT ? OFFSET ?""",
            (subject_id, limit, offset),
        ).fetchall()
        return [f for f in (_row_to_finding(r) for r in rows) if f is not None]

    def get_subject_findings_with_paths(
        self, subject_id: int, limit: int = 200, offset: int = 0
    ) -> list[tuple[Finding, str]]:
        """Return findings with their source's file_path, sorted by path."""
        rows = self.conn.execute(
            f"""SELECT {_FINDING_COLUMNS}, src.file_path AS source_path
                {_FINDING_FROM}
                LEFT JOIN sources src ON f.source_id = src.id
                WHERE fa.subject_id = ?
                ORDER BY src.file_path DESC, f.id
                LIMIT ? OFFSET ?""",
            (subject_id, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            finding = _row_to_finding(row)
            if finding is None:
                continue
            path = row["source_path"] or ""
            result.append((finding, path))
        return result

    def get_random_avatars(self, subject_ids: list[int]) -> dict[int, int]:
        """Pick one random finding ID per subject for avatar thumbnails."""
        if not subject_ids:
            return {}
        placeholders = ",".join("?" * len(subject_ids))
        rows = self.conn.execute(
            f"""SELECT subject_id, finding_id FROM (
                    SELECT fa.subject_id, fa.finding_id, ROW_NUMBER() OVER (
                        PARTITION BY fa.subject_id ORDER BY RANDOM()
                    ) AS rn FROM finding_assignment fa
                    WHERE fa.subject_id IN ({placeholders})
                ) WHERE rn = 1""",
            tuple(subject_ids),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    @_locked
    def delete_subject(self, subject_id: int) -> None:
        """Delete a subject. FK cascade clears finding_assignment rows
        (leaving those findings uncurated — no longer 'orphans' with
        person_id=NULL, they just vanish from the assignment table)."""
        self.conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
        self.conn.commit()

    @_locked
    def search_subjects(self, query: str, kind: str | None = None) -> list[Subject]:
        if kind:
            rows = self.conn.execute(
                """
                SELECT s.id, s.name, s.kind, COUNT(fa.finding_id) as face_count
                FROM subjects s
                LEFT JOIN finding_assignment fa ON fa.subject_id = s.id
                WHERE s.name LIKE ? AND s.kind = ?
                GROUP BY s.id ORDER BY s.name
                """,
                (f"%{query}%", kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT s.id, s.name, s.kind, COUNT(fa.finding_id) as face_count
                FROM subjects s
                LEFT JOIN finding_assignment fa ON fa.subject_id = s.id
                WHERE s.name LIKE ?
                GROUP BY s.id ORDER BY s.name
                """,
                (f"%{query}%",),
            ).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]
