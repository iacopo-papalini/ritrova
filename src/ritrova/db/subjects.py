"""Subject CRUD, assignment, and centroid mixin."""

from __future__ import annotations

import numpy as np

from ._base import _DBAccessor, _locked
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

    @_locked
    def get_subject(self, subject_id: int) -> Subject | None:
        row = self.conn.execute(
            """
            SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
            FROM subjects s LEFT JOIN findings f ON f.person_id = s.id
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
            SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
            FROM subjects s LEFT JOIN findings f ON f.person_id = s.id
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
            SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
            FROM subjects s LEFT JOIN findings f ON f.person_id = s.id
            WHERE s.kind = ?
            GROUP BY s.id ORDER BY s.name
            """,
            (kind,),
        ).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]

    @_locked
    def get_subject_centroids(
        self, kind: str = "person", embedding_dim: int | None = None
    ) -> list[tuple[int, str, np.ndarray]]:
        """Return [(subject_id, name, centroid)] for all subjects of a kind.

        Filters by embedding_dim to ensure only compatible embeddings are
        used in centroid computation (e.g. 512 for ArcFace, 768 for SigLIP).
        """
        species = self._species_for_kind(kind)
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"""
            SELECT f.person_id, s.name, f.embedding
            FROM findings f
            JOIN subjects s ON s.id = f.person_id
            WHERE f.person_id IS NOT NULL AND s.kind = ? AND {clause} AND {dim_clause}
            ORDER BY f.person_id
            """,
            (kind, *params, *dim_params),
        ).fetchall()

        from ritrova.embeddings import compute_centroid

        subject_embs: dict[int, tuple[str, list[np.ndarray]]] = {}
        for r in rows:
            sid = r["person_id"]
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

    @_locked
    def assign_finding_to_subject(
        self, finding_id: int, subject_id: int, *, force: bool = False
    ) -> None:
        """Assign a finding to a subject.

        Raises ValueError on species/kind mismatch unless force=True.
        Species is never mutated — the finding keeps its original species
        and embedding space.
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
        self.conn.execute(
            "UPDATE findings SET person_id = ? WHERE id = ?", (subject_id, finding_id)
        )
        self.conn.commit()

    @_locked
    def assign_cluster_to_subject(
        self, cluster_id: int, subject_id: int, *, force: bool = False
    ) -> None:
        """Assign all unassigned findings in a cluster to a subject.

        Raises ValueError on species/kind mismatch unless force=True.
        Species is never mutated.
        """
        subject = self.get_subject(subject_id)
        if subject and not force:
            findings = self.get_cluster_findings(cluster_id, limit=1)
            if findings and not self._is_species_kind_compatible(findings[0].species, subject.kind):
                raise ValueError(
                    f"Cannot assign {findings[0].species} finding to a {subject.kind} subject"
                )
        self.conn.execute(
            "UPDATE findings SET person_id = ? WHERE cluster_id = ? AND person_id IS NULL",
            (subject_id, cluster_id),
        )
        self.conn.commit()

    @_locked
    def merge_subjects(self, source_id: int, target_id: int) -> None:
        """Merge source subject into target: reassign findings, delete source."""
        self.conn.execute(
            "UPDATE findings SET person_id = ? WHERE person_id = ?",
            (target_id, source_id),
        )
        self.conn.execute("DELETE FROM subjects WHERE id = ?", (source_id,))
        self.conn.commit()

    @_locked
    def get_subject_findings(
        self, subject_id: int, limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        rows = self.conn.execute(
            """SELECT f.* FROM findings f
               LEFT JOIN sources s ON f.source_id = s.id
               WHERE f.person_id = ?
               ORDER BY s.taken_at DESC, f.id
               LIMIT ? OFFSET ?""",
            (subject_id, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Finding(**d))
        return result

    def get_subject_findings_with_paths(
        self, subject_id: int, limit: int = 200, offset: int = 0
    ) -> list[tuple[Finding, str]]:
        """Return findings with their source's file_path, sorted by path (date-based dirs)."""
        rows = self.conn.execute(
            """SELECT f.*, s.file_path AS source_path FROM findings f
               LEFT JOIN sources s ON f.source_id = s.id
               WHERE f.person_id = ?
               ORDER BY s.file_path DESC, f.id
               LIMIT ? OFFSET ?""",
            (subject_id, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            path = d.pop("source_path", "") or ""
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append((Finding(**d), path))
        return result

    def get_random_avatars(self, subject_ids: list[int]) -> dict[int, int]:
        """Pick one random finding ID per subject for avatar thumbnails."""
        if not subject_ids:
            return {}
        placeholders = ",".join("?" * len(subject_ids))
        rows = self.conn.execute(
            f"""SELECT person_id, id FROM (
                    SELECT person_id, id, ROW_NUMBER() OVER (
                        PARTITION BY person_id ORDER BY RANDOM()
                    ) AS rn FROM findings WHERE person_id IN ({placeholders})
                ) WHERE rn = 1""",
            tuple(subject_ids),
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    @_locked
    def delete_subject(self, subject_id: int) -> None:
        """Delete a subject and unassign their findings."""
        self.conn.execute("UPDATE findings SET person_id = NULL WHERE person_id = ?", (subject_id,))
        self.conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
        self.conn.commit()

    @_locked
    def search_subjects(self, query: str, kind: str | None = None) -> list[Subject]:
        if kind:
            rows = self.conn.execute(
                """
                SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
                FROM subjects s LEFT JOIN findings f ON f.person_id = s.id
                WHERE s.name LIKE ? AND s.kind = ?
                GROUP BY s.id ORDER BY s.name
                """,
                (f"%{query}%", kind),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT s.id, s.name, s.kind, COUNT(f.id) as face_count
                FROM subjects s LEFT JOIN findings f ON f.person_id = s.id
                WHERE s.name LIKE ?
                GROUP BY s.id ORDER BY s.name
                """,
                (f"%{query}%",),
            ).fetchall()
        return [
            Subject(id=r["id"], name=r["name"], kind=r["kind"], face_count=r["face_count"])
            for r in rows
        ]
