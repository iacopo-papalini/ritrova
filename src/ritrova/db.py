"""SQLite database for face recognition data."""

import functools
import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np

P = ParamSpec("P")
R = TypeVar("R")


def _locked(method: Callable[..., R]) -> Callable[..., R]:
    """Decorator: hold the DB lock for the entire method call."""

    @functools.wraps(method)
    def wrapper(self: FaceDB, *args: Any, **kwargs: Any) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Source:
    id: int
    file_path: str
    type: str
    width: int
    height: int
    taken_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class Finding:
    id: int
    source_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    embedding: np.ndarray
    person_id: int | None
    cluster_id: int | None
    confidence: float
    detected_at: str
    species: str = "human"
    frame_path: str | None = None


@dataclass
class Subject:
    id: int
    name: str
    kind: str = "person"
    face_count: int = 0


class FaceDB:
    def __init__(self, db_path: str | Path, base_dir: str | Path | None = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.base_dir: Path | None = Path(base_dir).resolve() if base_dir else None
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT UNIQUE NOT NULL,
                type TEXT NOT NULL DEFAULT 'photo',
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                taken_at TEXT,
                latitude REAL,
                longitude REAL
            );

            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                scan_type TEXT NOT NULL,
                scanned_at TEXT NOT NULL,
                UNIQUE (source_id, scan_type)
            );

            CREATE TABLE IF NOT EXISTS subjects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL DEFAULT 'person',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS findings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_w INTEGER NOT NULL,
                bbox_h INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                person_id INTEGER REFERENCES subjects(id) ON DELETE SET NULL,
                cluster_id INTEGER,
                confidence REAL NOT NULL DEFAULT 0.0,
                species TEXT NOT NULL DEFAULT 'human',
                detected_at TEXT NOT NULL,
                frame_path TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_findings_source ON findings(source_id);
            CREATE INDEX IF NOT EXISTS idx_findings_person ON findings(person_id);
            CREATE INDEX IF NOT EXISTS idx_findings_cluster ON findings(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(file_path);

            CREATE TABLE IF NOT EXISTS dismissed_findings (
                finding_id INTEGER PRIMARY KEY REFERENCES findings(id) ON DELETE CASCADE
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_subjects_name_kind
                ON subjects(name, kind);
        """)
        self.conn.commit()

    @_locked
    def run(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Execute SQL and commit. For ad-hoc queries from outside."""
        self.conn.execute(sql, params)
        self.conn.commit()

    @_locked
    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[Any]:
        """Execute SQL and return all rows."""
        return self.conn.execute(sql, params).fetchall()

    PET_SPECIES = ("dog", "cat", "other_pet")
    KIND_TO_SPECIES: dict[str, str] = {"person": "human", "pet": "pet"}

    def close(self) -> None:
        self.conn.close()

    def resolve_path(self, stored_path: str) -> Path:
        """Resolve a DB-stored path to an absolute filesystem path."""
        if stored_path.startswith("/"):
            return Path(stored_path)
        if ".." in stored_path.split("/"):
            msg = f"Path contains '..': {stored_path}"
            raise ValueError(msg)
        # tmp/ paths are app-generated (video frames, etc.) — relative to DB directory
        if stored_path.startswith("tmp/"):
            return self.db_path.parent / stored_path
        if self.base_dir is not None:
            return self.base_dir / stored_path
        return Path(stored_path)

    def resolve_finding_image(self, finding: Finding) -> Path:
        """Resolve the image file to use for a finding's thumbnail.

        Photo findings: the source file.
        Video findings: the extracted frame JPEG.
        """
        if finding.frame_path:
            return self.db_path.parent / finding.frame_path
        source = self.get_source(finding.source_id)
        if source:
            return self.resolve_path(source.file_path)
        msg = f"No source found for finding {finding.id}"
        raise ValueError(msg)

    def to_relative(self, absolute_path: str) -> str:
        """Convert an absolute path to a relative path (stripping base_dir prefix)."""
        if self.base_dir is None:
            return absolute_path
        try:
            return str(Path(absolute_path).resolve().relative_to(self.base_dir))
        except ValueError:
            return absolute_path

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    # --- Sources ---

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
        return cur.lastrowid

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
    def is_scanned(self, file_path: str, scan_type: str) -> bool:
        """Check if a source has been scanned with the given scan type."""
        row = self.conn.execute(
            """SELECT 1 FROM scans sc
               JOIN sources s ON s.id = sc.source_id
               WHERE s.file_path = ? AND sc.scan_type = ?""",
            (file_path, scan_type),
        ).fetchone()
        return row is not None

    @_locked
    def record_scan(self, source_id: int, scan_type: str) -> None:
        """Record that a scan has been performed on a source."""
        self.conn.execute(
            "INSERT OR IGNORE INTO scans (source_id, scan_type, scanned_at) VALUES (?, ?, ?)",
            (source_id, scan_type, self._now()),
        )
        self.conn.commit()

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

    # --- Findings ---

    @_locked
    def add_findings_batch(
        self,
        findings_data: list[tuple[int, tuple[int, int, int, int], np.ndarray, float]],
        species: str = "human",
        frame_path: str | None = None,
    ) -> None:
        """Add multiple findings in a single transaction."""
        now = self._now()
        self.conn.executemany(
            "INSERT INTO findings (source_id, bbox_x, bbox_y, bbox_w, bbox_h, "
            "embedding, confidence, detected_at, species, frame_path) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (sid, *bbox, emb.tobytes(), conf, now, species, frame_path)
                for sid, bbox, emb, conf in findings_data
            ],
        )
        self.conn.commit()

    @_locked
    def get_finding(self, finding_id: int) -> Finding | None:
        row = self.conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        if not row or row["embedding"] is None:
            return None
        d = dict(row)
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        return Finding(**d)

    @_locked
    def get_source_findings(self, source_id: int) -> list[Finding]:
        rows = self.conn.execute(
            "SELECT * FROM findings WHERE source_id = ?", (source_id,)
        ).fetchall()
        result = []
        for row in rows:
            if row["embedding"] is None:
                continue
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Finding(**d))
        return result

    @_locked
    def get_finding_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_all_embeddings(self, species: str = "human") -> list[tuple[int, np.ndarray]]:
        """Return all (finding_id, embedding) pairs for clustering, excluding dismissed."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE {clause} AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            params,
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def get_unassigned_embeddings(self, species: str = "human") -> list[tuple[int, np.ndarray]]:
        """Return (finding_id, embedding) for unassigned, non-dismissed findings only."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE person_id IS NULL AND {clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            params,
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def update_cluster_ids(self, finding_cluster_map: dict[int, int]) -> None:
        """Update cluster_id for multiple findings."""
        self.conn.executemany(
            "UPDATE findings SET cluster_id = ? WHERE id = ?",
            [(cid, fid) for fid, cid in finding_cluster_map.items()],
        )
        self.conn.commit()

    @_locked
    def clear_clusters(self, species: str | None = None) -> None:
        """Reset cluster assignments (preserves subject assignments)."""
        if species is None:
            self.conn.execute("UPDATE findings SET cluster_id = NULL")
        else:
            clause, params = self.species_filter(species)
            self.conn.execute(f"UPDATE findings SET cluster_id = NULL WHERE {clause}", params)
        self.conn.commit()

    @_locked
    def get_cluster_ids(self) -> list[int]:
        """All distinct non-null cluster IDs."""
        rows = self.conn.execute(
            "SELECT DISTINCT cluster_id FROM findings WHERE cluster_id IS NOT NULL ORDER BY cluster_id"
        ).fetchall()
        return [row[0] for row in rows]

    def species_filter(self, species: str) -> tuple[str, tuple[str, ...]]:
        """Return SQL clause and params for species filtering."""
        if species == "pet":
            placeholders = ",".join("?" * len(self.PET_SPECIES))
            return f"species IN ({placeholders})", self.PET_SPECIES
        return "species = ?", (species,)

    @_locked
    def get_unnamed_clusters(self, species: str = "human") -> list[dict[str, Any]]:
        """Clusters with no subject assigned, ordered by size."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT cluster_id, COUNT(*) as face_count,
                   GROUP_CONCAT(id) as finding_ids
            FROM findings
            WHERE cluster_id IS NOT NULL AND person_id IS NULL AND {clause}
            GROUP BY cluster_id
            HAVING COUNT(*) >= 2
            ORDER BY face_count DESC
        """,
            params,
        ).fetchall()
        result = []
        for row in rows:
            finding_ids = [int(x) for x in row["finding_ids"].split(",")]
            result.append(
                {
                    "cluster_id": row["cluster_id"],
                    "face_count": row["face_count"],
                    "sample_face_ids": finding_ids[:12],
                }
            )
        return result

    @_locked
    def get_singleton_findings(
        self, species: str = "human", limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        """Findings in clusters of size 1 or unclustered, unassigned, not dismissed."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT f.* FROM findings f
            WHERE f.person_id IS NULL AND {clause}
            AND f.id NOT IN (SELECT finding_id FROM dismissed_findings)
            AND (
                f.cluster_id IS NULL
                OR f.cluster_id IN (
                    SELECT cluster_id FROM findings
                    WHERE cluster_id IS NOT NULL AND person_id IS NULL
                    GROUP BY cluster_id HAVING COUNT(*) = 1
                )
            )
            LIMIT ? OFFSET ?
        """,
            (*params, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            if row["embedding"] is None:
                continue
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Finding(**d))
        return result

    @_locked
    def get_singleton_count(self, species: str = "human") -> int:
        clause, params = self.species_filter(species)
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM findings f
            WHERE f.person_id IS NULL AND {clause}
            AND f.id NOT IN (SELECT finding_id FROM dismissed_findings)
            AND (
                f.cluster_id IS NULL
                OR f.cluster_id IN (
                    SELECT cluster_id FROM findings
                    WHERE cluster_id IS NOT NULL AND person_id IS NULL
                    GROUP BY cluster_id HAVING COUNT(*) = 1
                )
            )
        """,
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_finding_count(self, cluster_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_findings(self, cluster_id: int, limit: int = 200) -> list[Finding]:
        rows = self.conn.execute(
            "SELECT * FROM findings WHERE cluster_id = ? LIMIT ?",
            (cluster_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Finding(**d))
        return result

    # --- Subjects ---

    def _species_for_kind(self, kind: str) -> str:
        """Map subject kind to the finding species filter value."""
        return self.KIND_TO_SPECIES[kind]

    def _is_species_kind_compatible(self, finding_species: str, subject_kind: str) -> bool:
        """Check if a finding's species is compatible with a subject's kind."""
        if subject_kind == "person":
            return finding_species == "human"
        if subject_kind == "pet":
            return finding_species in self.PET_SPECIES
        return True

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
    def get_subject_centroids(self, kind: str = "person") -> list[tuple[int, str, np.ndarray]]:
        """Return [(subject_id, name, centroid)] for all subjects of a kind."""
        species = self._species_for_kind(kind)
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT f.person_id, s.name, f.embedding
            FROM findings f
            JOIN subjects s ON s.id = f.person_id
            WHERE f.person_id IS NOT NULL AND s.kind = ? AND {clause}
            ORDER BY f.person_id
            """,
            (kind, *params),
        ).fetchall()

        from .embeddings import compute_centroid

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
    def assign_finding_to_subject(self, finding_id: int, subject_id: int) -> None:
        """Assign a finding to a subject, validating species/kind compatibility."""
        finding = self.get_finding(finding_id)
        subject = self.get_subject(subject_id)
        if (
            finding
            and subject
            and not self._is_species_kind_compatible(finding.species, subject.kind)
        ):
            raise ValueError(f"Cannot assign {finding.species} finding to a {subject.kind} subject")
        self.conn.execute(
            "UPDATE findings SET person_id = ? WHERE id = ?", (subject_id, finding_id)
        )
        self.conn.commit()

    @_locked
    def assign_cluster_to_subject(self, cluster_id: int, subject_id: int) -> None:
        """Assign all unassigned findings in a cluster to a subject."""
        subject = self.get_subject(subject_id)
        if subject:
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

    def get_sources_with_all_subjects(
        self, subject_ids: list[int], limit: int = 0, offset: int = 0
    ) -> list[Source]:
        """Find sources that contain ALL given subjects (intersection)."""
        if not subject_ids:
            return []
        placeholders = ",".join("?" * len(subject_ids))
        pagination = ""
        params: tuple[int | str, ...] = (*subject_ids, len(subject_ids))
        if limit > 0:
            pagination = " LIMIT ? OFFSET ?"
            params = (*params, limit, offset)
        rows = self.conn.execute(
            f"""
            SELECT s.* FROM sources s
            JOIN (
                SELECT source_id FROM findings
                WHERE person_id IN ({placeholders})
                GROUP BY source_id
                HAVING COUNT(DISTINCT person_id) = ?
            ) matched ON matched.source_id = s.id
            ORDER BY s.file_path DESC{pagination}
            """,
            params,
        ).fetchall()
        return [Source(**dict(r)) for r in rows]

    def count_sources_with_all_subjects(self, subject_ids: list[int]) -> int:
        """Count sources containing ALL given subjects."""
        if not subject_ids:
            return 0
        placeholders = ",".join("?" * len(subject_ids))
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT source_id FROM findings
                WHERE person_id IN ({placeholders})
                GROUP BY source_id
                HAVING COUNT(DISTINCT person_id) = ?
            )
            """,
            (*subject_ids, len(subject_ids)),
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_subject_sources(self, subject_id: int) -> list[Source]:
        """All unique sources containing a subject, newest first (by path)."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT s.* FROM sources s
            JOIN findings f ON f.source_id = s.id
            WHERE f.person_id = ?
            ORDER BY s.file_path DESC
            """,
            (subject_id,),
        ).fetchall()
        return [Source(**dict(r)) for r in rows]

    @_locked
    def dismiss_findings(self, finding_ids: list[int]) -> None:
        """Mark findings as non-faces (statues, paintings, etc.)."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO dismissed_findings (finding_id) VALUES (?)",
            [(fid,) for fid in finding_ids],
        )
        self.conn.executemany(
            "UPDATE findings SET cluster_id = NULL, person_id = NULL WHERE id = ?",
            [(fid,) for fid in finding_ids],
        )
        self.conn.commit()

    @_locked
    def unassign_finding(self, finding_id: int) -> None:
        """Remove subject assignment from a finding."""
        self.conn.execute("UPDATE findings SET person_id = NULL WHERE id = ?", (finding_id,))
        self.conn.commit()

    @_locked
    def exclude_findings(self, finding_ids: list[int], cluster_id: int) -> None:
        """Remove findings from a cluster (set cluster_id to NULL)."""
        placeholders = ",".join("?" * len(finding_ids))
        self.conn.execute(
            f"UPDATE findings SET cluster_id = NULL WHERE id IN ({placeholders}) "
            f"AND cluster_id = ?",
            (*finding_ids, cluster_id),
        )
        self.conn.commit()

    @_locked
    def merge_clusters(self, source_id: int, target_id: int) -> None:
        """Move all findings from source cluster to target cluster."""
        self.conn.execute(
            "UPDATE findings SET cluster_id = ? WHERE cluster_id = ?",
            (target_id, source_id),
        )
        self.conn.commit()

    @_locked
    def get_cluster_finding_ids(self, cluster_id: int) -> list[int]:
        """Return all finding IDs in a cluster."""
        rows = self.conn.execute(
            "SELECT id FROM findings WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def delete_subject(self, subject_id: int) -> None:
        """Delete a subject and unassign their findings."""
        self.conn.execute("UPDATE findings SET person_id = NULL WHERE person_id = ?", (subject_id,))
        self.conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))
        self.conn.commit()

    @_locked
    def get_unclustered_embeddings(self, species: str = "human") -> list[tuple[int, np.ndarray]]:
        """Return (finding_id, embedding) for unclustered, unassigned, non-dismissed findings."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE person_id IS NULL AND cluster_id IS NULL AND {clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            params,
        ).fetchall()
        return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]

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

    # --- Stats ---

    def _count(self, sql: str, params: tuple[str, ...] = ()) -> int:
        row = self.conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_stats(self, species: str = "human") -> dict[str, int]:
        clause, params = self.species_filter(species)
        return {
            "total_sources": self._count("SELECT COUNT(*) FROM sources WHERE type = 'photo'"),
            "total_findings": self._count(f"SELECT COUNT(*) FROM findings WHERE {clause}", params),
            "total_subjects": self._count("SELECT COUNT(*) FROM subjects"),
            "named_findings": self._count(
                f"SELECT COUNT(*) FROM findings WHERE person_id IS NOT NULL AND {clause}",
                params,
            ),
            "unnamed_clusters": self._count(
                f"SELECT COUNT(DISTINCT cluster_id) FROM findings "
                f"WHERE cluster_id IS NOT NULL AND person_id IS NULL AND {clause}",
                params,
            ),
            "unclustered_findings": self._count(
                f"SELECT COUNT(*) FROM findings WHERE cluster_id IS NULL AND {clause} "
                f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
                params,
            ),
            "dismissed_findings": self._count("SELECT COUNT(*) FROM dismissed_findings"),
        }

    # --- Export ---

    @_locked
    def export_json(self) -> str:
        """Export as JSON: subject -> sources -> finding rectangles."""
        subjects = self.get_subjects()
        data: dict[str, list[Any]] = {"subjects": [], "unnamed_findings": []}

        for subject in subjects:
            findings = self.get_subject_findings(subject.id, limit=10000)
            sources_map: dict[str, list[dict[str, Any]]] = {}
            for finding in findings:
                source = self.get_source(finding.source_id)
                if source:
                    sources_map.setdefault(source.file_path, []).append(
                        {
                            "bbox": [
                                finding.bbox_x,
                                finding.bbox_y,
                                finding.bbox_w,
                                finding.bbox_h,
                            ],
                            "confidence": finding.confidence,
                        }
                    )
            data["subjects"].append(
                {
                    "id": subject.id,
                    "name": subject.name,
                    "kind": subject.kind,
                    "sources": [
                        {"file_path": fp, "findings": rects} for fp, rects in sources_map.items()
                    ],
                }
            )

        unnamed = self.conn.execute("SELECT * FROM findings WHERE person_id IS NULL").fetchall()
        for row in unnamed:
            source = self.get_source(row["source_id"])
            if source:
                data["unnamed_findings"].append(
                    {
                        "source": source.file_path,
                        "bbox": [
                            row["bbox_x"],
                            row["bbox_y"],
                            row["bbox_w"],
                            row["bbox_h"],
                        ],
                        "cluster_id": row["cluster_id"],
                    }
                )

        return json.dumps(data, indent=2)
