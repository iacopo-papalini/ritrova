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
    embedding_dim: int = 0
    # 0 only as an in-memory default for tests/fixtures that don't round-trip via SQLite.
    # Real DB rows always have scan_id NOT NULL after the migration.
    scan_id: int = 0


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
        # Migrations (idempotent — errors suppressed if already applied)
        import contextlib

        for migration in [
            "ALTER TABLE scans ADD COLUMN detection_strategy TEXT NOT NULL DEFAULT 'unknown'",
            "ALTER TABLE findings ADD COLUMN embedding_dim INTEGER NOT NULL DEFAULT 0",
            # Step A: add the column nullable so existing rows stay valid;
            # backfill below will populate, then a table-rebuild enforces NOT NULL.
            "ALTER TABLE findings ADD COLUMN scan_id INTEGER REFERENCES scans(id) ON DELETE CASCADE",
        ]:
            with contextlib.suppress(sqlite3.OperationalError):
                self.conn.execute(migration)
        # Index on scan_id supports DELETE-cascade and `find_scans` lookups.
        with contextlib.suppress(sqlite3.OperationalError):
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_scan ON findings(scan_id)")
        self.conn.commit()

        self._backfill_finding_scan_ids()
        self._enforce_finding_scan_id_not_null()

    def _backfill_finding_scan_ids(self) -> None:
        """One-time backfill: link every legacy finding to a scan.

        Strategy:
          1. 512-dim embedding → human scan on the same source.
          2. 768-dim embedding → pet   scan on the same source.
          3. Orphans (no matching scan): synthesize a `legacy_backfill` scan row
             of the right type on that source and link the finding to it.

        Idempotent: only touches rows where `scan_id IS NULL`.
        """
        # Quick exit if nothing to do (avoids the orphan SELECT on every startup).
        null_count = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id IS NULL"
        ).fetchone()[0]
        if not null_count:
            return

        # Step 1: 512-dim → human scan.
        self.conn.execute(
            """
            UPDATE findings
            SET scan_id = (
                SELECT sc.id FROM scans sc
                WHERE sc.source_id = findings.source_id AND sc.scan_type = 'human'
                LIMIT 1
            )
            WHERE scan_id IS NULL AND length(embedding) / 4 = 512
            """
        )
        # Step 2: 768-dim → pet scan.
        self.conn.execute(
            """
            UPDATE findings
            SET scan_id = (
                SELECT sc.id FROM scans sc
                WHERE sc.source_id = findings.source_id AND sc.scan_type = 'pet'
                LIMIT 1
            )
            WHERE scan_id IS NULL AND length(embedding) / 4 = 768
            """
        )
        self.conn.commit()

        # Step 3: synthesize scans for any remaining orphans.
        orphans = self.conn.execute(
            """
            SELECT DISTINCT source_id, length(embedding) / 4 AS dim
            FROM findings WHERE scan_id IS NULL
            """
        ).fetchall()
        now = self._now()
        for source_id, dim in orphans:
            scan_type = "human" if dim == 512 else "pet"
            cur = self.conn.execute(
                "INSERT INTO scans (source_id, scan_type, scanned_at, detection_strategy) "
                "VALUES (?, ?, ?, 'legacy_backfill')",
                (source_id, scan_type, now),
            )
            self.conn.execute(
                "UPDATE findings SET scan_id = ? "
                "WHERE source_id = ? AND scan_id IS NULL AND length(embedding) / 4 = ?",
                (cur.lastrowid, source_id, dim),
            )
        self.conn.commit()

    def _enforce_finding_scan_id_not_null(self) -> None:
        """Enforce `findings.scan_id NOT NULL` via table rebuild.

        SQLite can't `ALTER COLUMN ... SET NOT NULL`. Idempotent via PRAGMA introspection:
        if `scan_id` is already non-nullable, this is a no-op. Aborts if any row still has
        NULL `scan_id` (signals a broken backfill rather than silently dropping rows).
        """
        info = self.conn.execute("PRAGMA table_info(findings)").fetchall()
        scan_col = next((c for c in info if c["name"] == "scan_id"), None)
        if scan_col is None or scan_col["notnull"] == 1:
            return  # column missing entirely (shouldn't happen) or already enforced

        null_count = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id IS NULL"
        ).fetchone()[0]
        if null_count:
            msg = (
                f"Cannot enforce findings.scan_id NOT NULL: {null_count} rows still NULL "
                "after backfill — investigate before proceeding."
            )
            raise RuntimeError(msg)

        self.conn.executescript(
            """
            BEGIN;
            CREATE TABLE findings_new (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
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
                frame_path TEXT,
                embedding_dim INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO findings_new
                SELECT id, source_id, scan_id, bbox_x, bbox_y, bbox_w, bbox_h,
                       embedding, person_id, cluster_id, confidence, species,
                       detected_at, frame_path, embedding_dim
                FROM findings;
            DROP TABLE findings;
            ALTER TABLE findings_new RENAME TO findings;
            CREATE INDEX idx_findings_source  ON findings(source_id);
            CREATE INDEX idx_findings_person  ON findings(person_id);
            CREATE INDEX idx_findings_cluster ON findings(cluster_id);
            CREATE INDEX idx_findings_scan    ON findings(scan_id);
            COMMIT;
            """
        )

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
    def record_scan(
        self, source_id: int, scan_type: str, detection_strategy: str = "unknown"
    ) -> int:
        """Insert a scan row and return its id.

        Callers MUST gate on `is_scanned()` first (UNIQUE on (source_id, scan_type)
        will raise otherwise). Rescans must `delete_scan()` the old one first.
        """
        cur = self.conn.execute(
            "INSERT INTO scans (source_id, scan_type, scanned_at, detection_strategy) "
            "VALUES (?, ?, ?, ?)",
            (source_id, scan_type, self._now(), detection_strategy),
        )
        self.conn.commit()
        scan_id = cur.lastrowid
        if scan_id is None:
            msg = "INSERT returned no lastrowid for scan"
            raise RuntimeError(msg)
        return scan_id

    @_locked
    def delete_scan(self, scan_id: int) -> dict[str, int]:
        """Delete a scan and (via ON DELETE CASCADE) all its findings.

        Returns counts {`deleted_findings`, `deleted_with_assignments`} for caller
        confirmation prompts. Raises ValueError if the scan doesn't exist.
        """
        scan = self.conn.execute("SELECT id FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if not scan:
            msg = f"No such scan: {scan_id}"
            raise ValueError(msg)
        deleted_findings = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id = ?", (scan_id,)
        ).fetchone()[0]
        deleted_with_assignments = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id = ? AND person_id IS NOT NULL",
            (scan_id,),
        ).fetchone()[0]
        # ON DELETE CASCADE on findings.scan_id handles the children.
        self.conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        self.conn.commit()
        return {
            "deleted_findings": int(deleted_findings),
            "deleted_with_assignments": int(deleted_with_assignments),
        }

    @_locked
    def find_scans(
        self, scan_id: int | None = None, source_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """List scans matching the given filters (AND semantics; both optional).

        Returns dicts with keys: id, source_id, source_path, scan_type, scanned_at,
        detection_strategy, finding_count.

        `source_pattern` uses SQLite GLOB syntax (`*`, `?`) on `sources.file_path`.
        """
        clauses = []
        params: list[Any] = []
        if scan_id is not None:
            clauses.append("sc.id = ?")
            params.append(scan_id)
        if source_pattern is not None:
            clauses.append("s.file_path GLOB ?")
            params.append(source_pattern)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"""
            SELECT sc.id, sc.source_id, s.file_path AS source_path, sc.scan_type,
                   sc.scanned_at, sc.detection_strategy,
                   (SELECT COUNT(*) FROM findings f WHERE f.scan_id = sc.id) AS finding_count
            FROM scans sc
            JOIN sources s ON s.id = sc.source_id
            {where}
            ORDER BY sc.scanned_at DESC, sc.id
            """,
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]

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
        *,
        scan_id: int,
        species: str = "human",
        frame_path: str | None = None,
    ) -> None:
        """Add multiple findings in a single transaction.

        `scan_id` is keyword-only and required: every finding must trace to the scan
        that produced it, so a faulty scan can be pruned without losing curation
        elsewhere on the same source.
        """
        now = self._now()
        self.conn.executemany(
            "INSERT INTO findings (source_id, bbox_x, bbox_y, bbox_w, bbox_h, "
            "embedding, confidence, detected_at, species, frame_path, embedding_dim, scan_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (sid, *bbox, emb.tobytes(), conf, now, species, frame_path, len(emb), scan_id)
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

    def _dim_filter(self, embedding_dim: int | None) -> tuple[str, tuple[int, ...]]:
        """Return SQL clause and params for embedding dimension filtering."""
        if embedding_dim is None:
            return "1=1", ()
        return "embedding_dim = ?", (embedding_dim,)

    @_locked
    def get_all_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Return all (finding_id, embedding) pairs for clustering, excluding dismissed."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE {clause} AND {dim_clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            (*params, *dim_params),
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def get_unassigned_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Return (finding_id, embedding) for unassigned, non-dismissed findings only."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE person_id IS NULL AND {clause} AND {dim_clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            (*params, *dim_params),
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
    def assign_finding_to_subject(
        self, finding_id: int, subject_id: int, *, force: bool = False
    ) -> None:
        """Assign a finding to a subject.

        Raises ValueError on species/kind mismatch unless force=True.
        Species is never mutated — the finding keeps its original species
        and embedding space. Centroid queries filter by embedding_dim to
        exclude incompatible findings automatically.
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

    def _together_query(
        self, subject_ids: list[int], alone: bool = False
    ) -> tuple[str, tuple[int | str, ...]]:
        """Build the inner subquery for together (shared by get and count)."""
        placeholders = ",".join("?" * len(subject_ids))
        n = len(subject_ids)
        if alone:
            # All named findings on the source — require exactly the requested set
            return (
                f"""SELECT source_id FROM findings
                    WHERE person_id IS NOT NULL
                    GROUP BY source_id
                    HAVING COUNT(DISTINCT CASE WHEN person_id IN ({placeholders}) THEN person_id END) = ?
                    AND COUNT(DISTINCT person_id) = ?""",
                (*subject_ids, n, n),
            )
        # At least these subjects (may include others)
        return (
            f"""SELECT source_id FROM findings
                WHERE person_id IN ({placeholders})
                GROUP BY source_id
                HAVING COUNT(DISTINCT person_id) = ?""",
            (*subject_ids, n),
        )

    def get_sources_with_all_subjects(
        self,
        subject_ids: list[int],
        limit: int = 0,
        offset: int = 0,
        alone: bool = False,
    ) -> list[Source]:
        """Find sources that contain ALL given subjects.

        If alone=True, exclude sources that also contain other named subjects.
        """
        if not subject_ids:
            return []
        inner, params = self._together_query(subject_ids, alone)
        pagination = ""
        full_params: tuple[int | str, ...] = params
        if limit > 0:
            pagination = " LIMIT ? OFFSET ?"
            full_params = (*params, limit, offset)
        rows = self.conn.execute(
            f"""
            SELECT s.* FROM sources s
            JOIN ({inner}) matched ON matched.source_id = s.id
            ORDER BY s.file_path DESC{pagination}
            """,
            full_params,
        ).fetchall()
        return [Source(**dict(r)) for r in rows]

    def count_sources_with_all_subjects(self, subject_ids: list[int], alone: bool = False) -> int:
        """Count sources containing ALL given subjects."""
        if not subject_ids:
            return 0
        inner, params = self._together_query(subject_ids, alone)
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM ({inner})",
            params,
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
    def get_subject_sources_with_findings(
        self, subject_id: int, source_type: str
    ) -> list[tuple[Source, list[Finding]]]:
        """Sources of a given type containing the subject, each paired with the subject's findings on it.

        Used by the Videos tab on the subject detail page — groups findings by source so each video
        card can show a thumbnail (from one of the findings) and a count of appearances.
        """
        # Explicit aliases: `SELECT s.*, f.*` collides on `id` and `dict(sqlite3.Row)` silently keeps
        # only the first one (source's), which made every Finding inherit the source's id.
        rows = self.conn.execute(
            """
            SELECT
                s.id        AS s_id,
                s.file_path AS s_file_path,
                s.type      AS s_type,
                s.width     AS s_width,
                s.height    AS s_height,
                s.taken_at  AS s_taken_at,
                s.latitude  AS s_latitude,
                s.longitude AS s_longitude,
                f.id            AS f_id,
                f.bbox_x        AS f_bbox_x,
                f.bbox_y        AS f_bbox_y,
                f.bbox_w        AS f_bbox_w,
                f.bbox_h        AS f_bbox_h,
                f.embedding     AS f_embedding,
                f.person_id     AS f_person_id,
                f.cluster_id    AS f_cluster_id,
                f.confidence    AS f_confidence,
                f.detected_at   AS f_detected_at,
                f.species       AS f_species,
                f.frame_path    AS f_frame_path,
                f.embedding_dim AS f_embedding_dim,
                f.scan_id       AS f_scan_id
            FROM sources s
            JOIN findings f ON f.source_id = s.id
            WHERE f.person_id = ? AND s.type = ?
            ORDER BY s.file_path DESC, f.id
            """,
            (subject_id, source_type),
        ).fetchall()

        grouped: dict[int, tuple[Source, list[Finding]]] = {}
        for row in rows:
            if row["f_embedding"] is None:
                continue
            source_id = row["s_id"]
            if source_id not in grouped:
                source = Source(
                    id=source_id,
                    file_path=row["s_file_path"],
                    type=row["s_type"],
                    width=row["s_width"],
                    height=row["s_height"],
                    taken_at=row["s_taken_at"],
                    latitude=row["s_latitude"],
                    longitude=row["s_longitude"],
                )
                grouped[source_id] = (source, [])

            finding = Finding(
                id=row["f_id"],
                source_id=source_id,
                bbox_x=row["f_bbox_x"],
                bbox_y=row["f_bbox_y"],
                bbox_w=row["f_bbox_w"],
                bbox_h=row["f_bbox_h"],
                embedding=np.frombuffer(row["f_embedding"], dtype=np.float32),
                person_id=row["f_person_id"],
                cluster_id=row["f_cluster_id"],
                confidence=row["f_confidence"],
                detected_at=row["f_detected_at"],
                species=row["f_species"] or "human",
                frame_path=row["f_frame_path"],
                embedding_dim=row["f_embedding_dim"] or 0,
                scan_id=row["f_scan_id"],
            )
            grouped[source_id][1].append(finding)

        return list(grouped.values())

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
    def get_unclustered_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Return (finding_id, embedding) for unclustered, unassigned, non-dismissed findings."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE person_id IS NULL AND cluster_id IS NULL AND {clause} AND {dim_clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            (*params, *dim_params),
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
