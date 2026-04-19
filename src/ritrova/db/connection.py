"""FaceDB: assembled class from mixins, schema creation, and migrations."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._base import _locked
from .circles import CirclesMixin
from .clusters import ClusterMixin
from .curation import CurationMixin
from .descriptions import DescriptionMixin
from .findings import FindingMixin
from .maintenance import MaintenanceMixin
from .paths import PathMixin
from .scans import ScanMixin
from .sources import SourceMixin
from .subjects import SubjectMixin
from .undo_support import UndoMixin

# Logical FK metadata for scans.scan_type — kept in sync with the shipping
# pipelines via ``_sync_scan_types`` on every DB open. Retire entries by
# setting ``retired_at`` rather than deleting them: historic scans keep
# pointing at the row for audit / A/B lineage.
KNOWN_SCAN_TYPES: list[dict[str, str | None]] = [
    {
        "name": "human",
        "pipeline": "arcface(buffalo_l, det_size=640)",
        "outputs": "findings(species=human, 512-dim embedding, bbox, confidence)",
        "introduced_at": "2026-04",
        "retired_at": None,
        "notes": (
            "Legacy face-only pipeline. No MIN_FACE_SIZE filter, no quality "
            "thresholds. Still the authoritative face baseline for pre-composite rows."
        ),
    },
    {
        "name": "pet",
        "pipeline": "yolo(yolo11m, classes={cat=15,dog=16}) -> siglip(base-patch16-224)",
        "outputs": "findings(species=dog|cat|other_pet, 768-dim embedding, bbox, confidence)",
        "introduced_at": "2026-04",
        "retired_at": None,
        "notes": "Legacy pet-only pipeline.",
    },
    {
        "name": "composite",
        "pipeline": (
            "qwen2.5-vl-7b(mlx,4-bit) prefilter -> arcface(buffalo_l, min_face_size=50, "
            "sharpness+edge filters) -> yolo+siglip -> opus-mt-tc-big-en-it -> dedup"
        ),
        "outputs": "mixed findings + descriptions(it caption + it tags)",
        "introduced_at": "2026-04-14",
        "retired_at": "2026-04-18",
        "notes": "Retired; see ADR-011. Italian translation quality + face recall loss.",
    },
    {
        "name": "subjects",
        "pipeline": (
            "arcface(buffalo_l, det_size=640, min_face_size=50, sharpness+edge filters) "
            "-> yolo11m+siglip-base-patch16-224 -> dedup"
        ),
        "outputs": "findings(species=human|dog|cat|other_pet, embedding, bbox, confidence)",
        "introduced_at": "2026-04-18",
        "retired_at": None,
        "notes": "Default `ritrova analyse`. Unifies human+pet under one scan.",
    },
    {
        "name": "subjects+captions",
        "pipeline": (
            "qwen2.5-vl-7b(mlx,4-bit) prefilter -> arcface(min_face_size=50) "
            "-> yolo+siglip -> opus-mt-tc-big-en-it -> dedup"
        ),
        "outputs": "subjects findings + descriptions(it caption + it tags)",
        "introduced_at": "2026-04-18",
        "retired_at": None,
        "notes": "Opt-in via `ritrova analyse --caption`. Apple Silicon only (MLX).",
    },
]


class FaceDB(
    PathMixin,
    SourceMixin,
    ScanMixin,
    FindingMixin,
    SubjectMixin,
    ClusterMixin,
    CurationMixin,
    DescriptionMixin,
    CirclesMixin,
    UndoMixin,
    MaintenanceMixin,
):
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
        for migration in [
            "ALTER TABLE scans ADD COLUMN detection_strategy TEXT NOT NULL DEFAULT 'unknown'",
            "ALTER TABLE findings ADD COLUMN embedding_dim INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE findings ADD COLUMN scan_id INTEGER REFERENCES scans(id) ON DELETE CASCADE",
            "ALTER TABLE findings ADD COLUMN frame_number INTEGER NOT NULL DEFAULT 0",
        ]:
            with contextlib.suppress(sqlite3.OperationalError):
                self.conn.execute(migration)
        with contextlib.suppress(sqlite3.OperationalError):
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_scan ON findings(scan_id)")

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS descriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                scan_id INTEGER NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
                caption TEXT NOT NULL,
                tags TEXT NOT NULL,
                generated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_descriptions_source ON descriptions(source_id);
            CREATE INDEX IF NOT EXISTS idx_descriptions_scan ON descriptions(scan_id);
            CREATE INDEX IF NOT EXISTS idx_descriptions_tags ON descriptions(tags);

            -- Documentation-only table. `scans.scan_type` is a logical FK to
            -- `scan_types.name` (no hard FK, so historic scan_type strings
            -- continue to exist even when their row here is retired).
            CREATE TABLE IF NOT EXISTS scan_types (
                name            TEXT PRIMARY KEY,
                pipeline        TEXT NOT NULL,
                outputs         TEXT NOT NULL,
                introduced_at   TEXT,
                retired_at      TEXT,
                notes           TEXT
            );

            -- FEAT-27: circles as a filtering metadata axis on subjects.
            CREATE TABLE IF NOT EXISTS circles (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at  TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS subject_circles (
                subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
                circle_id  INTEGER NOT NULL REFERENCES circles(id)  ON DELETE CASCADE,
                added_at   TEXT NOT NULL,
                PRIMARY KEY (subject_id, circle_id)
            );
            CREATE INDEX IF NOT EXISTS idx_subject_circles_circle
                ON subject_circles(circle_id);
        """)
        self.conn.commit()

        self._backfill_finding_scan_ids()
        self._enforce_finding_scan_id_not_null()
        self._sync_scan_types()
        self._ensure_strangers_circle()

    def _ensure_strangers_circle(self) -> None:
        """Seed the 'Strangers' circle so mark-stranger has somewhere to file subjects."""
        self.conn.execute(
            "INSERT OR IGNORE INTO circles (name, description, created_at) VALUES (?, ?, ?)",
            (
                "Strangers",
                "Auto-created faces the user explicitly marked as unknown.",
                self._now(),
            ),
        )
        self.conn.commit()

    def _sync_scan_types(self) -> None:
        """INSERT OR REPLACE the known scan_types entries.

        Idempotent on each open — keeps the scan_types catalog aligned with
        the code's current understanding of pipelines and their outputs.
        When a new scan_type ships, add it to ``KNOWN_SCAN_TYPES`` below and
        the next DB open picks it up.
        """
        for row in KNOWN_SCAN_TYPES:
            self.conn.execute(
                "INSERT OR REPLACE INTO scan_types "
                "(name, pipeline, outputs, introduced_at, retired_at, notes) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    row["name"],
                    row["pipeline"],
                    row["outputs"],
                    row.get("introduced_at"),
                    row.get("retired_at"),
                    row.get("notes"),
                ),
            )
        self.conn.commit()

    def _backfill_finding_scan_ids(self) -> None:
        """One-time backfill: link every legacy finding to a scan.

        Strategy:
          1. 512-dim embedding -> human scan on the same source.
          2. 768-dim embedding -> pet scan on the same source.
          3. Orphans (no matching scan): synthesize a `legacy_backfill` scan row
             of the right type on that source and link the finding to it.

        Idempotent: only touches rows where `scan_id IS NULL`.
        """
        null_count = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id IS NULL"
        ).fetchone()[0]
        if not null_count:
            return

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

        SQLite can't `ALTER COLUMN ... SET NOT NULL`. Idempotent via PRAGMA introspection.
        """
        info = self.conn.execute("PRAGMA table_info(findings)").fetchall()
        scan_col = next((c for c in info if c["name"] == "scan_id"), None)
        if scan_col is None or scan_col["notnull"] == 1:
            return

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
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                frame_number INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO findings_new
                SELECT id, source_id, scan_id, bbox_x, bbox_y, bbox_w, bbox_h,
                       embedding, person_id, cluster_id, confidence, species,
                       detected_at, frame_path, embedding_dim, 0
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

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _count(self, sql: str, params: tuple[str, ...] = ()) -> int:
        row = self.conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self.conn.close()
