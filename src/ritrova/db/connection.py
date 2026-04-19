"""FaceDB: assembled class from mixins, schema creation, and migrations."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._base import _locked
from .assignments import AssignmentMixin
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
    AssignmentMixin,
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
                confidence REAL NOT NULL DEFAULT 0.0,
                species TEXT NOT NULL DEFAULT 'human',
                detected_at TEXT NOT NULL,
                frame_path TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_findings_source ON findings(source_id);
            CREATE INDEX IF NOT EXISTS idx_sources_path ON sources(file_path);

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

            -- Apr 2026 refactor: split mutable curation state off the
            -- immutable `findings` table so the CV output stays pristine.
            -- `cluster_findings` holds per-finding cluster membership (one
            -- row per clustered finding; no row = unclustered). `finding_
            -- assignment` holds user-authored curation: exactly one of
            -- `subject_id` (named) or `exclusion_reason` (stranger /
            -- not_a_face) per row. Absence of a row = uncurated.
            CREATE TABLE IF NOT EXISTS cluster_findings (
                finding_id INTEGER PRIMARY KEY REFERENCES findings(id) ON DELETE CASCADE,
                cluster_id INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_cluster_findings_cluster
                ON cluster_findings(cluster_id);

            CREATE TABLE IF NOT EXISTS finding_assignment (
                finding_id       INTEGER PRIMARY KEY REFERENCES findings(id) ON DELETE CASCADE,
                subject_id       INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
                exclusion_reason TEXT,
                curated_at       TEXT NOT NULL,
                CHECK ((subject_id IS NULL) <> (exclusion_reason IS NULL)),
                CHECK (exclusion_reason IS NULL
                       OR exclusion_reason IN ('stranger', 'not_a_face'))
            );
            CREATE INDEX IF NOT EXISTS idx_finding_assignment_subject
                ON finding_assignment(subject_id)
                WHERE subject_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_finding_assignment_reason
                ON finding_assignment(exclusion_reason)
                WHERE exclusion_reason IS NOT NULL;
        """)
        self.conn.commit()

        self._backfill_finding_scan_ids()
        # Migrate to finding_assignment / cluster_findings BEFORE the
        # scan_id-NOT-NULL rebuild. The rebuild DROPs the findings table
        # to re-create it, which would cascade-wipe any dependent rows.
        # Running the migration first means:
        #   - dismissed_findings rows (pre-refactor, FK into findings) are
        #     copied into finding_assignment before the DROP wipes them;
        #   - the DROP+rebuild runs inside a foreign-keys-off block (see
        #     _enforce_finding_scan_id_not_null) so the fresh
        #     finding_assignment + cluster_findings rows also survive.
        self._migrate_to_finding_assignment()
        self._enforce_finding_scan_id_not_null()
        self._sync_scan_types()
        self._drop_obsolete_finding_columns()

    def _drop_obsolete_finding_columns(self) -> None:
        """Drop the denormalized columns and the obsolete `dismissed_findings`
        table. Runs after ``_migrate_to_finding_assignment`` so the copy-out
        is already done. Each statement is idempotent — a second open after
        the first successful drop is a no-op (OperationalError swallowed).
        """
        for stmt in (
            "DROP INDEX IF EXISTS idx_findings_person",
            "DROP INDEX IF EXISTS idx_findings_cluster",
            "ALTER TABLE findings DROP COLUMN person_id",
            "ALTER TABLE findings DROP COLUMN cluster_id",
            "DROP TABLE IF EXISTS dismissed_findings",
        ):
            with contextlib.suppress(sqlite3.OperationalError):
                self.conn.execute(stmt)
        self.conn.commit()

    def _migrate_to_finding_assignment(self) -> None:
        """One-shot migration (Apr 2026): populate `cluster_findings` and
        `finding_assignment` from the old `findings.cluster_id` /
        `findings.person_id` columns and from `dismissed_findings`.

        Subjects in the auto-seeded `Strangers` circle are dissolved: their
        findings become `exclusion_reason='stranger'` rows, the subjects
        themselves are deleted, and the circle row is removed (the user
        agreed the Strangers circle should disappear in this refactor —
        strangers are now a finding-level state, not a subject grouping).

        Idempotent: if either new table already has rows, assume migration
        ran previously and return. Fresh DBs (both tables empty, findings
        table also empty of person_id/cluster_id data) skip through cheaply.
        """
        already = self.conn.execute(
            "SELECT EXISTS(SELECT 1 FROM finding_assignment) "
            "OR EXISTS(SELECT 1 FROM cluster_findings)"
        ).fetchone()[0]
        if already:
            return
        # Fresh DB: no legacy columns/table to migrate from.
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(findings)").fetchall()}
        has_legacy_findings = "person_id" in cols or "cluster_id" in cols
        has_dismissed = (
            self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='dismissed_findings'"
            ).fetchone()
            is not None
        )
        if not has_legacy_findings and not has_dismissed:
            return

        now = self._now()
        cur = self.conn.cursor()
        has_person = "person_id" in cols
        has_cluster = "cluster_id" in cols

        # Dissolve Strangers-circle subjects first so their findings get the
        # 'stranger' reason below instead of being mis-migrated as active
        # assignments. Collect the subject_ids up front.
        stranger_subject_ids = (
            [
                row[0]
                for row in cur.execute(
                    "SELECT sc.subject_id FROM subject_circles sc "
                    "JOIN circles c ON c.id = sc.circle_id "
                    "WHERE c.name = 'Strangers'"
                ).fetchall()
            ]
            if has_person
            else []
        )

        cur.execute("BEGIN")

        if has_cluster:
            cur.execute(
                "INSERT INTO cluster_findings(finding_id, cluster_id) "
                "SELECT id, cluster_id FROM findings WHERE cluster_id IS NOT NULL"
            )

        if has_person and stranger_subject_ids:
            placeholders = ",".join("?" * len(stranger_subject_ids))
            if has_cluster:
                cur.execute(
                    f"DELETE FROM cluster_findings WHERE finding_id IN "
                    f"(SELECT id FROM findings WHERE person_id IN ({placeholders}))",
                    tuple(stranger_subject_ids),
                )
            cur.execute(
                f"INSERT INTO finding_assignment"
                f"(finding_id, subject_id, exclusion_reason, curated_at) "
                f"SELECT id, NULL, 'stranger', ? FROM findings "
                f"WHERE person_id IN ({placeholders})",
                (now, *stranger_subject_ids),
            )
            cur.execute(
                f"DELETE FROM subjects WHERE id IN ({placeholders})",
                tuple(stranger_subject_ids),
            )

        if has_person:
            # `OR IGNORE` so stranger findings (which got a 'stranger' row
            # inserted above) don't trip the UNIQUE constraint — SQLite's
            # cascade doesn't fire inside an open BEGIN for deferred FK
            # processing the way one would hope, so the source rows may
            # still have person_id set to a now-deleted subject id. Those
            # rows are already curated with exclusion_reason='stranger', so
            # the IGNORE is correct: the existing row wins.
            cur.execute(
                "INSERT OR IGNORE INTO finding_assignment"
                "(finding_id, subject_id, exclusion_reason, curated_at) "
                "SELECT id, person_id, NULL, ? FROM findings "
                "WHERE person_id IS NOT NULL",
                (now,),
            )

        if has_dismissed:
            # INNER JOIN on findings skips orphaned dismissed rows (their FK
            # target is missing, so the insert would fail). OR IGNORE guards
            # against the double-curated-and-dismissed edge case.
            cur.execute(
                "INSERT OR IGNORE INTO finding_assignment"
                "(finding_id, subject_id, exclusion_reason, curated_at) "
                "SELECT df.finding_id, NULL, 'not_a_face', ? "
                "FROM dismissed_findings df "
                "JOIN findings f ON f.id = df.finding_id",
                (now,),
            )

        # Drop the Strangers circle row itself (auto-seeded; now obsolete).
        cur.execute("DELETE FROM circles WHERE name = 'Strangers'")

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

        # Apr 2026 refactor: person_id/cluster_id are gone from findings.
        # The rebuild preserves only the current (post-refactor) columns.
        # If this runs on a pre-refactor DB, _migrate_to_finding_assignment
        # has already copied that state out; if it runs on a fresh DB, the
        # old columns don't exist to begin with.
        has_person = any(c["name"] == "person_id" for c in info)
        has_cluster = any(c["name"] == "cluster_id" for c in info)
        legacy_cols = ""
        legacy_select = ""
        if has_person:
            legacy_cols += ", person_id INTEGER"
            legacy_select += ", person_id"
        if has_cluster:
            legacy_cols += ", cluster_id INTEGER"
            legacy_select += ", cluster_id"
        # FKs off during the rebuild so DROP TABLE findings doesn't
        # cascade-wipe the sibling tables that reference findings.id
        # (finding_assignment, cluster_findings, dismissed_findings).
        # SQLite's recommended table-rebuild dance per
        # https://www.sqlite.org/lang_altertable.html#otheralter
        self.conn.executescript(
            f"""
            PRAGMA foreign_keys=OFF;
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
                confidence REAL NOT NULL DEFAULT 0.0,
                species TEXT NOT NULL DEFAULT 'human',
                detected_at TEXT NOT NULL,
                frame_path TEXT,
                embedding_dim INTEGER NOT NULL DEFAULT 0,
                frame_number INTEGER NOT NULL DEFAULT 0
                {legacy_cols}
            );
            INSERT INTO findings_new
                SELECT id, source_id, scan_id, bbox_x, bbox_y, bbox_w, bbox_h,
                       embedding, confidence, species,
                       detected_at, frame_path, embedding_dim, 0
                       {legacy_select}
                FROM findings;
            DROP TABLE findings;
            ALTER TABLE findings_new RENAME TO findings;
            CREATE INDEX idx_findings_source  ON findings(source_id);
            CREATE INDEX idx_findings_scan    ON findings(scan_id);
            COMMIT;
            PRAGMA foreign_keys=ON;
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
