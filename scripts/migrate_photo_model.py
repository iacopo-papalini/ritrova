#!/usr/bin/env python3
"""Migrate photos/faces schema to sources/findings/scans model.

Run OFFLINE against the DB file before deploying code changes.
Idempotent: safe to run more than once.

Usage:
    python scripts/migrate_photo_model.py data/faces.db /path/to/photos/dir
"""
# ruff: noqa: T201

import sqlite3
import sys
from pathlib import Path


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def migrate(db_path: Path, base_dir: str) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=OFF")

    if not table_exists(conn, "photos"):
        if table_exists(conn, "sources"):
            print("Already migrated (sources table exists, photos does not).")
            conn.close()
            return
        print("ERROR: neither 'photos' nor 'sources' table found")
        sys.exit(1)

    if table_exists(conn, "sources"):
        print("sources table already exists — migration may have partially run.")
        print("Restore from backup and re-run.")
        sys.exit(1)

    # Ensure base_dir ends with /
    if not base_dir.endswith("/"):
        base_dir += "/"

    print(f"Base dir: {base_dir}")

    # ── Pre-migration stats ──
    photo_count = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    dismissed_count = conn.execute("SELECT COUNT(*) FROM dismissed_faces").fetchone()[0]
    abs_count = conn.execute("SELECT COUNT(*) FROM photos WHERE file_path LIKE '/%'").fetchone()[0]
    pets_count = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE file_path LIKE '%__pets'"
    ).fetchone()[0]
    nofaces_count = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE file_path LIKE '__nofaces_%'"
    ).fetchone()[0]

    print("\n── Pre-migration stats ──")
    print(
        f"  Photos: {photo_count} (absolute: {abs_count}, __pets: {pets_count}, __nofaces: {nofaces_count})"
    )
    print(f"  Faces: {face_count}")
    print(f"  Dismissed: {dismissed_count}")

    # ── Step 1: Normalize absolute paths ──
    print("\nStep 1: Normalizing absolute paths...")
    base_len = len(base_dir)
    cur = conn.execute(
        "UPDATE photos SET file_path = SUBSTR(file_path, ?) "
        "WHERE file_path LIKE ? AND file_path NOT LIKE '%__pets'",
        (base_len + 1, base_dir + "%"),
    )
    conn.commit()
    print(f"  Normalized {cur.rowcount} absolute paths")

    # ── Step 2: Create new tables ──
    print("\nStep 2: Creating new tables...")
    conn.executescript("""
        CREATE TABLE sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE NOT NULL,
            type TEXT NOT NULL DEFAULT 'photo',
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            taken_at TEXT,
            latitude REAL,
            longitude REAL
        );

        CREATE TABLE scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id INTEGER NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            scan_type TEXT NOT NULL,
            scanned_at TEXT NOT NULL,
            UNIQUE (source_id, scan_type)
        );

        CREATE TABLE findings (
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

        CREATE INDEX idx_findings_source ON findings(source_id);
        CREATE INDEX idx_findings_person ON findings(person_id);
        CREATE INDEX idx_findings_cluster ON findings(cluster_id);

        CREATE TABLE dismissed_findings (
            finding_id INTEGER PRIMARY KEY REFERENCES findings(id) ON DELETE CASCADE
        );
    """)

    # ── Step 3a: Migrate regular photos → sources ──
    print("\nStep 3a: Migrating photos → sources...")
    cur = conn.execute("""
        INSERT INTO sources (id, file_path, type, width, height, taken_at, latitude, longitude)
        SELECT id, file_path, 'photo', width, height, taken_at, latitude, longitude
        FROM photos
        WHERE video_path IS NULL
        AND file_path NOT LIKE '%__pets'
        AND file_path NOT LIKE '__nofaces_%'
    """)
    conn.commit()
    photo_sources = cur.rowcount
    print(f"  Migrated {photo_sources} photo sources")

    # ── Step 3b: Migrate videos → one source row per video ──
    print("Step 3b: Migrating videos → sources...")
    cur = conn.execute("""
        INSERT INTO sources (file_path, type, width, height, taken_at, latitude, longitude)
        SELECT video_path, 'video',
               MAX(width), MAX(height), MIN(taken_at), MAX(latitude), MAX(longitude)
        FROM photos
        WHERE video_path IS NOT NULL
        AND file_path NOT LIKE '%__pets'
        GROUP BY video_path
    """)
    conn.commit()
    video_sources = cur.rowcount
    print(f"  Migrated {video_sources} video sources")

    # ── Step 3c: Create sources for pet-only photos (no matching real photo) ──
    print("Step 3c: Creating sources for pet-only photos...")
    cur = conn.execute("""
        INSERT OR IGNORE INTO sources (file_path, type, width, height, taken_at, latitude, longitude)
        SELECT REPLACE(file_path, '__pets', ''), 'photo', width, height, taken_at, latitude, longitude
        FROM photos
        WHERE file_path LIKE '%__pets'
        AND REPLACE(file_path, '__pets', '') NOT IN (SELECT file_path FROM sources)
    """)
    conn.commit()
    pet_only_sources = cur.rowcount
    print(f"  Created {pet_only_sources} sources for pet-only photos")

    # ── Step 4: Populate scans ──
    print("\nStep 4: Populating scans...")

    # Photo human scans
    cur = conn.execute("""
        INSERT INTO scans (source_id, scan_type, scanned_at)
        SELECT s.id, 'human', p.scanned_at
        FROM photos p
        JOIN sources s ON s.file_path = p.file_path AND s.type = 'photo'
        WHERE p.video_path IS NULL
        AND p.file_path NOT LIKE '%__pets'
        AND p.file_path NOT LIKE '__nofaces_%'
    """)
    conn.commit()
    human_photo_scans = cur.rowcount

    # Video human scans
    cur = conn.execute("""
        INSERT INTO scans (source_id, scan_type, scanned_at)
        SELECT s.id, 'human', MIN(p.scanned_at)
        FROM photos p
        JOIN sources s ON s.file_path = p.video_path AND s.type = 'video'
        WHERE p.video_path IS NOT NULL
        AND p.file_path NOT LIKE '%__pets'
        GROUP BY s.id
    """)
    conn.commit()
    human_video_scans = cur.rowcount

    # Pet scans
    cur = conn.execute("""
        INSERT INTO scans (source_id, scan_type, scanned_at)
        SELECT s.id, 'pet', p_pet.scanned_at
        FROM photos p_pet
        JOIN sources s ON s.file_path = REPLACE(p_pet.file_path, '__pets', '')
        WHERE p_pet.file_path LIKE '%__pets'
    """)
    conn.commit()
    pet_scans = cur.rowcount

    print(f"  Human photo scans: {human_photo_scans}")
    print(f"  Human video scans: {human_video_scans}")
    print(f"  Pet scans: {pet_scans}")

    # ── Step 5: Migrate faces → findings ──
    print("\nStep 5: Migrating faces → findings...")

    # Photo findings (preserve IDs)
    cur = conn.execute("""
        INSERT INTO findings (id, source_id, bbox_x, bbox_y, bbox_w, bbox_h,
                              embedding, person_id, cluster_id, confidence, species,
                              detected_at, frame_path)
        SELECT f.id, f.photo_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
               f.embedding, f.person_id, f.cluster_id, f.confidence, f.species,
               f.detected_at, NULL
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        WHERE p.video_path IS NULL
        AND p.file_path NOT LIKE '%__pets'
    """)
    conn.commit()
    photo_findings = cur.rowcount

    # Video findings (new IDs, frame_path = old photo file_path)
    cur = conn.execute("""
        INSERT INTO findings (source_id, bbox_x, bbox_y, bbox_w, bbox_h,
                              embedding, person_id, cluster_id, confidence, species,
                              detected_at, frame_path)
        SELECT s.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
               f.embedding, f.person_id, f.cluster_id, f.confidence, f.species,
               f.detected_at, p.file_path
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        JOIN sources s ON s.file_path = p.video_path AND s.type = 'video'
        WHERE p.video_path IS NOT NULL
        AND p.file_path NOT LIKE '%__pets'
    """)
    conn.commit()
    video_findings = cur.rowcount

    # Pet findings (new IDs, merge into real source)
    cur = conn.execute("""
        INSERT INTO findings (source_id, bbox_x, bbox_y, bbox_w, bbox_h,
                              embedding, person_id, cluster_id, confidence, species,
                              detected_at, frame_path)
        SELECT s.id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h,
               f.embedding, f.person_id, f.cluster_id, f.confidence, f.species,
               f.detected_at, NULL
        FROM faces f
        JOIN photos p ON p.id = f.photo_id
        JOIN sources s ON s.file_path = REPLACE(p.file_path, '__pets', '')
        WHERE p.file_path LIKE '%__pets'
    """)
    conn.commit()
    pet_findings = cur.rowcount

    total_findings = photo_findings + video_findings + pet_findings
    print(f"  Photo findings: {photo_findings}")
    print(f"  Video findings: {video_findings}")
    print(f"  Pet findings: {pet_findings}")
    print(f"  Total: {total_findings}")

    # ── Step 6: Migrate dismissed_faces ──
    print("\nStep 6: Migrating dismissed faces...")
    # Photo dismissed (IDs preserved)
    cur = conn.execute("""
        INSERT INTO dismissed_findings (finding_id)
        SELECT df.face_id
        FROM dismissed_faces df
        JOIN findings fi ON fi.id = df.face_id
    """)
    conn.commit()
    dismissed_preserved = cur.rowcount

    # Video/pet dismissed need ID mapping — build via embedding + source match
    # For video findings: match by (source_id, bbox_x, bbox_y, bbox_w, bbox_h)
    cur = conn.execute("""
        INSERT OR IGNORE INTO dismissed_findings (finding_id)
        SELECT fi.id
        FROM dismissed_faces df
        JOIN faces f ON f.id = df.face_id
        JOIN photos p ON p.id = f.photo_id
        JOIN sources s ON (
            (s.file_path = p.video_path AND s.type = 'video')
            OR (s.file_path = REPLACE(p.file_path, '__pets', '') AND p.file_path LIKE '%__pets')
        )
        JOIN findings fi ON fi.source_id = s.id
            AND fi.bbox_x = f.bbox_x AND fi.bbox_y = f.bbox_y
            AND fi.bbox_w = f.bbox_w AND fi.bbox_h = f.bbox_h
            AND fi.species = f.species
        WHERE df.face_id NOT IN (SELECT finding_id FROM dismissed_findings)
    """)
    conn.commit()
    dismissed_remapped = cur.rowcount
    print(f"  Preserved: {dismissed_preserved}, Remapped: {dismissed_remapped}")

    # ── Step 7: Fix video frame paths ──
    print("\nStep 7: Fixing video frame paths...")
    cur = conn.execute("""
        UPDATE findings
        SET frame_path = SUBSTR(frame_path, LENGTH('face_recog/data/') + 1)
        WHERE frame_path LIKE 'face_recog/data/%'
    """)
    conn.commit()
    print(f"  Fixed {cur.rowcount} frame paths")

    # ── Step 8: Drop old tables ──
    print("\nStep 8: Dropping old tables...")
    conn.executescript("""
        DROP TABLE IF EXISTS dismissed_faces;
        DROP TABLE IF EXISTS faces;
        DROP TABLE IF EXISTS photos;
    """)

    # ── Verification ──
    print("\n── Post-migration stats ──")
    source_count = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
    finding_count = conn.execute("SELECT COUNT(*) FROM findings").fetchone()[0]
    scan_count = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    dismissed_new = conn.execute("SELECT COUNT(*) FROM dismissed_findings").fetchone()[0]

    print(f"  Sources: {source_count}")
    print(f"  Findings: {finding_count} (was {face_count} faces)")
    print(f"  Scans: {scan_count}")
    print(f"  Dismissed: {dismissed_new} (was {dismissed_count})")

    by_type = conn.execute("SELECT type, COUNT(*) FROM sources GROUP BY type").fetchall()
    print("\n  Sources by type:")
    for t, c in by_type:
        print(f"    {t}: {c}")

    by_scan = conn.execute("SELECT scan_type, COUNT(*) FROM scans GROUP BY scan_type").fetchall()
    print("\n  Scans by type:")
    for t, c in by_scan:
        print(f"    {t}: {c}")

    abs_paths = conn.execute("SELECT COUNT(*) FROM sources WHERE file_path LIKE '/%'").fetchone()[0]
    pets_paths = conn.execute(
        "SELECT COUNT(*) FROM sources WHERE file_path LIKE '%__pets'"
    ).fetchone()[0]
    print(f"\n  Absolute paths remaining: {abs_paths}")
    print(f"  __pets paths remaining: {pets_paths}")

    if finding_count != face_count:
        print(f"\n  WARNING: finding count ({finding_count}) != face count ({face_count})")
        lost = face_count - finding_count
        print(f"  {lost} faces were not migrated (likely on __nofaces_ or orphaned rows)")

    conn.execute("PRAGMA foreign_keys=ON")
    conn.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <path-to-faces.db> <photos-base-dir>")
        sys.exit(1)
    db_path = Path(sys.argv[1])
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)
    base_dir = sys.argv[2]
    migrate(db_path, base_dir)
