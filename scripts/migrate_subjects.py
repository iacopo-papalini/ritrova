#!/usr/bin/env python3
"""Migrate 'persons' table to 'subjects' with a 'kind' column.

Run OFFLINE against the DB file before deploying code changes.
Idempotent: safe to run more than once.

Usage:
    python scripts/migrate_subjects.py data/faces.db
"""
# ruff: noqa: T201

import sqlite3
import sys
from pathlib import Path


def check_table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def check_column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(c[1] == column for c in cols)


def migrate(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    already_renamed = check_table_exists(conn, "subjects")
    has_persons = check_table_exists(conn, "persons")

    if not already_renamed and not has_persons:
        print("ERROR: neither 'persons' nor 'subjects' table found")
        sys.exit(1)

    if already_renamed:
        print("Table already renamed to 'subjects'.")
    else:
        print("Renaming 'persons' -> 'subjects'...")
        conn.execute("ALTER TABLE persons RENAME TO subjects")
        conn.commit()

    if not check_column_exists(conn, "subjects", "kind"):
        print("Adding 'kind' column...")
        conn.execute("ALTER TABLE subjects ADD COLUMN kind TEXT NOT NULL DEFAULT 'person'")
        conn.commit()

        print("Backfilling kind='pet' from face species majority vote...")
        cur = conn.execute("""
            UPDATE subjects SET kind = 'pet'
            WHERE id IN (
                SELECT person_id FROM faces
                WHERE person_id IS NOT NULL
                GROUP BY person_id
                HAVING SUM(CASE WHEN species IN ('dog','cat','other_pet') THEN 1 ELSE 0 END) * 2
                       > COUNT(*)
            )
        """)
        conn.commit()
        print(f"  {cur.rowcount} subjects set to kind='pet'")
    else:
        print("Column 'kind' already exists.")

    # Unique index on (name, kind)
    existing_indexes = [r[1] for r in conn.execute("PRAGMA index_list(subjects)").fetchall()]
    if "idx_subjects_name_kind" not in existing_indexes:
        # Check for collisions first
        dupes = conn.execute("""
            SELECT name, kind, COUNT(*) as cnt
            FROM subjects GROUP BY name, kind HAVING cnt > 1
        """).fetchall()
        if dupes:
            print("WARNING: duplicate (name, kind) pairs found — cannot create unique index:")
            for name, kind, cnt in dupes:
                print(f"  name={name!r}, kind={kind!r}, count={cnt}")
            print("Resolve duplicates manually, then re-run.")
        else:
            print("Creating unique index on (name, kind)...")
            conn.execute("CREATE UNIQUE INDEX idx_subjects_name_kind ON subjects(name, kind)")
            conn.commit()
    else:
        print("Unique index already exists.")

    # ── Verification ──
    print("\n── Verification ──")
    total = conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
    pet_count = conn.execute("SELECT COUNT(*) FROM subjects WHERE kind = 'pet'").fetchone()[0]
    person_count = conn.execute("SELECT COUNT(*) FROM subjects WHERE kind = 'person'").fetchone()[0]
    named_faces = conn.execute("SELECT COUNT(*) FROM faces WHERE person_id IS NOT NULL").fetchone()[
        0
    ]

    print(f"  Total subjects: {total}")
    print(f"  Persons: {person_count}, Pets: {pet_count}")
    print(f"  Named faces: {named_faces}")

    # Show close-call backfills (40-60% split)
    close_calls = conn.execute("""
        SELECT s.id, s.name, s.kind,
               SUM(CASE WHEN f.species IN ('dog','cat','other_pet') THEN 1 ELSE 0 END) as pet_faces,
               SUM(CASE WHEN f.species = 'human' THEN 1 ELSE 0 END) as human_faces
        FROM subjects s
        JOIN faces f ON f.person_id = s.id
        GROUP BY s.id
        HAVING MIN(pet_faces, human_faces) * 1.0 / (pet_faces + human_faces) > 0.3
    """).fetchall()

    if close_calls:
        print("\n  Close-call backfills (mixed species, review manually):")
        for sid, name, kind, pet_f, human_f in close_calls:
            print(f"    id={sid} name={name!r} kind={kind} pet_faces={pet_f} human_faces={human_f}")

    conn.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path-to-faces.db>")
        sys.exit(1)
    db_path = Path(sys.argv[1])
    if not db_path.exists():
        print(f"ERROR: {db_path} not found")
        sys.exit(1)
    migrate(db_path)
