#!/usr/bin/env python3
"""Add detection_strategy to scans and embedding_dim to findings.

Run OFFLINE against the DB file. Idempotent.

Usage:
    python scripts/migrate_embedding_dim.py data/faces.db
"""
# ruff: noqa: T201

import sqlite3
import sys
from pathlib import Path


def col_exists(conn: sqlite3.Connection, table: str, col: str) -> bool:
    return any(c[1] == col for c in conn.execute(f"PRAGMA table_info({table})").fetchall())


def migrate(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")

    # scans.detection_strategy
    if not col_exists(conn, "scans", "detection_strategy"):
        print("Adding scans.detection_strategy...")
        conn.execute(
            "ALTER TABLE scans ADD COLUMN detection_strategy TEXT NOT NULL DEFAULT 'unknown'"
        )
        conn.execute("UPDATE scans SET detection_strategy = 'arcface_v1' WHERE scan_type = 'human'")
        conn.execute("UPDATE scans SET detection_strategy = 'siglip_v1' WHERE scan_type = 'pet'")
        conn.commit()
        print("  Done.")
    else:
        print("scans.detection_strategy already exists.")

    # findings.embedding_dim
    if not col_exists(conn, "findings", "embedding_dim"):
        print("Adding findings.embedding_dim...")
        conn.execute("ALTER TABLE findings ADD COLUMN embedding_dim INTEGER NOT NULL DEFAULT 0")
        conn.execute("UPDATE findings SET embedding_dim = LENGTH(embedding) / 4")
        conn.commit()
        print("  Done.")
    else:
        print("findings.embedding_dim already exists.")

    # Verify
    print("\n── Verification ──")
    for dim, cnt in conn.execute(
        "SELECT embedding_dim, COUNT(*) FROM findings GROUP BY embedding_dim"
    ).fetchall():
        print(f"  embedding_dim={dim}: {cnt} findings")
    for strat, cnt in conn.execute(
        "SELECT detection_strategy, COUNT(*) FROM scans GROUP BY detection_strategy"
    ).fetchall():
        print(f"  detection_strategy={strat!r}: {cnt} scans")

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
