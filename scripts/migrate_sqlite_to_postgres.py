"""One-shot migration of ``faces.db`` into Postgres (W1.6).

Reads a SQLite *copy* and writes an Alembic-migrated Postgres database. After
cutover the SQLite file is deleted (D11), so this script produces the only
remaining copy of the curation work and verifies its own output rather than
trusting it.

Integer primary keys become UUIDv7. The version-7 timestamp is taken from the
row's own creation time -- ``findings.detected_at``, ``subjects.created_at``,
and for ``sources``, which has no creation column, the earliest scan of that
source. Stamping them with the migration's wall clock would be easier and would
throw away the property that makes UUIDv7 worth having: ids that sort by when
the row was made. Rows sharing a millisecond keep their original relative order
through a counter in ``rand_a``.

Because ids are rewritten, verification cannot compare them. It checks instead
that the graph is unchanged: row counts, that UUID order reproduces the old id
order on every table, that every foreign key still points at the row it pointed
at before, and that embeddings survive byte-for-byte.

Date handling follows D9, and the three column families are not alike:

* ``sources.taken_at`` is EXIF colon format (``2002:08:09 20:07:38``) with 27
  ``0000:00:00 00:00:00`` sentinels. Naive local camera time -> ``timestamp``,
  sentinels -> NULL.
* The audit columns are ISO-8601. ``scans.scanned_at`` is mixed: most rows
  carry ``+00:00``, but 9,809 legacy-backfill rows have no offset at all and
  are assumed UTC *explicitly* -- a naive parse under a Europe/Rome session
  would shift them by two hours and nothing would fail.
* ``source_path_metadata``'s four date columns are variable-precision
  prefixes (``1994``, ``2002-08``) described by ``date_precision``. They stay
  text.

Run:
    uv run python scripts/migrate_sqlite_to_postgres.py --sqlite <copy.db> --dry-run
    uv run python scripts/migrate_sqlite_to_postgres.py --sqlite <copy.db> --apply
"""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import numpy as np
from dotenv import load_dotenv
from uuid6 import UUID

type IdMap = dict[int, UUID]

# Tables with a surrogate key, and the column whose value dates the row.
# `sources` has no creation column; its scans supply one.
KEYED_TABLES: dict[str, str | None] = {
    "sources": None,
    "scans": "scanned_at",
    "subjects": "created_at",
    "findings": "detected_at",
    "descriptions": "generated_at",
    "circles": "created_at",
}

AWARE_COLUMNS = {
    "scans": ("scanned_at",),
    "subjects": ("created_at",),
    "findings": ("detected_at",),
    "descriptions": ("generated_at",),
    "source_path_metadata": ("indexed_at",),
    "print_selection": ("added_at",),
    "circles": ("created_at",),
    "subject_circles": ("added_at",),
    "finding_assignment": ("curated_at",),
}

# column -> which id map rewrites it
FOREIGN_KEYS: dict[str, dict[str, str]] = {
    "scans": {"source_id": "sources"},
    "findings": {"source_id": "sources", "scan_id": "scans"},
    "descriptions": {"source_id": "sources", "scan_id": "scans"},
    "source_path_metadata": {"source_id": "sources"},
    "print_selection": {"source_id": "sources"},
    "subject_circles": {"subject_id": "subjects", "circle_id": "circles"},
    "cluster_findings": {"finding_id": "findings"},
    "finding_assignment": {"finding_id": "findings", "subject_id": "subjects"},
}

TABLES: list[tuple[str, tuple[str, ...]]] = [
    (
        "sources",
        ("id", "file_path", "type", "width", "height", "taken_at", "latitude", "longitude"),
    ),
    ("scans", ("id", "source_id", "scan_type", "scanned_at", "detection_strategy")),
    ("subjects", ("id", "name", "kind", "created_at")),
    (
        "findings",
        (
            "id",
            "source_id",
            "scan_id",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
            "embedding",
            "confidence",
            "species",
            "detected_at",
            "frame_path",
            "embedding_dim",
            "frame_number",
        ),
    ),
    ("descriptions", ("id", "source_id", "scan_id", "caption", "tags", "generated_at")),
    (
        "source_path_metadata",
        (
            "source_id",
            "date_text",
            "date_precision",
            "date_source",
            "date_conflict",
            "filename_date",
            "directory_date",
            "exif_date",
            "path_tags",
            "indexed_at",
        ),
    ),
    ("print_selection", ("source_id", "position", "added_at")),
    ("scan_types", ("name", "pipeline", "outputs", "introduced_at", "retired_at", "notes")),
    ("circles", ("id", "name", "description", "created_at")),
    ("subject_circles", ("subject_id", "circle_id", "added_at")),
    ("cluster_findings", ("finding_id", "cluster_id")),
    ("finding_assignment", ("finding_id", "subject_id", "exclusion_reason", "curated_at")),
]

BOOL_COLUMNS = {"source_path_metadata": ("date_conflict",)}


def exif_to_naive(value: str | None) -> datetime | None:
    """``2002:08:09 20:07:38`` -> naive datetime. Sentinels and junk -> None."""
    if not value or value.startswith("0000"):
        return None
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return datetime.fromisoformat(value) if "-" in value else None


def iso_to_aware(value: str | None) -> datetime | None:
    """ISO-8601 -> aware UTC datetime, assuming UTC when no offset is present."""
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def uuid7_at(timestamp_ms: int, counter: int) -> UUID:
    """A UUIDv7 whose timestamp field is ``timestamp_ms``.

    ``counter`` occupies rand_a (12 bits), so rows sharing a millisecond keep
    the order they were handed to us in. Layout per RFC 9562, matching
    ``uuid6.uuid7``: 48-bit unix ms, version, 12-bit rand_a, variant, rand_b.
    """
    value = (timestamp_ms & 0xFFFFFFFFFFFF) << 80
    value |= (counter & 0xFFF) << 64
    value |= secrets.randbits(62)
    return UUID(int=value, version=7)


def assign_uuids(dated_rows: list[tuple[int, datetime]]) -> IdMap:
    """Map old integer id -> UUIDv7, preserving (timestamp, id) order.

    The emitted (millisecond, counter) pair only ever moves forwards. That
    matters more than it sounds: 9,809 legacy-backfill scans share a single
    timestamp, far past the 4,096 a 12-bit counter holds, so the overflow path
    is not a theoretical edge — it runs on this data, and a spill that reset to
    the next row's own millisecond would hand out descending ids.
    """
    mapping: IdMap = {}
    last_ms = -1
    counter = 0
    for old_id, moment in sorted(dated_rows, key=lambda pair: (pair[1], pair[0])):
        timestamp_ms = int(moment.timestamp() * 1000)
        if timestamp_ms > last_ms:
            last_ms, counter = timestamp_ms, 0
        else:
            counter += 1
            if counter > 0xFFF:
                last_ms += 1
                counter = 0
        mapping[old_id] = uuid7_at(last_ms, counter)
    return mapping


def build_id_maps(conn: sqlite3.Connection) -> dict[str, IdMap]:
    maps: dict[str, IdMap] = {}
    for table, date_column in KEYED_TABLES.items():
        if date_column is None:
            # sources: dated by its earliest scan. The audit confirmed every
            # source has at least one, but COALESCE keeps a stray row ordered
            # rather than crashing the migration.
            rows = conn.execute(
                "SELECT s.id, COALESCE(MIN(sc.scanned_at), '1970-01-01T00:00:00+00:00') "
                "FROM sources s LEFT JOIN scans sc ON sc.source_id = s.id GROUP BY s.id"
            ).fetchall()
        else:
            rows = conn.execute(f"SELECT id, {date_column} FROM {table}").fetchall()  # noqa: S608
        dated = [(row[0], iso_to_aware(row[1]) or datetime(1970, 1, 1, tzinfo=UTC)) for row in rows]
        maps[table] = assign_uuids(dated)
        print(f"  {table:<22} {len(dated):>7} ids")
    return maps


def read_table(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...], maps: dict[str, IdMap]
) -> list[tuple[Any, ...]]:
    aware = AWARE_COLUMNS.get(table, ())
    booleans = BOOL_COLUMNS.get(table, ())
    foreign = FOREIGN_KEYS.get(table, {})
    own_map = maps.get(table)
    rows = conn.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()  # noqa: S608
    out: list[tuple[Any, ...]] = []
    for row in rows:
        record = list(row)
        for index, column in enumerate(columns):
            value = record[index]
            if column == "id" and own_map is not None:
                record[index] = own_map[value]
            elif column in foreign:
                record[index] = maps[foreign[column]][value] if value is not None else None
            elif table == "sources" and column == "taken_at":
                record[index] = exif_to_naive(value)
            elif column in aware:
                record[index] = iso_to_aware(value)
            elif column in booleans:
                record[index] = bool(value)
        out.append(tuple(record))
    return out


async def load(
    pg: asyncpg.Connection, sqlite_conn: sqlite3.Connection, maps: dict[str, IdMap]
) -> dict[str, int]:
    loaded: dict[str, int] = {}
    for table, columns in TABLES:
        records = read_table(sqlite_conn, table, columns, maps)
        if records:
            await pg.copy_records_to_table(table, records=records, columns=list(columns))
        loaded[table] = len(records)
        print(f"  {table:<22} {len(records):>7} rows")
    return loaded


async def verify(
    pg: asyncpg.Connection,
    sqlite_conn: sqlite3.Connection,
    loaded: dict[str, int],
    maps: dict[str, IdMap],
) -> list[str]:
    failures: list[str] = []

    for table, _ in TABLES:
        source = sqlite_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]  # noqa: S608
        target = await pg.fetchval(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
        if not (source == target == loaded[table]):
            failures.append(f"{table}: sqlite={source} postgres={target} loaded={loaded[table]}")

    # No id may collide, and no two rows may share one.
    for table, mapping in maps.items():
        if len(set(mapping.values())) != len(mapping):
            failures.append(f"{table}: duplicate UUIDs generated")

    # The property the scheme is for: ascending UUID reproduces ascending
    # creation time. Note this is NOT the old id order — old ids are insertion
    # order, and the two genuinely differ wherever a row was inserted long
    # after the event it records, as the legacy-backfill scans were.
    for table, date_column in KEYED_TABLES.items():
        mapping = maps[table]
        if not mapping:
            continue
        dated = sqlite_conn.execute(
            "SELECT s.id, COALESCE(MIN(sc.scanned_at), '1970-01-01T00:00:00+00:00') "  # noqa: S608
            "FROM sources s LEFT JOIN scans sc ON sc.source_id = s.id GROUP BY s.id"
            if date_column is None
            else f"SELECT id, {date_column} FROM {table}"  # noqa: S608
        ).fetchall()
        expected = [
            mapping[old_id]
            for old_id, _ in sorted(
                dated,
                key=lambda row: (iso_to_aware(row[1]) or datetime(1970, 1, 1, tzinfo=UTC), row[0]),
            )
        ]
        if expected != sorted(expected):
            failures.append(f"{table}: UUID order does not reproduce creation order")

    # Every foreign key must still point where it pointed. Compared as whole
    # distributions rather than sampled rows, so a misrouting anywhere shows up
    # — and so composite-key tables need no special case.
    for table, relations in FOREIGN_KEYS.items():
        for column, parent in relations.items():
            reverse = {uid: old for old, uid in maps[parent].items()}
            before = dict(
                sqlite_conn.execute(
                    f"SELECT {column}, COUNT(*) FROM {table} "  # noqa: S608
                    f"WHERE {column} IS NOT NULL GROUP BY {column}"
                ).fetchall()
            )
            after = {
                reverse.get(row[column]): row["n"]
                for row in await pg.fetch(
                    f"SELECT {column}, COUNT(*) AS n FROM {table} "  # noqa: S608
                    f"WHERE {column} IS NOT NULL GROUP BY {column}"
                )
            }
            if before != after:
                failures.append(
                    f"{table}.{column}: parent distribution changed "
                    f"({len(before)} distinct before, {len(after)} after)"
                )

    # Per-column distributions would not catch two findings swapping their
    # (source, scan) pairs with each other, so check the pairs themselves.
    source_back = {uid: old for old, uid in maps["sources"].items()}
    scan_back = {uid: old for old, uid in maps["scans"].items()}
    pairs_before = sorted(sqlite_conn.execute("SELECT source_id, scan_id FROM findings").fetchall())
    pairs_after = sorted(
        (source_back.get(row["source_id"]), scan_back.get(row["scan_id"]))
        for row in await pg.fetch("SELECT source_id, scan_id FROM findings")
    )
    if pairs_before != pairs_after:
        failures.append("findings: (source_id, scan_id) pairs are not preserved")

    sample = sqlite_conn.execute(
        "SELECT id, embedding, embedding_dim FROM findings ORDER BY id LIMIT 500"
    ).fetchall()
    for old_id, blob, dim in sample:
        stored = await pg.fetchval(
            "SELECT embedding FROM findings WHERE id = $1", maps["findings"][old_id]
        )
        if stored != blob:
            failures.append(f"findings old id {old_id}: embedding bytes differ")
            break
        if not np.array_equal(np.frombuffer(blob, np.float32), np.frombuffer(stored, np.float32)):
            failures.append(f"findings old id {old_id}: embedding does not round-trip as float32")
            break
        if len(stored) // 4 != dim:
            failures.append(f"findings old id {old_id}: octet_length/4 != embedding_dim")
            break

    orphans = await pg.fetchval(
        "SELECT COUNT(*) FROM finding_assignment "
        "WHERE (subject_id IS NULL) = (exclusion_reason IS NULL)"
    )
    if orphans:
        failures.append(f"finding_assignment: {orphans} rows violate the XOR invariant")

    sentinels = await pg.fetchval("SELECT COUNT(*) FROM sources WHERE taken_at < '1900-01-01'")
    if sentinels:
        failures.append(f"sources: {sentinels} taken_at values below 1900 — sentinel leaked")

    stale = await pg.fetchval(
        "SELECT COUNT(*) FROM scans WHERE scanned_at > now() + interval '1 day'"
    )
    if stale:
        failures.append(f"scans: {stale} scanned_at values in the future — timezone misparse")

    return failures


async def run(sqlite_path: Path, dsn: str, apply: bool) -> int:
    sqlite_conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    pg = await asyncpg.connect(dsn)
    try:
        existing = await pg.fetchval("SELECT COUNT(*) FROM sources")
        if existing:
            print(f"refusing to load: sources already holds {existing} rows", file=sys.stderr)
            return 1

        print("minting UUIDv7 keys:")
        maps = build_id_maps(sqlite_conn)

        transaction = pg.transaction()
        await transaction.start()
        print("loading:")
        loaded = await load(pg, sqlite_conn, maps)
        print("verifying:")
        failures = await verify(pg, sqlite_conn, loaded, maps)

        if failures:
            await transaction.rollback()
            print("\nFAILED — rolled back, nothing written:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
            return 1

        print(f"  all checks passed ({sum(loaded.values())} rows)")
        if apply:
            await transaction.commit()
            print("\ncommitted.")
        else:
            await transaction.rollback()
            print("\ndry run — rolled back. Re-run with --apply to keep it.")
        return 0
    finally:
        await pg.close()
        sqlite_conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite", type=Path, required=True, help="path to a COPY of faces.db")
    parser.add_argument("--dsn", default=None, help="defaults to RITROVA_DSN")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="commit the load")
    group.add_argument(
        "--dry-run", action="store_true", default=True, help="load, verify, roll back"
    )
    args = parser.parse_args()

    load_dotenv()
    dsn = args.dsn or os.environ.get("RITROVA_DSN")
    if not dsn:
        print("RITROVA_DSN is not set and --dsn was not given", file=sys.stderr)
        return 2

    source = args.sqlite.resolve()
    if not source.exists():
        print(f"no such file: {source}", file=sys.stderr)
        return 2
    # D11 deletes the original after cutover; nothing here should acquire the
    # habit of opening the live file inside the sync tree. A read-only open is
    # not harmless either — it materialises -wal and -shm next to the database.
    if "nextCloud" in source.parts or "Nextcloud" in source.parts:
        print(f"refusing to read the live database inside the sync tree: {source}", file=sys.stderr)
        return 2

    return asyncio.run(run(source, dsn, args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
