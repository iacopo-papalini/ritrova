"""Collapse duplicate findings across scan_types.

A duplicate group is findings on the same source with the same species whose
bboxes differ by less than 10% of `max(w_a, w_b)` on each of (x, y, w, h).
Within a group, the winner is the one with:

  1. Highest "curation weight":
        - has person_id AND is dismissed: 3
        - has person_id only:             2
        - is dismissed only:              1
        - neither:                        0
     — preserves manual assignments and dismissals over raw detection.
  2. Highest confidence among equals on (1).
  3. Lowest id as final tiebreak (deterministic).

Losers are deleted. Foreign-key cascades clean up `dismissed_findings`.

Run:
    uv run python scripts/dedup_findings.py --dry-run
    uv run python scripts/dedup_findings.py --apply
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections import defaultdict

from dotenv import load_dotenv


def find_groups(
    rows: list[sqlite3.Row],
) -> list[list[sqlite3.Row]]:
    """Greedy ±10% bbox grouping within one (source, species) partition."""
    groups: list[list[sqlite3.Row]] = []
    for r in rows:
        placed = False
        for g in groups:
            rep = g[0]
            max_w = max(r["bbox_w"], rep["bbox_w"])
            max_h = max(r["bbox_h"], rep["bbox_h"])
            if (
                abs(r["bbox_x"] - rep["bbox_x"]) < 0.1 * max_w
                and abs(r["bbox_y"] - rep["bbox_y"]) < 0.1 * max_h
                and abs(r["bbox_w"] - rep["bbox_w"]) < 0.1 * max_w
                and abs(r["bbox_h"] - rep["bbox_h"]) < 0.1 * max_h
            ):
                g.append(r)
                placed = True
                break
        if not placed:
            groups.append([r])
    return groups


def curation_weight(row: sqlite3.Row, dismissed_ids: set[int]) -> int:
    has_named = row["person_id"] is not None
    is_dismissed = row["id"] in dismissed_ids
    return (2 if has_named else 0) + (1 if is_dismissed else 0)


def pick_winner(group: list[sqlite3.Row], dismissed_ids: set[int]) -> sqlite3.Row:
    def key(r: sqlite3.Row) -> tuple[int, float, int]:
        # Higher curation weight + higher confidence wins; lowest id breaks ties
        # deterministically. Negate id so "higher first" returns the lowest.
        return (curation_weight(r, dismissed_ids), r["confidence"], -r["id"])

    return max(group, key=key)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Preview only.")
    parser.add_argument("--apply", action="store_true", help="Actually delete losers.")
    args = parser.parse_args()
    if args.dry_run == args.apply:
        raise SystemExit("Pass exactly one of --dry-run / --apply.")

    load_dotenv()
    db_path = os.environ.get("FACE_DB", "./data/faces.db")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    # Pull every finding with the columns we need. At ~110K rows this is fine.
    rows = conn.execute(
        "SELECT id, source_id, species, bbox_x, bbox_y, bbox_w, bbox_h, "
        "       confidence, person_id "
        "FROM findings ORDER BY source_id, species, id"
    ).fetchall()
    dismissed_ids: set[int] = {
        r["finding_id"] for r in conn.execute("SELECT finding_id FROM dismissed_findings")
    }
    print(f"Total findings: {len(rows)}   dismissed: {len(dismissed_ids)}")

    buckets: dict[tuple[int, str], list[sqlite3.Row]] = defaultdict(list)
    for r in rows:
        buckets[(r["source_id"], r["species"])].append(r)

    losers: list[int] = []
    duplicate_groups = 0
    kept_named_from_unnamed_duplicate = 0
    kept_dismissed_from_nondismissed = 0
    for bucket_rows in buckets.values():
        for group in find_groups(bucket_rows):
            if len(group) < 2:
                continue
            duplicate_groups += 1
            winner = pick_winner(group, dismissed_ids)
            for r in group:
                if r["id"] == winner["id"]:
                    continue
                losers.append(r["id"])
                if winner["person_id"] is not None and r["person_id"] is None:
                    kept_named_from_unnamed_duplicate += 1
                if (winner["id"] in dismissed_ids) and (r["id"] not in dismissed_ids):
                    kept_dismissed_from_nondismissed += 1

    print(f"Duplicate groups (2+ members): {duplicate_groups}")
    print(f"Findings marked for deletion: {len(losers)}")
    print(
        f"  curation wins: kept a named finding over an unnamed dupe: {kept_named_from_unnamed_duplicate}"
    )
    print(
        f"  curation wins: kept a dismissed over a non-dismissed dupe: {kept_dismissed_from_nondismissed}"
    )

    if args.dry_run:
        print("\nDry run — nothing deleted. Run with --apply to execute.")
        return

    # Apply in one transaction
    conn.execute("BEGIN")
    cur = conn.cursor()
    cur.executemany("DELETE FROM findings WHERE id = ?", [(fid,) for fid in losers])
    deleted = cur.rowcount
    conn.commit()
    print(f"\nDeleted {deleted} findings (cascades dismissed_findings by FK).")


if __name__ == "__main__":
    main()
