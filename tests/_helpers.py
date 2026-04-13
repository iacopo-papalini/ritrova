"""Test helpers shared across multiple test modules.

The most common need: insert findings without manually wiring up a scan row each
time. Production code wires scans explicitly (one per source × scan_type) so that
faulty scans can be pruned. Tests rarely care; this helper records-or-reuses the
correct scan row per (source, species) and then inserts the findings.
"""

from __future__ import annotations

import numpy as np

from ritrova.db import FaceDB


def add_findings(
    db: FaceDB,
    batch: list[tuple[int, tuple[int, int, int, int], np.ndarray, float]],
    *,
    species: str = "human",
    frame_path: str | None = None,
) -> None:
    """Find-or-record the appropriate scan per source, then insert the findings."""
    scan_type = "pet" if species in FaceDB.PET_SPECIES else "human"

    by_source: dict[int, list[tuple[int, tuple[int, int, int, int], np.ndarray, float]]] = {}
    for item in batch:
        by_source.setdefault(item[0], []).append(item)

    for source_id, items in by_source.items():
        scan_row = db.conn.execute(
            "SELECT id FROM scans WHERE source_id = ? AND scan_type = ?",
            (source_id, scan_type),
        ).fetchone()
        scan_id = int(scan_row["id"]) if scan_row else db.record_scan(source_id, scan_type)
        db.add_findings_batch(items, scan_id=scan_id, species=species, frame_path=frame_path)
