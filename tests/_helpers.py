"""Test helpers shared across multiple test modules.

The most common need: insert findings without manually wiring up a scan row each
time. Production code wires scans explicitly (one per source × scan_type) so that
faulty scans can be pruned. Tests rarely care; this helper records-or-reuses the
correct scan row per (source, species) and then inserts the findings.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np

from ritrova.db import FaceDB

# A syntactically valid UUIDv7 that no row ever has: lets a request reach the
# handler and 404 instead of being rejected by path validation.
MISSING_ID = UUID("00000000-0000-7000-8000-000000000000")


async def add_findings(
    db: FaceDB,
    batch: list[tuple[UUID, tuple[int, int, int, int], np.ndarray, float]],
    *,
    species: str = "human",
    frame_path: str | None = None,
) -> None:
    """Find-or-record the appropriate scan per source, then insert the findings."""
    scan_type = "pet" if species in FaceDB.PET_SPECIES else "human"

    by_source: dict[UUID, list[tuple[UUID, tuple[int, int, int, int], np.ndarray, float]]] = {}
    for item in batch:
        by_source.setdefault(item[0], []).append(item)

    for source_id, items in by_source.items():
        scan_row = await db.fetchrow(
            "SELECT id FROM scans WHERE source_id = ? AND scan_type = ?", source_id, scan_type
        )
        scan_id = scan_row["id"] if scan_row else await db.record_scan(source_id, scan_type)
        await db.add_findings_batch(items, scan_id=scan_id, species=species, frame_path=frame_path)
