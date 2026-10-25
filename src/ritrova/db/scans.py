"""Scan record mixin."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from uuid6 import uuid7

from ._base import _DBAccessor, glob_to_like


class ScanMixin(_DBAccessor):
    async def is_scanned(self, file_path: str, scan_type: str) -> bool:
        """Check if a source has been scanned with the given scan type."""
        found = await self.fetchval(
            """SELECT 1 FROM scans sc
               JOIN sources s ON s.id = sc.source_id
               WHERE s.file_path = ? AND sc.scan_type = ?""",
            file_path,
            scan_type,
        )
        return found is not None

    async def record_scan(
        self, source_id: UUID, scan_type: str, detection_strategy: str = "unknown"
    ) -> UUID:
        """Insert a scan row and return its id.

        Callers MUST gate on `is_scanned()` first (UNIQUE on (source_id, scan_type)
        will raise otherwise). Rescans must `delete_scan()` the old one first.
        """
        scan_id = uuid7()
        await self.execute(
            "INSERT INTO scans (id, source_id, scan_type, scanned_at, detection_strategy) "
            "VALUES (?, ?, ?, ?, ?)",
            scan_id,
            source_id,
            scan_type,
            self._now(),
            detection_strategy,
        )
        return scan_id

    async def delete_scan(self, scan_id: UUID) -> dict[str, int]:
        """Delete a scan and (via ON DELETE CASCADE) all its findings.

        Returns counts {`deleted_findings`, `deleted_with_assignments`} for caller
        confirmation prompts. Raises ValueError if the scan doesn't exist.
        """
        scan = await self.fetchval("SELECT id FROM scans WHERE id = ?", scan_id)
        if not scan:
            msg = f"No such scan: {scan_id}"
            raise ValueError(msg)
        deleted_findings = await self._count(
            "SELECT COUNT(*) FROM findings WHERE scan_id = ?", scan_id
        )
        deleted_with_assignments = await self._count(
            "SELECT COUNT(*) FROM findings f "
            "JOIN finding_assignment fa ON fa.finding_id = f.id "
            "WHERE f.scan_id = ? AND fa.subject_id IS NOT NULL",
            scan_id,
        )
        await self.execute("DELETE FROM scans WHERE id = ?", scan_id)
        return {
            "deleted_findings": deleted_findings,
            "deleted_with_assignments": deleted_with_assignments,
        }

    async def find_scans(
        self, scan_id: UUID | None = None, source_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """List scans matching the given filters (AND semantics; both optional).

        Returns dicts with keys: id, source_id, source_path, scan_type, scanned_at,
        detection_strategy, finding_count.

        `source_pattern` uses GLOB syntax (`*`, `?`) on `sources.file_path`.
        """
        clauses = []
        params: list[Any] = []
        if scan_id is not None:
            clauses.append("sc.id = ?")
            params.append(scan_id)
        if source_pattern is not None:
            # Postgres has no GLOB; the pattern is translated rather than the
            # CLI flag changed, so `--source-pattern '2024/*'` keeps working.
            clauses.append(r"s.file_path LIKE ? ESCAPE '\'")
            params.append(glob_to_like(source_pattern))
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = await self.fetch(
            f"""
            SELECT sc.id, sc.source_id, s.file_path AS source_path, sc.scan_type,
                   sc.scanned_at, sc.detection_strategy,
                   (SELECT COUNT(*) FROM findings f WHERE f.scan_id = sc.id) AS finding_count
            FROM scans sc
            JOIN sources s ON s.id = sc.source_id
            {where}
            ORDER BY sc.scanned_at DESC, sc.id
            """,
            *params,
        )
        return [dict(r) for r in rows]

    async def delete_describe_scans(self, source_id: UUID) -> None:
        """Delete all 'describe' scans for a source (cascades to descriptions)."""
        await self.execute(
            "DELETE FROM scans WHERE source_id = ? AND scan_type = 'describe'",
            source_id,
        )
