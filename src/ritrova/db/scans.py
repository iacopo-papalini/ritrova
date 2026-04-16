"""Scan record mixin."""

from __future__ import annotations

from typing import Any

from ._base import _DBAccessor, _locked


class ScanMixin(_DBAccessor):
    @_locked
    def is_scanned(self, file_path: str, scan_type: str) -> bool:
        """Check if a source has been scanned with the given scan type."""
        row = self.conn.execute(
            """SELECT 1 FROM scans sc
               JOIN sources s ON s.id = sc.source_id
               WHERE s.file_path = ? AND sc.scan_type = ?""",
            (file_path, scan_type),
        ).fetchone()
        return row is not None

    @_locked
    def record_scan(
        self, source_id: int, scan_type: str, detection_strategy: str = "unknown"
    ) -> int:
        """Insert a scan row and return its id.

        Callers MUST gate on `is_scanned()` first (UNIQUE on (source_id, scan_type)
        will raise otherwise). Rescans must `delete_scan()` the old one first.
        """
        cur = self.conn.execute(
            "INSERT INTO scans (source_id, scan_type, scanned_at, detection_strategy) "
            "VALUES (?, ?, ?, ?)",
            (source_id, scan_type, self._now(), detection_strategy),
        )
        self.conn.commit()
        scan_id = cur.lastrowid
        if scan_id is None:
            msg = "INSERT returned no lastrowid for scan"
            raise RuntimeError(msg)
        return scan_id

    @_locked
    def delete_scan(self, scan_id: int) -> dict[str, int]:
        """Delete a scan and (via ON DELETE CASCADE) all its findings.

        Returns counts {`deleted_findings`, `deleted_with_assignments`} for caller
        confirmation prompts. Raises ValueError if the scan doesn't exist.
        """
        scan = self.conn.execute("SELECT id FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if not scan:
            msg = f"No such scan: {scan_id}"
            raise ValueError(msg)
        deleted_findings = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id = ?", (scan_id,)
        ).fetchone()[0]
        deleted_with_assignments = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE scan_id = ? AND person_id IS NOT NULL",
            (scan_id,),
        ).fetchone()[0]
        self.conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        self.conn.commit()
        return {
            "deleted_findings": int(deleted_findings),
            "deleted_with_assignments": int(deleted_with_assignments),
        }

    @_locked
    def find_scans(
        self, scan_id: int | None = None, source_pattern: str | None = None
    ) -> list[dict[str, Any]]:
        """List scans matching the given filters (AND semantics; both optional).

        Returns dicts with keys: id, source_id, source_path, scan_type, scanned_at,
        detection_strategy, finding_count.

        `source_pattern` uses SQLite GLOB syntax (`*`, `?`) on `sources.file_path`.
        """
        clauses = []
        params: list[Any] = []
        if scan_id is not None:
            clauses.append("sc.id = ?")
            params.append(scan_id)
        if source_pattern is not None:
            clauses.append("s.file_path GLOB ?")
            params.append(source_pattern)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"""
            SELECT sc.id, sc.source_id, s.file_path AS source_path, sc.scan_type,
                   sc.scanned_at, sc.detection_strategy,
                   (SELECT COUNT(*) FROM findings f WHERE f.scan_id = sc.id) AS finding_count
            FROM scans sc
            JOIN sources s ON s.id = sc.source_id
            {where}
            ORDER BY sc.scanned_at DESC, sc.id
            """,
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]

    @_locked
    def delete_describe_scans(self, source_id: int) -> None:
        """Delete all 'describe' scans for a source (cascades to descriptions)."""
        self.conn.execute(
            "DELETE FROM scans WHERE source_id = ? AND scan_type = 'describe'",
            (source_id,),
        )
        self.conn.commit()
