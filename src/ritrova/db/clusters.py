"""Cluster query and mutation mixin.

Apr 2026 refactor: cluster membership lives on `cluster_findings`, not
`findings.cluster_id`. All writes delegate to AssignmentMixin
(`set_cluster_memberships`, `remove_cluster_memberships`, `merge_cluster
_memberships`). Reads JOIN cluster_findings.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import PurePosixPath
from typing import Any

from ._base import _DBAccessor, _locked
from .findings import _FINDING_COLUMNS, _FINDING_FROM, _row_to_finding
from .models import Finding

# YYYY-MM in any path component is the project's source-of-truth date
# (CLAUDE.md). Inlined here rather than importing from app/helpers.py so
# the DB layer keeps a clean dependency surface.
_PATH_DATE_RE = re.compile(r"(\d{4})-(\d{2})")


def _photo_month_index(file_path: str, taken_at: str | None) -> int | None:
    """Return ``year*12 + month`` for sort comparisons. Path date wins;
    ``taken_at`` (EXIF) is the fallback. Returns ``None`` when neither
    is parseable so the caller can sink such clusters to the bottom."""
    for part in reversed(PurePosixPath(file_path).parts):
        m = _PATH_DATE_RE.search(part)
        if m:
            return int(m.group(1)) * 12 + int(m.group(2))
    if taken_at and len(taken_at) >= 7:
        try:
            return int(taken_at[:4]) * 12 + int(taken_at[5:7])
        except ValueError:
            return None
    return None


class ClusterMixin(_DBAccessor):
    def update_cluster_ids(self, finding_cluster_map: dict[int, int]) -> None:
        """Bulk set cluster membership — thin alias over the AssignmentMixin
        batch helper. Kept on ClusterMixin for call-site readability."""
        self.set_cluster_memberships(finding_cluster_map)  # type: ignore[attr-defined]

    def clear_clusters(self, species: str | None = None) -> None:
        """Reset cluster assignments. Subject assignments are in a separate
        table and untouched."""
        if species is None:
            self.conn.execute("DELETE FROM cluster_findings")
            self.conn.commit()
        else:
            self.clear_species_cluster_memberships(species)  # type: ignore[attr-defined]

    @_locked
    def get_cluster_ids(self) -> list[int]:
        """All distinct cluster IDs that currently hold any findings."""
        rows = self.conn.execute(
            "SELECT DISTINCT cluster_id FROM cluster_findings ORDER BY cluster_id"
        ).fetchall()
        return [row[0] for row in rows]

    @_locked
    def get_unnamed_clusters(self, species: str = "human") -> list[dict[str, Any]]:
        """Clusters where every finding is uncurated, ordered by the
        average *photo age* of the cluster (newest first).

        "Uncurated" = no finding_assignment row. Clusters containing even
        one named or excluded finding are filtered out — the user is
        mid-curation there or has already told us that blob is done.
        Sample faces are returned biggest-first so the thumbnail preview
        shows the cluster's largest crops.

        Sort key per cluster: average ``year*12 + month`` over its
        sources, where the month comes from the path (``YYYY-MM`` in any
        directory component — the project's date-of-truth per CLAUDE.md)
        with ``sources.taken_at`` (EXIF) as a fallback. Clusters with no
        datable source sink to the bottom. Tiebreaker = total face area.
        """
        rows = self._unnamed_cluster_summary_rows(species)
        if not rows:
            return []
        cluster_ids = [r["cluster_id"] for r in rows]
        # Second query feeds the photo-age sort. Done separately so we can
        # keep the path / taken_at parsing in Python (no GROUP_CONCAT-of-
        # paths shenanigans, no SQL date parsing on a TEXT column).
        placeholders = ",".join("?" * len(cluster_ids))
        age_rows = self.conn.execute(
            f"""
            SELECT cf.cluster_id, s.file_path, s.taken_at
            FROM cluster_findings cf
            JOIN findings f ON f.id = cf.finding_id
            JOIN sources s ON s.id = f.source_id
            WHERE cf.cluster_id IN ({placeholders})
            """,
            tuple(cluster_ids),
        ).fetchall()
        per_cluster_months: dict[int, list[int]] = {}
        for ar in age_rows:
            month_idx = _photo_month_index(ar["file_path"], ar["taken_at"])
            if month_idx is not None:
                per_cluster_months.setdefault(ar["cluster_id"], []).append(month_idx)
        avg_month = {cid: sum(months) / len(months) for cid, months in per_cluster_months.items()}
        result = []
        for row in rows:
            pairs = [p.split(":") for p in row["finding_area_pairs"].split(",")]
            pairs.sort(key=lambda p: int(p[1]), reverse=True)
            finding_ids = [int(p[0]) for p in pairs]
            result.append(
                {
                    "cluster_id": row["cluster_id"],
                    "face_count": row["face_count"],
                    "total_area": int(row["total_area"] or 0),
                    "sample_face_ids": finding_ids[:12],
                }
            )
        # Newest first; clusters with no datable sources go to the bottom.
        # Tiebreaker = total face area (the previous default).
        result.sort(
            key=lambda r: (avg_month.get(r["cluster_id"], -1.0), r["total_area"]),
            reverse=True,
        )
        return result

    @_locked
    def get_unnamed_cluster_count(self, species: str = "human") -> int:
        """Count clusters shown by ``get_unnamed_clusters``.

        This is the reviewable unnamed-cluster count used by the dashboard
        and the cluster list; keep it tied to the same summary query so the
        UI cannot drift into different definitions.
        """
        return len(self._unnamed_cluster_summary_rows(species))

    def _unnamed_cluster_summary_rows(self, species: str) -> list[sqlite3.Row]:
        """Shared source of truth for reviewable unnamed clusters."""
        clause, params = self.species_filter(species)
        return self.conn.execute(
            f"""
            SELECT cf.cluster_id,
                   COUNT(*) AS face_count,
                   SUM(CAST(f.bbox_w AS INTEGER) * CAST(f.bbox_h AS INTEGER)) AS total_area,
                   GROUP_CONCAT(f.id || ':' || (f.bbox_w * f.bbox_h)) AS finding_area_pairs
            FROM cluster_findings cf
            JOIN findings f ON f.id = cf.finding_id
            LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
            WHERE fa.finding_id IS NULL AND {clause}
            GROUP BY cf.cluster_id
            HAVING COUNT(*) >= 2
            """,
            params,
        ).fetchall()

    @_locked
    def get_singleton_findings(
        self, species: str = "human", limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        """Findings that are uncurated and either unclustered or in a
        size-1 cluster. The singletons page uses this to show
        detector-false-positives and isolated faces that couldn't cluster."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT {_FINDING_COLUMNS}
            {_FINDING_FROM}
            WHERE fa.finding_id IS NULL AND {clause}
              AND (
                cf.finding_id IS NULL
                OR cf.cluster_id IN (
                    SELECT cf2.cluster_id FROM cluster_findings cf2
                    LEFT JOIN finding_assignment fa2 ON fa2.finding_id = cf2.finding_id
                    WHERE fa2.finding_id IS NULL
                    GROUP BY cf2.cluster_id HAVING COUNT(*) = 1
                )
              )
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()
        return [f for f in (_row_to_finding(r) for r in rows) if f is not None]

    @_locked
    def get_singleton_count(self, species: str = "human") -> int:
        clause, params = self.species_filter(species)
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM findings f
            LEFT JOIN finding_assignment fa ON fa.finding_id = f.id
            LEFT JOIN cluster_findings cf ON cf.finding_id = f.id
            WHERE fa.finding_id IS NULL AND {clause}
              AND (
                cf.finding_id IS NULL
                OR cf.cluster_id IN (
                    SELECT cf2.cluster_id FROM cluster_findings cf2
                    LEFT JOIN finding_assignment fa2 ON fa2.finding_id = cf2.finding_id
                    WHERE fa2.finding_id IS NULL
                    GROUP BY cf2.cluster_id HAVING COUNT(*) = 1
                )
              )
            """,
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_finding_count(self, cluster_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM cluster_findings WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_findings(self, cluster_id: int, limit: int = 200) -> list[Finding]:
        rows = self.conn.execute(
            f"""SELECT {_FINDING_COLUMNS}
                {_FINDING_FROM}
                WHERE cf.cluster_id = ?
                LIMIT ?""",
            (cluster_id, limit),
        ).fetchall()
        return [f for f in (_row_to_finding(r) for r in rows) if f is not None]

    @_locked
    def get_cluster_finding_ids(self, cluster_id: int) -> list[int]:
        rows = self.conn.execute(
            "SELECT finding_id FROM cluster_findings WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def get_cluster_face_stubs(
        self, cluster_id: int, limit: int = 200, offset: int = 0
    ) -> list[tuple[int, int]]:
        """Paginated ``(finding_id, source_id)`` for every finding in a cluster.

        Routes use this to render a face grid + link each tile back to its
        photo without materialising full ``Finding`` rows. Goes through
        ``_FINDING_FROM`` so curation-state changes can't leak around the
        JOIN convention.
        """
        rows = self.conn.execute(
            f"SELECT f.id, f.source_id {_FINDING_FROM} WHERE cf.cluster_id = ? LIMIT ? OFFSET ?",
            (cluster_id, limit, offset),
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def merge_clusters(self, source_id: int, target_id: int) -> None:
        """Move all findings from source cluster to target cluster."""
        self.merge_cluster_memberships(source_id, target_id)  # type: ignore[attr-defined]

    @_locked
    def split_cluster_findings(
        self, source_cluster_id: int, finding_ids: list[int]
    ) -> tuple[int, list[int]] | None:
        """Move selected findings from one cluster into a fresh cluster id.

        Returns ``(new_cluster_id, moved_finding_ids)``. Findings that are
        no longer in ``source_cluster_id`` are ignored so stale client
        selections cannot move unrelated rows.
        """
        if not finding_ids:
            return None
        placeholders = ",".join("?" * len(finding_ids))
        rows = self.conn.execute(
            f"""
            SELECT finding_id
            FROM cluster_findings
            WHERE cluster_id = ? AND finding_id IN ({placeholders})
            """,
            (source_cluster_id, *finding_ids),
        ).fetchall()
        moved_ids = [int(r[0]) for r in rows]
        if not moved_ids:
            return None
        row = self.conn.execute(
            "SELECT COALESCE(MAX(cluster_id), -1) + 1 FROM cluster_findings"
        ).fetchone()
        new_cluster_id = int(row[0]) if row else 0
        moved_placeholders = ",".join("?" * len(moved_ids))
        self.conn.execute(
            f"""
            UPDATE cluster_findings
            SET cluster_id = ?
            WHERE cluster_id = ? AND finding_id IN ({moved_placeholders})
            """,
            (new_cluster_id, source_cluster_id, *moved_ids),
        )
        self.conn.commit()
        return new_cluster_id, moved_ids
