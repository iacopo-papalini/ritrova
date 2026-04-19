"""Cluster query and mutation mixin.

Apr 2026 refactor: cluster membership lives on `cluster_findings`, not
`findings.cluster_id`. All writes delegate to AssignmentMixin
(`set_cluster_memberships`, `remove_cluster_memberships`, `merge_cluster
_memberships`). Reads JOIN cluster_findings.
"""

from __future__ import annotations

from typing import Any

from ._base import _DBAccessor, _locked
from .findings import _FINDING_COLUMNS, _FINDING_FROM, _row_to_finding
from .models import Finding


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
        """Clusters where every finding is uncurated, ordered by total face
        area (biggest first).

        "Uncurated" = no finding_assignment row. Clusters containing even
        one named or excluded finding are filtered out — the user is
        mid-curation there or has already told us that blob is done.
        Sample faces are returned biggest-first so the thumbnail preview
        shows the cluster's largest crops.
        """
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
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
            ORDER BY total_area DESC
            """,
            params,
        ).fetchall()
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
        return result

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

    def merge_clusters(self, source_id: int, target_id: int) -> None:
        """Move all findings from source cluster to target cluster."""
        self.merge_cluster_memberships(source_id, target_id)  # type: ignore[attr-defined]
