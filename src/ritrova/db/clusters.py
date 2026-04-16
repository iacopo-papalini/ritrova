"""Cluster query and mutation mixin."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._base import _DBAccessor, _locked
from .models import Finding


class ClusterMixin(_DBAccessor):
    @_locked
    def update_cluster_ids(self, finding_cluster_map: dict[int, int]) -> None:
        """Update cluster_id for multiple findings."""
        self.conn.executemany(
            "UPDATE findings SET cluster_id = ? WHERE id = ?",
            [(cid, fid) for fid, cid in finding_cluster_map.items()],
        )
        self.conn.commit()

    @_locked
    def clear_clusters(self, species: str | None = None) -> None:
        """Reset cluster assignments (preserves subject assignments)."""
        if species is None:
            self.conn.execute("UPDATE findings SET cluster_id = NULL")
        else:
            clause, params = self.species_filter(species)
            self.conn.execute(f"UPDATE findings SET cluster_id = NULL WHERE {clause}", params)
        self.conn.commit()

    @_locked
    def get_cluster_ids(self) -> list[int]:
        """All distinct non-null cluster IDs."""
        rows = self.conn.execute(
            "SELECT DISTINCT cluster_id FROM findings WHERE cluster_id IS NOT NULL ORDER BY cluster_id"
        ).fetchall()
        return [row[0] for row in rows]

    @_locked
    def get_unnamed_clusters(self, species: str = "human") -> list[dict[str, Any]]:
        """Clusters with no subject assigned, ordered by size."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT cluster_id, COUNT(*) as face_count,
                   GROUP_CONCAT(id) as finding_ids
            FROM findings
            WHERE cluster_id IS NOT NULL AND person_id IS NULL AND {clause}
            GROUP BY cluster_id
            HAVING COUNT(*) >= 2
            ORDER BY face_count DESC
        """,
            params,
        ).fetchall()
        result = []
        for row in rows:
            finding_ids = [int(x) for x in row["finding_ids"].split(",")]
            result.append(
                {
                    "cluster_id": row["cluster_id"],
                    "face_count": row["face_count"],
                    "sample_face_ids": finding_ids[:12],
                }
            )
        return result

    @_locked
    def get_singleton_findings(
        self, species: str = "human", limit: int = 200, offset: int = 0
    ) -> list[Finding]:
        """Findings in clusters of size 1 or unclustered, unassigned, not dismissed."""
        clause, params = self.species_filter(species)
        rows = self.conn.execute(
            f"""
            SELECT f.* FROM findings f
            WHERE f.person_id IS NULL AND {clause}
            AND f.id NOT IN (SELECT finding_id FROM dismissed_findings)
            AND (
                f.cluster_id IS NULL
                OR f.cluster_id IN (
                    SELECT cluster_id FROM findings
                    WHERE cluster_id IS NOT NULL AND person_id IS NULL
                    GROUP BY cluster_id HAVING COUNT(*) = 1
                )
            )
            LIMIT ? OFFSET ?
        """,
            (*params, limit, offset),
        ).fetchall()
        result = []
        for row in rows:
            if row["embedding"] is None:
                continue
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Finding(**d))
        return result

    @_locked
    def get_singleton_count(self, species: str = "human") -> int:
        clause, params = self.species_filter(species)
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) FROM findings f
            WHERE f.person_id IS NULL AND {clause}
            AND f.id NOT IN (SELECT finding_id FROM dismissed_findings)
            AND (
                f.cluster_id IS NULL
                OR f.cluster_id IN (
                    SELECT cluster_id FROM findings
                    WHERE cluster_id IS NOT NULL AND person_id IS NULL
                    GROUP BY cluster_id HAVING COUNT(*) = 1
                )
            )
        """,
            params,
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_finding_count(self, cluster_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM findings WHERE cluster_id = ?",
            (cluster_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_cluster_findings(self, cluster_id: int, limit: int = 200) -> list[Finding]:
        rows = self.conn.execute(
            "SELECT * FROM findings WHERE cluster_id = ? LIMIT ?",
            (cluster_id, limit),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
            result.append(Finding(**d))
        return result

    @_locked
    def get_cluster_finding_ids(self, cluster_id: int) -> list[int]:
        """Return all finding IDs in a cluster."""
        rows = self.conn.execute(
            "SELECT id FROM findings WHERE cluster_id = ?", (cluster_id,)
        ).fetchall()
        return [r[0] for r in rows]

    @_locked
    def merge_clusters(self, source_id: int, target_id: int) -> None:
        """Move all findings from source cluster to target cluster."""
        self.conn.execute(
            "UPDATE findings SET cluster_id = ? WHERE cluster_id = ?",
            (target_id, source_id),
        )
        self.conn.commit()
