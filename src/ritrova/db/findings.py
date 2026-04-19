"""Finding CRUD and embedding query mixin."""

from __future__ import annotations

import numpy as np

from ._base import _DBAccessor, _locked
from .models import Finding


class FindingMixin(_DBAccessor):
    @_locked
    def add_findings_batch(
        self,
        findings_data: list[tuple[int, tuple[int, int, int, int], np.ndarray, float]],
        *,
        scan_id: int,
        species: str = "human",
        frame_path: str | None = None,
        frame_number: int = 0,
    ) -> None:
        """Add multiple findings in a single transaction.

        `scan_id` is keyword-only and required: every finding must trace to the scan
        that produced it, so a faulty scan can be pruned without losing curation
        elsewhere on the same source.
        """
        now = self._now()
        self.conn.executemany(
            "INSERT INTO findings (source_id, bbox_x, bbox_y, bbox_w, bbox_h, "
            "embedding, confidence, detected_at, species, frame_path, "
            "embedding_dim, scan_id, frame_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    sid,
                    *bbox,
                    emb.tobytes(),
                    conf,
                    now,
                    species,
                    frame_path,
                    len(emb),
                    scan_id,
                    frame_number,
                )
                for sid, bbox, emb, conf in findings_data
            ],
        )
        self.conn.commit()

    @_locked
    def get_finding(self, finding_id: int) -> Finding | None:
        row = self.conn.execute("SELECT * FROM findings WHERE id = ?", (finding_id,)).fetchone()
        if not row or row["embedding"] is None:
            return None
        d = dict(row)
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
        return Finding(**d)

    @_locked
    def get_source_findings(self, source_id: int) -> list[Finding]:
        rows = self.conn.execute(
            "SELECT * FROM findings WHERE source_id = ?", (source_id,)
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
    def get_finding_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_all_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Return all (finding_id, embedding) pairs for clustering, excluding dismissed."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE {clause} AND {dim_clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            (*params, *dim_params),
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def get_unassigned_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Return (finding_id, embedding) for unassigned, non-dismissed findings only."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE person_id IS NULL AND {clause} AND {dim_clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            (*params, *dim_params),
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def get_unclustered_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Return (finding_id, embedding) for unclustered, unassigned, non-dismissed findings."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM findings "
            f"WHERE person_id IS NULL AND cluster_id IS NULL AND {clause} AND {dim_clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings)",
            (*params, *dim_params),
        ).fetchall()
        return [(r[0], np.frombuffer(r[1], dtype=np.float32)) for r in rows]

    @_locked
    def has_unclustered_findings(self, species: str = "human") -> bool:
        """Cheap existence check mirroring get_unclustered_embeddings's filter.

        Used by the subject-detail view to decide whether to render the
        "Show similar unclustered" link — full similarity scan is too slow
        to run per page-render, but a zero/non-zero gate is one row away.
        """
        clause, params = self.species_filter(species)
        row = self.conn.execute(
            f"SELECT 1 FROM findings "
            f"WHERE person_id IS NULL AND cluster_id IS NULL AND {clause} "
            f"AND id NOT IN (SELECT finding_id FROM dismissed_findings) "
            f"LIMIT 1",
            params,
        ).fetchone()
        return row is not None
