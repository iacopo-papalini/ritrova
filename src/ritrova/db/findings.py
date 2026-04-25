"""Finding CRUD and embedding query mixin.

Apr 2026 refactor: curation state (subject assignment + exclusion) and
cluster membership now live on `finding_assignment` and `cluster_findings`.
All read queries LEFT JOIN both side tables and alias their columns back
onto the Finding dataclass's fields (`subject_id`, `cluster_id`,
`exclusion_reason`).
"""

from __future__ import annotations

import numpy as np

from ._base import _DBAccessor, _locked
from .models import Finding

# SELECT list that projects a Finding-shaped row by joining in the
# curation/cluster state. Use with `_FINDING_FROM` so the LEFT JOINs line
# up. Any query that needs to filter on subject/exclusion/cluster can add
# predicates on `fa.*` / `cf.*` directly.
_FINDING_COLUMNS = (
    "f.id, f.source_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h, "
    "f.embedding, f.confidence, f.detected_at, f.species, "
    "f.frame_path, f.embedding_dim, f.scan_id, f.frame_number, "
    "fa.subject_id AS subject_id, "
    "fa.exclusion_reason AS exclusion_reason, "
    "cf.cluster_id AS cluster_id"
)
_FINDING_FROM = (
    "FROM findings f "
    "LEFT JOIN finding_assignment fa ON fa.finding_id = f.id "
    "LEFT JOIN cluster_findings cf ON cf.finding_id = f.id"
)


def _row_to_finding(row: object) -> Finding | None:
    """Build a Finding from a row produced by a `SELECT _FINDING_COLUMNS`."""
    emb_bytes = row["embedding"]  # type: ignore[index]
    if emb_bytes is None:
        return None
    return Finding(
        id=row["id"],  # type: ignore[index]
        source_id=row["source_id"],  # type: ignore[index]
        bbox_x=row["bbox_x"],  # type: ignore[index]
        bbox_y=row["bbox_y"],  # type: ignore[index]
        bbox_w=row["bbox_w"],  # type: ignore[index]
        bbox_h=row["bbox_h"],  # type: ignore[index]
        embedding=np.frombuffer(emb_bytes, dtype=np.float32),
        subject_id=row["subject_id"],  # type: ignore[index]
        cluster_id=row["cluster_id"],  # type: ignore[index]
        confidence=row["confidence"],  # type: ignore[index]
        detected_at=row["detected_at"],  # type: ignore[index]
        species=row["species"],  # type: ignore[index]
        frame_path=row["frame_path"],  # type: ignore[index]
        embedding_dim=row["embedding_dim"],  # type: ignore[index]
        scan_id=row["scan_id"],  # type: ignore[index]
        frame_number=row["frame_number"],  # type: ignore[index]
        exclusion_reason=row["exclusion_reason"],  # type: ignore[index]
    )


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
        row = self.conn.execute(
            f"SELECT {_FINDING_COLUMNS} {_FINDING_FROM} WHERE f.id = ?",
            (finding_id,),
        ).fetchone()
        if not row:
            return None
        return _row_to_finding(row)

    @_locked
    def get_source_findings(self, source_id: int) -> list[Finding]:
        rows = self.conn.execute(
            f"SELECT {_FINDING_COLUMNS} {_FINDING_FROM} WHERE f.source_id = ?",
            (source_id,),
        ).fetchall()
        return [f for f in (_row_to_finding(r) for r in rows) if f is not None]

    @_locked
    def get_finding_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM findings").fetchone()
        return int(row[0]) if row else 0

    @_locked
    def add_manual_finding(
        self,
        source_id: int,
        bbox: tuple[int, int, int, int],
        embedding: np.ndarray,
        *,
        scan_id: int,
        species: str,
        confidence: float = 1.0,
    ) -> int:
        """Insert a single manually-drawn finding (FEAT-29) and return its id.

        Separate entry point from ``add_findings_batch`` because the caller is a
        user gesture — one row at a time, needs the row id back to wire up the
        undo payload, and never carries a ``frame_path`` (photo sources only
        for the MVP).
        """
        cur = self.conn.execute(
            "INSERT INTO findings (source_id, bbox_x, bbox_y, bbox_w, bbox_h, "
            "embedding, confidence, detected_at, species, frame_path, "
            "embedding_dim, scan_id, frame_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 0)",
            (
                source_id,
                *bbox,
                embedding.tobytes(),
                confidence,
                self._now(),
                species,
                len(embedding),
                scan_id,
            ),
        )
        self.conn.commit()
        assert cur.lastrowid is not None
        return cur.lastrowid

    @_locked
    def delete_finding(self, finding_id: int) -> None:
        """Delete a single finding row. ON DELETE CASCADE clears
        ``finding_assignment`` and ``cluster_findings`` entries.

        Used by the ``DeleteManualFindingPayload`` undo path — no
        no-op / not-found error handling, since the payload is only
        ever registered with a freshly-created finding id.
        """
        self.conn.execute("DELETE FROM findings WHERE id = ?", (finding_id,))
        self.conn.commit()

    @_locked
    def get_latest_scan_id(self, source_id: int, scan_type: str) -> int | None:
        """Return the newest ``scans.id`` for ``(source_id, scan_type)`` or None.

        Added for FEAT-29: manual findings attach to the most recent
        ``subjects`` scan on the source. ``scans`` has a UNIQUE
        (source_id, scan_type) constraint so "newest" is also "only",
        but we order by id for future-proofing.
        """
        row = self.conn.execute(
            "SELECT id FROM scans WHERE source_id = ? AND scan_type = ? ORDER BY id DESC LIMIT 1",
            (source_id, scan_type),
        ).fetchone()
        return int(row[0]) if row else None

    @_locked
    def get_all_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """Return all (finding_id, embedding) pairs for clustering.

        Excludes excluded findings (stranger OR not_a_face) — they shouldn't
        feed the clustering algorithm.
        """
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT f.id, f.embedding {_FINDING_FROM} "
            f"WHERE {clause} AND {dim_clause} "
            f"AND fa.exclusion_reason IS NULL",
            (*params, *dim_params),
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def get_unassigned_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """(finding_id, embedding) for findings with no subject assignment
        and no exclusion — i.e. the findings still in the curation queue."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT f.id, f.embedding {_FINDING_FROM} "
            f"WHERE fa.finding_id IS NULL AND {clause} AND {dim_clause}",
            (*params, *dim_params),
        ).fetchall()
        return [(row[0], np.frombuffer(row[1], dtype=np.float32)) for row in rows]

    @_locked
    def get_unclustered_embeddings(
        self, species: str = "human", embedding_dim: int | None = None
    ) -> list[tuple[int, np.ndarray]]:
        """(finding_id, embedding) for uncurated, unclustered findings."""
        clause, params = self.species_filter(species)
        dim_clause, dim_params = self._dim_filter(embedding_dim)
        rows = self.conn.execute(
            f"SELECT f.id, f.embedding {_FINDING_FROM} "
            f"WHERE fa.finding_id IS NULL AND cf.finding_id IS NULL "
            f"AND {clause} AND {dim_clause}",
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
            f"SELECT 1 {_FINDING_FROM} "
            f"WHERE fa.finding_id IS NULL AND cf.finding_id IS NULL AND {clause} "
            f"LIMIT 1",
            params,
        ).fetchone()
        return row is not None
