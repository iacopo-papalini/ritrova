"""Curation mutations and together-query mixin."""

from __future__ import annotations

import numpy as np

from ._base import _DBAccessor, _locked
from .models import Finding, Source


class CurationMixin(_DBAccessor):
    @_locked
    def dismiss_findings(self, finding_ids: list[int]) -> None:
        """Mark findings as non-faces (statues, paintings, etc.)."""
        self.conn.executemany(
            "INSERT OR IGNORE INTO dismissed_findings (finding_id) VALUES (?)",
            [(fid,) for fid in finding_ids],
        )
        self.conn.executemany(
            "UPDATE findings SET cluster_id = NULL, person_id = NULL WHERE id = ?",
            [(fid,) for fid in finding_ids],
        )
        self.conn.commit()

    @_locked
    def unassign_finding(self, finding_id: int) -> None:
        """Remove subject assignment from a finding."""
        self.conn.execute("UPDATE findings SET person_id = NULL WHERE id = ?", (finding_id,))
        self.conn.commit()

    @_locked
    def exclude_findings(self, finding_ids: list[int], cluster_id: int) -> None:
        """Remove findings from a cluster (set cluster_id to NULL)."""
        placeholders = ",".join("?" * len(finding_ids))
        self.conn.execute(
            f"UPDATE findings SET cluster_id = NULL WHERE id IN ({placeholders}) "
            f"AND cluster_id = ?",
            (*finding_ids, cluster_id),
        )
        self.conn.commit()

    def _together_query(
        self, subject_ids: list[int], alone: bool = False
    ) -> tuple[str, tuple[int | str, ...]]:
        """Build the inner subquery for together (shared by get and count)."""
        placeholders = ",".join("?" * len(subject_ids))
        n = len(subject_ids)
        if alone:
            return (
                f"""SELECT source_id FROM findings
                    WHERE person_id IS NOT NULL
                    GROUP BY source_id
                    HAVING COUNT(DISTINCT CASE WHEN person_id IN ({placeholders}) THEN person_id END) = ?
                    AND COUNT(DISTINCT person_id) = ?""",
                (*subject_ids, n, n),
            )
        return (
            f"""SELECT source_id FROM findings
                WHERE person_id IN ({placeholders})
                GROUP BY source_id
                HAVING COUNT(DISTINCT person_id) = ?""",
            (*subject_ids, n),
        )

    def get_sources_with_all_subjects(
        self,
        subject_ids: list[int],
        limit: int = 0,
        offset: int = 0,
        alone: bool = False,
        source_type: str | None = None,
    ) -> list[Source]:
        """Find sources that contain ALL given subjects.

        If alone=True, exclude sources that also contain other named subjects.
        If source_type is given ("photo" / "video"), only return sources of that type.
        """
        if not subject_ids:
            return []
        inner, params = self._together_query(subject_ids, alone)
        type_filter = ""
        full_params: tuple[int | str, ...] = params
        if source_type:
            type_filter = " WHERE s.type = ?"
            full_params = (*full_params, source_type)
        pagination = ""
        if limit > 0:
            pagination = " LIMIT ? OFFSET ?"
            full_params = (*full_params, limit, offset)
        rows = self.conn.execute(
            f"""
            SELECT s.* FROM sources s
            JOIN ({inner}) matched ON matched.source_id = s.id
            {type_filter}
            ORDER BY s.file_path DESC{pagination}
            """,
            full_params,
        ).fetchall()
        return [Source(**dict(r)) for r in rows]

    def count_sources_with_all_subjects(
        self,
        subject_ids: list[int],
        alone: bool = False,
        source_type: str | None = None,
    ) -> int:
        """Count sources containing ALL given subjects, optionally filtered by type."""
        if not subject_ids:
            return 0
        inner, params = self._together_query(subject_ids, alone)
        if source_type:
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM sources s JOIN ({inner}) m ON m.source_id = s.id "
                f"WHERE s.type = ?",
                (*params, source_type),
            ).fetchone()
        else:
            row = self.conn.execute(f"SELECT COUNT(*) FROM ({inner})", params).fetchone()
        return int(row[0]) if row else 0

    @_locked
    def get_subject_sources(self, subject_id: int) -> list[Source]:
        """All unique sources containing a subject, newest first (by path)."""
        rows = self.conn.execute(
            """
            SELECT DISTINCT s.* FROM sources s
            JOIN findings f ON f.source_id = s.id
            WHERE f.person_id = ?
            ORDER BY s.file_path DESC
            """,
            (subject_id,),
        ).fetchall()
        return [Source(**dict(r)) for r in rows]

    @_locked
    def get_subject_sources_with_findings(
        self, subject_id: int, source_type: str
    ) -> list[tuple[Source, list[Finding]]]:
        """Sources of a given type containing the subject, each paired with the subject's findings on it.

        Used by the Videos tab on the subject detail page — groups findings by source so each video
        card can show a thumbnail (from one of the findings) and a count of appearances.
        """
        rows = self.conn.execute(
            """
            SELECT
                s.id        AS s_id,
                s.file_path AS s_file_path,
                s.type      AS s_type,
                s.width     AS s_width,
                s.height    AS s_height,
                s.taken_at  AS s_taken_at,
                s.latitude  AS s_latitude,
                s.longitude AS s_longitude,
                f.id            AS f_id,
                f.bbox_x        AS f_bbox_x,
                f.bbox_y        AS f_bbox_y,
                f.bbox_w        AS f_bbox_w,
                f.bbox_h        AS f_bbox_h,
                f.embedding     AS f_embedding,
                f.person_id     AS f_person_id,
                f.cluster_id    AS f_cluster_id,
                f.confidence    AS f_confidence,
                f.detected_at   AS f_detected_at,
                f.species       AS f_species,
                f.frame_path    AS f_frame_path,
                f.embedding_dim AS f_embedding_dim,
                f.scan_id       AS f_scan_id
            FROM sources s
            JOIN findings f ON f.source_id = s.id
            WHERE f.person_id = ? AND s.type = ?
            ORDER BY s.file_path DESC, f.id
            """,
            (subject_id, source_type),
        ).fetchall()

        grouped: dict[int, tuple[Source, list[Finding]]] = {}
        for row in rows:
            if row["f_embedding"] is None:
                continue
            source_id = row["s_id"]
            if source_id not in grouped:
                source = Source(
                    id=source_id,
                    file_path=row["s_file_path"],
                    type=row["s_type"],
                    width=row["s_width"],
                    height=row["s_height"],
                    taken_at=row["s_taken_at"],
                    latitude=row["s_latitude"],
                    longitude=row["s_longitude"],
                )
                grouped[source_id] = (source, [])

            finding = Finding(
                id=row["f_id"],
                source_id=source_id,
                bbox_x=row["f_bbox_x"],
                bbox_y=row["f_bbox_y"],
                bbox_w=row["f_bbox_w"],
                bbox_h=row["f_bbox_h"],
                embedding=np.frombuffer(row["f_embedding"], dtype=np.float32),
                person_id=row["f_person_id"],
                cluster_id=row["f_cluster_id"],
                confidence=row["f_confidence"],
                detected_at=row["f_detected_at"],
                species=row["f_species"] or "human",
                frame_path=row["f_frame_path"],
                embedding_dim=row["f_embedding_dim"] or 0,
                scan_id=row["f_scan_id"],
            )
            grouped[source_id][1].append(finding)

        return list(grouped.values())
