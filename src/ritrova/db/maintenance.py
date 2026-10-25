"""Health check, stats, and export mixin."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from ._base import _DBAccessor
from .models import OrphanReport


class MaintenanceMixin(_DBAccessor):
    async def find_orphans(self) -> OrphanReport:
        """Scan for dangling-child rows. Never mutates. See ``OrphanReport``."""
        findings_no_source = [
            r[0]
            for r in await self.fetch(
                "SELECT f.id FROM findings f "
                "LEFT JOIN sources s ON f.source_id = s.id "
                "WHERE s.id IS NULL"
            )
        ]
        findings_no_scan = [
            r[0]
            for r in await self.fetch(
                "SELECT f.id FROM findings f "
                "LEFT JOIN scans sc ON f.scan_id = sc.id "
                "WHERE sc.id IS NULL"
            )
        ]
        scans_no_source = [
            r[0]
            for r in await self.fetch(
                "SELECT sc.id FROM scans sc "
                "LEFT JOIN sources s ON sc.source_id = s.id "
                "WHERE s.id IS NULL"
            )
        ]
        return OrphanReport(
            findings_missing_source=findings_no_source,
            findings_missing_scan=findings_no_scan,
            scans_missing_source=scans_no_source,
        )

    async def delete_orphans(self, report: OrphanReport) -> None:
        """Delete every id in ``report``. Single transaction so a partial
        failure rolls back cleanly."""
        if report.total == 0:
            return

        async def _delete(table: str, column: str, ids: list[UUID]) -> None:
            if ids:
                await self.execute(f"DELETE FROM {table} WHERE {column} = ANY(?)", ids)  # noqa: S608

        async with self.transaction():
            await _delete("findings", "id", report.findings_missing_source)
            await _delete("findings", "id", report.findings_missing_scan)
            await _delete("scans", "id", report.scans_missing_source)

    async def get_stats(self, species: str = "human") -> dict[str, int]:
        clause, params = self.species_filter(species)
        return {
            "total_sources": await self._count("SELECT COUNT(*) FROM sources WHERE type = 'photo'"),
            "total_findings": await self._count(
                f"SELECT COUNT(*) FROM findings WHERE {clause}", *params
            ),
            "total_subjects": await self._count("SELECT COUNT(*) FROM subjects"),
            "named_findings": await self._count(
                f"SELECT COUNT(*) FROM findings f "
                f"JOIN finding_assignment fa ON fa.finding_id = f.id "
                f"WHERE fa.subject_id IS NOT NULL AND {clause}",
                *params,
            ),
            "unnamed_clusters": await self.get_unnamed_cluster_count(species),
            "unclustered_findings": await self._count(
                f"SELECT COUNT(*) FROM findings f "
                f"LEFT JOIN cluster_findings cf ON cf.finding_id = f.id "
                f"LEFT JOIN finding_assignment fa ON fa.finding_id = f.id "
                f"WHERE cf.finding_id IS NULL AND {clause} "
                f"AND (fa.exclusion_reason IS NULL OR fa.exclusion_reason <> 'not_a_face')",
                *params,
            ),
            "dismissed_findings": await self._count(
                "SELECT COUNT(*) FROM finding_assignment WHERE exclusion_reason = 'not_a_face'"
            ),
        }

    async def export_json(self) -> str:
        """Export as JSON: subject -> sources -> finding rectangles."""
        subjects = await self.get_subjects()
        data: dict[str, list[Any]] = {"subjects": [], "unnamed_findings": []}

        for subject in subjects:
            findings = await self.get_subject_findings(subject.id, limit=10000)
            sources_map: dict[str, list[dict[str, Any]]] = {}
            for finding in findings:
                source = await self.get_source(finding.source_id)
                if source:
                    sources_map.setdefault(source.file_path, []).append(
                        {
                            "bbox": [
                                finding.bbox_x,
                                finding.bbox_y,
                                finding.bbox_w,
                                finding.bbox_h,
                            ],
                            "confidence": finding.confidence,
                        }
                    )
            data["subjects"].append(
                {
                    "id": str(subject.id),
                    "name": subject.name,
                    "kind": subject.kind,
                    "sources": [
                        {"file_path": fp, "findings": rects} for fp, rects in sources_map.items()
                    ],
                }
            )

        unnamed = await self.fetch(
            "SELECT f.*, cf.cluster_id AS cluster_id_v FROM findings f "
            "LEFT JOIN finding_assignment fa ON fa.finding_id = f.id "
            "LEFT JOIN cluster_findings cf ON cf.finding_id = f.id "
            "WHERE fa.subject_id IS NULL"
        )
        for row in unnamed:
            source = await self.get_source(row["source_id"])
            if source:
                data["unnamed_findings"].append(
                    {
                        "source": source.file_path,
                        "bbox": [
                            row["bbox_x"],
                            row["bbox_y"],
                            row["bbox_w"],
                            row["bbox_h"],
                        ],
                        "cluster_id": row["cluster_id_v"],
                    }
                )

        # UUID and datetime have no JSON form of their own.
        return json.dumps(data, indent=2, default=str)
