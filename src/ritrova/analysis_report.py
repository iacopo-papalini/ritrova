"""HTML reporting for completed analysis runs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from .db import FaceDB


@dataclass(frozen=True)
class AnalysisReport:
    """A written analysis report."""

    path: Path
    source_count: int
    finding_count: int
    named_subject_count: int


def default_analysis_report_path(db_path: Path) -> Path:
    """Return the default timestamped report path for an analysis run."""

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return db_path.parent / "reports" / f"analysis-{stamp}.html"


def write_analysis_report(
    db: FaceDB,
    scan_ids: list[int],
    output_path: Path | None = None,
) -> AnalysisReport | None:
    """Write an HTML report for the given scan ids.

    Returns ``None`` when there are no persisted scans to report.
    """

    if not scan_ids:
        return None

    path = output_path or default_analysis_report_path(db.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    placeholders = ",".join("?" for _ in scan_ids)
    scan_params = tuple(scan_ids)

    source_rows = [
        dict(row)
        for row in db.conn.execute(
            f"""
            SELECT
                sc.id AS scan_id,
                sc.scanned_at,
                s.file_path,
                s.type AS source_type,
                SUM(CASE WHEN f.species = 'human' THEN 1 ELSE 0 END) AS humans,
                SUM(CASE WHEN f.species = 'dog' THEN 1 ELSE 0 END) AS dogs,
                SUM(CASE WHEN f.species = 'cat' THEN 1 ELSE 0 END) AS cats,
                SUM(CASE WHEN f.species NOT IN ('human', 'dog', 'cat') THEN 1 ELSE 0 END) AS other,
                COUNT(f.id) AS total
            FROM scans sc
            JOIN sources s ON s.id = sc.source_id
            LEFT JOIN findings f ON f.scan_id = sc.id
            WHERE sc.id IN ({placeholders})
            GROUP BY sc.id, s.id
            ORDER BY s.type, s.file_path
            """,
            scan_params,
        ).fetchall()
    ]

    species_rows = [
        dict(row)
        for row in db.conn.execute(
            f"""
            SELECT s.type AS source_type, f.species, COUNT(*) AS findings
            FROM findings f
            JOIN sources s ON s.id = f.source_id
            WHERE f.scan_id IN ({placeholders})
            GROUP BY s.type, f.species
            ORDER BY s.type, f.species
            """,
            scan_params,
        ).fetchall()
    ]

    subject_rows = [
        dict(row)
        for row in db.conn.execute(
            f"""
            SELECT sub.id, sub.name, sub.kind, f.species, COUNT(*) AS findings
            FROM findings f
            JOIN finding_assignment fa ON fa.finding_id = f.id
            JOIN subjects sub ON sub.id = fa.subject_id
            WHERE f.scan_id IN ({placeholders})
            GROUP BY sub.id, sub.name, sub.kind, f.species
            ORDER BY findings DESC, sub.name
            """,
            scan_params,
        ).fetchall()
    ]

    excluded_rows = [
        dict(row)
        for row in db.conn.execute(
            f"""
            SELECT fa.exclusion_reason, f.species, COUNT(*) AS findings
            FROM findings f
            JOIN finding_assignment fa ON fa.finding_id = f.id
            WHERE f.scan_id IN ({placeholders}) AND fa.exclusion_reason IS NOT NULL
            GROUP BY fa.exclusion_reason, f.species
            ORDER BY findings DESC, fa.exclusion_reason, f.species
            """,
            scan_params,
        ).fetchall()
    ]

    cluster_rows = [
        dict(row)
        for row in db.conn.execute(
            f"""
            SELECT
                f.species,
                COUNT(*) AS findings,
                SUM(CASE WHEN cf.cluster_id IS NOT NULL THEN 1 ELSE 0 END) AS clustered,
                SUM(CASE WHEN cf.cluster_id IS NULL THEN 1 ELSE 0 END) AS unclustered
            FROM findings f
            LEFT JOIN cluster_findings cf ON cf.finding_id = f.id
            WHERE f.scan_id IN ({placeholders})
            GROUP BY f.species
            ORDER BY f.species
            """,
            scan_params,
        ).fetchall()
    ]

    likely_subject_rows = [
        dict(row)
        for row in db.conn.execute(
            f"""
            WITH new_findings AS (
                SELECT f.id, f.species, cf.cluster_id
                FROM findings f
                LEFT JOIN cluster_findings cf ON cf.finding_id = f.id
                WHERE f.scan_id IN ({placeholders})
            )
            SELECT
                sub.id,
                sub.name,
                sub.kind,
                nf.species,
                COUNT(DISTINCT nf.id) AS matching_findings
            FROM new_findings nf
            JOIN cluster_findings cf_new ON cf_new.finding_id = nf.id
            JOIN cluster_findings cf_peer ON cf_peer.cluster_id = cf_new.cluster_id
            JOIN finding_assignment fa
              ON fa.finding_id = cf_peer.finding_id
             AND fa.subject_id IS NOT NULL
            JOIN subjects sub ON sub.id = fa.subject_id
            WHERE cf_peer.finding_id != nf.id
            GROUP BY sub.id, sub.name, sub.kind, nf.species
            ORDER BY matching_findings DESC, sub.name
            """,
            scan_params,
        ).fetchall()
    ]

    source_counts = Counter(str(row["source_type"]) for row in source_rows)
    finding_count = sum(int(row["total"]) for row in source_rows)
    clustered_count = sum(int(row["clustered"]) for row in cluster_rows)
    species_counts = Counter(
        {str(row["species"]): int(row["findings"]) for row in _sum_species(species_rows)}
    )
    first_scan = min(str(row["scanned_at"]) for row in source_rows)
    last_scan = max(str(row["scanned_at"]) for row in source_rows)

    html = _render_html(
        source_rows=source_rows,
        species_rows=species_rows,
        subject_rows=subject_rows,
        excluded_rows=excluded_rows,
        cluster_rows=cluster_rows,
        likely_subject_rows=likely_subject_rows,
        source_counts=source_counts,
        species_counts=species_counts,
        finding_count=finding_count,
        clustered_count=clustered_count,
        first_scan=first_scan,
        last_scan=last_scan,
        scan_ids=scan_ids,
    )
    path.write_text(html, encoding="utf-8")
    return AnalysisReport(
        path=path,
        source_count=len(source_rows),
        finding_count=finding_count,
        named_subject_count=len(
            {int(row["id"]) for row in subject_rows}
            | {int(row["id"]) for row in likely_subject_rows}
        ),
    )


def _sum_species(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row["species"])] += int(row["findings"])
    return [{"species": species, "findings": count} for species, count in counts.items()]


def _render_html(
    *,
    source_rows: list[dict[str, Any]],
    species_rows: list[dict[str, Any]],
    subject_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    cluster_rows: list[dict[str, Any]],
    likely_subject_rows: list[dict[str, Any]],
    source_counts: Counter[str],
    species_counts: Counter[str],
    finding_count: int,
    clustered_count: int,
    first_scan: str,
    last_scan: str,
    scan_ids: list[int],
) -> str:
    source_summary = ", ".join(
        f"{count} {source_type}{'' if count == 1 else 's'}"
        for source_type, count in sorted(source_counts.items())
    )
    scan_range = f"{min(scan_ids)}-{max(scan_ids)}" if len(scan_ids) > 1 else str(scan_ids[0])
    direct_subjects = len({int(row["id"]) for row in subject_rows})
    likely_subjects = len({int(row["id"]) for row in likely_subject_rows})
    subject_note = (
        f"{direct_subjects} named subject(s) directly assigned in this run."
        if subject_rows
        else f"{likely_subjects} likely known subject(s) found through clustering."
        if likely_subject_rows
        else (
            "No named subjects are assigned yet. Analysis detected findings; "
            "cluster or review them next to attach names."
        )
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Ritrova Analysis Report</title>
  <style>
    :root {{ color-scheme: light; --bg: #f7f4ee; --text: #201f1c; --muted: #6d675e; --line: #d7d0c5; --panel: #fffdf8; }}
    body {{ margin: 0; font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 32px 24px 48px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    .muted {{ color: var(--muted); }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 24px 0; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ display: block; margin-top: 4px; font-size: 26px; font-weight: 700; }}
    .note {{ background: #fff7e5; border: 1px solid #e6c985; border-radius: 8px; padding: 12px 14px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eee7db; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    tr:last-child td {{ border-bottom: 0; }}
    .path {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
  </style>
</head>
<body>
<main>
  <h1>Ritrova Analysis Report</h1>
  <p class="muted">Scan ids {escape(scan_range)}, scanned {escape(first_scan)} to {escape(last_scan)}.</p>
  <section class="cards" aria-label="Run summary">
    <div class="card"><span class="label">Sources</span><span class="value">{len(source_rows)}</span><span class="muted">{escape(source_summary)}</span></div>
    <div class="card"><span class="label">Findings</span><span class="value">{finding_count}</span><span class="muted">Persisted detections</span></div>
    <div class="card"><span class="label">Clustered</span><span class="value">{clustered_count}</span><span class="muted">{finding_count - clustered_count} unclustered</span></div>
    <div class="card"><span class="label">People</span><span class="value">{species_counts.get("human", 0)}</span><span class="muted">Human detections</span></div>
    <div class="card"><span class="label">Pets</span><span class="value">{species_counts.get("dog", 0) + species_counts.get("cat", 0)}</span><span class="muted">{species_counts.get("dog", 0)} dogs, {species_counts.get("cat", 0)} cats</span></div>
  </section>
  <div class="note"><strong>Named subjects found:</strong> {escape(subject_note)}</div>
  {_table("Likely Known Subjects", ["Subject", "Kind", "Species", "Clustered findings"], _likely_subject_cells(likely_subject_rows))}
  {_table("Clustering", ["Species", "Findings", "Clustered", "Unclustered"], _cluster_cells(cluster_rows))}
  {_table("Assigned Subjects", ["Subject", "Kind", "Species", "Findings"], _subject_cells(subject_rows))}
  {_table("Species Breakdown", ["Source type", "Species", "Findings"], _species_cells(species_rows))}
  {_table("Excluded Findings", ["Reason", "Species", "Findings"], _excluded_cells(excluded_rows))}
  {_table("Sources", ["Source", "Type", "Humans", "Dogs", "Cats", "Other", "Total"], _source_cells(source_rows))}
</main>
</body>
</html>
"""


def _table(title: str, headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f"<h2>{escape(title)}</h2><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _source_cells(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            f'<span class="path">{escape(str(row["file_path"]))}</span>',
            escape(str(row["source_type"])),
            str(int(row["humans"])),
            str(int(row["dogs"])),
            str(int(row["cats"])),
            str(int(row["other"])),
            str(int(row["total"])),
        ]
        for row in rows
    ]


def _species_cells(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [escape(str(row["source_type"])), escape(str(row["species"])), str(int(row["findings"]))]
        for row in rows
    ]


def _subject_cells(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            escape(str(row["name"])),
            escape(str(row["kind"])),
            escape(str(row["species"])),
            str(int(row["findings"])),
        ]
        for row in rows
    ]


def _likely_subject_cells(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            escape(str(row["name"])),
            escape(str(row["kind"])),
            escape(str(row["species"])),
            str(int(row["matching_findings"])),
        ]
        for row in rows
    ]


def _cluster_cells(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            escape(str(row["species"])),
            str(int(row["findings"])),
            str(int(row["clustered"])),
            str(int(row["unclustered"])),
        ]
        for row in rows
    ]


def _excluded_cells(rows: list[dict[str, Any]]) -> list[list[str]]:
    return [
        [
            escape(str(row["exclusion_reason"])),
            escape(str(row["species"])),
            str(int(row["findings"])),
        ]
        for row in rows
    ]
