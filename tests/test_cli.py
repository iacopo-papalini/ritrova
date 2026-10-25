"""Tests for the click CLI commands added with the scan↔finding linkage refactor.

Covers `scans list`, `scans prune`, and the single-source `analyse <path>`
path that replaced the retired `rescan` command. The detector models are not
exercised here — those paths are integration-tested via the scanner unit
tests; the CLI tests focus on argument handling, confirmation, and the
DB-side cascade.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import numpy as np
from click.testing import CliRunner, Result

from ritrova.cli import cli
from ritrova.db import FaceDB

from ._helpers import add_findings
from .conftest import TEST_DSN


async def _seed(db: FaceDB) -> dict[str, UUID]:
    """Seed two photo sources, each with one human scan + finding."""
    photo1 = await db.add_source("/2024/jan/a.jpg", width=100, height=100)
    photo2 = await db.add_source("/2025/feb/b.jpg", width=100, height=100)
    await add_findings(
        db,
        [
            (photo1, (0, 0, 10, 10), np.ones(512, dtype=np.float32) / 22.6, 0.9),
            (photo2, (0, 0, 10, 10), np.ones(512, dtype=np.float32) / 22.6, 0.9),
        ],
    )
    return {"photo1": photo1, "photo2": photo2}


async def _run(args: list[str], data_dir: Path, input_text: str | None = None) -> Result:
    """Invoke the CLI off the test's loop — each command drives its own
    ``asyncio.run``, which cannot nest inside a running loop."""
    runner = CliRunner()
    return await asyncio.to_thread(
        runner.invoke,
        cli,
        ["--dsn", TEST_DSN, "--data-dir", str(data_dir), *args],
        input=input_text,
    )


async def test_scans_list_outputs_rows(db: FaceDB, tmp_path: Path) -> None:
    _ = await _seed(db)
    result = await _run(["scans", "list"], tmp_path)
    assert result.exit_code == 0, result.output
    assert "/2024/jan/a.jpg" in result.output
    assert "/2025/feb/b.jpg" in result.output
    assert "2 scan(s)" in result.output


async def test_scans_list_filters_by_pattern(db: FaceDB, tmp_path: Path) -> None:
    _ = await _seed(db)
    result = await _run(["scans", "list", "--source-pattern", "*2024*"], tmp_path)
    assert result.exit_code == 0, result.output
    assert "/2024/jan/a.jpg" in result.output
    assert "/2025/feb/b.jpg" not in result.output


async def test_scans_prune_requires_filter(db: FaceDB, tmp_path: Path) -> None:
    _ = await _seed(db)
    result = await _run(["scans", "prune"], tmp_path)
    assert result.exit_code != 0
    assert "refusing to prune everything" in result.output


async def test_scans_prune_aborts_on_no(db: FaceDB, tmp_path: Path) -> None:
    _ = await _seed(db)
    result = await _run(["scans", "prune", "--source-pattern", "*"], tmp_path, input_text="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert len(await db.find_scans()) == 2  # nothing removed


async def test_scans_prune_by_id_with_yes(db: FaceDB, tmp_path: Path) -> None:
    _ = await _seed(db)
    scan_id = (await db.find_scans(source_pattern="*a.jpg"))[0]["id"]
    result = await _run(["scans", "prune", "--scan-id", str(scan_id), "-y"], tmp_path)
    assert result.exit_code == 0, result.output
    assert "Pruned 1 scan(s)" in result.output
    remaining = await db.find_scans()
    assert len(remaining) == 1
    assert "/2025/feb/b.jpg" in remaining[0]["source_path"]


async def test_scans_prune_by_pattern_intersection(db: FaceDB, tmp_path: Path) -> None:
    """Both filters → only the intersection is pruned."""
    _ = await _seed(db)
    a_scan_id = (await db.find_scans(source_pattern="*a.jpg"))[0]["id"]
    # Pattern matches /2025/* but scan_id points to /2024/a.jpg → no overlap.
    result = await _run(
        [
            "scans",
            "prune",
            "--scan-id",
            str(a_scan_id),
            "--source-pattern",
            "*2025*",
            "-y",
        ],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert "(no scans match)" in result.output
    assert len(await db.find_scans()) == 2


# ── `ritrova analyse <path>` — the single-source re-analysis path ────
# (replaces the retired `rescan` command)


async def _seed_subjects_scan(
    db: FaceDB, tmp_path: Path, *, with_assignment: bool
) -> tuple[Path, UUID, UUID]:
    """Set up a DB + on-disk file with one 'subjects' scan and one finding.

    Returns ``(photo_path, source_id, finding_id)``. When
    ``with_assignment`` is True, the finding is tagged as manually assigned
    (non-null ``finding_assignment.subject_id``) — this is what trips the
    safety prompt on ``analyse <path> --force``.
    """
    photo_path = tmp_path / "fake.jpg"
    photo_path.write_bytes(b"x")  # exists on disk; detectors are skipped via --no-*

    db.base_dir = tmp_path.resolve()
    sid = await db.add_source(db.to_relative(str(photo_path)), width=100, height=100)
    # Explicit scan_type='subjects' — the default effective_scan_type for
    # ``analyse`` without --caption. The helper's ``human``/``pet`` scan
    # rows would not intersect the safety prompt's IN-query.
    scan_id = await db.record_scan(sid, "subjects")
    await db.add_findings_batch(
        [(sid, (0, 0, 10, 10), np.ones(512, dtype=np.float32) / 22.6, 0.9)],
        scan_id=scan_id,
        species="human",
    )
    findings = await db.get_source_findings(sid)
    assert len(findings) == 1
    finding_id = findings[0].id

    if with_assignment:
        # Create a subject and wire the finding to it — the prompt pivots on
        # a non-null finding_assignment.subject_id.
        subj_id = await db.create_subject("Test Subject")
        await db.assign_finding_to_subject(finding_id, subj_id)
    return photo_path, sid, finding_id


async def test_analyse_missing_source_errors(db: FaceDB, tmp_path: Path) -> None:
    """Positional [SOURCES] arg uses ``click.Path(exists=True)`` — click rejects it."""
    _ = await _seed(db)
    result = await _run(
        ["--photos-dir", str(tmp_path), "analyse", str(tmp_path / "nope.jpg")],
        tmp_path,
    )
    # click.Path(exists=True) renders a standard "does not exist" error.
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower()


async def test_analyse_force_assigned_aborts_on_no(db: FaceDB, tmp_path: Path) -> None:
    """--force on a source with a manual assignment must prompt; 'n' leaves DB unchanged."""
    photo_path, sid, _ = await _seed_subjects_scan(db, tmp_path, with_assignment=True)
    findings_before = len(await db.get_source_findings(sid))
    scans_before = len(await db.find_scans())

    result = await _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--force",
            "--no-faces",
            "--no-pets",
            "--no-cluster",
        ],
        tmp_path,
        input_text="n\n",
    )
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output
    assert "manual subject assignments" in result.output
    assert len(await db.get_source_findings(sid)) == findings_before  # untouched
    assert len(await db.find_scans()) == scans_before


async def test_analyse_force_yes_skips_prompt(db: FaceDB, tmp_path: Path) -> None:
    """-y must skip the safety prompt even when an assignment would be lost."""
    photo_path, _sid, _ = await _seed_subjects_scan(db, tmp_path, with_assignment=True)

    result = await _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--force",
            "-y",
            "--no-faces",
            "--no-pets",
            "--no-cluster",
        ],
        tmp_path,
    )
    # The invalid-bytes jpg trips OSError inside photo_frames → counted as an
    # error, pipeline doesn't persist, but the command completes cleanly and
    # — critically — never showed the confirm prompt.
    assert result.exit_code == 0, result.output
    assert "Proceed?" not in result.output
    assert "Aborted" not in result.output


async def test_analyse_without_force_skips_already_scanned(db: FaceDB, tmp_path: Path) -> None:
    """Without --force, analyse <path> on an already-scanned source is a no-op."""
    photo_path, sid, _ = await _seed_subjects_scan(db, tmp_path, with_assignment=False)
    findings_before = len(await db.get_source_findings(sid))

    result = await _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--no-faces",
            "--no-pets",
            "--no-cluster",
        ],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert "Skipping already-scanned source" in result.output
    assert len(await db.get_source_findings(sid)) == findings_before  # untouched


async def test_analyse_persists_zero_result_scan(db: FaceDB, tmp_path: Path) -> None:
    """A successful analysis with no findings still records a scan row."""
    from PIL import Image

    photo_path = tmp_path / "empty.jpg"
    Image.new("RGB", (16, 16), "white").save(photo_path)

    result = await _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--no-faces",
            "--no-pets",
            "--no-cluster",
        ],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert "findings=0" in result.output
    assert "errors=0" in result.output
    scans = await db.find_scans()
    assert len(scans) == 1
    assert scans[0]["source_path"] == "empty.jpg"
    assert scans[0]["scan_type"] == "subjects"
    assert scans[0]["detection_strategy"] == "dedup"
    assert scans[0]["finding_count"] == 0
    source = await db.get_source_by_path("empty.jpg")
    assert source is not None
    assert await db.get_source_findings(source.id) == []


async def test_analyse_writes_html_report_by_default(tmp_path: Path) -> None:
    """analyse writes a per-run HTML report for persisted scans unless disabled."""
    from PIL import Image

    photo_path = tmp_path / "empty.jpg"
    Image.new("RGB", (16, 16), "white").save(photo_path)

    result = await _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--no-faces",
            "--no-pets",
        ],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert "Clustering analysed findings" in result.output
    assert "Analysis report:" in result.output

    reports = list((tmp_path / "reports").glob("analysis-*.html"))
    assert len(reports) == 1
    html = reports[0].read_text()
    assert "Ritrova Analysis Report" in html
    assert "empty.jpg" in html


async def test_analyse_report_can_be_disabled(tmp_path: Path) -> None:
    from PIL import Image

    photo_path = tmp_path / "empty.jpg"
    Image.new("RGB", (16, 16), "white").save(photo_path)

    result = await _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--no-faces",
            "--no-pets",
            "--no-analysis-report",
            "--no-cluster",
        ],
        tmp_path,
    )
    assert result.exit_code == 0, result.output
    assert "Analysis report:" not in result.output
    assert not (tmp_path / "reports").exists()


def test_analyse_help_shows_sources_argument() -> None:
    """`analyse --help` must advertise the positional [SOURCES] argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "--help"])
    assert result.exit_code == 0, result.output
    # click renders variadic args as [SOURCES]... in the usage line.
    assert "[SOURCES]..." in result.output


# ── `ritrova doctor` ──────────────────────────────────────────────────


async def _orphan_a_source(db: FaceDB, source_id: UUID) -> None:
    """Delete a source row with FK triggers off so children become orphans.

    Postgres cannot reach this state the way the original SQLite incident
    did (``PRAGMA foreign_keys=OFF``), so the nearest equivalent is used to
    manufacture it: ``session_replication_role = replica`` suspends FK
    triggers for one session. The detector is still worth testing for rows
    that predate the migration.
    """
    async with db.acquire() as connection:
        await connection.execute("SET session_replication_role = replica")
        try:
            await connection.execute("DELETE FROM sources WHERE id = $1", source_id)
        finally:
            await connection.execute("SET session_replication_role = DEFAULT")


async def test_doctor_clean_db_reports_healthy_and_exits_zero(db: FaceDB, tmp_path: Path) -> None:
    _ = await _seed(db)
    result = await _run(["doctor"], tmp_path)
    assert result.exit_code == 0, result.output
    assert "No orphans found" in result.output


async def test_doctor_detects_orphans_and_exits_non_zero_without_fix(
    db: FaceDB, tmp_path: Path
) -> None:
    sources = await _seed(db)
    await _orphan_a_source(db, sources["photo1"])

    result = await _run(["doctor"], tmp_path)
    # Report-only mode: non-zero exit when dirty so shell pipelines can gate.
    assert result.exit_code == 1, result.output
    assert "findings with missing source: 1" in result.output
    assert "scans with missing source: 1" in result.output
    assert "Total orphans: 2" in result.output
    assert "Run with --fix to delete" in result.output

    # DB is untouched — no --fix was passed.
    assert len((await db.find_orphans()).findings_missing_source) == 1


async def test_doctor_fix_deletes_orphans(db: FaceDB, tmp_path: Path) -> None:
    sources = await _seed(db)
    await _orphan_a_source(db, sources["photo1"])

    result = await _run(["doctor", "--fix", "-y"], tmp_path)
    assert result.exit_code == 0, result.output
    assert "Deleted 2 orphan row(s)" in result.output
    report = await db.find_orphans()
    assert report.total == 0
    # Sanity: the healthy source + its scan + its finding are untouched.
    remaining = await db.get_source_findings(sources["photo2"])
    assert len(remaining) == 1


async def test_doctor_fix_without_yes_aborts_on_no_input(db: FaceDB, tmp_path: Path) -> None:
    sources = await _seed(db)
    await _orphan_a_source(db, sources["photo1"])

    result = await _run(["doctor", "--fix"], tmp_path, input_text="n\n")
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output

    # Nothing was deleted — orphans still there.
    assert (await db.find_orphans()).total == 2


async def test_doctor_dismissed_orphan_category_is_noop_post_refactor(
    db: FaceDB, tmp_path: Path
) -> None:
    """The dismissed_findings table is gone post-Apr-2026 refactor —
    exclusion_reason='not_a_face' lives on finding_assignment now, and
    that FK has ON DELETE CASCADE so orphans can't exist. The doctor's
    dismissed-orphan category is retained as an empty list for callers
    that still check it."""
    await _seed(db)
    result = await _run(["doctor", "--fix", "-y"], tmp_path)
    assert result.exit_code == 0, result.output
    # Either reports zero orphans or doesn't mention the category at all.
    assert "dismissed_findings with missing finding: 0" in result.output or (
        "dismissed_findings" not in result.output
    )
