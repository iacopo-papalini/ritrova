"""Tests for the click CLI commands added with the scan↔finding linkage refactor.

Covers `scans list`, `scans prune`, and the single-source `analyse <path>`
path that replaced the retired `rescan` command. The detector models are not
exercised here — those paths are integration-tested via the scanner unit
tests; the CLI tests focus on argument handling, confirmation, and the
DB-side cascade.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from click.testing import CliRunner, Result

from ritrova.cli import cli
from ritrova.db import FaceDB

from ._helpers import add_findings


def _seed(tmp_path: Path) -> tuple[Path, FaceDB, dict[str, int]]:
    """Set up a test DB with two photo sources, each with one human scan + finding."""
    db_path = tmp_path / "test.db"
    db = FaceDB(db_path)
    photo1 = db.add_source("/2024/jan/a.jpg", width=100, height=100)
    photo2 = db.add_source("/2025/feb/b.jpg", width=100, height=100)
    add_findings(
        db,
        [
            (photo1, (0, 0, 10, 10), np.ones(512, dtype=np.float32) / 22.6, 0.9),
            (photo2, (0, 0, 10, 10), np.ones(512, dtype=np.float32) / 22.6, 0.9),
        ],
    )
    sources = {"photo1": photo1, "photo2": photo2}
    db.close()
    return db_path, FaceDB(db_path), sources


def _run(args: list[str], db_path: Path, input_text: str | None = None) -> Result:
    runner = CliRunner()
    return runner.invoke(cli, ["--db", str(db_path), *args], input=input_text)


def test_scans_list_outputs_rows(tmp_path: Path) -> None:
    db_path, db, _ = _seed(tmp_path)
    db.close()
    result = _run(["scans", "list"], db_path)
    assert result.exit_code == 0, result.output
    assert "/2024/jan/a.jpg" in result.output
    assert "/2025/feb/b.jpg" in result.output
    assert "2 scan(s)" in result.output


def test_scans_list_filters_by_pattern(tmp_path: Path) -> None:
    db_path, db, _ = _seed(tmp_path)
    db.close()
    result = _run(["scans", "list", "--source-pattern", "*2024*"], db_path)
    assert result.exit_code == 0, result.output
    assert "/2024/jan/a.jpg" in result.output
    assert "/2025/feb/b.jpg" not in result.output


def test_scans_prune_requires_filter(tmp_path: Path) -> None:
    db_path, db, _ = _seed(tmp_path)
    db.close()
    result = _run(["scans", "prune"], db_path)
    assert result.exit_code != 0
    assert "refusing to prune everything" in result.output


def test_scans_prune_aborts_on_no(tmp_path: Path) -> None:
    db_path, db, _ = _seed(tmp_path)
    db.close()
    result = _run(["scans", "prune", "--source-pattern", "*"], db_path, input_text="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output

    db = FaceDB(db_path)
    assert len(db.find_scans()) == 2  # nothing removed
    db.close()


def test_scans_prune_by_id_with_yes(tmp_path: Path) -> None:
    db_path, db, _ = _seed(tmp_path)
    scan_id = db.find_scans(source_pattern="*a.jpg")[0]["id"]
    db.close()
    result = _run(["scans", "prune", "--scan-id", str(scan_id), "-y"], db_path)
    assert result.exit_code == 0, result.output
    assert "Pruned 1 scan(s)" in result.output

    db = FaceDB(db_path)
    remaining = db.find_scans()
    assert len(remaining) == 1
    assert "/2025/feb/b.jpg" in remaining[0]["source_path"]
    db.close()


def test_scans_prune_by_pattern_intersection(tmp_path: Path) -> None:
    """Both filters → only the intersection is pruned."""
    db_path, db, _ = _seed(tmp_path)
    a_scan_id = db.find_scans(source_pattern="*a.jpg")[0]["id"]
    db.close()
    # Pattern matches /2025/* but scan_id points to /2024/a.jpg → no overlap.
    result = _run(
        [
            "scans",
            "prune",
            "--scan-id",
            str(a_scan_id),
            "--source-pattern",
            "*2025*",
            "-y",
        ],
        db_path,
    )
    assert result.exit_code == 0, result.output
    assert "(no scans match)" in result.output

    db = FaceDB(db_path)
    assert len(db.find_scans()) == 2
    db.close()


# ── `ritrova analyse <path>` — the single-source re-analysis path ────
# (replaces the retired `rescan` command)


def _seed_subjects_scan(tmp_path: Path, *, with_assignment: bool) -> tuple[Path, Path, int, int]:
    """Set up a DB + on-disk file with one 'subjects' scan and one finding.

    Returns ``(db_path, photo_path, source_id, finding_id)``. When
    ``with_assignment`` is True, the finding is tagged as manually assigned
    (non-null ``finding_assignment.subject_id``) — this is what trips the
    safety prompt on ``analyse <path> --force``.
    """
    db_path = tmp_path / "test.db"
    photo_path = tmp_path / "fake.jpg"
    photo_path.write_bytes(b"x")  # exists on disk; detectors are skipped via --no-*

    db = FaceDB(db_path, base_dir=tmp_path)
    sid = db.add_source(db.to_relative(str(photo_path)), width=100, height=100)
    # Explicit scan_type='subjects' — the default effective_scan_type for
    # ``analyse`` without --caption. The helper's ``human``/``pet`` scan
    # rows would not intersect the safety prompt's IN-query.
    scan_id = db.record_scan(sid, "subjects")
    db.add_findings_batch(
        [(sid, (0, 0, 10, 10), np.ones(512, dtype=np.float32) / 22.6, 0.9)],
        scan_id=scan_id,
        species="human",
    )
    findings = db.get_source_findings(sid)
    assert len(findings) == 1
    finding_id = findings[0].id

    if with_assignment:
        # Create a subject and wire the finding to it — the prompt pivots on
        # a non-null finding_assignment.subject_id.
        subj_id = db.create_subject("Test Subject")
        db.assign_finding_to_subject(finding_id, subj_id)
    db.close()
    return db_path, photo_path, sid, finding_id


def test_analyse_missing_source_errors(tmp_path: Path) -> None:
    """Positional [SOURCES] arg uses ``click.Path(exists=True)`` — click rejects it."""
    db_path, db, _ = _seed(tmp_path)
    db.close()
    result = _run(
        ["--photos-dir", str(tmp_path), "analyse", str(tmp_path / "nope.jpg")],
        db_path,
    )
    # click.Path(exists=True) renders a standard "does not exist" error.
    assert result.exit_code != 0
    assert "does not exist" in result.output.lower()


def test_analyse_force_assigned_aborts_on_no(tmp_path: Path) -> None:
    """--force on a source with a manual assignment must prompt; 'n' leaves DB unchanged."""
    db_path, photo_path, sid, _ = _seed_subjects_scan(tmp_path, with_assignment=True)
    db = FaceDB(db_path, base_dir=tmp_path)
    findings_before = len(db.get_source_findings(sid))
    scans_before = len(db.find_scans())
    db.close()

    result = _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--force",
            "--no-faces",
            "--no-pets",
        ],
        db_path,
        input_text="n\n",
    )
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output
    assert "manual subject assignments" in result.output

    db = FaceDB(db_path, base_dir=tmp_path)
    assert len(db.get_source_findings(sid)) == findings_before  # untouched
    assert len(db.find_scans()) == scans_before
    db.close()


def test_analyse_force_yes_skips_prompt(tmp_path: Path) -> None:
    """-y must skip the safety prompt even when an assignment would be lost."""
    db_path, photo_path, _sid, _ = _seed_subjects_scan(tmp_path, with_assignment=True)

    result = _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--force",
            "-y",
            "--no-faces",
            "--no-pets",
        ],
        db_path,
    )
    # The invalid-bytes jpg trips OSError inside photo_frames → counted as an
    # error, pipeline doesn't persist, but the command completes cleanly and
    # — critically — never showed the confirm prompt.
    assert result.exit_code == 0, result.output
    assert "Proceed?" not in result.output
    assert "Aborted" not in result.output


def test_analyse_without_force_skips_already_scanned(tmp_path: Path) -> None:
    """Without --force, analyse <path> on an already-scanned source is a no-op."""
    db_path, photo_path, sid, _ = _seed_subjects_scan(tmp_path, with_assignment=False)
    db = FaceDB(db_path, base_dir=tmp_path)
    findings_before = len(db.get_source_findings(sid))
    db.close()

    result = _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--no-faces",
            "--no-pets",
        ],
        db_path,
    )
    assert result.exit_code == 0, result.output
    assert "Skipping already-scanned source" in result.output

    db = FaceDB(db_path, base_dir=tmp_path)
    assert len(db.get_source_findings(sid)) == findings_before  # untouched
    db.close()


def test_analyse_persists_zero_result_scan(tmp_path: Path) -> None:
    """A successful analysis with no findings still records a scan row."""
    from PIL import Image

    db_path = tmp_path / "test.db"
    photo_path = tmp_path / "empty.jpg"
    Image.new("RGB", (16, 16), "white").save(photo_path)

    result = _run(
        [
            "--photos-dir",
            str(tmp_path),
            "analyse",
            str(photo_path),
            "--no-faces",
            "--no-pets",
        ],
        db_path,
    )
    assert result.exit_code == 0, result.output
    assert "findings=0" in result.output
    assert "errors=0" in result.output

    db = FaceDB(db_path, base_dir=tmp_path)
    scans = db.find_scans()
    assert len(scans) == 1
    assert scans[0]["source_path"] == "empty.jpg"
    assert scans[0]["scan_type"] == "subjects"
    assert scans[0]["detection_strategy"] == "dedup"
    assert scans[0]["finding_count"] == 0
    source = db.get_source_by_path("empty.jpg")
    assert source is not None
    assert db.get_source_findings(source.id) == []
    db.close()


def test_analyse_help_shows_sources_argument() -> None:
    """`analyse --help` must advertise the positional [SOURCES] argument."""
    runner = CliRunner()
    result = runner.invoke(cli, ["analyse", "--help"])
    assert result.exit_code == 0, result.output
    # click renders variadic args as [SOURCES]... in the usage line.
    assert "[SOURCES]..." in result.output


# ── `ritrova doctor` ──────────────────────────────────────────────────


def _orphan_a_source(db_path: Path, source_id: int) -> None:
    """Delete a source row with FKs OFF so children become orphans.

    This simulates exactly the scenario that produced the user's 15-orphan
    incident: a ``sqlite3`` CLI session (or any connection that didn't set
    ``PRAGMA foreign_keys=ON``) issuing a DELETE on ``sources``.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
    conn.commit()
    conn.close()


def test_doctor_clean_db_reports_healthy_and_exits_zero(tmp_path: Path) -> None:
    db_path, db, _ = _seed(tmp_path)
    db.close()
    result = _run(["doctor"], db_path)
    assert result.exit_code == 0, result.output
    assert "No orphans found" in result.output


def test_doctor_detects_orphans_and_exits_non_zero_without_fix(tmp_path: Path) -> None:
    db_path, db, sources = _seed(tmp_path)
    db.close()
    _orphan_a_source(db_path, sources["photo1"])

    result = _run(["doctor"], db_path)
    # Report-only mode: non-zero exit when dirty so shell pipelines can gate.
    assert result.exit_code == 1, result.output
    assert "findings with missing source: 1" in result.output
    assert "scans with missing source: 1" in result.output
    assert "Total orphans: 2" in result.output
    assert "Run with --fix to delete" in result.output

    # DB is untouched — no --fix was passed.
    db = FaceDB(db_path)
    assert len(db.find_orphans().findings_missing_source) == 1
    db.close()


def test_doctor_fix_deletes_orphans(tmp_path: Path) -> None:
    db_path, db, sources = _seed(tmp_path)
    db.close()
    _orphan_a_source(db_path, sources["photo1"])

    result = _run(["doctor", "--fix", "-y"], db_path)
    assert result.exit_code == 0, result.output
    assert "Deleted 2 orphan row(s)" in result.output

    db = FaceDB(db_path)
    report = db.find_orphans()
    assert report.total == 0
    # Sanity: the healthy source + its scan + its finding are untouched.
    remaining = db.get_source_findings(sources["photo2"])
    assert len(remaining) == 1
    db.close()


def test_doctor_fix_without_yes_aborts_on_no_input(tmp_path: Path) -> None:
    db_path, db, sources = _seed(tmp_path)
    db.close()
    _orphan_a_source(db_path, sources["photo1"])

    result = _run(["doctor", "--fix"], db_path, input_text="n\n")
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output

    # Nothing was deleted — orphans still there.
    db = FaceDB(db_path)
    assert db.find_orphans().total == 2
    db.close()


def test_doctor_dismissed_orphan_category_is_noop_post_refactor(tmp_path: Path) -> None:
    """The dismissed_findings table is gone post-Apr-2026 refactor —
    exclusion_reason='not_a_face' lives on finding_assignment now, and
    that FK has ON DELETE CASCADE so orphans can't exist. The doctor's
    dismissed-orphan category is retained as an empty list for callers
    that still check it."""
    db_path, _, _ = _seed(tmp_path)
    result = _run(["doctor", "--fix", "-y"], db_path)
    assert result.exit_code == 0, result.output
    # Either reports zero orphans or doesn't mention the category at all.
    assert "dismissed_findings with missing finding: 0" in result.output or (
        "dismissed_findings" not in result.output
    )
