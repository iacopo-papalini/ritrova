"""Tests for the click CLI commands added with the scan↔finding linkage refactor.

Covers `scans list`, `scans prune`, and the `rescan` argument validation. The
detector models are not exercised here — those paths are integration-tested via
the scanner unit tests; the CLI tests focus on argument handling, confirmation,
and the DB-side cascade.
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


def test_rescan_unknown_source_errors(tmp_path: Path) -> None:
    db_path, db, _ = _seed(tmp_path)
    db.close()
    # Use absolute path that doesn't exist on disk.
    result = _run(
        ["--photos-dir", str(tmp_path), "rescan", str(tmp_path / "nope.jpg")],
        db_path,
    )
    assert result.exit_code != 0
    assert "File not found" in result.output


def test_rescan_aborts_on_no(tmp_path: Path) -> None:
    """Rescan asks before destroying findings — answering 'n' leaves DB intact."""
    db_path = tmp_path / "test.db"
    photo_path = tmp_path / "fake.jpg"
    photo_path.write_bytes(b"x")  # has to exist on disk; rescan checks before prompting

    # Seed: add source by its eventual stored (relative) path, plus a scan + finding.
    db = FaceDB(db_path, base_dir=tmp_path)
    sid = db.add_source(db.to_relative(str(photo_path)), width=100, height=100)
    add_findings(db, [(sid, (0, 0, 10, 10), np.ones(512, dtype=np.float32) / 22.6, 0.9)])
    findings_before = len(db.get_source_findings(sid))
    assert findings_before == 1
    db.close()

    result = _run(
        ["--photos-dir", str(tmp_path), "rescan", str(photo_path)],
        db_path,
        input_text="n\n",
    )
    assert result.exit_code == 0, result.output
    assert "Aborted" in result.output

    db = FaceDB(db_path, base_dir=tmp_path)
    assert len(db.get_source_findings(sid)) == findings_before  # untouched
    db.close()


def test_rescan_no_scans_message(tmp_path: Path) -> None:
    """If the source exists but has no matching scan, rescan exits cleanly."""
    db_path = tmp_path / "test.db"
    photo_path = tmp_path / "x.jpg"
    photo_path.write_bytes(b"x")
    db = FaceDB(db_path, base_dir=tmp_path)
    db.add_source(db.to_relative(str(photo_path)), width=100, height=100)
    db.close()

    result = _run(
        ["--photos-dir", str(tmp_path), "rescan", str(photo_path)],
        db_path,
    )
    assert result.exit_code == 0, result.output
    assert "no matching scans" in result.output
