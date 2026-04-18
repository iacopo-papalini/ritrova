"""Rescan sources where legacy `human`/`pet` scans found faces but the new
`subjects` pipeline returned zero findings.

Motivation (ADR-011 + scripts/investigations/aquarium_face_loss.py):
the edge-density filter in the old detector.py silently dropped legitimate
faces on dim/low-contrast photos. The main subjects rescan ran with that
filter still in place and left ~2,400 sources without any findings even
though legacy had recorded some. With the edge-density filter now removed,
we want a targeted second pass that hits only those sources — not another
full-archive rescan.

Targets are selected with:
    legacy (scan_type IN ('human','pet')) has ≥1 finding on the source
  AND subjects has 0 findings on the source (or no subjects scan at all)

For each target the existing analyse pipeline is invoked in-process
(FaceDetection → PetDetection → Dedup). A matching `subjects` scan is
deleted first if present (`--force`-style behaviour) so the rewrite is
idempotent.

Run:
    uv run python scripts/rescan_losers.py                 # full run
    uv run python scripts/rescan_losers.py --dry-run       # list targets only
    uv run python scripts/rescan_losers.py --sample 20     # first 20 targets
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Silence model-loading chatter.
for noisy in ("insightface", "onnxruntime", "ultralytics"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from ritrova.analysis import (  # noqa: E402
    AnalysisPersister,
    AnalysisPipelineBuilder,
    SourceAnalysis,
    photo_frames,
    video_frames,
)
from ritrova.analysis_steps import (  # noqa: E402
    DeduplicationStep,
    FaceDetectionStep,
    PetDetectionStep,
)
from ritrova.db import FaceDB  # noqa: E402
from ritrova.detector import FaceDetector  # noqa: E402
from ritrova.pet_detector import PetDetector  # noqa: E402
from ritrova.scanner import get_exif_date, get_exif_gps  # noqa: E402

SCAN_TYPE = "subjects"


def find_losers(db: FaceDB) -> list[tuple[int, str, str]]:
    """Source ids + stored paths + source_type for sources that need redoing."""
    rows = db.conn.execute(
        """
        WITH legacy_sources AS (
          SELECT DISTINCT sc.source_id
          FROM scans sc JOIN findings f ON f.scan_id = sc.id
          WHERE sc.scan_type IN ('human','pet')
        ),
        subjects_with_findings AS (
          SELECT DISTINCT sc.source_id
          FROM scans sc JOIN findings f ON f.scan_id = sc.id
          WHERE sc.scan_type = 'subjects'
        )
        SELECT s.id, s.file_path, s.type
        FROM sources s
        WHERE s.id IN legacy_sources
          AND s.id NOT IN subjects_with_findings
        ORDER BY s.id
        """
    ).fetchall()
    return [(row["id"], row["file_path"], row["type"]) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dry-run", action="store_true", help="List targets, don't rescan.")
    parser.add_argument(
        "--sample", type=int, default=0, help="Process only first N targets (0 = all)."
    )
    parser.add_argument(
        "--type",
        choices=("photo", "video", "all"),
        default="all",
        help="Restrict targets to one source type (default: all).",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between video frames.")
    args = parser.parse_args()

    load_dotenv()
    db_path = os.environ.get("FACE_DB", "./data/faces.db")
    photos_dir = os.environ.get("PHOTOS_DIR")
    if not photos_dir:
        raise SystemExit("PHOTOS_DIR must be set (in .env or the environment).")

    db = FaceDB(db_path, base_dir=photos_dir)
    targets = find_losers(db)
    print(f"Targets (legacy found faces, subjects found zero): {len(targets)}")

    if args.type != "all":
        targets = [t for t in targets if t[2] == args.type]
        print(f"Filtered to type={args.type!r}: {len(targets)} targets.")

    if args.sample > 0:
        targets = targets[: args.sample]
        print(f"Limiting to first {len(targets)} (--sample={args.sample}).")

    if args.dry_run:
        for _, path, kind in targets[:50]:
            print(f"  [{kind}] {path}")
        if len(targets) > 50:
            print(f"  … {len(targets) - 50} more")
        return

    if not targets:
        print("Nothing to do.")
        return

    # Build the same default pipeline as `ritrova analyse` (no captions).
    print("Loading face detection model…")
    face_det = FaceDetector()
    print("Loading pet detection models…")
    pet_det = PetDetector()

    builder = AnalysisPipelineBuilder()
    builder.add_step(FaceDetectionStep(face_det))
    builder.add_step(PetDetectionStep(pet_det))
    builder.add_step(DeduplicationStep())
    pipeline = builder.build()

    frames_dir = Path(db_path).parent / "tmp" / "frames"
    persister = AnalysisPersister(db, frames_dir=frames_dir)

    t_start = time.monotonic()
    processed = 0
    recovered_findings = 0
    errors = 0

    for source_id, stored_path, source_type in targets:
        source_path = db.resolve_path(stored_path)
        if not source_path.exists():
            errors += 1
            continue

        frames = (
            video_frames(source_path, interval_sec=args.interval)
            if source_type == "video"
            else photo_frames(source_path)
        )

        initial = SourceAnalysis(source_path=stored_path, source_type=source_type)
        if source_type == "photo":
            initial.taken_at = get_exif_date(source_path)
            gps = get_exif_gps(source_path)
            if gps:
                initial.latitude, initial.longitude = gps

        try:
            result = pipeline.analyse_source(
                source_path,
                source_type=source_type,
                frames=frames,
                initial_state=initial,
            )
        except OSError as e:
            print(f"  [error] {stored_path}: {e}")
            errors += 1
            continue

        n_findings = len(result.findings)
        # --force-style: drop an existing subjects scan on this source if any
        existing = db.conn.execute(
            "SELECT id FROM scans WHERE source_id=? AND scan_type=?",
            (source_id, SCAN_TYPE),
        ).fetchone()
        if existing:
            db.delete_scan(existing["id"])

        if n_findings > 0:
            persister.persist(result, strategy_id=pipeline.strategy_id, scan_type=SCAN_TYPE)
            recovered_findings += n_findings

        processed += 1
        if processed % 50 == 0 or processed == len(targets):
            elapsed = time.monotonic() - t_start
            rate = processed / elapsed if elapsed > 0 else 0.0
            remaining = len(targets) - processed
            eta = remaining / rate if rate > 0 else 0.0
            print(
                f"  [{processed}/{len(targets)}] recovered_findings={recovered_findings}  "
                f"errors={errors}  {rate:.1f}/s  ETA {eta / 60:.1f}m"
            )

    print(
        f"Done. processed={processed}  recovered_findings={recovered_findings}  "
        f"errors={errors}  wall_time={(time.monotonic() - t_start) / 60:.1f}m"
    )


if __name__ == "__main__":
    main()
