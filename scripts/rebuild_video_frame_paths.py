"""One-shot: backfill missing ``findings.frame_path`` for video sources.

Why this script exists
----------------------
Pre-fix ``analysis.py`` cached video frame images ONLY when a frame's
post-step ``state.findings`` count grew. ``DeduplicationStep`` runs as
the last per-frame step and compresses near-duplicate detections, so
frames whose net length didn't grow had their image silently dropped.
The surviving finding's row was then written with ``frame_path=NULL``,
which crashes ``crop_face_thumbnail`` (PIL can't open a .mp4).

The fix in ``analysis.py`` caches the frame image after each individual
detection step (before dedup) so future scans never produce these
broken rows. This script repairs the rows that were already written.

What it does
------------
1. Find every ``findings`` row where the source is a video and
   ``frame_path IS NULL`` (and ``frame_number IS NOT NULL`` — manual
   findings on photos use ``frame_number=NULL`` and don't apply).
2. Group by source, sort frame numbers ascending, open each video
   exactly once, sequentially read frames (matches what the detector
   originally saw — safer than ``CAP_PROP_POS_FRAMES`` seeking on
   variable-bitrate videos).
3. Save each target frame to
   ``<db-dir>/tmp/frames/vid_<md5(source_path)[:10]>_<frame_number>.jpg``
   — the same naming ``analysis.AnalysisPersister.persist`` uses, so
   subsequent re-scans don't double-write.
4. Update the row: ``UPDATE findings SET frame_path=? WHERE id=?``.

Idempotent — re-running on a clean DB is a no-op.

Usage
-----
    uv run python scripts/rebuild_video_frame_paths.py --dry-run
    uv run python scripts/rebuild_video_frame_paths.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import cv2
from PIL import Image

# Ensure ``ritrova`` is importable when run from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ritrova.db import FaceDB  # noqa: E402

logger = logging.getLogger("rebuild_video_frame_paths")


def _frame_filename(source_rel_path: str, frame_number: int) -> str:
    """Match the naming used by ``AnalysisPersister.persist`` so a future
    re-scan reuses the same JPGs and doesn't double-write."""
    vid_hash = hashlib.md5(source_rel_path.encode()).hexdigest()[:10]  # noqa: S324
    return f"vid_{vid_hash}_{frame_number}.jpg"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="./data/faces.db", help="SQLite DB path.")
    parser.add_argument(
        "--photos-dir",
        default=None,
        help=(
            "Root directory the DB stores paths relative to. Defaults to the "
            "PHOTOS_DIR environment variable; required if not set."
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually write JPGs and update rows. Default is a dry-run.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    photos_dir = args.photos_dir
    if not photos_dir:
        import os

        photos_dir = os.environ.get("PHOTOS_DIR")
    if not photos_dir:
        logger.error("Set --photos-dir or the PHOTOS_DIR env var.")
        return 2

    db = FaceDB(args.db, base_dir=photos_dir)
    frames_dir = Path(args.db).resolve().parent / "tmp" / "frames"
    if args.apply:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Pull (finding_id, source_id, file_path, frame_number) for every
    # candidate row. Sort by source so we process one video at a time.
    rows = db.conn.execute(
        """
        SELECT f.id, f.source_id, s.file_path, f.frame_number
        FROM findings f
        JOIN sources s ON s.id = f.source_id
        WHERE s.type = 'video'
          AND f.frame_path IS NULL
          AND f.frame_number IS NOT NULL
        ORDER BY f.source_id, f.frame_number
        """,
    ).fetchall()
    if not rows:
        logger.info("No broken video findings — nothing to do.")
        db.close()
        return 0
    logger.info(
        "Found %d broken finding(s) across %d video source(s).",
        len(rows),
        len({r[1] for r in rows}),
    )

    # Group target frame_numbers per source.
    by_source: dict[int, list[tuple[int, str, int]]] = {}
    for finding_id, source_id, file_path, frame_number in rows:
        by_source.setdefault(source_id, []).append((finding_id, file_path, frame_number))

    fixed = 0
    skipped_already_on_disk = 0
    failed_sources: list[tuple[int, str, str]] = []  # (source_id, file_path, reason)
    failed_frames: list[tuple[int, str, int, str]] = []  # (finding_id, ..., reason)

    for source_id, group in by_source.items():
        file_path = group[0][1]
        absolute = db.resolve_path(file_path)
        if not absolute.exists():
            failed_sources.append((source_id, file_path, "file missing on disk"))
            logger.warning("source %d: file missing — %s", source_id, absolute)
            continue
        target_frames = sorted({fn for _fid, _fp, fn in group})
        # Map frame_number -> [finding_ids needing it]. Multiple findings can
        # share a frame_number when several subjects appeared in the same
        # sampled frame.
        finding_ids_per_frame: dict[int, list[int]] = {}
        for fid, _fp, fn in group:
            finding_ids_per_frame.setdefault(fn, []).append(fid)

        # Pre-existing JPGs on disk count as "already fixed" — only the DB
        # row needs the path written.
        captures_needed: list[int] = []
        prebuilt: dict[int, str] = {}
        for fn in target_frames:
            jpg_name = _frame_filename(file_path, fn)
            jpg_path = frames_dir / jpg_name
            relative_to_db = str(jpg_path.relative_to(Path(args.db).resolve().parent))
            if jpg_path.exists():
                prebuilt[fn] = relative_to_db
                skipped_already_on_disk += len(finding_ids_per_frame[fn])
            else:
                captures_needed.append(fn)

        captured: dict[int, str] = dict(prebuilt)
        if captures_needed:
            cap = cv2.VideoCapture(str(absolute))
            if not cap.isOpened():
                cap.release()
                failed_sources.append((source_id, file_path, "cv2 failed to open"))
                logger.warning("source %d: cv2 could not open — %s", source_id, absolute)
                continue
            wanted = set(captures_needed)
            max_wanted = max(wanted)
            cur_idx = 0
            try:
                while wanted:
                    ret, bgr = cap.read()
                    if not ret:
                        for missing in sorted(wanted):
                            for fid in finding_ids_per_frame[missing]:
                                failed_frames.append(
                                    (fid, file_path, missing, "EOF before frame reached")
                                )
                        logger.warning(
                            "source %d: EOF at frame %d, %d frame(s) unreached: %s",
                            source_id,
                            cur_idx,
                            len(wanted),
                            sorted(wanted),
                        )
                        break
                    if cur_idx in wanted:
                        rgb = bgr[:, :, ::-1]
                        img = Image.fromarray(rgb)
                        jpg_name = _frame_filename(file_path, cur_idx)
                        jpg_path = frames_dir / jpg_name
                        relative_to_db = str(jpg_path.relative_to(Path(args.db).resolve().parent))
                        if args.apply:
                            img.save(str(jpg_path), "JPEG", quality=85)
                            logger.debug("wrote %s", jpg_path)
                        captured[cur_idx] = relative_to_db
                        wanted.discard(cur_idx)
                    cur_idx += 1
                    if cur_idx > max_wanted:
                        break
            finally:
                cap.release()

        # Update the rows. Even in --dry-run we count what would change.
        for fn, relative_to_db in captured.items():
            for finding_id in finding_ids_per_frame[fn]:
                if args.apply:
                    db.conn.execute(
                        "UPDATE findings SET frame_path = ? WHERE id = ?",
                        (relative_to_db, finding_id),
                    )
                fixed += 1

    if args.apply:
        db.conn.commit()

    logger.info(
        "Done. fixed=%d (of which %d already had the JPG on disk), failed_sources=%d, failed_frames=%d. %s",
        fixed,
        skipped_already_on_disk,
        len(failed_sources),
        len(failed_frames),
        "Wrote changes." if args.apply else "Dry-run; re-run with --apply to commit.",
    )
    if failed_sources:
        logger.info("Sources skipped:")
        for source_id, file_path, reason in failed_sources:
            logger.info("  source=%d  reason=%s  path=%s", source_id, reason, file_path)
    if failed_frames:
        logger.info("Frames unreachable:")
        for finding_id, file_path, frame_number, reason in failed_frames:
            logger.info(
                "  finding=%d  frame=%d  reason=%s  path=%s",
                finding_id,
                frame_number,
                reason,
                file_path,
            )

    db.close()
    return 0 if not failed_sources and not failed_frames else 1


if __name__ == "__main__":
    sys.exit(main())
