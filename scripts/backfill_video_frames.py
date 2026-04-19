"""Backfill ``frame_path`` on video findings that never had one saved.

Cause: the pipeline used to skip caching frame 0 of videos (analysis.py
``frame.frame_number > 0`` guard), and some older code paths didn't save
frames at all. The UI's thumbnail endpoint then tried to PIL-open the
video file itself and crashed.

Strategy: for every ``findings`` row where ``sources.type='video'`` and
``frame_path`` is NULL/empty, re-open the source video, seek to
``frame_number``, write the JPEG under the same scheme the persister uses
(``vid_<md5[:10]>_<frame_number>.jpg``), and update the row.

Run:
    uv run python scripts/backfill_video_frames.py --dry-run
    uv run python scripts/backfill_video_frames.py --apply
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sqlite3
from pathlib import Path

import cv2
from dotenv import load_dotenv
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.apply:
        raise SystemExit("Pass exactly one of --dry-run / --apply.")

    load_dotenv()
    db_path = Path(os.environ.get("FACE_DB", "./data/faces.db"))
    photos_dir = os.environ.get("PHOTOS_DIR")
    if not photos_dir:
        raise SystemExit("PHOTOS_DIR must be set.")
    photos_root = Path(photos_dir)
    frames_dir = db_path.parent / "tmp" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")

    rows = conn.execute(
        """
        SELECT f.id, f.source_id, f.frame_number, s.file_path
        FROM findings f JOIN sources s ON s.id = f.source_id
        WHERE s.type = 'video' AND (f.frame_path IS NULL OR f.frame_path = '')
        ORDER BY s.file_path, f.frame_number
        """
    ).fetchall()
    print(f"Video findings missing frame_path: {len(rows)}")

    # Group by (source_path, video) so we open each video once.
    by_source: dict[str, list[sqlite3.Row]] = {}
    for r in rows:
        by_source.setdefault(r["file_path"], []).append(r)

    fixed = 0
    already_on_disk = 0
    video_missing = 0
    frame_missing = 0
    updates: list[tuple[str, int]] = []  # (frame_path_rel, finding_id)

    for rel_path, group in by_source.items():
        vid_path = photos_root / rel_path
        if not vid_path.exists():
            video_missing += len(group)
            continue

        vid_hash = hashlib.md5(rel_path.encode()).hexdigest()[:10]  # noqa: S324
        needed_frames = {r["frame_number"] for r in group}
        extracted: dict[int, str] = {}

        # Fast path: if the JPEG already exists on disk, just reuse it.
        for fn in list(needed_frames):
            jpg = frames_dir / f"vid_{vid_hash}_{fn}.jpg"
            if jpg.exists():
                extracted[fn] = str(jpg.relative_to(db_path.parent))
                needed_frames.discard(fn)
                already_on_disk += 1

        if needed_frames:
            cap = cv2.VideoCapture(str(vid_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for fn in sorted(needed_frames):
                if fn >= total:
                    frame_missing += 1
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
                ret, bgr = cap.read()
                if not ret:
                    frame_missing += 1
                    continue
                rgb = bgr[:, :, ::-1]
                img = Image.fromarray(rgb)
                jpg = frames_dir / f"vid_{vid_hash}_{fn}.jpg"
                img.save(str(jpg), "JPEG", quality=85)
                extracted[fn] = str(jpg.relative_to(db_path.parent))
            cap.release()

        for r in group:
            fp = extracted.get(r["frame_number"])
            if fp is None:
                continue
            updates.append((fp, r["id"]))
            fixed += 1

    print(f"  fixable (frame extracted or already on disk): {fixed}")
    print(f"    of which already on disk (no re-extract):   {already_on_disk}")
    print(f"  video file missing from disk:                 {video_missing}")
    print(f"  frame unreadable (seek/read failed):          {frame_missing}")

    if args.dry_run:
        print("\nDry run — no DB writes. Run with --apply to persist.")
        return

    conn.execute("BEGIN")
    conn.executemany("UPDATE findings SET frame_path = ? WHERE id = ?", updates)
    conn.commit()
    print(f"\nUpdated {len(updates)} findings.")


if __name__ == "__main__":
    main()
