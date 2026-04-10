"""CLI entry point for ritrova."""

from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()


@click.group()
@click.option(
    "--db",
    default="./faces.db",
    help="Path to SQLite database",
    envvar="FACE_DB",
)
@click.option(
    "--photos-dir",
    default=None,
    help="Root directory for photos (paths stored relative to this)",
    envvar="PHOTOS_DIR",
    type=click.Path(file_okay=False),
)
@click.pass_context
def cli(ctx: click.Context, db: str, photos_dir: str | None) -> None:
    """Ritrova — find again: face and pet recognition for photo collections."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db
    ctx.obj["photos_dir"] = photos_dir


def _require_photos_dir(ctx: click.Context) -> str:
    photos_dir: str | None = ctx.obj["photos_dir"]
    if not photos_dir:
        raise click.UsageError("Set --photos-dir or PHOTOS_DIR environment variable")
    return photos_dir


@cli.command()
@click.option("--min-confidence", default=0.65, help="Minimum detection confidence")
@click.pass_context
def scan(ctx: click.Context, min_confidence: float) -> None:
    """Scan photos directory and detect all faces."""
    from .db import FaceDB
    from .detector import FaceDetector
    from .scanner import scan_photos

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    print(f"Database: {ctx.obj['db_path']}")
    print(f"Scanning: {photos_dir}")
    print("Loading face detection model (first run downloads ~300 MB)...")

    detector = FaceDetector()
    result = scan_photos(db, Path(photos_dir), detector, min_confidence)

    print(
        f"\nDone! processed={result['processed']}  "
        f"faces={result['faces_found']}  "
        f"skipped={result['skipped']}  errors={result['errors']}"
    )
    db.close()


@cli.command()
@click.option("--min-confidence", default=0.7, help="Minimum YOLO detection confidence")
@click.pass_context
def scan_pets(ctx: click.Context, min_confidence: float) -> None:
    """Scan photos for dogs and cats using YOLO + SigLIP."""
    from .db import FaceDB
    from .pet_detector import PetDetector
    from .scanner import scan_pets as _scan_pets

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    print(f"Database: {ctx.obj['db_path']}")
    print(f"Scanning for pets in: {photos_dir}")
    print("Loading YOLO + SigLIP models (first run downloads them)...")

    detector = PetDetector()
    result = _scan_pets(db, Path(photos_dir), detector, min_confidence)

    print(
        f"\nDone! processed={result['processed']}  "
        f"pets={result['pets_found']}  "
        f"skipped={result['skipped']}  errors={result['errors']}"
    )
    db.close()


@cli.command()
@click.option("--min-confidence", default=0.65, help="Minimum detection confidence")
@click.option("--interval", default=2.0, help="Seconds between sampled frames")
@click.pass_context
def scan_videos(ctx: click.Context, min_confidence: float, interval: float) -> None:
    """Scan videos: extract frames, detect faces, one per person per clip."""
    from .db import FaceDB
    from .detector import FaceDetector
    from .scanner import scan_videos as _scan_videos

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    frames_dir = Path(ctx.obj["db_path"]).parent / "tmp" / "frames"
    print(f"Database: {ctx.obj['db_path']}")
    print(f"Scanning videos in: {photos_dir}")
    print("Loading face detection model...")

    detector = FaceDetector()
    result = _scan_videos(
        db,
        Path(photos_dir),
        detector,
        frames_dir,
        min_confidence=min_confidence,
        interval_sec=interval,
    )

    print(
        f"\nDone! processed={result['processed']}  "
        f"faces={result['faces_found']}  "
        f"skipped={result['skipped']}  errors={result['errors']}"
    )
    db.close()


SPECIES_THRESHOLDS = {
    "human": 0.45,  # ArcFace 512-dim: well-separated
    "dog": 0.20,  # SigLIP 768-dim: much denser, needs tighter threshold
    "cat": 0.20,
}


@cli.command()
@click.option(
    "--threshold",
    default=None,
    type=float,
    help="Override cosine distance threshold (default: per-species)",
)
@click.option("--min-size", default=2, help="Minimum faces per cluster")
@click.pass_context
def cluster(ctx: click.Context, threshold: float | None, min_size: int) -> None:
    """Cluster all detected faces by embedding similarity (humans + pets)."""
    from .cluster import cluster_faces
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])

    for species, default_thresh in SPECIES_THRESHOLDS.items():
        t = threshold if threshold is not None else default_thresh
        print(f"\n── {species} (threshold={t}) ──")
        result = cluster_faces(db, threshold=t, min_size=min_size, species=species)
        print(f"  Total faces:      {result['total_faces']}")
        print(f"  Clusters formed:  {result['clusters']}")
        print(f"  Noise (outliers): {result['noise']}")
        if result.get("largest_cluster"):
            print(f"  Largest cluster:  {result['largest_cluster']} faces")

    db.close()


@cli.command()
@click.option("--min-similarity", default=50.0, help="Minimum centroid similarity %")
@click.option("--species", default="human", help="Species to process")
@click.pass_context
def auto_assign(ctx: click.Context, min_similarity: float, species: str) -> None:
    """Bulk-assign unnamed clusters to existing named persons."""
    from .cluster import auto_assign as _auto_assign
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    result = _auto_assign(db, min_similarity=min_similarity / 100, species=species)

    print(
        f"\nAssigned {result['assigned_clusters']} clusters "
        f"({result['assigned_faces']} faces), "
        f"{result['unmatched']} unmatched"
    )
    db.close()


@cli.command()
@click.option("--min-similarity", default=70.0, help="Minimum centroid similarity %")
@click.option("--species", default="human", help="Species to process")
@click.pass_context
def auto_merge(ctx: click.Context, min_similarity: float, species: str) -> None:
    """Auto-merge unnamed clusters whose centroids are highly similar."""
    from .cluster import auto_merge_clusters
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    result = auto_merge_clusters(db, min_similarity=min_similarity / 100, species=species)

    print(
        f"\nMerged {result['merged']} cluster pairs "
        f"({result['faces_moved']} faces moved), "
        f"{result['remaining_clusters']} clusters remaining"
    )
    db.close()


@cli.command()
@click.option("--min-size", default=50, help="Minimum face width/height in pixels")
@click.option("--min-sharpness", default=30.0, help="Minimum Laplacian variance (focus blur)")
@click.option("--min-edges", default=2.0, help="Minimum Canny edge density % (motion blur)")
@click.option("--dry-run", is_flag=True, help="Show what would be dismissed without acting")
@click.pass_context
def cleanup(
    ctx: click.Context, min_size: int, min_sharpness: float, min_edges: float, dry_run: bool
) -> None:
    """Dismiss tiny and blurry faces from the database."""
    import cv2
    from PIL import Image, ImageFile, ImageOps

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])

    # Find unassigned, non-dismissed faces
    rows = db.query(
        "SELECT f.id, f.photo_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h "
        "FROM faces f "
        "WHERE f.person_id IS NULL "
        "AND f.id NOT IN (SELECT face_id FROM dismissed_faces)"
    )
    total = len(rows)
    print(f"Checking {total} unassigned faces...")

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    import numpy as np

    tiny_ids = []
    check_rows = []
    for r in rows:
        fid, pid, bx, by, bw, bh = r[0], r[1], r[2], r[3], r[4], r[5]
        if bw < min_size or bh < min_size:
            tiny_ids.append(fid)
        else:
            check_rows.append((fid, pid, bx, by, bw, bh))

    print(f"  Tiny (<{min_size}px): {len(tiny_ids)} — instant dismiss")
    print(f"  Checking sharpness on {len(check_rows)} faces...")

    bad_quality_ids = []
    lock = threading.Lock()
    done = [0]
    focus_blur = [0]
    motion_blur = [0]

    def check_quality(row: tuple[int, ...]) -> tuple[int | None, str | None]:
        fid, pid, bx, by, bw, bh = row
        photo = db.get_photo(pid)
        if not photo:
            return None, None
        resolved = db.resolve_path(photo.file_path)
        if not resolved.exists():
            return None, None
        try:
            raw_img = Image.open(resolved)
            oriented = ImageOps.exif_transpose(raw_img)
            crop = oriented.crop((bx, by, bx + bw, by + bh))
            gray = np.array(crop.convert("L"))
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            if lap < min_sharpness:
                return fid, "focus"
            edge_pct = (cv2.Canny(gray, 50, 150) > 0).mean() * 100
            if edge_pct < min_edges:
                return fid, "motion"
            return None, None
        except OSError:
            return None, None

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(check_quality, r): r for r in check_rows}
        for future in as_completed(futures):
            fid, reason = future.result()
            if fid is not None:
                bad_quality_ids.append(fid)
                if reason == "focus":
                    focus_blur[0] += 1
                else:
                    motion_blur[0] += 1
            with lock:
                done[0] += 1
                if done[0] % 500 == 0:
                    print(
                        f"\r  [{done[0]}/{len(check_rows)}] out_of_focus={focus_blur[0]} featureless={motion_blur[0]}",
                        end="",
                        flush=True,
                    )

    to_dismiss = tiny_ids + bad_quality_ids
    print(f"\n  Tiny (<{min_size}px): {len(tiny_ids)}")
    print(f"  Out of focus (laplacian <{min_sharpness}): {focus_blur[0]}")
    print(f"  Featureless (edges <{min_edges}%): {motion_blur[0]}")
    print(f"  Total to dismiss: {len(to_dismiss)}")

    if dry_run:
        print("Dry run — no changes made.")
    elif to_dismiss:
        db.dismiss_faces(to_dismiss)
        print(f"Dismissed {len(to_dismiss)} faces.")

    db.close()


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8787, type=int, help="Port to listen on")
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the web UI for browsing and naming faces."""
    import uvicorn

    from .app import create_app

    app = create_app(ctx.obj["db_path"], photos_dir=ctx.obj["photos_dir"])
    print(f"Ritrova → http://localhost:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option("--output", "-o", default="-", help="Output file (- for stdout)")
@click.pass_context
def export(ctx: click.Context, output: str) -> None:
    """Export database as JSON."""
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    data = db.export_json()

    if output == "-":
        print(data)
    else:
        Path(output).write_text(data)
        print(f"Exported to {output}")
    db.close()


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show database statistics."""
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    s = db.get_stats()

    print(f"Photos scanned:    {s['total_photos']}")
    print(f"Faces detected:    {s['total_faces']}")
    print(f"Persons named:     {s['total_persons']}")
    print(f"Named faces:       {s['named_faces']}")
    print(f"Unnamed clusters:  {s['unnamed_clusters']}")
    print(f"Unclustered faces: {s['unclustered_faces']}")
    db.close()


@cli.command("backfill-gps")
@click.pass_context
def backfill_gps(ctx: click.Context) -> None:
    """Read GPS from EXIF for all photos missing coordinates."""
    from .db import FaceDB
    from .scanner import get_exif_gps

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    rows = db.query("SELECT id, file_path FROM photos WHERE latitude IS NULL")
    updated = 0
    total = len(rows)
    for i, r in enumerate(rows, 1):
        pid, fp = r[0], r[1]
        resolved = db.resolve_path(fp)
        if not resolved.exists():
            continue
        gps = get_exif_gps(resolved)
        if gps:
            db.run(
                "UPDATE photos SET latitude = ?, longitude = ? WHERE id = ?",
                (gps[0], gps[1], pid),
            )
            updated += 1
        if i % 500 == 0 or i == total:
            print(f"\r  [{i}/{total}] updated={updated}", end="", flush=True)
    print(f"\nUpdated {updated} of {total} photos with GPS coordinates.")
    db.close()


@cli.command("migrate-paths")
@click.pass_context
def migrate_paths(ctx: click.Context) -> None:
    """Rewrite absolute paths in the DB to relative (using --photos-dir as base)."""
    from .db import FaceDB

    photos_dir = ctx.obj["photos_dir"]
    if not photos_dir:
        print("Error: --photos-dir is required for migration")
        raise SystemExit(1)

    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    base = str(db.base_dir) + "/"

    rows = db.query("SELECT id, file_path, video_path FROM photos")
    migrated = 0
    for r in rows:
        pid, fp, vp = r[0], r[1], r[2]
        new_fp = fp
        new_vp = vp

        if fp and fp.startswith(base):
            new_fp = fp[len(base) :]
        if vp and vp.startswith(base):
            new_vp = vp[len(base) :]

        if new_fp != fp or new_vp != vp:
            db.run(
                "UPDATE photos SET file_path = ?, video_path = ? WHERE id = ?",
                (new_fp, new_vp, pid),
            )
            migrated += 1

    print(f"Migrated {migrated} of {len(rows)} photo paths to relative.")
    db.close()
