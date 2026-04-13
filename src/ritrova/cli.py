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
@click.option("--kind", default="person", help="Subject kind: person or pet")
@click.pass_context
def auto_assign(ctx: click.Context, min_similarity: float, kind: str) -> None:
    """Bulk-assign unnamed clusters to existing named subjects."""
    from .cluster import auto_assign as _auto_assign
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    result = _auto_assign(db, min_similarity=min_similarity / 100, kind=kind)

    print(
        f"\nAssigned {result['assigned_clusters']} clusters "
        f"({result['assigned_faces']} faces), "
        f"{result['unmatched']} unmatched"
    )
    db.close()


@cli.command()
@click.option("--min-similarity", default=70.0, help="Minimum centroid similarity %")
@click.option("--kind", default="person", help="Subject kind: person or pet")
@click.pass_context
def auto_merge(ctx: click.Context, min_similarity: float, kind: str) -> None:
    """Auto-merge unnamed clusters whose centroids are highly similar."""
    from .cluster import auto_merge_clusters
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    result = auto_merge_clusters(db, min_similarity=min_similarity / 100, kind=kind)

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
        "SELECT f.id, f.source_id, f.bbox_x, f.bbox_y, f.bbox_w, f.bbox_h "
        "FROM findings f "
        "WHERE f.person_id IS NULL "
        "AND f.id NOT IN (SELECT finding_id FROM dismissed_findings)"
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
        photo = db.get_source(pid)
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
        db.dismiss_findings(to_dismiss)
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

    print(f"Sources scanned:   {s['total_sources']}")
    print(f"Findings detected: {s['total_findings']}")
    print(f"Subjects named:    {s['total_subjects']}")
    print(f"Named findings:    {s['named_findings']}")
    print(f"Unnamed clusters:  {s['unnamed_clusters']}")
    print(f"Unclustered:       {s['unclustered_findings']}")
    db.close()


@cli.group()
def scans() -> None:
    """Inspect and prune scan records (and the findings they own)."""


@scans.command("list")
@click.option("--source-pattern", default=None, help="GLOB pattern on source path (e.g. '2024/*')")
@click.pass_context
def scans_list(ctx: click.Context, source_pattern: str | None) -> None:
    """List scans, optionally filtered by a source path pattern."""
    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    rows = db.find_scans(source_pattern=source_pattern)
    if not rows:
        click.echo("(no scans match)")
    else:
        click.echo(f"{'id':>6}  {'type':6s}  {'findings':>8}  {'scanned_at':24s}  source")
        for r in rows:
            click.echo(
                f"{r['id']:>6}  {r['scan_type']:6s}  {r['finding_count']:>8}  "
                f"{r['scanned_at']:24s}  {r['source_path']}"
            )
        click.echo(f"\n{len(rows)} scan(s)")
    db.close()


@scans.command("prune")
@click.option("--scan-id", type=int, default=None, help="Prune a single scan by id")
@click.option(
    "--source-pattern",
    default=None,
    help="GLOB pattern on source path; prunes every matching scan",
)
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
@click.pass_context
def scans_prune(
    ctx: click.Context, scan_id: int | None, source_pattern: str | None, yes: bool
) -> None:
    """Prune scans (and their findings via cascade). Requires --scan-id and/or --source-pattern."""
    if scan_id is None and source_pattern is None:
        msg = "Provide --scan-id and/or --source-pattern (refusing to prune everything)."
        raise click.UsageError(msg)

    from .db import FaceDB

    db = FaceDB(ctx.obj["db_path"], base_dir=ctx.obj["photos_dir"])
    targets = db.find_scans(scan_id=scan_id, source_pattern=source_pattern)
    if not targets:
        click.echo("(no scans match)")
        db.close()
        return

    n_findings = sum(t["finding_count"] for t in targets)
    n_assigned = db.query(
        f"SELECT COUNT(*) FROM findings WHERE person_id IS NOT NULL AND scan_id IN "  # noqa: S608
        f"({','.join(str(t['id']) for t in targets)})"
    )[0][0]

    click.echo(f"About to prune {len(targets)} scan(s):")
    for t in targets[:10]:
        click.echo(f"  scan {t['id']:>6} [{t['scan_type']}]  {t['source_path']}")
    if len(targets) > 10:
        click.echo(f"  ... and {len(targets) - 10} more")
    click.echo(
        f"\nThis will delete {n_findings} finding(s), of which "
        f"{n_assigned} have manual subject assignments."
    )
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        db.close()
        return

    for t in targets:
        db.delete_scan(t["id"])
    click.echo(f"Pruned {len(targets)} scan(s) ({n_findings} findings).")
    db.close()


@cli.command()
@click.argument("source")
@click.option(
    "--scan-type",
    default="all",
    type=click.Choice(["all", "human", "pet"]),
    help="Which scan to redo (default all existing on the source)",
)
@click.option("-y", "--yes", is_flag=True, help="Skip the confirmation prompt")
@click.pass_context
def rescan(ctx: click.Context, source: str, scan_type: str, yes: bool) -> None:
    """Rescan a single source: delete existing scans + findings, run fresh detection."""
    from .db import FaceDB

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)

    # Resolve to a stored path (relative to PHOTOS_DIR) and the absolute path.
    abs_path = Path(source)
    if not abs_path.is_absolute():
        abs_path = (Path(photos_dir) / source).resolve()
    if not abs_path.exists():
        raise click.UsageError(f"File not found: {abs_path}")
    stored_path = db.to_relative(str(abs_path))
    src = db.get_source_by_path(stored_path)
    if src is None:
        raise click.UsageError(
            f"Source not in DB: {stored_path}\nRun `ritrova scan` (or scan-pets/scan-videos) "
            "to ingest new files first."
        )

    existing = [s for s in db.find_scans() if s["source_id"] == src.id]
    if scan_type != "all":
        existing = [s for s in existing if s["scan_type"] == scan_type]
    if not existing:
        click.echo(f"(no matching scans on {stored_path})")
        db.close()
        return

    n_findings = sum(t["finding_count"] for t in existing)
    n_assigned = db.query(
        f"SELECT COUNT(*) FROM findings WHERE person_id IS NOT NULL AND scan_id IN "  # noqa: S608
        f"({','.join(str(t['id']) for t in existing)})"
    )[0][0]
    click.echo(
        f"About to rescan {stored_path}\n"
        f"Replacing {len(existing)} scan(s): {sorted({t['scan_type'] for t in existing})}\n"
        f"This will delete {n_findings} finding(s), of which "
        f"{n_assigned} have manual subject assignments."
    )
    if not yes and not click.confirm("Proceed?", default=False):
        click.echo("Aborted.")
        db.close()
        return

    types_to_redo = {t["scan_type"] for t in existing}
    for t in existing:
        db.delete_scan(t["id"])

    # Re-run only the appropriate single-source scanners.
    from .scanner import (
        scan_one_photo_for_human,
        scan_one_photo_for_pets,
        scan_one_video_for_human,
    )

    if src.type == "video":
        if "human" in types_to_redo:
            from .detector import FaceDetector

            detector = FaceDetector()
            frames_dir = Path(ctx.obj["db_path"]).parent / "tmp" / "frames"
            ok, n = scan_one_video_for_human(db, abs_path, detector, frames_dir)
            click.echo(f"  video human scan: {'ok' if ok else 'FAILED'}, {n} face(s)")
    else:
        if "human" in types_to_redo:
            from .detector import FaceDetector

            detector = FaceDetector()
            ok, n = scan_one_photo_for_human(db, abs_path, detector)
            click.echo(f"  photo human scan: {'ok' if ok else 'FAILED'}, {n} face(s)")
        if "pet" in types_to_redo:
            from .pet_detector import PetDetector

            pet_detector = PetDetector()
            ok, n = scan_one_photo_for_pets(db, abs_path, pet_detector)
            click.echo(f"  photo pet scan:   {'ok' if ok else 'FAILED'}, {n} pet(s)")

    db.close()


@cli.command("backfill-gps")
@click.pass_context
def backfill_gps(ctx: click.Context) -> None:
    """Read GPS from EXIF for all photos missing coordinates."""
    from .db import FaceDB
    from .scanner import get_exif_gps

    photos_dir = _require_photos_dir(ctx)
    db = FaceDB(ctx.obj["db_path"], base_dir=photos_dir)
    rows = db.query("SELECT id, file_path FROM sources WHERE latitude IS NULL")
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
                "UPDATE sources SET latitude = ?, longitude = ? WHERE id = ?",
                (gps[0], gps[1], pid),
            )
            updated += 1
        if i % 500 == 0 or i == total:
            print(f"\r  [{i}/{total}] updated={updated}", end="", flush=True)
    print(f"\nUpdated {updated} of {total} sources with GPS coordinates.")
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

    rows = db.query("SELECT id, file_path FROM sources")
    migrated = 0
    for r in rows:
        sid, fp = r[0], r[1]
        if fp and fp.startswith(base):
            new_fp = fp[len(base) :]
            db.run("UPDATE sources SET file_path = ? WHERE id = ?", (new_fp, sid))
            migrated += 1

    print(f"Migrated {migrated} of {len(rows)} source paths to relative.")
    db.close()
