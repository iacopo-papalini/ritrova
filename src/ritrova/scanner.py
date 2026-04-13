"""Photo and video scanning pipeline."""

import hashlib
import logging
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import Base as ExifBase

from .db import FaceDB
from .detector import FaceDetector

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def get_exif_date(image_path: Path) -> str | None:
    """Extract date taken from EXIF data."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            if exif:
                date_str = exif.get(ExifBase.DateTimeOriginal) or exif.get(ExifBase.DateTime)
                if date_str:
                    return str(date_str)
    except OSError:
        pass
    return None


def _dms_to_decimal(dms: tuple[float, ...], ref: str) -> float:
    """Convert degrees/minutes/seconds to decimal degrees."""
    d, m, s = (float(x) for x in dms)
    decimal = d + m / 60 + s / 3600
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal


_GPS_IFD = 0x8825


def get_exif_gps(image_path: Path) -> tuple[float, float] | None:
    """Extract GPS lat/lon from EXIF. Returns (lat, lon) or None."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            gps_ifd = exif.get_ifd(_GPS_IFD)
            if not gps_ifd:
                return None
            # GPS IFD tags: 1=LatRef, 2=Lat, 3=LonRef, 4=Lon
            lat_ref = gps_ifd.get(1)
            lat_dms = gps_ifd.get(2)
            lon_ref = gps_ifd.get(3)
            lon_dms = gps_ifd.get(4)
            if not (lat_ref and lat_dms and lon_ref and lon_dms):
                return None
            lat = _dms_to_decimal(lat_dms, lat_ref)
            lon = _dms_to_decimal(lon_dms, lon_ref)
            return (lat, lon)
    except OSError:
        return None


EXCLUDE_MARKER = ".fr_exclude"


def _excluded_dirs(root: Path) -> set[Path]:
    """Find all directories under root containing a .fr_exclude marker file."""
    return {p.parent.resolve() for p in root.rglob(EXCLUDE_MARKER)}


def _is_excluded(path: Path, excluded: set[Path]) -> bool:
    """Check if path is inside any excluded directory."""
    resolved = path.resolve()
    return any(resolved == ex or ex in resolved.parents for ex in excluded)


def find_images(photos_dir: Path) -> list[Path]:
    """Find all image files in directory tree, skipping dirs with .fr_exclude."""
    excluded = _excluded_dirs(photos_dir)
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(photos_dir.rglob(f"*{ext}"))
        images.extend(photos_dir.rglob(f"*{ext.upper()}"))
    seen: set[str] = set()
    unique = []
    for p in sorted(images):
        key = str(p).lower()
        if key not in seen:
            if excluded and _is_excluded(p, excluded):
                continue
            seen.add(key)
            unique.append(p)
    return unique


def find_videos(photos_dir: Path) -> list[Path]:
    """Find all video files in directory tree, skipping dirs with .fr_exclude."""
    excluded = _excluded_dirs(photos_dir)
    videos: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(photos_dir.rglob(f"*{ext}"))
        videos.extend(photos_dir.rglob(f"*{ext.upper()}"))
    seen: set[str] = set()
    unique = []
    for p in sorted(videos):
        key = str(p).lower()
        if key not in seen:
            if excluded and _is_excluded(p, excluded):
                continue
            seen.add(key)
            unique.append(p)
    return unique


def scan_one_photo_for_human(
    db: FaceDB,
    image_path: Path,
    detector: FaceDetector,
    min_confidence: float = 0.65,
) -> tuple[bool, int]:
    """Run a human-face scan on a single photo. Returns (success, n_faces).

    Caller is responsible for `is_scanned()` gating (bulk loop) or for
    `delete_scan()`-ing the prior scan first (rescan command). On success a
    new scan row is recorded and findings are linked to it.
    """
    stored_path = db.to_relative(str(image_path.resolve()))
    try:
        detected_faces, width, height = detector.detect(image_path)
    except OSError:
        logger.warning("Error reading %s", image_path)
        return (False, 0)
    if width == 0 and height == 0:
        return (False, 0)

    taken_at = get_exif_date(image_path)
    gps = get_exif_gps(image_path)
    lat, lon = gps if gps else (None, None)
    source_id = db.get_or_create_source(
        stored_path,
        source_type="photo",
        width=width,
        height=height,
        taken_at=taken_at,
        latitude=lat,
        longitude=lon,
    )
    scan_id = db.record_scan(source_id, "human", detection_strategy="arcface_v1")

    batch = [
        (source_id, face["bbox"], face["embedding"], face["confidence"])
        for face in detected_faces
        if cast(float, face["confidence"]) >= min_confidence
    ]
    if batch:
        db.add_findings_batch(batch, scan_id=scan_id)
    return (True, len(batch))


def scan_photos(
    db: FaceDB,
    photos_dir: Path,
    detector: FaceDetector,
    min_confidence: float = 0.65,
) -> dict[str, int]:
    """Scan all photos, detect faces, store in DB. Returns stats dict."""
    images = find_images(photos_dir)
    total = len(images)
    logger.info("Found %d images to process", total)

    skipped = 0
    processed = 0
    faces_found = 0
    errors = 0

    for i, image_path in enumerate(images, 1):
        stored_path = db.to_relative(str(image_path.resolve()))

        if db.is_scanned(stored_path, "human"):
            skipped += 1
            if i % 500 == 0 or i == total:
                print(
                    f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                    f"faces={faces_found} errors={errors}",
                    end="",
                    flush=True,
                )
            continue

        success, n_faces = scan_one_photo_for_human(db, image_path, detector, min_confidence)
        if not success:
            errors += 1
        else:
            faces_found += n_faces
            processed += 1

        if i % 100 == 0 or i == total:
            print(
                f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                f"faces={faces_found} errors={errors}",
                end="",
                flush=True,
            )

    print()
    return {
        "processed": processed,
        "skipped": skipped,
        "faces_found": faces_found,
        "errors": errors,
    }


def _is_duplicate(embedding: np.ndarray, seen: list[np.ndarray], threshold: float = 0.6) -> bool:
    """Check if this face embedding is too similar to one already seen."""
    return any(float(embedding @ emb) > threshold for emb in seen)


def _update_or_add_face(
    unique_faces: list[dict[str, Any]],
    emb: np.ndarray,
    bbox: tuple[int, int, int, int],
    conf: float,
    frame: np.ndarray,
    w: int,
    h: int,
    dedup_threshold: float,
) -> None:
    """Add a face to the unique list, or update if it's a higher-confidence duplicate."""
    seen_embs = [uf["embedding"] for uf in unique_faces]
    if _is_duplicate(emb, seen_embs, dedup_threshold):
        for uf in unique_faces:
            if float(emb @ uf["embedding"]) > dedup_threshold and conf > uf["confidence"]:
                uf.update(
                    frame=frame.copy(),
                    bbox=bbox,
                    confidence=conf,
                    embedding=emb,
                    width=w,
                    height=h,
                )
                break
    else:
        unique_faces.append(
            {
                "frame": frame.copy(),
                "bbox": bbox,
                "embedding": emb,
                "confidence": conf,
                "width": w,
                "height": h,
            }
        )


def _extract_video_faces(
    video_path: str,
    detector: FaceDetector,
    min_confidence: float,
    interval_sec: float,
    dedup_threshold: float,
) -> list[dict[str, Any]] | None:
    """Extract unique faces from a video. Returns None on error."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        return None

    frame_interval = max(1, int(fps * interval_sec))
    unique_faces: list[dict[str, Any]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        raw_faces = detector.app.get(frame)

        for face in raw_faces:
            conf = float(face.det_score)
            if conf < min_confidence:
                continue
            emb = face.normed_embedding.astype(np.float32)
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            _update_or_add_face(unique_faces, emb, bbox, conf, frame, w, h, dedup_threshold)

        frame_idx += 1

    cap.release()
    return unique_faces


def scan_one_video_for_human(
    db: FaceDB,
    video_path: Path,
    detector: FaceDetector,
    frames_dir: Path,
    min_confidence: float = 0.65,
    interval_sec: float = 2.0,
    dedup_threshold: float = 0.6,
) -> tuple[bool, int]:
    """Run a human-face scan on a single video. Returns (success, n_faces)."""
    abs_video = str(video_path.resolve())
    stored_video = db.to_relative(abs_video)
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        unique_faces = _extract_video_faces(
            abs_video, detector, min_confidence, interval_sec, dedup_threshold
        )
    except OSError:
        logger.warning("Error processing video %s", video_path)
        return (False, 0)
    if unique_faces is None:
        return (False, 0)

    h_frame, w_frame = (
        (unique_faces[0]["height"], unique_faces[0]["width"]) if unique_faces else (0, 0)
    )
    source_id = db.get_or_create_source(
        stored_video, source_type="video", width=w_frame, height=h_frame
    )
    scan_id = db.record_scan(source_id, "human", detection_strategy="arcface_v1")

    vid_hash = hashlib.md5(abs_video.encode()).hexdigest()[:10]
    for j, uf in enumerate(unique_faces):
        frame_file = frames_dir / f"vid_{vid_hash}_{j}.jpg"
        rgb = cv2.cvtColor(uf["frame"], cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(frame_file), "JPEG", quality=85)
        rel_frame = str(frame_file.relative_to(db.db_path.parent))
        db.add_findings_batch(
            [(source_id, uf["bbox"], uf["embedding"], uf["confidence"])],
            scan_id=scan_id,
            frame_path=rel_frame,
        )
    return (True, len(unique_faces))


def scan_videos(
    db: FaceDB,
    photos_dir: Path,
    detector: FaceDetector,
    frames_dir: Path,
    min_confidence: float = 0.65,
    interval_sec: float = 2.0,
    dedup_threshold: float = 0.6,
) -> dict[str, int]:
    """Scan videos: extract frames, detect faces, deduplicate per video."""
    videos = find_videos(photos_dir)
    total = len(videos)
    logger.info("Found %d videos to process", total)

    skipped = 0
    processed = 0
    faces_found = 0
    errors = 0

    for i, video_path in enumerate(videos, 1):
        abs_video = str(video_path.resolve())
        stored_video = db.to_relative(abs_video)

        if db.is_scanned(stored_video, "human"):
            skipped += 1
            if i % 50 == 0 or i == total:
                print(
                    f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                    f"faces={faces_found} errors={errors}",
                    end="",
                    flush=True,
                )
            continue

        success, n_faces = scan_one_video_for_human(
            db, video_path, detector, frames_dir, min_confidence, interval_sec, dedup_threshold
        )
        if not success:
            errors += 1
        else:
            faces_found += n_faces
            processed += 1

        if i % 10 == 0 or i == total:
            print(
                f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                f"faces={faces_found} errors={errors}",
                end="",
                flush=True,
            )

    print()
    return {
        "processed": processed,
        "skipped": skipped,
        "faces_found": faces_found,
        "errors": errors,
    }


def scan_one_photo_for_pets(
    db: FaceDB,
    image_path: Path,
    pet_detector: Any,
    min_confidence: float = 0.5,
) -> tuple[bool, int]:
    """Run a pet scan on a single photo. Returns (success, n_pets)."""
    from PIL import Image as _Image
    from PIL import ImageOps as _ImageOps

    stored_path = db.to_relative(str(image_path.resolve()))
    try:
        detected = pet_detector.detect(image_path)
        good = [p for p in detected if p["confidence"] >= min_confidence]
        with _Image.open(image_path) as img:
            oriented = _ImageOps.exif_transpose(img)
            w, h = oriented.size
    except OSError, ValueError:
        logger.warning("Error processing %s", image_path)
        return (False, 0)

    source_id = db.get_or_create_source(stored_path, source_type="photo", width=w, height=h)
    scan_id = db.record_scan(source_id, "pet", detection_strategy="siglip_v1")
    for pet in good:
        db.add_findings_batch(
            [(source_id, pet["bbox"], pet["embedding"], pet["confidence"])],
            scan_id=scan_id,
            species=pet["species"],
        )
    return (True, len(good))


def scan_pets(
    db: FaceDB,
    photos_dir: Path,
    pet_detector: Any,
    min_confidence: float = 0.5,
) -> dict[str, int]:
    """Scan all photos for dogs and cats using YOLO + SigLIP."""
    images = find_images(photos_dir)
    total = len(images)
    logger.info("Found %d images to scan for pets", total)

    skipped = 0
    processed = 0
    pets_found = 0
    errors = 0

    for i, image_path in enumerate(images, 1):
        stored_path = db.to_relative(str(image_path.resolve()))

        if db.is_scanned(stored_path, "pet"):
            skipped += 1
            if i % 500 == 0 or i == total:
                print(
                    f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                    f"pets={pets_found} errors={errors}",
                    end="",
                    flush=True,
                )
            continue

        success, n_pets = scan_one_photo_for_pets(db, image_path, pet_detector, min_confidence)
        if not success:
            errors += 1
        else:
            pets_found += n_pets
            processed += 1

        if i % 50 == 0 or i == total:
            print(
                f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                f"pets={pets_found} errors={errors}",
                end="",
                flush=True,
            )

    print()
    return {
        "processed": processed,
        "skipped": skipped,
        "pets_found": pets_found,
        "errors": errors,
    }
