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


def find_images(photos_dir: Path) -> list[Path]:
    """Find all image files in directory tree."""
    images: list[Path] = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(photos_dir.rglob(f"*{ext}"))
        images.extend(photos_dir.rglob(f"*{ext.upper()}"))
    seen: set[str] = set()
    unique = []
    for p in sorted(images):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def find_videos(photos_dir: Path) -> list[Path]:
    """Find all video files in directory tree."""
    videos: list[Path] = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(photos_dir.rglob(f"*{ext}"))
        videos.extend(photos_dir.rglob(f"*{ext.upper()}"))
    seen: set[str] = set()
    unique = []
    for p in sorted(videos):
        key = str(p).lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


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
        abs_path = str(image_path.resolve())

        if db.is_photo_scanned(abs_path):
            skipped += 1
            if i % 500 == 0 or i == total:
                print(
                    f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                    f"faces={faces_found} errors={errors}",
                    end="",
                    flush=True,
                )
            continue

        try:
            detected_faces, width, height = detector.detect(image_path)
            if width == 0 and height == 0:
                errors += 1
                continue

            taken_at = get_exif_date(image_path)
            photo_id = db.add_photo(abs_path, width, height, taken_at)

            batch = [
                (photo_id, face["bbox"], face["embedding"], face["confidence"])
                for face in detected_faces
                if cast(float, face["confidence"]) >= min_confidence
            ]

            if batch:
                db.add_faces_batch(batch)
                faces_found += len(batch)

            processed += 1

        except OSError:
            logger.warning("Error reading %s", image_path)
            errors += 1
            continue

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

    frames_dir.mkdir(parents=True, exist_ok=True)

    skipped = 0
    processed = 0
    faces_found = 0
    errors = 0

    for i, video_path in enumerate(videos, 1):
        abs_video = str(video_path.resolve())

        if db.is_video_scanned(abs_video):
            skipped += 1
            if i % 50 == 0 or i == total:
                print(
                    f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                    f"faces={faces_found} errors={errors}",
                    end="",
                    flush=True,
                )
            continue

        try:
            unique_faces = _extract_video_faces(
                abs_video,
                detector,
                min_confidence,
                interval_sec,
                dedup_threshold,
            )
            if unique_faces is None:
                errors += 1
                continue

            if not unique_faces:
                vid_hash = hashlib.md5(abs_video.encode()).hexdigest()[:10]
                db.add_photo(f"__nofaces_{vid_hash}", 0, 0, video_path=abs_video)
                processed += 1
                continue

            vid_hash = hashlib.md5(abs_video.encode()).hexdigest()[:10]
            for j, uf in enumerate(unique_faces):
                frame_path = frames_dir / f"vid_{vid_hash}_{j}.jpg"
                rgb = cv2.cvtColor(uf["frame"], cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(str(frame_path), "JPEG", quality=85)

                photo_id = db.add_photo(
                    str(frame_path),
                    uf["width"],
                    uf["height"],
                    video_path=abs_video,
                )
                db.add_faces_batch([(photo_id, uf["bbox"], uf["embedding"], uf["confidence"])])
                faces_found += 1

            processed += 1

        except OSError:
            logger.warning("Error processing video %s", video_path)
            errors += 1
            continue

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


def scan_pets(
    db: FaceDB,
    photos_dir: Path,
    pet_detector: Any,
    min_confidence: float = 0.5,
) -> dict[str, int]:
    """Scan all photos for dogs and cats using YOLO + SigLIP."""
    from PIL import Image as _Image
    from PIL import ImageOps as _ImageOps

    images = find_images(photos_dir)
    total = len(images)
    logger.info("Found %d images to scan for pets", total)

    skipped = 0
    processed = 0
    pets_found = 0
    errors = 0

    for i, image_path in enumerate(images, 1):
        abs_path = str(image_path.resolve())

        if db.is_pet_scanned(abs_path):
            skipped += 1
            if i % 500 == 0 or i == total:
                print(
                    f"\r  [{i}/{total}] processed={processed} skipped={skipped} "
                    f"pets={pets_found} errors={errors}",
                    end="",
                    flush=True,
                )
            continue

        try:
            detected = pet_detector.detect(image_path)
            good = [p for p in detected if p["confidence"] >= min_confidence]

            with _Image.open(image_path) as img:
                oriented = _ImageOps.exif_transpose(img)
                w, h = oriented.size

            photo_id = db.add_photo(abs_path + "__pets", w, h)

            for pet in good:
                db.add_faces_batch(
                    [(photo_id, pet["bbox"], pet["embedding"], pet["confidence"])],
                    species=pet["species"],
                )
                pets_found += 1

            processed += 1

        except Exception:
            logger.warning("Error processing %s", image_path)
            errors += 1
            continue

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
