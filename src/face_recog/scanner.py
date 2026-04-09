"""Photo and video scanning pipeline."""

import hashlib
import logging
from pathlib import Path

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
    images = []
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
    videos = []
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
) -> dict:
    """Scan all photos, detect faces, store in DB. Returns stats dict."""
    images = find_images(photos_dir)
    total = len(images)
    print(f"Found {total} images to process")

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
                if face["confidence"] >= min_confidence
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
    for emb in seen:
        if float(embedding @ emb) > threshold:
            return True
    return False


def scan_videos(
    db: FaceDB,
    photos_dir: Path,
    detector: FaceDetector,
    frames_dir: Path,
    min_confidence: float = 0.65,
    interval_sec: float = 2.0,
    dedup_threshold: float = 0.6,
) -> dict:
    """Scan videos: extract frames, detect faces, deduplicate per video."""
    videos = find_videos(photos_dir)
    total = len(videos)
    print(f"Found {total} videos to process")

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
            cap = cv2.VideoCapture(abs_video)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                cap.release()
                errors += 1
                continue

            frame_interval = max(1, int(fps * interval_sec))
            # Collect unique faces: embedding + best frame + best bbox + best conf
            unique_faces: list[dict] = []
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

                    # Check for duplicate
                    seen_embs = [uf["embedding"] for uf in unique_faces]
                    if _is_duplicate(emb, seen_embs, dedup_threshold):
                        # Update if this detection has higher confidence
                        for uf in unique_faces:
                            if float(emb @ uf["embedding"]) > dedup_threshold:
                                if conf > uf["confidence"]:
                                    uf["frame"] = frame.copy()
                                    uf["bbox"] = bbox
                                    uf["confidence"] = conf
                                    uf["embedding"] = emb
                                    uf["width"] = w
                                    uf["height"] = h
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

                frame_idx += 1

            cap.release()

            if not unique_faces:
                # No faces — still register so we skip next time
                vid_hash = hashlib.md5(abs_video.encode()).hexdigest()[:10]
                db.add_photo(f"__nofaces_{vid_hash}", 0, 0, video_path=abs_video)
                processed += 1
                continue

            # Save each unique face's frame as a JPEG and store in DB
            for j, uf in enumerate(unique_faces):
                vid_hash = hashlib.md5(abs_video.encode()).hexdigest()[:10]
                frame_filename = f"vid_{vid_hash}_{j}.jpg"
                frame_path = frames_dir / frame_filename
                # Convert BGR to RGB for saving
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
    pet_detector,
    min_confidence: float = 0.5,
) -> dict:
    """Scan all photos for dogs and cats using YOLO + SigLIP."""
    from PIL import Image as _Image, ImageOps as _ImageOps

    images = find_images(photos_dir)
    total = len(images)
    print(f"Found {total} images to scan for pets")

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
                img = _ImageOps.exif_transpose(img)
                w, h = img.size

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
