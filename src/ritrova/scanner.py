"""Photo and video file discovery + EXIF extraction.

The single-source scanning helpers (``scan_one_photo_for_human`` et al)
were retired in ADR-012 §M3 step 4 — every write path now goes through
``AnalysisPipeline`` + ``AnalysisPersister`` in ``analysis.py``. What
remains here: walking the photo tree for ``ritrova analyse`` +
``rescan``, and the EXIF readers used by the ingest side.
"""

import logging
from pathlib import Path

from PIL import Image
from PIL.ExifTags import Base as ExifBase

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
    """Extract GPS lat/lon from EXIF. Returns (lat, lon) or None.

    Some cameras and editing tools write malformed GPS tags (e.g. an
    IFDRational with a zero denominator). PIL only raises when we cast to
    float, deep inside the per-image scan loop — without this guard one bad
    photo crashes the whole scan-photos pass.
    """
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
    except OSError, ZeroDivisionError, ValueError, TypeError:
        return None


EXCLUDE_MARKER = ".fr_exclude"


def _excluded_dirs(root: Path) -> set[Path]:
    """Find all directories under root containing a .fr_exclude marker file."""
    return {p.parent.resolve() for p in root.rglob(EXCLUDE_MARKER)}


def _is_excluded(path: Path, excluded: set[Path]) -> bool:
    """Check if path is inside any excluded directory."""
    resolved = path.resolve()
    return any(resolved == ex or ex in resolved.parents for ex in excluded)


def _is_skippable(path: Path) -> bool:
    """True if a discovered file should be skipped (zero bytes, broken symlink, …)."""
    try:
        return path.stat().st_size == 0
    except OSError:
        # stat failed: broken symlink, permissions, vanished mid-scan — skip silently.
        return True


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
            if _is_skippable(p):
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
            if _is_skippable(p):
                continue
            seen.add(key)
            unique.append(p)
    return unique
