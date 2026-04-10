"""Image serving utilities: raw streaming, resizing, thumbnail cropping."""

import io
from pathlib import Path

from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True


def stream_raw(resolved: Path) -> tuple[io.BufferedReader, str]:
    """Return file handle + media type for raw file streaming."""
    media = "image/jpeg" if resolved.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return open(resolved, "rb"), media  # noqa: SIM115


def resize_photo(resolved: Path, max_size: int) -> io.BytesIO:
    """Load, orient, resize a photo. Returns JPEG bytes in a BytesIO buffer."""
    with Image.open(resolved) as raw_img:
        oriented = ImageOps.exif_transpose(raw_img)
        oriented.thumbnail((max_size, max_size))
        buf = io.BytesIO()
        oriented.save(buf, "JPEG", quality=85)
    buf.seek(0)
    return buf


def crop_face_thumbnail(
    resolved: Path, bbox_x: int, bbox_y: int, bbox_w: int, bbox_h: int, size: int
) -> io.BytesIO:
    """Crop a face region with padding, resize to thumbnail. Returns JPEG bytes."""
    with Image.open(resolved) as raw_img:
        oriented = ImageOps.exif_transpose(raw_img)
        pad_w = bbox_w * 0.3
        pad_h = bbox_h * 0.3
        x1 = max(0, bbox_x - pad_w)
        y1 = max(0, bbox_y - pad_h)
        x2 = min(oriented.width, bbox_x + bbox_w + pad_w)
        y2 = min(oriented.height, bbox_y + bbox_h + pad_h)
        crop = oriented.crop((int(x1), int(y1), int(x2), int(y2)))
    crop.thumbnail((size, size))
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=85)
    buf.seek(0)
    return buf
