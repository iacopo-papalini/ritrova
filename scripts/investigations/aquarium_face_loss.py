"""Investigate face-detection discrepancy between legacy and current pipelines.

Observed: on certain photos (notably the 2015 Aquarium di Genova series), the
legacy `human` scan from ~a week ago recorded 3 faces, but today's `subjects`
scan records 0. This script reproduces the gap on one photo and splits the
question into testable sub-problems:

  1. Are `FaceDetector.detect(path)` and `FaceDetector.detect_image(pil)`
     equivalent today? (They take different inputs but should converge on the
     same BGR array inside ArcFace.)
  2. What does the raw ArcFace backbone (`self.app.get(bgr)`) return for
     this photo, BEFORE any MIN_FACE_SIZE / sharpness / edge-density filters
     are applied?
  3. Of those raw faces, which ones pass each filter and which ones get
     dropped? Which filter is responsible for the "lost" faces?

Run:  uv run python scripts/investigations/aquarium_face_loss.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# Silence model-loading chatter so the signal is easy to read.
for noisy in ("insightface", "onnxruntime", "ultralytics"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from ritrova.detector import (  # noqa: E402
    _BASE_SHARPNESS,
    _REF_AREA,
    MIN_FACE_SIZE,
    FaceDetector,
)

PHOTOS_ROOT = Path("/Users/iacopo.papalini/nextCloud/Photos")

# Four aquarium photos where legacy recorded faces and subjects recorded 0.
CASES: list[tuple[Path, int]] = [
    (PHOTOS_ROOT / "2015/2015-05-17.Acquario.Genova/20150517_132917.jpg", 3),
    (PHOTOS_ROOT / "2015/2015-05-17.Acquario.Genova/20150517_132916(0).jpg", 3),
    (PHOTOS_ROOT / "2015/2015-05-17.Acquario.Genova/20150517_132915.jpg", 2),
    (PHOTOS_ROOT / "2015/2015-05-17.Acquario.Genova/20150517_131642.jpg", 1),
]


def load_bgr_legacy(path: Path) -> np.ndarray:
    """Reproduce FaceDetector._load_image exactly."""
    raw = Image.open(path)
    oriented = ImageOps.exif_transpose(raw)
    img = oriented.convert("RGB")
    return np.array(img)[:, :, ::-1]


def load_bgr_pipeline(path: Path) -> np.ndarray:
    """Reproduce what photo_frames() hands to FaceDetectionStep."""
    raw = Image.open(path)
    img = ImageOps.exif_transpose(raw)
    if img is None:
        img = raw
    img = img.convert("RGB")
    return np.array(img)[:, :, ::-1]


def explain_filter_outcome(img: np.ndarray, bbox_int: tuple[int, int, int, int]) -> str:
    x1, y1, x2, y2 = bbox_int
    fw, fh = x2 - x1, y2 - y1
    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return f"REJECTED[min_face_size]: {fw}x{fh} < {MIN_FACE_SIZE}"
    area = fw * fh
    scale = _REF_AREA / max(area, _REF_AREA)
    crop_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(crop_gray, cv2.CV_64F).var()
    sharpness_threshold = _BASE_SHARPNESS * scale
    if sharpness < sharpness_threshold:
        return (
            f"REJECTED[sharpness]: {sharpness:.2f} < {sharpness_threshold:.4f} "
            f"(area={area}, scale={scale:.4f})"
        )
    # Compute the (now-disabled) Canny edge density for context only.
    edges = cv2.Canny(crop_gray, 50, 150)
    edge_density = (edges > 0).mean()
    return (
        f"KEPT     sharpness={sharpness:.1f} (thr={sharpness_threshold:.3f})  "
        f"edges(info-only)={edge_density:.4f}"
    )


def main() -> None:
    print("Loading FaceDetector…")
    detector = FaceDetector()

    for path, legacy_count in CASES:
        print("\n" + "=" * 78)
        print(f"{path.relative_to(PHOTOS_ROOT)}  (legacy recorded {legacy_count} faces)")
        print("=" * 78)

        # Byte-for-byte check of the two input paths.
        legacy_bgr = load_bgr_legacy(path)
        pipeline_bgr = load_bgr_pipeline(path)
        identical = np.array_equal(legacy_bgr, pipeline_bgr)
        print(
            f"  load equivalence: legacy vs pipeline BGR arrays identical = {identical}  "
            f"(shape={legacy_bgr.shape})"
        )
        if not identical:
            diff = legacy_bgr.astype(int) - pipeline_bgr.astype(int)
            print(f"    max|diff|={np.abs(diff).max()}  n_diff={int((diff != 0).any(-1).sum())}")

        # Raw ArcFace output, before any of the quality filters.
        raw = detector.app.get(legacy_bgr)
        print(f"  raw ArcFace faces (pre-filter): {len(raw)}")
        h, w = legacy_bgr.shape[:2]
        for i, face in enumerate(raw):
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            verdict = explain_filter_outcome(legacy_bgr, (x1, y1, x2, y2))
            print(
                f"    face {i}: bbox=({x1},{y1},{x2 - x1},{y2 - y1})  "
                f"det={face.det_score:.3f}  {verdict}"
            )

        # Both top-level methods, to confirm they collapse to the same count.
        r_legacy = detector.detect(path)
        r_new = detector.detect_image(Image.fromarray(pipeline_bgr[:, :, ::-1]))
        print(f"  detect(path):        {len(r_legacy.detections)} faces kept")
        print(f"  detect_image(pil):   {len(r_new.detections)} faces kept")


if __name__ == "__main__":
    main()
