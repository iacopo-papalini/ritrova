"""Face detection and embedding extraction using InsightFace."""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

# Some JPEGs are slightly truncated but still perfectly viewable
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


MIN_FACE_SIZE = 50  # pixels — smaller faces are unrecognizable
MIN_SHARPNESS = 30.0  # Laplacian variance — below this is out-of-focus
MIN_EDGE_DENSITY = 0.02  # Canny edge ratio — below this is motion blur / featureless


class FaceDetector:
    def __init__(self, det_size: int = 640):
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=(det_size, det_size))

    def _load_image(self, image_path: Path) -> np.ndarray | None:
        """Load image with EXIF orientation applied, return BGR array."""
        try:
            raw: Image.Image = Image.open(image_path)
            oriented: Image.Image = ImageOps.exif_transpose(raw)
            img = oriented.convert("RGB")
            return np.array(img)[:, :, ::-1]  # RGB -> BGR for insightface
        except OSError:
            logger.warning("Could not read image: %s", image_path)
            return None

    def detect(self, image_path: Path) -> tuple[list[dict[str, object]], int, int]:
        """Detect faces in an image.

        Returns:
            (faces, width, height) where each face dict has keys:
            bbox (x, y, w, h), embedding (np.ndarray), confidence (float)
        """
        img = self._load_image(image_path)
        if img is None:
            return [], 0, 0

        h, w = img.shape[:2]
        raw_faces = self.app.get(img)

        faces = []
        for face in raw_faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1

            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue

            crop_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(crop_gray, cv2.CV_64F).var() < MIN_SHARPNESS:
                continue
            edges = cv2.Canny(crop_gray, 50, 150)
            if (edges > 0).mean() < MIN_EDGE_DENSITY:
                continue

            faces.append(
                {
                    "bbox": (int(x1), int(y1), int(fw), int(fh)),
                    "embedding": face.normed_embedding.astype(np.float32),
                    "confidence": float(face.det_score),
                }
            )

        return faces, w, h
