"""Face detection and embedding extraction using InsightFace."""

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile, ImageOps

from .detection import Detection, DetectionResult

# Some JPEGs are slightly truncated but still perfectly viewable
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


MIN_FACE_SIZE = 50  # pixels — smaller faces are unrecognizable

# Quality thresholds at the reference size (50x50 = 2500 px²).
# For larger faces, thresholds scale down proportionally to area — a 200x200
# face needs 1/16th the sharpness score of a 50x50 face to pass. This avoids
# rejecting large clear faces where smooth skin = low edge density.
_REF_AREA = 2500.0  # 50x50
_BASE_SHARPNESS = 30.0  # Laplacian variance at reference size
_BASE_EDGE_DENSITY = 0.02  # Canny edge ratio at reference size


class FaceDetector:
    def __init__(self, det_size: int = 640):
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        providers = [
            provider
            for provider in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
            if provider in ort.get_available_providers()
        ]
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
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

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect faces from a file path (opens the file).

        Use ``detect_image`` when you already have a loaded PIL Image.
        """
        img = self._load_image(image_path)
        if img is None:
            return DetectionResult(detections=[], width=0, height=0)
        return self._detect_bgr(img)

    def detect_image(self, pil_image: Image.Image) -> DetectionResult:
        """Detect faces from an already-loaded PIL RGB Image."""
        bgr = np.array(pil_image)[:, :, ::-1]
        return self._detect_bgr(bgr)

    def _detect_bgr(self, img: np.ndarray) -> DetectionResult:
        """Core detection on a BGR numpy array."""
        h, w = img.shape[:2]
        raw_faces = self.app.get(img)

        faces: list[Detection] = []
        for face in raw_faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            fw, fh = x2 - x1, y2 - y1

            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue

            area = fw * fh
            scale = _REF_AREA / max(area, _REF_AREA)
            crop_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(crop_gray, cv2.CV_64F).var() < _BASE_SHARPNESS * scale:
                continue
            edges = cv2.Canny(crop_gray, 50, 150)
            if (edges > 0).mean() < _BASE_EDGE_DENSITY * scale:
                continue

            faces.append(
                Detection(
                    bbox=(int(x1), int(y1), int(fw), int(fh)),
                    embedding=face.normed_embedding.astype(np.float32),
                    confidence=float(face.det_score),
                )
            )

        return DetectionResult(detections=faces, width=w, height=h)
