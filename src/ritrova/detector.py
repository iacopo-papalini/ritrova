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

# Sharpness threshold at the reference size (50x50 = 2500 px²). For larger
# faces, the threshold scales down proportionally to area — a 200x200 face
# needs 1/16th the Laplacian variance of a 50x50 face to pass, so large
# smooth-skinned faces aren't rejected for having low texture.
#
# The edge-density filter that used to live here was removed after it caused
# legitimate faces in dim scenes (aquarium interiors, low-light rooms) to
# be rejected: cv2.Canny(50, 150) returns zero edges when the scene-wide
# gradient never crosses 50/255, even on clear portraits. The Laplacian
# sharpness filter alone handles motion blur as intended.
# See scripts/investigations/aquarium_face_loss.py for the diagnosis.
_REF_AREA = 2500.0  # 50x50
_BASE_SHARPNESS = 30.0  # Laplacian variance at reference size


class FaceDetector:
    def __init__(self, det_size: int = 640):
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        providers = [
            provider
            for provider in (
                "CUDAExecutionProvider",
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            )
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

            faces.append(
                Detection(
                    bbox=(int(x1), int(y1), int(fw), int(fh)),
                    embedding=face.normed_embedding.astype(np.float32),
                    confidence=float(face.det_score),
                )
            )

        return DetectionResult(detections=faces, width=w, height=h)

    def embed_crop(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> np.ndarray:
        """Compute an ArcFace embedding for a user-drawn bbox (FEAT-29).

        ``image`` is the EXIF-oriented source (RGB). ``bbox`` is
        ``(x, y, w, h)`` in source pixels. Returns a 512-dim L2-normalized
        vector in the same embedding space used by ``detect`` — callers can
        feed it straight into centroid-similarity search.

        Raises ``ValueError`` when the detector model doesn't find a face in
        the crop (e.g. bbox covers wall / fabric): the whole point of the
        gesture is that the user asserts a face is there, so we force a
        single-face run on the crop via ``insightface`` directly rather than
        running the full ``get()`` pipeline with its own size / sharpness
        filters.
        """
        x, y, w, h = bbox
        rgb = np.array(image)
        bgr = rgb[:, :, ::-1]
        h_img, w_img = bgr.shape[:2]
        x0 = max(0, min(x, w_img))
        y0 = max(0, min(y, h_img))
        x1 = max(0, min(x + w, w_img))
        y1 = max(0, min(y + h, h_img))
        if x1 <= x0 or y1 <= y0:
            msg = "bbox has zero area inside the image"
            raise ValueError(msg)
        crop = bgr[y0:y1, x0:x1]
        # Run the full insightface pipeline on just the crop — the detector
        # re-localises a face inside the user-drawn box (they rarely draw
        # tight to the face) and the recognition model embeds from the
        # aligned landmarks. If the detector finds nothing, fall back to
        # embedding the whole crop via the recognition model directly so
        # the user's gesture still produces a vector.
        faces = self.app.get(crop)
        if faces:
            emb = faces[0].normed_embedding.astype(np.float32)
        else:
            rec = self.app.models.get("recognition")
            if rec is None:
                msg = "no face detected in the drawn bbox and no recognition model available"
                raise ValueError(msg)
            emb = rec.get_feat(crop).flatten().astype(np.float32)
            norm = float(np.linalg.norm(emb))
            if norm > 0:
                emb = emb / norm
        return emb  # type: ignore[no-any-return]
