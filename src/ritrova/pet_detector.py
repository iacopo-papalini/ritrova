"""Pet detection (YOLO) and embedding extraction (SigLIP) for dogs and cats."""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, ImageOps
from ultralytics import YOLO  # type: ignore[attr-defined]

from .detection import Detection, DetectionResult

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

# COCO class IDs: 15 = cat, 16 = dog
COCO_CAT = 15
COCO_DOG = 16
SPECIES_MAP = {COCO_CAT: "cat", COCO_DOG: "dog"}


class PetDetector:
    def __init__(
        self,
        yolo_model: str = "data/yolo11m.pt",
        siglip_model: str = "google/siglip-base-patch16-224",
    ):
        # YOLO for detection
        self.yolo = YOLO(yolo_model)

        # SigLIP for embeddings
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(siglip_model)  # type: ignore[no-untyped-call]
        self.siglip = AutoModel.from_pretrained(siglip_model)
        self.siglip.eval()
        if torch.backends.mps.is_available():
            self.siglip = self.siglip.to("mps")
            self.device = "mps"
        else:
            self.device = "cpu"

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect dogs and cats in an image.

        Returns a ``DetectionResult`` with each pet's bounding box, embedding,
        confidence, and species, plus the source image dimensions.
        """
        try:
            img_raw = Image.open(image_path)
            img = ImageOps.exif_transpose(img_raw)
            img_rgb = img.convert("RGB")
        except OSError:
            logger.warning("Could not read image: %s", image_path)
            return DetectionResult(detections=[], width=0, height=0)

        w_img, h_img = img_rgb.size

        # YOLO detection — filter for cat/dog only
        results = self.yolo(img_rgb, verbose=False, classes=[COCO_CAT, COCO_DOG])

        pets: list[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                species = SPECIES_MAP.get(cls_id)
                if not species:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w_img, int(x2)), min(h_img, int(y2))

                crop = img_rgb.crop((x1, y1, x2, y2))
                embedding = self._embed(crop)

                pets.append(
                    Detection(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        embedding=embedding,
                        confidence=conf,
                        species=species,
                    )
                )

        return DetectionResult(detections=pets, width=w_img, height=h_img)

    def _embed(self, crop: Image.Image) -> np.ndarray:
        """Get SigLIP embedding for a cropped pet image."""
        inputs = self.processor(images=crop, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items() if v is not None}
        with torch.no_grad():
            vision_outputs = self.siglip.vision_model(**inputs)
            # Pool: use the CLS-like pooler output, or mean-pool patch tokens
            emb = vision_outputs.pooler_output
            if emb is None:
                # Fallback: mean pool the last hidden state
                emb = vision_outputs.last_hidden_state.mean(dim=1)
        emb = emb[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb  # type: ignore[no-any-return]
