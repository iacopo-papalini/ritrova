"""Pet detection (YOLO) and embedding extraction (SigLIP) for dogs and cats."""

import logging
from pathlib import Path

import numpy as np
import torch
import torchvision
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
        self.yolo_device = self._choose_yolo_device()

        # SigLIP for embeddings
        from transformers import AutoModel, AutoProcessor

        # backend="pil" replaces the deprecated ``use_fast=False`` in modern
        # transformers. SigLIP's PIL path gives slightly different (and more
        # stable) resize/normalise numerics than torchvision — keep PIL.
        self.processor = AutoProcessor.from_pretrained(siglip_model, backend="pil")  # type: ignore[no-untyped-call]
        self.siglip = AutoModel.from_pretrained(siglip_model)
        self.siglip.eval()
        if torch.cuda.is_available():
            self.siglip = self.siglip.to("cuda")
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.siglip = self.siglip.to("mps")
            self.device = "mps"
        else:
            self.device = "cpu"

    def _choose_yolo_device(self) -> str:
        """Pick the safest YOLO runtime device for this environment."""
        if torch.cuda.is_available() and not torchvision.__version__.endswith("+cpu"):
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            logger.warning(
                "CUDA torch detected but torchvision is CPU-only (%s); "
                "running YOLO pet detection on CPU to avoid CUDA NMS failures.",
                torchvision.__version__,
            )
        return "cpu"

    def detect(self, image_path: Path) -> DetectionResult:
        """Detect pets from a file path (opens the file).

        Use ``detect_image`` when you already have a loaded PIL Image.
        """
        try:
            img_raw = Image.open(image_path)
            img = ImageOps.exif_transpose(img_raw)
            img_rgb = img.convert("RGB")
        except OSError:
            logger.warning("Could not read image: %s", image_path)
            return DetectionResult(detections=[], width=0, height=0)
        return self.detect_image(img_rgb)

    def detect_image(self, pil_image: Image.Image) -> DetectionResult:
        """Detect dogs and cats from an already-loaded PIL RGB Image."""
        w_img, h_img = pil_image.size

        results = self.yolo(
            pil_image,
            verbose=False,
            classes=[COCO_CAT, COCO_DOG],
            device=self.yolo_device,
        )

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

                crop = pil_image.crop((x1, y1, x2, y2))
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
