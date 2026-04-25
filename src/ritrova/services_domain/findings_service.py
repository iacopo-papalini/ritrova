"""Findings-aggregate coordinator: manual bbox → finding row (FEAT-29).

``create_manual`` owns the user-asserted-bbox path: load the source image,
embed the crop in the correct vector space, insert a fresh ``findings``
row tied to the most-recent ``subjects`` scan on the source, rank it
against every named subject's centroid, and return both the row id and
the top suggestion (if any) plus an undo receipt whose payload deletes
the row.

The detector singletons are lazy — loaded on first use, kept for the
lifetime of the app process. Loading InsightFace / YOLO / SigLIP each
takes a few seconds, so one-shot per-request instantiation would make
the manual-finding UX feel broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageOps

from ..cluster import nearest_named_subject
from ..db import FaceDB
from ..undo import DeleteManualFindingPayload, UndoStore
from .receipts import UndoReceipt

if TYPE_CHECKING:
    from ..detector import FaceDetector
    from ..pet_detector import PetDetector


@dataclass(frozen=True)
class Suggestion:
    """Top-1 subject match for a manual finding's embedding."""

    subject_id: int
    name: str
    similarity_pct: float


@dataclass(frozen=True)
class CreateManualResult:
    """Return shape of ``FindingsService.create_manual``."""

    finding_id: int
    suggestion: Suggestion | None
    receipt: UndoReceipt


class ManualFindingError(ValueError):
    """Raised when a manual-finding request fails validation.

    Routers translate to 422 with ``{"error": <message>}``. Subclasses
    ``ValueError`` so callers that only care about "bad input" keep a
    single ``except`` branch.
    """


_VALID_SPECIES = {"human", "dog", "cat"}
_MIN_BBOX_SIDE = 20


class FindingsService:
    """Coordinator for finding-level creation (FEAT-29)."""

    def __init__(self, db: FaceDB, undo: UndoStore) -> None:
        self._db = db
        self._undo = undo
        # Lazy detector singletons — see module docstring.
        self._face_detector: FaceDetector | None = None
        self._pet_detector: PetDetector | None = None

    def create_manual(
        self,
        source_id: int,
        bbox: tuple[int, int, int, int],
        species: str,
    ) -> CreateManualResult:
        """Insert one manually-drawn finding on a photo source.

        Raises ``ManualFindingError`` on any validation failure (bad
        species, bbox outside source, too-small bbox, video source,
        missing ``subjects`` scan). All DB work happens inside one
        ``db.transaction()`` per ADR-012 §M4.
        """
        if species not in _VALID_SPECIES:
            msg = f"Unsupported species: {species!r}. Expected one of {sorted(_VALID_SPECIES)}."
            raise ManualFindingError(msg)
        x, y, w, h = bbox
        if w < _MIN_BBOX_SIDE or h < _MIN_BBOX_SIDE:
            msg = f"Bbox is too small ({w}×{h}). Minimum side is {_MIN_BBOX_SIDE} pixels."
            raise ManualFindingError(msg)
        source = self._db.get_source(source_id)
        if source is None:
            msg = f"Source {source_id} not found."
            raise ManualFindingError(msg)
        if source.type != "photo":
            msg = (
                "Manual findings on video sources are not supported yet — "
                "use the frame-scrubber in Phase 2."
            )
            raise ManualFindingError(msg)
        if source.width <= 0 or source.height <= 0:
            msg = "Source has no recorded dimensions; re-scan it first."
            raise ManualFindingError(msg)
        if x < 0 or y < 0 or x + w > source.width or y + h > source.height:
            msg = (
                f"Bbox ({x},{y},{w},{h}) falls outside source dimensions "
                f"({source.width}×{source.height})."
            )
            raise ManualFindingError(msg)
        scan_id = self._db.get_latest_scan_id(source_id, "subjects")
        if scan_id is None:
            msg = "Source has no 'subjects' scan — analyse the source first."
            raise ManualFindingError(msg)

        image_path = self._db.resolve_path(source.file_path)
        if not image_path.exists():
            msg = "Source file is missing on disk."
            raise ManualFindingError(msg)
        with Image.open(image_path) as raw:
            oriented = ImageOps.exif_transpose(raw)
            pil_image = oriented.convert("RGB")

        embedding = self._embed(pil_image, bbox, species)

        with self._db.transaction():
            finding_id = self._db.add_manual_finding(
                source_id=source_id,
                bbox=bbox,
                embedding=embedding,
                scan_id=scan_id,
                species=species,
                confidence=1.0,
            )

        match = nearest_named_subject(self._db, embedding, species=species)
        suggestion = (
            Suggestion(subject_id=match[0], name=match[1], similarity_pct=match[2])
            if match is not None
            else None
        )

        message = "Added manual face" if species == "human" else f"Added manual {species}"
        token = self._undo.put(
            description=message,
            payload=DeleteManualFindingPayload(finding_id=finding_id),
        )
        return CreateManualResult(
            finding_id=finding_id,
            suggestion=suggestion,
            receipt=UndoReceipt(token=token, message=message),
        )

    # ── Detector plumbing ──────────────────────────────────────────────

    def _embed(
        self,
        pil_image: Image.Image,
        bbox: tuple[int, int, int, int],
        species: str,
    ) -> np.ndarray:
        """Pick the right model and return a normalized embedding."""
        if species == "human":
            return self._get_face_detector().embed_crop(pil_image, bbox)
        return self._get_pet_detector().embed_crop(pil_image, bbox)

    def _get_face_detector(self) -> FaceDetector:
        if self._face_detector is None:
            # Imported lazily — the model load cost shouldn't hit import time.
            from ..detector import FaceDetector

            self._face_detector = FaceDetector()
        return self._face_detector

    def _get_pet_detector(self) -> PetDetector:
        if self._pet_detector is None:
            from ..pet_detector import PetDetector

            self._pet_detector = PetDetector()
        return self._pet_detector

    # ── Test seam ──────────────────────────────────────────────────────
    #
    # Tests that don't want to load the real models monkey-patch
    # ``_embed`` directly; exposing ``_face_detector`` / ``_pet_detector``
    # as plain attributes lets them pre-seed a stub too.
