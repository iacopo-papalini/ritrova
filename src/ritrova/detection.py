"""Shared types for detection results.

Both ``FaceDetector`` and ``PetDetector`` return a ``DetectionResult``
containing a list of ``Detection`` objects and the source image dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Detection:
    """A single detected face or pet in an image."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h)
    embedding: np.ndarray
    confidence: float
    species: str = "human"


@dataclass(slots=True)
class DetectionResult:
    """Output of a detector: detections plus source image dimensions."""

    detections: list[Detection]
    width: int
    height: int
