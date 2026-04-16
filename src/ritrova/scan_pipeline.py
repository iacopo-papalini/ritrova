"""Generic scan pipeline: Template Method for batch scanning.

Each concrete pipeline defines *what* to discover, how to check idempotency,
and how to process a single item.  The loop, progress reporting, and stats
accumulation are shared.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .db import FaceDB

if TYPE_CHECKING:
    from .detector import FaceDetector
    from .pet_detector import PetDetector


@dataclass
class ScanStats:
    """Accumulated counters for a pipeline run."""

    processed: int = 0
    skipped: int = 0
    found: int = 0
    errors: int = 0


class ScanPipeline[T](ABC):
    """Template Method: the bulk-scan loop is fixed; subclasses define the steps.

    The type parameter ``T`` is the candidate type yielded by ``discover``
    (e.g. ``Path`` for file-based scans, ``int`` for source-id-based scans).
    """

    def __init__(self, db: FaceDB) -> None:
        self.db = db

    @abstractmethod
    def discover(self) -> list[T]:
        """Return the ordered list of candidates to process."""

    @abstractmethod
    def is_already_done(self, candidate: T) -> bool:
        """Idempotency gate — return True to skip."""

    @abstractmethod
    def process_one(self, candidate: T) -> int:
        """Process a single candidate.

        Return the count of items found (>= 0 on success, -1 to record an error).
        """

    @property
    def progress_interval(self) -> int:
        """Print progress every N items (override for slow pipelines)."""
        return 100

    @property
    def skip_progress_interval(self) -> int:
        """Print progress every N *skipped* items (can be higher to reduce noise)."""
        return 500

    @property
    def found_label(self) -> str:
        """Label for the ``found`` counter in progress output (e.g. 'faces', 'pets')."""
        return "found"

    def run(self) -> ScanStats:
        """Execute the pipeline. Returns accumulated stats."""
        candidates = self.discover()
        total = len(candidates)
        stats = ScanStats()

        for i, candidate in enumerate(candidates, 1):
            if self.is_already_done(candidate):
                stats.skipped += 1
                if i % self.skip_progress_interval == 0 or i == total:
                    self._report(i, total, stats)
                continue

            result = self.process_one(candidate)
            if result < 0:
                stats.errors += 1
            else:
                stats.found += result
                stats.processed += 1

            if i % self.progress_interval == 0 or i == total:
                self._report(i, total, stats)

        print()
        return stats

    def _report(self, i: int, total: int, stats: ScanStats) -> None:
        print(
            f"\r  [{i}/{total}] processed={stats.processed} skipped={stats.skipped} "
            f"{self.found_label}={stats.found} errors={stats.errors}",
            end="",
            flush=True,
        )


# ── Concrete pipelines ─────────────────────────────────────────────────


class HumanPhotoScan(ScanPipeline[Path]):
    """Scan photos for human faces using ArcFace."""

    def __init__(
        self,
        db: FaceDB,
        photos_dir: Path,
        detector: FaceDetector,
        min_confidence: float = 0.65,
    ) -> None:
        super().__init__(db)
        self.photos_dir = photos_dir
        self.detector = detector
        self.min_confidence = min_confidence

    def discover(self) -> list[Path]:
        from .scanner import find_images

        return find_images(self.photos_dir)

    def is_already_done(self, image_path: Path) -> bool:
        stored = self.db.to_relative(str(image_path.resolve()))
        return self.db.is_scanned(stored, "human")

    def process_one(self, image_path: Path) -> int:
        from .scanner import scan_one_photo_for_human

        success, n = scan_one_photo_for_human(
            self.db, image_path, self.detector, self.min_confidence
        )
        return n if success else -1

    @property
    def found_label(self) -> str:
        return "faces"


class PetPhotoScan(ScanPipeline[Path]):
    """Scan photos for dogs and cats using YOLO + SigLIP."""

    def __init__(
        self,
        db: FaceDB,
        photos_dir: Path,
        pet_detector: PetDetector,
        min_confidence: float = 0.5,
    ) -> None:
        super().__init__(db)
        self.photos_dir = photos_dir
        self.pet_detector = pet_detector
        self.min_confidence = min_confidence

    def discover(self) -> list[Path]:
        from .scanner import find_images

        return find_images(self.photos_dir)

    def is_already_done(self, image_path: Path) -> bool:
        stored = self.db.to_relative(str(image_path.resolve()))
        return self.db.is_scanned(stored, "pet")

    def process_one(self, image_path: Path) -> int:
        from .scanner import scan_one_photo_for_pets

        success, n = scan_one_photo_for_pets(
            self.db, image_path, self.pet_detector, self.min_confidence
        )
        return n if success else -1

    @property
    def progress_interval(self) -> int:
        return 50

    @property
    def found_label(self) -> str:
        return "pets"


class HumanVideoScan(ScanPipeline[Path]):
    """Scan videos for human faces, extracting and deduplicating frames."""

    def __init__(
        self,
        db: FaceDB,
        photos_dir: Path,
        detector: FaceDetector,
        frames_dir: Path,
        min_confidence: float = 0.65,
        interval_sec: float = 2.0,
        dedup_threshold: float = 0.6,
    ) -> None:
        super().__init__(db)
        self.photos_dir = photos_dir
        self.detector = detector
        self.frames_dir = frames_dir
        self.min_confidence = min_confidence
        self.interval_sec = interval_sec
        self.dedup_threshold = dedup_threshold

    def discover(self) -> list[Path]:
        from .scanner import find_videos

        return find_videos(self.photos_dir)

    def is_already_done(self, video_path: Path) -> bool:
        stored = self.db.to_relative(str(video_path.resolve()))
        return self.db.is_scanned(stored, "human")

    def process_one(self, video_path: Path) -> int:
        from .scanner import scan_one_video_for_human

        success, n = scan_one_video_for_human(
            self.db,
            video_path,
            self.detector,
            self.frames_dir,
            self.min_confidence,
            self.interval_sec,
            self.dedup_threshold,
        )
        return n if success else -1

    @property
    def progress_interval(self) -> int:
        return 10

    @property
    def skip_progress_interval(self) -> int:
        return 50

    @property
    def found_label(self) -> str:
        return "faces"
