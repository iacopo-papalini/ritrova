"""Domain dataclasses for the persistence layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Source:
    id: int
    file_path: str
    type: str
    width: int
    height: int
    taken_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None


@dataclass
class Finding:
    id: int
    source_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    embedding: np.ndarray
    person_id: int | None  # populated via LEFT JOIN finding_assignment.subject_id
    cluster_id: int | None  # populated via LEFT JOIN cluster_findings.cluster_id
    confidence: float
    detected_at: str
    species: str = "human"
    frame_path: str | None = None
    embedding_dim: int = 0
    # 0 only as an in-memory default for tests/fixtures that don't round-trip via SQLite.
    # Real DB rows always have scan_id NOT NULL after the migration.
    scan_id: int = 0
    frame_number: int = 0
    # Apr 2026 refactor: curation state lives on finding_assignment now.
    # exclusion_reason is None when the finding is either uncurated or
    # assigned to a subject; it's 'stranger' or 'not_a_face' when excluded.
    exclusion_reason: str | None = None


@dataclass
class Subject:
    id: int
    name: str
    kind: str = "person"
    face_count: int = 0


@dataclass
class OrphanReport:
    """Dangling-child rows found by ``FaceDB.find_orphans``.

    These exist when someone (sqlite3 CLI, a GUI, a helper script) issued a
    DELETE from a connection without ``PRAGMA foreign_keys=ON`` — cascade
    didn't fire and children survived their parents. The app connection
    enforces FKs, so the app itself can't produce these.
    """

    findings_missing_source: list[int]
    findings_missing_scan: list[int]
    scans_missing_source: list[int]
    dismissed_missing_finding: list[int]

    @property
    def total(self) -> int:
        return (
            len(self.findings_missing_source)
            + len(self.findings_missing_scan)
            + len(self.scans_missing_source)
            + len(self.dismissed_missing_finding)
        )


@dataclass
class Description:
    id: int
    source_id: int
    scan_id: int
    caption: str
    tags: set[str]
    generated_at: str
