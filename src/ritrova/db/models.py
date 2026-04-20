"""Domain dataclasses for the persistence layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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


# ── Curation union ────────────────────────────────────────────────────
#
# The XOR invariant on finding_assignment (CHECK constraint: exactly one of
# `subject_id` or `exclusion_reason` is non-null per row, or the row is
# absent entirely) is enforced at the DB level but lost to the caller as
# two sibling Optionals on Finding. The Curation union reifies it: every
# Finding has exactly one curation state, and callers that pattern-match
# on it get exhaustiveness for free.
#
# Raw `Finding.subject_id` and `Finding.exclusion_reason` fields remain for
# callers that still read them directly; M3 migrates them behind
# Finding.curation as they come up naturally.


@dataclass(frozen=True)
class Uncurated:
    """No finding_assignment row exists for this finding."""


@dataclass(frozen=True)
class AssignedTo:
    """Finding is claimed by a subject (``finding_assignment.subject_id``)."""

    subject_id: int


@dataclass(frozen=True)
class Excluded:
    """Finding is flagged as non-face (``not_a_face``) or not-a-known-person
    (``stranger``). It still exists on the source, but it's hidden from
    clustering, merge-suggestions, and the curation queue."""

    reason: Literal["stranger", "not_a_face"]


Curation = Uncurated | AssignedTo | Excluded


@dataclass
class Finding:
    id: int
    source_id: int
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    embedding: np.ndarray
    subject_id: int | None  # populated via LEFT JOIN finding_assignment.subject_id
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

    @property
    def curation(self) -> Curation:
        """Typed view of the XOR invariant on (subject_id, exclusion_reason).

        Prefer this over reading the raw fields when you want exhaustiveness:
        ``match finding.curation: case AssignedTo(subject_id=sid): ...``.
        """
        if self.subject_id is not None:
            return AssignedTo(subject_id=self.subject_id)
        if self.exclusion_reason is not None:
            return Excluded(reason=self.exclusion_reason)  # type: ignore[arg-type]
        return Uncurated()


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

    @property
    def total(self) -> int:
        return (
            len(self.findings_missing_source)
            + len(self.findings_missing_scan)
            + len(self.scans_missing_source)
        )


@dataclass
class Description:
    id: int
    source_id: int
    scan_id: int
    caption: str
    tags: set[str]
    generated_at: str
