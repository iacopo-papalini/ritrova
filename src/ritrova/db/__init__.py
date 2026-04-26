"""SQLite database for face recognition data.

Public API — import from ``ritrova.db`` as before::

    from ritrova.db import FaceDB, Finding, Source, Subject
"""

from .circles import Circle
from .connection import FaceDB
from .curation import PruneReport
from .models import (
    Description,
    Finding,
    OrphanReport,
    PrintSelectionItem,
    Source,
    SourcePathMetadata,
    Subject,
)

__all__ = [
    "Circle",
    "Description",
    "FaceDB",
    "Finding",
    "OrphanReport",
    "PrintSelectionItem",
    "PruneReport",
    "Source",
    "SourcePathMetadata",
    "Subject",
]
