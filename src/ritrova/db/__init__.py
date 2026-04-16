"""SQLite database for face recognition data.

Public API — import from ``ritrova.db`` as before::

    from ritrova.db import FaceDB, Finding, Source, Subject
"""

from .connection import FaceDB
from .curation import PruneReport
from .models import Description, Finding, OrphanReport, Source, Subject

__all__ = [
    "Description",
    "FaceDB",
    "Finding",
    "OrphanReport",
    "PruneReport",
    "Source",
    "Subject",
]
