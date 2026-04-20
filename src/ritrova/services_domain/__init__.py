"""Domain-service layer above ``FaceDB`` (ADR-012 §M3).

Each service is a thin coordinator over an aggregate (findings, subjects,
clusters, circles). Mutating methods perform the full
snapshot-then-mutate-then-register-undo dance internally and return an
``UndoReceipt`` so routers can forward it verbatim to the HTTP client.

The rules (ADR-012 §M3 non-negotiables):

* Each service file ≤ 200 lines.
* Every mutating method returns an ``UndoReceipt`` (token + human-facing
  description) or, where undo doesn't apply, ``None``.
* Services never wrap SQL in a repository pattern — ``FaceDB`` remains
  the SQL layer. A service method is one of: (a) read DB → return DTO,
  (b) mutate DB → return ``UndoReceipt``.
* No caching, no ORM, no query builders.
* The snapshot + mutate split stays two lock-acquisitions for now; M4
  collapses them into a single transaction.
"""

from __future__ import annotations

from .circles_service import CirclesService
from .cluster_service import ClusterService
from .curation_service import CurationService
from .receipts import SpeciesMismatch, UndoReceipt, UndoToken
from .subject_service import SubjectService

__all__ = [
    "CirclesService",
    "ClusterService",
    "CurationService",
    "SpeciesMismatch",
    "SubjectService",
    "UndoReceipt",
    "UndoToken",
]
