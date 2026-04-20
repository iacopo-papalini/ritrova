"""Shared return types for the domain-service layer.

``UndoReceipt`` is what every mutating service method returns — the token
to use against ``/api/undo/pop`` plus the human-facing description. The
ADR-012 §M3 rationale: services own the describe-string generation so
routers never reimplement it and callers get one consistent shape back.

``SpeciesMismatch`` is the typed 409 exception raised when a finding's
species can't legitimately belong to the target subject's kind. Routers
catch it and render the existing wire contract
(``{"error": ..., "needs_confirm": true}`` → 409).
"""

from __future__ import annotations

from dataclasses import dataclass

UndoToken = str
"""A type alias for the token string.

Prefer ``UndoReceipt`` at the service boundary — this alias only exists so
router signatures can say ``UndoToken`` when they deliberately want just
the string (e.g. the 303 redirect path that stores the toast via
``/api/undo/peek`` on the next page load).
"""


@dataclass(frozen=True)
class UndoReceipt:
    """Service-method return value for every undoable mutation.

    ``token`` is what the HTTP client sends to ``/api/undo/pop``.
    ``message`` is the toast string the UI shows immediately.
    """

    token: UndoToken
    message: str


class SpeciesMismatch(ValueError):
    """Raised by ``SubjectService.claim_faces`` / ``ClusterService.assign_cluster``
    when a caller tries to put a ``dog`` finding on a ``person`` subject
    (or vice-versa) without passing ``force=True``.

    Subclasses ``ValueError`` so legacy call sites that previously caught
    ``ValueError`` keep working. Routers check for this specific class
    and translate to the 409 ``{error, needs_confirm: true}`` wire shape.
    """
