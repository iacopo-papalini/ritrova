"""Finding-level mutation endpoints.

Read / image endpoints for findings live in ``images.py`` (thumbnail,
frame, info) and ``pages.py`` (singleton grid partial) — the separation
mirrors who the consumer is, not which DB table.

Every mutating endpoint forwards to the domain-service layer
(``services_domain``) which owns the snapshot+mutate+register-undo
dance — see ADR-012 §M3.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Form
from fastapi.responses import JSONResponse

from ..deps import get_curation_service, get_db, get_subject_service

router = APIRouter()


@router.post("/api/findings/swap")
def swap_findings(
    face_ids: list[int] = Body(...),
    target_subject_id: int = Body(...),
) -> JSONResponse:
    """Move a batch of findings onto a target subject.

    Same undo shape as claim-faces; kept as a separate route so the UI
    can label the toast "Swapped" rather than "Claimed".
    """
    receipt = get_subject_service().swap_findings(face_ids, target_subject_id)
    if receipt is None:
        return JSONResponse({"ok": True, "swapped": 0})
    return JSONResponse(
        {
            "ok": True,
            "swapped": len(face_ids),
            "undo_token": receipt.token,
            "message": receipt.message,
        }
    )


@router.post("/api/findings/mark-stranger")
def mark_findings_stranger(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
    """Flag an ad-hoc batch of findings as strangers.

    Writes ``exclusion_reason='stranger'`` on each one (overwriting any
    prior uncurated state) and drops their cluster_findings rows so they
    vanish from clustering / merge-suggestions / auto-assign. Reversible
    via the returned undo token; prior cluster membership is snapshotted
    inside the service and restored on undo.
    """
    receipt = get_curation_service().mark_strangers(face_ids)
    if receipt is None:
        return JSONResponse({"ok": True, "marked": 0})
    return JSONResponse(
        {
            "ok": True,
            "marked": len(face_ids),
            "undo_token": receipt.token,
            "message": receipt.message,
        }
    )


@router.post("/api/findings/dismiss")
def dismiss_findings(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
    """Mark findings as non-faces (statues, paintings, dogs, etc.)."""
    receipt = get_curation_service().dismiss_findings(face_ids)
    if receipt is None:
        return JSONResponse({"ok": True, "dismissed": 0})
    return JSONResponse(
        {
            "ok": True,
            "dismissed": len(face_ids),
            "undo_token": receipt.token,
            "message": receipt.message,
        }
    )


@router.post("/api/findings/exclude")
def exclude_findings(
    face_ids: list[int] = Body(..., embed=True),
    cluster_id: int = Body(..., embed=True),
) -> JSONResponse:
    receipt = get_curation_service().exclude_findings_from_cluster(cluster_id, face_ids)
    if receipt is None:
        return JSONResponse({"ok": True, "excluded": 0})
    return JSONResponse(
        {
            "ok": True,
            "excluded": len(face_ids),
            "undo_token": receipt.token,
            "message": receipt.message,
        }
    )


@router.post("/api/findings/{finding_id}/assign")
def assign_finding(
    finding_id: int, subject_id: int = Form(...), force: bool = Form(False)
) -> JSONResponse:
    """Assign a single finding to a subject.

    Single-finding / non-undoable path — the client reloads the photo
    page after a success. Mirrors the on-the-wire 409 + ``needs_confirm``
    contract for the cross-species case. Routes straight through FaceDB
    rather than ``SubjectService`` so we don't register a one-finding
    undo that the client isn't prepared to consume.
    """
    try:
        get_db().assign_finding_to_subject(finding_id, subject_id, force=force)
    except ValueError as e:
        return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
    return JSONResponse({"ok": True})


@router.post("/api/findings/unassign")
def unassign_findings_batch(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
    """Unassign multiple findings from their current subjects in one call."""
    receipt = get_curation_service().unassign_findings(face_ids)
    if receipt is None:
        return JSONResponse({"ok": True, "unassigned": 0})
    return JSONResponse(
        {
            "ok": True,
            "unassigned": len(face_ids),
            "undo_token": receipt.token,
            "message": receipt.message,
        }
    )


@router.post("/api/findings/{finding_id}/unassign")
def unassign_finding(finding_id: int) -> JSONResponse:
    """Single-finding unassign.

    Emits the "Removed face from {name}" toast that the photo page
    expects (distinct from the batch "Removed N face(s)" phrasing).
    """
    receipt = get_curation_service().unassign_finding(finding_id)
    if receipt is None:
        return JSONResponse({"ok": True})
    return JSONResponse({"ok": True, "undo_token": receipt.token, "message": receipt.message})
