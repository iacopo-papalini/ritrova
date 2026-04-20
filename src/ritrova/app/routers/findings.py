"""Finding-level mutation endpoints.

Read / image endpoints for findings live in ``images.py`` (thumbnail,
frame, info) and ``pages.py`` (singleton grid partial) — the separation
mirrors who the consumer is, not which DB table.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Form
from fastapi.responses import JSONResponse

from ...undo import (
    DismissPayload,
    FindingFieldsSnapshot,
    FindingSubjectSnapshot,
    RestoreClusterPayload,
    RestoreFromStrangerBatchPayload,
    RestoreSubjectIdsPayload,
)
from ..deps import get_db, get_undo_store
from ..helpers import describe_findings_dismiss, describe_findings_reassign

router = APIRouter()


@router.post("/api/findings/swap")
def swap_findings(
    face_ids: list[int] = Body(...),
    target_subject_id: int = Body(...),
) -> JSONResponse:
    db = get_db()
    undo_store = get_undo_store()
    prior = db.snapshot_findings_fields(face_ids)
    snapshots = [FindingSubjectSnapshot(finding_id=fid, subject_id=sid) for fid, sid, _cid in prior]
    for fid in face_ids:
        db.assign_finding_to_subject(fid, target_subject_id)
    subject = db.get_subject(target_subject_id)
    name = subject.name if subject else f"#{target_subject_id}"
    message = describe_findings_reassign("Swapped", name, len(face_ids))
    token = undo_store.put(
        description=message,
        payload=RestoreSubjectIdsPayload(snapshots=snapshots),
    )
    return JSONResponse(
        {"ok": True, "swapped": len(face_ids), "undo_token": token, "message": message}
    )


@router.post("/api/findings/mark-stranger")
def mark_findings_stranger(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
    """Flag an ad-hoc batch of findings as strangers.

    Writes `exclusion_reason='stranger'` on each one (overwriting any
    prior uncurated state) and drops their cluster_findings rows so
    they vanish from clustering / merge-suggestions / auto-assign.
    Reversible: prior cluster membership is snapshotted and restored
    on undo. Findings that had no cluster (singletons) return to the
    uncurated-unclustered state."""
    db = get_db()
    undo_store = get_undo_store()
    if not face_ids:
        return JSONResponse({"ok": True, "marked": 0})
    rows = db.snapshot_findings_fields(face_ids)
    snapshots = [
        FindingFieldsSnapshot(finding_id=fid, subject_id=sid, cluster_id=cid)
        for fid, sid, cid in rows
    ]
    db.set_exclusions(face_ids, "stranger")
    db.remove_cluster_memberships(face_ids)
    n = len(face_ids)
    noun = "face" if n == 1 else "faces"
    message = f"Marked {n} {noun} as stranger"
    token = undo_store.put(
        description=message,
        payload=RestoreFromStrangerBatchPayload(snapshots=snapshots),
    )
    return JSONResponse({"ok": True, "marked": n, "undo_token": token, "message": message})


@router.post("/api/findings/dismiss")
def dismiss_findings(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
    """Mark findings as non-faces (statues, paintings, dogs, etc.)."""
    db = get_db()
    undo_store = get_undo_store()
    if not face_ids:
        return JSONResponse({"ok": True, "dismissed": 0})
    rows = db.snapshot_findings_fields(face_ids)
    snapshots = [
        FindingFieldsSnapshot(finding_id=fid, subject_id=sid, cluster_id=cid)
        for fid, sid, cid in rows
    ]
    db.dismiss_findings(face_ids)
    message = describe_findings_dismiss(len(face_ids))
    token = undo_store.put(
        description=message,
        payload=DismissPayload(snapshots=snapshots),
    )
    return JSONResponse(
        {"ok": True, "dismissed": len(face_ids), "undo_token": token, "message": message}
    )


@router.post("/api/findings/exclude")
def exclude_findings(
    face_ids: list[int] = Body(..., embed=True),
    cluster_id: int = Body(..., embed=True),
) -> JSONResponse:
    db = get_db()
    undo_store = get_undo_store()
    if not face_ids:
        return JSONResponse({"ok": True, "excluded": 0})
    db.exclude_findings(face_ids, cluster_id=cluster_id)
    noun = "face" if len(face_ids) == 1 else "faces"
    message = f"Excluded {len(face_ids)} {noun} from cluster #{cluster_id}"
    token = undo_store.put(
        description=message,
        payload=RestoreClusterPayload(cluster_id=cluster_id, finding_ids=face_ids),
    )
    return JSONResponse(
        {"ok": True, "excluded": len(face_ids), "undo_token": token, "message": message}
    )


@router.post("/api/findings/{finding_id}/assign")
def assign_finding(
    finding_id: int, subject_id: int = Form(...), force: bool = Form(False)
) -> JSONResponse:
    db = get_db()
    try:
        db.assign_finding_to_subject(finding_id, subject_id, force=force)
    except ValueError as e:
        return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
    return JSONResponse({"ok": True})


@router.post("/api/findings/unassign")
def unassign_findings_batch(face_ids: list[int] = Body(..., embed=True)) -> JSONResponse:
    """Unassign multiple findings from their current subjects in one call."""
    db = get_db()
    undo_store = get_undo_store()
    if not face_ids:
        return JSONResponse({"ok": True, "unassigned": 0})
    prior = db.snapshot_findings_fields(face_ids)
    snapshots = [
        FindingSubjectSnapshot(finding_id=fid, subject_id=sid)
        for fid, sid, _cid in prior
        if sid is not None
    ]
    db.unassign_findings(face_ids)
    n = len(face_ids)
    message = f"Removed {n} face{'s' if n != 1 else ''}"
    token = undo_store.put(
        description=message,
        payload=RestoreSubjectIdsPayload(snapshots=snapshots),
    )
    return JSONResponse({"ok": True, "unassigned": n, "undo_token": token, "message": message})


@router.post("/api/findings/{finding_id}/unassign")
def unassign_finding(finding_id: int) -> JSONResponse:
    db = get_db()
    undo_store = get_undo_store()
    prior_subject_id = db.get_finding_subject_id(finding_id)
    db.unassign_finding(finding_id)
    if prior_subject_id is not None:
        subject = db.get_subject(prior_subject_id)
        name = subject.name if subject else f"#{prior_subject_id}"
        message = f"Removed face from {name}"
        token = undo_store.put(
            description=message,
            payload=RestoreSubjectIdsPayload(
                snapshots=[
                    FindingSubjectSnapshot(finding_id=finding_id, subject_id=prior_subject_id)
                ]
            ),
        )
        return JSONResponse({"ok": True, "undo_token": token, "message": message})
    return JSONResponse({"ok": True})
