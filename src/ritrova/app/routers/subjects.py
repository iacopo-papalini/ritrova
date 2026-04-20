"""Subject-level endpoints: CRUD, claim-faces, and the typeahead feed."""

from __future__ import annotations

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...undo import (
    FindingPersonSnapshot,
    RestorePersonIdsPayload,
    ResurrectSubjectPayload,
    SubjectSnapshot,
)
from ..deps import get_db, get_templates, get_undo_store
from ..helpers import (
    describe_findings_reassign,
    describe_subject_delete,
    describe_subject_merge,
    group_by_month,
)

router = APIRouter()


@router.post("/api/subjects/{subject_id}/claim-faces")
def claim_faces(
    subject_id: int,
    face_ids: list[int] = Body(..., embed=True),
    force: bool = Body(False, embed=True),
) -> JSONResponse:
    db = get_db()
    undo_store = get_undo_store()
    prior = db.snapshot_findings_fields(face_ids)
    snapshots = [FindingPersonSnapshot(finding_id=fid, person_id=pid) for fid, pid, _cid in prior]
    try:
        for fid in face_ids:
            db.assign_finding_to_subject(fid, subject_id, force=force)
    except ValueError as e:
        return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
    subject = db.get_subject(subject_id)
    name = subject.name if subject else f"#{subject_id}"
    message = describe_findings_reassign("Claimed", name, len(face_ids))
    token = undo_store.put(
        description=message,
        payload=RestorePersonIdsPayload(snapshots=snapshots),
    )
    return JSONResponse(
        {"ok": True, "claimed": len(face_ids), "undo_token": token, "message": message}
    )


@router.get("/api/subjects/{subject_id}/findings")
def subject_faces_api(subject_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
    db = get_db()
    rows = db.query(
        "SELECT f.id, f.source_id FROM findings f "
        "JOIN finding_assignment fa ON fa.finding_id = f.id "
        "WHERE fa.subject_id = ? LIMIT ? OFFSET ?",
        (subject_id, limit, offset),
    )
    return JSONResponse([{"id": r[0], "source_id": r[1]} for r in rows])


@router.get("/api/subjects/{subject_id}/findings-html", response_class=HTMLResponse)
def subject_faces_html(
    request: Request, subject_id: int, offset: int = 0, limit: int = 200
) -> HTMLResponse:
    db = get_db()
    findings_with_paths = db.get_subject_findings_with_paths(subject_id, limit=limit, offset=offset)
    finding_groups = group_by_month(findings_with_paths, key="findings")
    subject = db.get_subject(subject_id)
    total = subject.face_count if subject else 0
    return get_templates().TemplateResponse(
        name="partials/subject_finding_grid.html",
        context={
            "finding_groups": finding_groups,
            "subject_id": subject_id,
            "offset": offset,
            "limit": limit,
            "total": total,
            "findings_count": len(findings_with_paths),
        },
        request=request,
    )


@router.post("/api/subjects/{subject_id}/rename")
def rename_subject(subject_id: int, name: str = Form(...)) -> RedirectResponse:
    db = get_db()
    subject = db.get_subject(subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    # DB→URL boundary: singular subject.kind ("person"/"pet") ->
    # plural URL kind ("people"/"pets"). Inlined per ADR-012 M0.5.
    kind = "pets" if subject.kind == "pet" else "people"
    db.rename_subject(subject_id, name)
    return RedirectResponse(f"/{kind}/{subject_id}", status_code=303)


@router.post("/api/subjects/merge")
def merge_subjects(source_id: int = Form(...), target_id: int = Form(...)) -> RedirectResponse:
    db = get_db()
    undo_store = get_undo_store()
    if source_id == target_id:
        raise HTTPException(400, "Cannot merge subject with themselves")
    target = db.get_subject(target_id)
    if not target:
        raise HTTPException(404, "Target subject not found")
    # Snapshot the source subject row and all its findings BEFORE merge
    # destroys both. merge_subjects flips person_id source->target on
    # every finding, then DELETEs the source row.
    source_row = db.get_subject_row(source_id)
    if not source_row:
        raise HTTPException(404, "Source subject not found")
    moved_ids = db.get_subject_finding_ids(source_id)
    source_snapshot = SubjectSnapshot(
        id=source_row[0], name=source_row[1], kind=source_row[2], created_at=source_row[3]
    )
    # DB→URL boundary (see rename_subject above).
    kind = "pets" if target.kind == "pet" else "people"
    db.merge_subjects(source_id, target_id)
    undo_store.put(
        description=describe_subject_merge(source_snapshot.name, target.name, len(moved_ids)),
        payload=ResurrectSubjectPayload(subject=source_snapshot, finding_ids=moved_ids),
    )
    return RedirectResponse(f"/{kind}/{target_id}", status_code=303)


@router.post("/api/subjects/{subject_id}/delete")
def delete_subject(subject_id: int) -> RedirectResponse:
    """Unassign all findings and delete the subject."""
    db = get_db()
    undo_store = get_undo_store()
    subject = db.get_subject(subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    # Snapshot the full row + every assigned finding_id BEFORE delete
    # destroys the row and NULLs the person_ids.
    row = db.get_subject_row(subject_id)
    assert row is not None  # get_subject hit means the row exists
    finding_ids = db.get_subject_finding_ids(subject_id)
    snapshot = SubjectSnapshot(id=row[0], name=row[1], kind=row[2], created_at=row[3])
    # DB→URL boundary (see rename_subject above).
    kind = "pets" if subject.kind == "pet" else "people"
    db.delete_subject(subject_id)
    undo_store.put(
        description=describe_subject_delete(snapshot.name, len(finding_ids)),
        payload=ResurrectSubjectPayload(subject=snapshot, finding_ids=finding_ids),
    )
    return RedirectResponse(f"/{kind}", status_code=303)


@router.post("/api/subjects/create")
def create_subject_api(
    name: str = Body(..., embed=True),
    kind: str = Body("person", embed=True),
) -> JSONResponse:
    """Create a subject and return its data. Used by typeahead picker."""
    db = get_db()
    subject_id = db.create_subject(name, kind=kind)
    subject = db.get_subject(subject_id)
    assert subject is not None
    return JSONResponse(
        {
            "id": subject.id,
            "name": subject.name,
            "kind": subject.kind,
            "face_count": subject.face_count,
        }
    )


@router.get("/api/subjects/all")
def all_subjects_api() -> JSONResponse:
    """All subjects (people and pets) for typeahead components, with avatar face ID."""
    db = get_db()
    subjects = db.get_subjects()
    avatars = db.get_random_avatars([s.id for s in subjects])
    return JSONResponse(
        [
            {
                "id": s.id,
                "name": s.name,
                "kind": s.kind,
                "face_count": s.face_count,
                "face_id": avatars.get(s.id),
            }
            for s in subjects
        ]
    )
