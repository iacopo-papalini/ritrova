"""Subject-level endpoints: CRUD, claim-faces, and the typeahead feed.

Mutating endpoints forward to ``SubjectService`` (ADR-012 §M3).
``claim-faces`` translates a ``SpeciesMismatch`` into the HTTP 409
``{error, needs_confirm: true}`` contract — the wire behaviour the
client already depends on.
"""

from __future__ import annotations

from fastapi import APIRouter, Body, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...services_domain import SpeciesMismatch
from ..deps import get_db, get_subject_service, get_templates
from ..helpers import group_by_month

router = APIRouter()


@router.post("/api/subjects/{subject_id}/claim-faces")
def claim_faces(
    subject_id: int,
    face_ids: list[int] = Body(..., embed=True),
    force: bool = Body(False, embed=True),
) -> JSONResponse:
    try:
        receipt = get_subject_service().claim_faces(subject_id, face_ids, force=force)
    except SpeciesMismatch as e:
        return JSONResponse({"error": str(e), "needs_confirm": True}, status_code=409)
    if receipt is None:
        return JSONResponse({"ok": True, "claimed": 0})
    return JSONResponse(
        {
            "ok": True,
            "claimed": len(face_ids),
            "undo_token": receipt.token,
            "message": receipt.message,
        }
    )


@router.get("/api/subjects/{subject_id}/findings")
def subject_faces_api(subject_id: int, offset: int = 0, limit: int = 200) -> JSONResponse:
    db = get_db()
    stubs = db.get_subject_face_stubs(subject_id, limit=limit, offset=offset)
    return JSONResponse([{"id": fid, "source_id": sid} for fid, sid in stubs])


@router.get("/api/subjects/{subject_id}/findings-html", response_class=HTMLResponse)
def subject_faces_html(
    request: Request,
    subject_id: int,
    offset: int = 0,
    limit: int = 200,
    dense: int = 1,
) -> HTMLResponse:
    """HTMX fragment for infinite-scroll pagination of a subject's face grid.

    ``dense=1`` (default) emits one continuous grid with inline month-label
    cells. ``dense=0`` emits the old ``<h3>`` month headers + a fresh grid
    per month — kept for the "Group by month" toggle.
    """
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
            "dense": bool(dense),
        },
        request=request,
    )


@router.post("/api/subjects/{subject_id}/rename", response_model=None)
def rename_subject(
    request: Request, subject_id: int, name: str = Form(...)
) -> RedirectResponse | JSONResponse:
    db = get_db()
    subject = db.get_subject(subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    # DB→URL boundary: singular subject.kind ("person"/"pet") ->
    # plural URL kind ("people"/"pets"). Inlined per ADR-012 M0.5.
    kind = "pets" if subject.kind == "pet" else "people"
    get_subject_service().rename_subject(subject_id, name)
    # HTMX / JSON clients (e.g. inline-rename on subject-detail) get the
    # new name back and patch the DOM — matches the circle-rename contract.
    if request.headers.get("hx-request") or request.headers.get("accept", "").startswith(
        "application/json"
    ):
        return JSONResponse({"ok": True, "name": name.strip()})
    return RedirectResponse(f"/{kind}/{subject_id}", status_code=303)


@router.post("/api/subjects/merge")
def merge_subjects(source_id: int = Form(...), target_id: int = Form(...)) -> RedirectResponse:
    db = get_db()
    # Resolve the target subject for the redirect kind *before* calling
    # the service — the service mutates merge semantics but the target
    # row is unchanged. Errors fall back to ValueError → 404.
    target = db.get_subject(target_id)
    if not target:
        raise HTTPException(404, "Target subject not found")
    try:
        get_subject_service().merge_subjects(source_id, target_id)
    except ValueError as e:
        msg = str(e)
        if "themselves" in msg:
            raise HTTPException(400, msg) from e
        raise HTTPException(404, msg) from e
    kind = "pets" if target.kind == "pet" else "people"
    return RedirectResponse(f"/{kind}/{target_id}", status_code=303)


@router.post("/api/subjects/{subject_id}/delete")
def delete_subject(subject_id: int) -> RedirectResponse:
    """Unassign all findings and delete the subject."""
    db = get_db()
    subject = db.get_subject(subject_id)
    if not subject:
        raise HTTPException(404, "Subject not found")
    kind = "pets" if subject.kind == "pet" else "people"
    get_subject_service().delete_subject(subject_id)
    return RedirectResponse(f"/{kind}", status_code=303)


@router.post("/api/subjects/create")
def create_subject_api(
    name: str = Body(..., embed=True),
    kind: str = Body("person", embed=True),
) -> JSONResponse:
    """Create a subject and return its data. Used by typeahead picker."""
    db = get_db()
    subject_id, _receipt = get_subject_service().create_subject(name, kind=kind)
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
