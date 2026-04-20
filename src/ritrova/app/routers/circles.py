"""Circles admin + membership (FEAT-27)."""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from ...undo import (
    AddSubjectToCirclePayload,
    RecreateCirclePayload,
    RemoveSubjectFromCirclePayload,
)
from ..deps import get_db, get_undo_store

router = APIRouter()


@router.post("/api/circles/create")
def create_circle_api(
    name: str = Form(...),
    description: str = Form(""),
) -> RedirectResponse:
    db = get_db()
    circle_id = db.create_circle(name, description=description.strip() or None)
    # No undo for create — delete is the obvious reverse and user has that button.
    return RedirectResponse(f"/circles/{circle_id}", status_code=303)


@router.post("/api/circles/{circle_id}/rename", response_model=None)
def rename_circle_api(
    request: Request, circle_id: int, name: str = Form(...)
) -> RedirectResponse | JSONResponse:
    db = get_db()
    db.rename_circle(circle_id, name)
    if request.headers.get("hx-request") or request.headers.get("accept", "").startswith(
        "application/json"
    ):
        return JSONResponse({"ok": True, "name": name.strip()})
    return RedirectResponse(f"/circles/{circle_id}", status_code=303)


@router.post("/api/circles/{circle_id}/delete", response_model=None)
def delete_circle_api(request: Request, circle_id: int) -> JSONResponse | RedirectResponse:
    db = get_db()
    undo_store = get_undo_store()
    circle = db.get_circle(circle_id)
    if circle is None:
        raise HTTPException(404, "Circle not found")
    member_ids = db.get_circle_subject_ids(circle_id)
    db.delete_circle(circle_id)
    token = undo_store.put(
        description=f"Deleted circle '{circle.name}' ({len(member_ids)} members)",
        payload=RecreateCirclePayload(
            name=circle.name,
            description=circle.description,
            member_subject_ids=member_ids,
        ),
    )
    if request.headers.get("HX-Request"):
        return JSONResponse({"ok": True, "undo_token": token})
    return RedirectResponse("/circles", status_code=303)


@router.post("/api/subjects/{subject_id}/circles/{circle_id}/add")
def add_subject_to_circle_api(request: Request, subject_id: int, circle_id: int) -> JSONResponse:
    db = get_db()
    undo_store = get_undo_store()
    subject = db.get_subject(subject_id)
    circle = db.get_circle(circle_id)
    if subject is None or circle is None:
        raise HTTPException(404)
    added = db.add_subject_to_circle(subject_id, circle_id)
    if not added:
        return JSONResponse({"ok": True, "already": True})
    token = undo_store.put(
        description=f"Added {subject.name} to {circle.name}",
        payload=RemoveSubjectFromCirclePayload(subject_id=subject_id, circle_id=circle_id),
    )
    return JSONResponse({"ok": True, "undo_token": token})


@router.post("/api/subjects/{subject_id}/circles/{circle_id}/remove")
def remove_subject_from_circle_api(
    request: Request, subject_id: int, circle_id: int
) -> JSONResponse:
    db = get_db()
    undo_store = get_undo_store()
    subject = db.get_subject(subject_id)
    circle = db.get_circle(circle_id)
    if subject is None or circle is None:
        raise HTTPException(404)
    removed = db.remove_subject_from_circle(subject_id, circle_id)
    if not removed:
        return JSONResponse({"ok": True, "already": True})
    token = undo_store.put(
        description=f"Removed {subject.name} from {circle.name}",
        payload=AddSubjectToCirclePayload(subject_id=subject_id, circle_id=circle_id),
    )
    return JSONResponse({"ok": True, "undo_token": token})


@router.get("/api/circles/all")
def list_circles_api() -> JSONResponse:
    db = get_db()
    circles = db.list_circles()
    return JSONResponse(
        {"circles": [{"id": c.id, "name": c.name, "member_count": c.member_count} for c in circles]}
    )
