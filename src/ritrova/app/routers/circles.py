"""Circles admin + membership (FEAT-27).

Mutating endpoints forward to ``CirclesService`` (ADR-012 §M3).
"""

from __future__ import annotations

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from ..deps import get_circles_service, get_db

router = APIRouter()


@router.post("/api/circles/create")
def create_circle_api(
    name: str = Form(...),
    description: str = Form(""),
) -> RedirectResponse:
    circle_id, _ = get_circles_service().create_circle(
        name, description=description.strip() or None
    )
    return RedirectResponse(f"/circles/{circle_id}", status_code=303)


@router.post("/api/circles/{circle_id}/rename", response_model=None)
def rename_circle_api(
    request: Request, circle_id: int, name: str = Form(...)
) -> RedirectResponse | JSONResponse:
    get_circles_service().rename_circle(circle_id, name)
    if request.headers.get("hx-request") or request.headers.get("accept", "").startswith(
        "application/json"
    ):
        return JSONResponse({"ok": True, "name": name.strip()})
    return RedirectResponse(f"/circles/{circle_id}", status_code=303)


@router.post("/api/circles/{circle_id}/delete", response_model=None)
def delete_circle_api(request: Request, circle_id: int) -> JSONResponse | RedirectResponse:
    try:
        receipt = get_circles_service().delete_circle(circle_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    if request.headers.get("HX-Request"):
        return JSONResponse({"ok": True, "undo_token": receipt.token, "message": receipt.message})
    return RedirectResponse("/circles", status_code=303)


@router.post("/api/subjects/{subject_id}/circles/{circle_id}/add")
def add_subject_to_circle_api(request: Request, subject_id: int, circle_id: int) -> JSONResponse:
    try:
        receipt = get_circles_service().add_subject(circle_id, subject_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    if receipt is None:
        return JSONResponse({"ok": True, "already": True})
    return JSONResponse({"ok": True, "undo_token": receipt.token, "message": receipt.message})


@router.post("/api/subjects/{subject_id}/circles/{circle_id}/remove")
def remove_subject_from_circle_api(
    request: Request, subject_id: int, circle_id: int
) -> JSONResponse:
    try:
        receipt = get_circles_service().remove_subject(circle_id, subject_id)
    except ValueError as e:
        raise HTTPException(404, str(e)) from e
    if receipt is None:
        return JSONResponse({"ok": True, "already": True})
    return JSONResponse({"ok": True, "undo_token": receipt.token, "message": receipt.message})


@router.get("/api/circles/all")
def list_circles_api() -> JSONResponse:
    db = get_db()
    circles = db.list_circles()
    return JSONResponse(
        {"circles": [{"id": c.id, "name": c.name, "member_count": c.member_count} for c in circles]}
    )
