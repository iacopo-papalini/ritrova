"""Global single-slot undo endpoints (FEAT-5)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from ..deps import get_db, get_undo_store

router = APIRouter()


@router.get("/api/undo/peek")
def peek_undo() -> JSONResponse:
    """Return the currently pending undo, if any. Used by the client on
    page load (e.g. after a redirect) to restore the toast."""
    entry = get_undo_store().peek()
    if entry is None:
        return JSONResponse({"pending": False})
    return JSONResponse({"pending": True, "token": entry.token, "message": entry.description})


@router.post("/api/undo/{token}")
def apply_undo(token: str) -> JSONResponse:
    """Consume the pending undo matching ``token`` and invert its effect.

    Returns 404 if the token is missing, already consumed, or past its TTL.
    Single-shot: a successful undo clears the slot.
    """
    entry = get_undo_store().pop(token)
    if entry is None:
        raise HTTPException(404, "Undo token missing, already used, or expired")
    entry.payload.undo(get_db())
    return JSONResponse({"ok": True, "undone": entry.description})
